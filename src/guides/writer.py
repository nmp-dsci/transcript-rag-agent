"""Write and revise field guides with the Claude Agent SDK.

The contract, from the plan:

* **Stateless.** Every stage is a fresh ``query()``; nothing is resumed. All
  the state a stage needs is a file under ``guides/<slug>/`` — the exported
  corpus, the evidence JSON, the current page — so a killed run restarts from
  its last completed stage and a revision is reproducible from the repo.
* **Corpus-only.** The agent gets Read/Write/Edit/Glob/Grep, confined by a
  permission callback to the guide directory, plus one in-process
  ``retrieve_chunks`` tool over the hybrid retriever. Web and shell tools are
  disallowed; ``allow_web`` adds the two web tools for one run and records it.
* **Verified before published.** The verifier resolves every cite with no
  LLM; a page that fails gets a bounded fix pass, then the run errors.

The SDK call itself sits behind ``agent_fn`` so every stage is testable with
a fake that writes canned files.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.guides.catalog import (
    GuidePaths,
    Manifest,
    list_versions,
    read_manifest,
    write_index,
    write_manifest,
)
from src.guides.export import ChunksFn, ExportSummary, export_corpus, load_export
from src.guides.prompts import (
    composer_prompt,
    composer_system_prompt,
    extractor_prompt,
    extractor_system_prompt,
    fix_prompt,
    load_skill,
    markdown_prompt,
    reviser_prompt,
    reviser_system_prompt,
)
from src.guides.scope import cluster_videos
from src.guides.tools import (
    QUALIFIED_TOOL,
    RetrievalLog,
    RetrieveChunksFn,
    build_retrieval_server,
    describe_tool_call,
)
from src.guides.verify import ChunkLookup, Report, verify_html, write_claims

DEFAULT_EXTRACT_MODEL = "claude-sonnet-5"
DEFAULT_COMPOSE_MODEL = "claude-opus-5"

FILE_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
WEB_TOOLS = ["WebSearch", "WebFetch"]
#: Never available to a guide agent, whatever the run says: the shell and
#: anything that could reach outside the guide directory or spawn more agents.
ALWAYS_DISALLOWED = ["Bash", "Task", "NotebookEdit", "KillShell", "BashOutput", "TodoWrite"]

OnEvent = Callable[[dict[str, Any]], None]


@dataclass
class AgentRequest:
    label: str
    prompt: str
    system_prompt: str
    model: str
    cwd: Path
    allowed_tools: list[str]
    disallowed_tools: list[str]
    default_video_ids: list[str] | None
    max_turns: int = 200


@dataclass
class AgentResult:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float | None = None
    duration_ms: int | None = None
    num_turns: int = 0
    error: str | None = None


#: ``agent_fn(request, retrieve, on_event) -> result``. The SDK implementation
#: is :func:`run_with_sdk`; tests inject a fake that writes files.
AgentFn = Callable[[AgentRequest, RetrieveChunksFn, OnEvent], Awaitable[AgentResult]]


@dataclass
class WriterConfig:
    extract_model: str = DEFAULT_EXTRACT_MODEL
    compose_model: str = DEFAULT_COMPOSE_MODEL
    allow_web: bool = False
    max_fix_passes: int = 2
    max_clusters: int = 6
    parallel_extract: bool = True
    example_path: Path | None = None
    #: A stage that runs longer than this is abandoned (the SDK session is
    #: closed) and the run errors; the next run resumes from the last file.
    stage_timeout_seconds: float = 2400.0
    #: The subscription's rate limit surfaces as ``rate_limit``; one retry after
    #: a pause turns a transient refusal into a finished stage.
    rate_limit_retries: int = 1
    rate_limit_pause_seconds: float = 120.0


def _inside(root: Path, candidate: str) -> bool:
    try:
        target = (
            (root / candidate).resolve()
            if not Path(candidate).is_absolute()
            else Path(candidate).resolve()
        )
    except OSError:
        return False
    return target == root.resolve() or root.resolve() in target.parents


def make_confinement_hook(root: Path) -> Any:
    """A ``PreToolUse`` hook that keeps every file tool inside the guide directory.

    A hook rather than ``can_use_tool``: an ``allowed_tools`` entry that names a
    whole tool auto-approves it before the permission callback is consulted,
    so the callback would never see a Read outside the directory. Hooks run on
    every call regardless.
    """

    async def confine(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        tool_name = str(hook_input.get("tool_name", ""))
        tool_input = hook_input.get("tool_input") or {}
        if tool_name in FILE_TOOLS:
            for key in ("file_path", "path", "notebook_path", "pattern"):
                value = tool_input.get(key)
                if isinstance(value, str) and value and not _inside(root, value):
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"{tool_name} is confined to the guide directory; refused {value}"
                            ),
                        }
                    }
        return {}

    return confine


def sdk_env() -> dict[str, str]:
    """Bill the run to the subscription, never to a pay-as-you-go key.

    The CLI prefers ``ANTHROPIC_API_KEY`` when both are set, and this machine's
    key is a different account with no credit — so when an OAuth token exists
    the API key is blanked for the subprocess only.
    """
    import logging
    import os

    env: dict[str, str] = {}
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = ""
    elif os.environ.get("ANTHROPIC_API_KEY"):
        logging.getLogger(__name__).warning(
            "CLAUDE_CODE_OAUTH_TOKEN is not set; this run will bill ANTHROPIC_API_KEY's "
            "pay-as-you-go account instead of the subscription"
        )
    return env


async def run_with_sdk(
    request: AgentRequest, retrieve: RetrieveChunksFn, on_event: OnEvent
) -> AgentResult:
    """One fresh SDK session. No resume, no continue — see the module docstring."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    log = RetrievalLog()
    server = build_retrieval_server(
        retrieve,
        default_video_ids=request.default_video_ids,
        log=log,
        on_call=lambda call: on_event({"type": "tool", "label": request.label, **call}),
    )
    options = ClaudeAgentOptions(
        model=request.model,
        system_prompt=request.system_prompt,
        cwd=str(request.cwd),
        allowed_tools=list(request.allowed_tools),
        disallowed_tools=list(request.disallowed_tools),
        mcp_servers={"corpus": server},
        permission_mode="bypassPermissions",
        hooks={"PreToolUse": [HookMatcher(hooks=[make_confinement_hook(request.cwd)])]},
        max_turns=request.max_turns,
        continue_conversation=False,
        resume=None,
        setting_sources=[],
        env=sdk_env(),
    )
    result = AgentResult()
    chunks: list[str] = []
    # The client (not ``query()``) because the permission guard needs the
    # streaming transport. One client per stage, never reconnected: stateless.
    async with ClaudeSDKClient(options=options) as client:
        await client.query(request.prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                if message.error:
                    result.error = message.error
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        call = {
                            "name": block.name,
                            "summary": describe_tool_call(
                                block.name, block.input, str(request.cwd)
                            ),
                        }
                        result.tool_calls.append(call)
                        on_event({"type": "tool", "label": request.label, **call})
            elif isinstance(message, ResultMessage):
                result.cost_usd = message.total_cost_usd
                result.duration_ms = message.duration_ms
                result.num_turns = message.num_turns
                if message.is_error:
                    result.error = result.error or (
                        "; ".join(message.errors) if message.errors else message.subtype
                    )
    result.text = "".join(chunks)
    result.tool_calls.extend({"name": QUALIFIED_TOOL, **call} for call in log.calls)
    return result


class GuideWriter:
    """Runs the stages for one guide directory. One instance per run."""

    def __init__(
        self,
        paths: GuidePaths,
        *,
        chunks_for: ChunksFn,
        retrieve: RetrieveChunksFn,
        known_videos: set[str],
        chunk_lookup: ChunkLookup,
        config: WriterConfig | None = None,
        agent_fn: AgentFn | None = None,
        on_event: OnEvent | None = None,
        skill_text: str | None = None,
    ) -> None:
        self.paths = paths
        self.chunks_for = chunks_for
        self.retrieve = retrieve
        self.known_videos = known_videos
        self.chunk_lookup = chunk_lookup
        self.config = config or WriterConfig()
        self.agent_fn = agent_fn or run_with_sdk
        self._on_event = on_event or (lambda event: None)
        self.skill = skill_text if skill_text is not None else load_skill()
        self.run_log: Path | None = None
        self.costs: list[dict[str, Any]] = []

    # -- events ------------------------------------------------------------
    def emit(self, stage: str, status: str, message: str = "", **data: Any) -> None:
        event = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "stage": stage,
            "status": status,
            "message": message,
            **data,
        }
        self._record(event)
        self._on_event(event)

    def _record(self, event: dict[str, Any]) -> None:
        if self.run_log is None:
            return
        with self.run_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _tool_event(self, event: dict[str, Any]) -> None:
        stamped = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "stage": event.get("label", "agent"),
            "status": "tool",
            "message": event.get("summary") or event.get("question") or event.get("name", ""),
            **{k: v for k, v in event.items() if k not in ("type", "label")},
        }
        self._record(stamped)
        self._on_event(stamped)

    def _open_run_log(self, kind: str) -> None:
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_log = self.paths.runs / f"{stamp}-{kind}.jsonl"

    # -- tool lists ----------------------------------------------------------
    def _tools(self, *, edit: bool) -> tuple[list[str], list[str]]:
        allowed = ["Read", "Glob", "Grep", "Write"] + (["Edit"] if edit else []) + [QUALIFIED_TOOL]
        disallowed = list(ALWAYS_DISALLOWED)
        if self.config.allow_web:
            allowed += WEB_TOOLS
        else:
            disallowed += WEB_TOOLS
        return allowed, disallowed

    async def _run(self, request: AgentRequest) -> AgentResult:
        attempts = 0
        while True:
            attempts += 1
            try:
                result = await asyncio.wait_for(
                    self.agent_fn(request, self.retrieve, self._tool_event),
                    timeout=self.config.stage_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"{request.label}: no result after {self.config.stage_timeout_seconds:.0f}s"
                ) from exc
            if result.error == "rate_limit" and attempts <= self.config.rate_limit_retries:
                self.emit(
                    request.label,
                    "progress",
                    f"rate limited; retrying in {self.config.rate_limit_pause_seconds:.0f}s",
                    attempt=attempts,
                )
                await asyncio.sleep(self.config.rate_limit_pause_seconds)
                continue
            break
        self.costs.append(
            {
                "label": request.label,
                "model": request.model,
                "cost_usd": result.cost_usd,
                "duration_ms": result.duration_ms,
                "turns": result.num_turns,
                "tool_calls": len(result.tool_calls),
            }
        )
        if result.error:
            raise RuntimeError(f"{request.label}: {result.error}")
        return result

    # -- stages ----------------------------------------------------------------
    def stage_export(
        self, video_ids: list[str], videos_meta: list[dict[str, Any]]
    ) -> ExportSummary:
        self.emit("export", "start", f"exporting {len(video_ids)} videos")
        meta = {str(v.get("video_id")): v for v in videos_meta}
        ordered = [meta.get(vid, {"video_id": vid}) for vid in video_ids]
        clusters = cluster_videos(ordered, max_clusters=self.config.max_clusters)
        summary = export_corpus(self.paths, video_ids, self.chunks_for, clusters=clusters)
        self.emit(
            "export",
            "done",
            f"{summary.chunk_count} chunks across {len(video_ids)} videos in {len(clusters)} clusters",
            chunk_count=summary.chunk_count,
            clusters=len(clusters),
            videos=[vars(v) for v in summary.videos],
        )
        return summary

    async def stage_extract(self, topic: str, summary: ExportSummary) -> list[Path]:
        self.paths.evidence.mkdir(parents=True, exist_ok=True)
        allowed, disallowed = self._tools(edit=False)
        system = extractor_system_prompt(self.skill, allow_web=self.config.allow_web)
        requests: list[tuple[str, AgentRequest, Path]] = []
        for number, cluster in enumerate(summary.clusters, 1):
            name = f"cluster-{number}"
            output = self.paths.evidence / f"{name}.json"
            if output.is_file() and self._evidence_matches(output, cluster):
                self.emit("extract", "skip", f"{name}: evidence exists", cluster=name)
                continue
            requests.append(
                (
                    name,
                    AgentRequest(
                        label=f"extract:{name}",
                        prompt=extractor_prompt(
                            topic=topic,
                            cluster_name=name,
                            files=[f"corpus/{vid}.md" for vid in cluster],
                            video_ids=list(cluster),
                            output_path=f"evidence/{name}.json",
                        ),
                        system_prompt=system,
                        model=self.config.extract_model,
                        cwd=self.paths.dir,
                        allowed_tools=allowed,
                        disallowed_tools=disallowed,
                        default_video_ids=list(cluster),
                    ),
                    output,
                )
            )
        self.emit(
            "extract",
            "start",
            f"{len(requests)} extraction passes ({self.config.extract_model})",
            total=len(summary.clusters),
            pending=len(requests),
        )

        cluster_videos_by_name = {
            f"cluster-{n}": list(c) for n, c in enumerate(summary.clusters, 1)
        }

        async def one(name: str, request: AgentRequest, output: Path) -> None:
            self.emit("extract", "progress", f"{name}: reading", cluster=name)
            await self._run(request)
            if not output.is_file():
                raise RuntimeError(f"{name}: the extractor wrote no {output.name}")
            claims = self._verify_evidence(output, cluster_videos_by_name[name])
            self.emit(
                "extract",
                "progress",
                f"{name}: {claims} verified claims",
                cluster=name,
                claims=claims,
            )

        if self.config.parallel_extract:
            await asyncio.gather(*(one(*item) for item in requests))
        else:
            for item in requests:
                await one(*item)
        files = sorted(self.paths.evidence.glob("cluster-*.json"))
        total = sum(self._count_claims(path) for path in files)
        self.emit(
            "extract",
            "done",
            f"{total} verified claims in {len(files)} evidence files",
            claims=total,
        )
        return files

    @staticmethod
    def _evidence_matches(path: Path, cluster: list[str]) -> bool:
        """Only skip re-extraction when the cached evidence covers this exact cluster."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return list(data.get("videos") or []) == list(cluster)

    def _verify_evidence(self, path: Path, video_ids: list[str] | None = None) -> int:
        """Mark each claim ``verified`` (chunk exists and quote occurs). Returns the count."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"{path.name} is not valid JSON: {exc}") from exc
        if video_ids is not None:
            data["videos"] = list(video_ids)
        cache: dict[str, dict[int, str]] = {}
        verified = 0
        for claim in data.get("claims", []):
            video_id = str(claim.get("video_id") or "")
            if video_id not in cache:
                cache[video_id] = {
                    int(r["chunk_index"]): str(r.get("text") or "")
                    for r in (self.chunk_lookup(video_id) if video_id in self.known_videos else [])
                }
            try:
                index = int(claim.get("chunk_index"))
            except (TypeError, ValueError):
                index = -1
            text = cache[video_id].get(index)
            quote = " ".join(str(claim.get("quote") or "").split()).lower()
            ok = text is not None and (not quote or quote in " ".join(text.split()).lower())
            claim["verified"] = bool(ok)
            verified += int(bool(ok))
        data["verified_claims"] = verified
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return verified

    @staticmethod
    def _count_claims(path: Path) -> int:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return 0
        return int(data.get("verified_claims") or 0)

    def _gaps(self) -> list[str]:
        gaps: list[str] = []
        for path in sorted(self.paths.evidence.glob("cluster-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for gap in data.get("gaps", []) or []:
                if isinstance(gap, str) and gap.strip() and gap not in gaps:
                    gaps.append(gap.strip())
        return gaps

    async def stage_compose(
        self,
        *,
        topic: str,
        title: str,
        summary: ExportSummary,
        evidence_files: list[Path],
        compiled_at: str,
    ) -> None:
        self.emit("compose", "start", f"composing with {self.config.compose_model}")
        allowed, disallowed = self._tools(edit=True)
        sources = [
            {"video_id": v.video_id, "title": v.title, "channel": v.channel_name}
            for v in summary.videos
        ]
        example = None
        if self.config.example_path and self.config.example_path.is_file():
            example = str(self.config.example_path)
        request = AgentRequest(
            label="compose",
            prompt=composer_prompt(
                topic=topic,
                title=title,
                slug=self.paths.slug,
                evidence_files=[f"evidence/{p.name}" for p in evidence_files],
                corpus_files=[v.path for v in summary.videos],
                video_count=len(summary.videos),
                chunk_count=summary.chunk_count,
                cluster_count=len(summary.clusters),
                compiled_at=compiled_at,
                sources=sources,
                example_path=example,
            ),
            system_prompt=composer_system_prompt(self.skill, allow_web=self.config.allow_web),
            model=self.config.compose_model,
            cwd=self.paths.dir,
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            default_video_ids=[v.video_id for v in summary.videos],
        )
        await self._run(request)
        if not self.paths.html.is_file():
            raise RuntimeError("compose: the composer wrote no guide.html")
        self.emit("compose", "done", "guide.html written", markdown=self.paths.markdown.is_file())

    def verify(self) -> Report:
        html = self.paths.html.read_text(encoding="utf-8")
        report = verify_html(html, known_videos=self.known_videos, chunk_lookup=self.chunk_lookup)
        write_claims(self.paths.claims, report)
        return report

    async def stage_verify(self, default_video_ids: list[str]) -> Report:
        self.emit("verify", "start", "resolving cites")
        report = self.verify()
        passes = 0
        while not report.ok and passes < self.config.max_fix_passes:
            passes += 1
            self.emit(
                "verify",
                "progress",
                f"{len(report.invalid)} invalid cites, {len(report.structure_errors)} structure errors — fix pass {passes}",
                invalid=len(report.invalid),
                structure_errors=list(report.structure_errors),
            )
            allowed, disallowed = self._tools(edit=True)
            request = AgentRequest(
                label=f"fix:{passes}",
                prompt=fix_prompt(
                    structure_errors=list(report.structure_errors),
                    invalid=[claim.to_dict() for claim in report.invalid],
                ),
                system_prompt=composer_system_prompt(self.skill, allow_web=self.config.allow_web),
                model=self.config.compose_model,
                cwd=self.paths.dir,
                allowed_tools=allowed,
                disallowed_tools=disallowed,
                default_video_ids=default_video_ids,
            )
            await self._run(request)
            report = self.verify()
        if not report.ok:
            self.emit(
                "verify",
                "error",
                f"still failing after {passes} fix passes: {len(report.invalid)} invalid cites",
                invalid=len(report.invalid),
            )
            raise RuntimeError(
                f"verify: {len(report.invalid)} cites do not resolve and "
                f"{len(report.structure_errors)} structure errors remain after {passes} fix passes"
            )
        self.emit(
            "verify",
            "done",
            f"cites {report.valid}/{report.total} resolve",
            valid=report.valid,
            total=report.total,
            fix_passes=passes,
        )
        return report

    def stage_publish(
        self,
        *,
        topic: str,
        title: str,
        summary: ExportSummary,
        report: Report,
        compiled_at: str,
        version: int,
        manifest: Manifest | None = None,
    ) -> Manifest:
        self.emit("publish", "start", f"publishing v{version}")
        from src.guides.normalize import normalize_page

        page = self.paths.html.read_text(encoding="utf-8")
        normalized = normalize_page(page, title=title)
        if normalized != page:
            self.paths.html.write_text(normalized, encoding="utf-8")
            self.emit("publish", "progress", "normalized page chrome (wrap / nav / bridge)")
        self.paths.versions.mkdir(parents=True, exist_ok=True)
        version_path = self.paths.version_html(version)
        if version_path.is_file():
            raise RuntimeError(f"{version_path} already exists; versions are immutable")
        shutil.copyfile(self.paths.html, version_path)
        subtitle = ""
        if manifest is None:
            from src.guides.importer import extract_subtitle

            subtitle = extract_subtitle(self.paths.html.read_text(encoding="utf-8"))
        result = manifest or Manifest(slug=self.paths.slug, title=title, topic=topic)
        result.title = title
        result.topic = topic
        if subtitle:
            result.subtitle = subtitle
        result.status = "published"
        result.current_version = version
        result.compiled_at = compiled_at
        result.model = {
            "extractor": self.config.extract_model,
            "composer": self.config.compose_model,
            "reviser": self.config.compose_model,
        }
        result.video_ids = [v.video_id for v in summary.videos]
        result.sources = [
            {
                "video_id": v.video_id,
                "title": v.title,
                "channel": v.channel_name,
                "contributed": self._contributed(v.video_id),
                "url": f"https://www.youtube.com/watch?v={v.video_id}",
            }
            for v in summary.videos
        ]
        result.chunk_count = summary.chunk_count
        result.cluster_count = len(summary.clusters)
        result.cite_total = report.total
        result.cite_valid = report.valid
        result.gaps = self._gaps()
        result.web_allowed = self.config.allow_web
        result.provenance = {
            **result.provenance,
            "pipeline": "guides write",
            "skill": "field-guide-writer",
            "cite_level": "chunk",
            "sections": report.sections,
            "runs": [
                *(result.provenance.get("runs") or []),
                self.run_log.name if self.run_log else None,
            ],
            "costs": [*(result.provenance.get("costs") or []), *self.costs],
        }
        write_manifest(self.paths, result)
        write_index(self.paths.root)
        self.emit("publish", "done", f"v{version} published", version=version)
        return result

    def _contributed(self, video_id: str) -> str:
        for path in sorted(self.paths.evidence.glob("cluster-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for note in data.get("video_notes", []) or []:
                if note.get("video_id") == video_id and note.get("contributed"):
                    return str(note["contributed"])
        return ""

    async def stage_markdown(self, *, title: str, video_ids: list[str]) -> None:
        """Write ``guide.md`` from the published page (the agent-facing copy)."""
        self.emit("markdown", "start", f"writing guide.md with {self.config.compose_model}")
        allowed, disallowed = self._tools(edit=False)
        request = AgentRequest(
            label="markdown",
            prompt=markdown_prompt(title=title),
            system_prompt=composer_system_prompt(self.skill, allow_web=False),
            model=self.config.compose_model,
            cwd=self.paths.dir,
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            default_video_ids=video_ids,
            max_turns=40,
        )
        await self._run(request)
        if not self.paths.markdown.is_file():
            raise RuntimeError("markdown: the agent wrote no guide.md")
        self.emit("markdown", "done", "guide.md written")

    def write_markdown(self, *, title: str, video_ids: list[str]) -> None:
        self._open_run_log("markdown")
        asyncio.run(self.stage_markdown(title=title, video_ids=video_ids))

    # -- entry points --------------------------------------------------------------
    async def write_async(
        self,
        *,
        topic: str,
        title: str,
        video_ids: list[str],
        videos_meta: list[dict[str, Any]],
        compiled_at: str | None = None,
    ) -> Manifest:
        compiled_at = compiled_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        self.paths.dir.mkdir(parents=True, exist_ok=True)
        self._open_run_log("write")
        self.emit("scope", "done", f"{len(video_ids)} videos confirmed", video_ids=list(video_ids))
        try:
            summary = load_export(self.paths)
            if summary is None or [v.video_id for v in summary.videos] != list(video_ids):
                summary = self.stage_export(video_ids, videos_meta)
            else:
                self.emit(
                    "export", "skip", "corpus already exported", chunk_count=summary.chunk_count
                )
            evidence = await self.stage_extract(topic, summary)
            if not self.paths.html.is_file():
                await self.stage_compose(
                    topic=topic,
                    title=title,
                    summary=summary,
                    evidence_files=evidence,
                    compiled_at=compiled_at,
                )
            else:
                self.emit("compose", "skip", "guide.html already written")
            report = await self.stage_verify([v.video_id for v in summary.videos])
            versions = list_versions(self.paths)
            version = (max(versions) + 1) if versions else 1
            return self.stage_publish(
                topic=topic,
                title=title,
                summary=summary,
                report=report,
                compiled_at=compiled_at,
                version=version,
                manifest=read_manifest(self.paths),
            )
        except Exception as exc:
            self.emit("run", "error", str(exc))
            raise

    def write(self, **kwargs: Any) -> Manifest:
        return asyncio.run(self.write_async(**kwargs))

    async def revise_async(self, *, comment_ids: list[str] | None = None) -> Manifest:
        """Apply the open comments as one tracked batch and publish the next version.

        Stateless like every other stage: the reviser gets the current page,
        the claims, the evidence files and the comments as files and text, and
        must return a receipt that names every comment exactly once. A receipt
        that does not is a failed run, not a partial one.
        """
        from src.guides.comments import (
            ReceiptError,
            open_comments,
            parse_receipt,
            resolve_comments,
            write_receipt,
        )

        manifest = read_manifest(self.paths)
        if manifest is None:
            raise RuntimeError(f"revise: guides/{self.paths.slug} has no manifest")
        pending = open_comments(self.paths)
        if comment_ids:
            wanted = set(comment_ids)
            pending = [c for c in pending if c["id"] in wanted]
        if not pending:
            raise RuntimeError("revise: no open comments to apply")
        self._open_run_log("revise")
        versions = list_versions(self.paths)
        next_version = (max(versions) + 1) if versions else manifest.current_version + 1
        receipt_path = self.paths.dir / "receipt.json"
        if receipt_path.exists():
            receipt_path.unlink()
        self.emit(
            "revise",
            "start",
            f"applying {len(pending)} comments with {self.config.compose_model}",
            comments=[c["id"] for c in pending],
            version=next_version,
        )
        try:
            evidence = sorted(self.paths.evidence.glob("cluster-*.json"))
            corpus = [
                f"corpus/{vid}.md"
                for vid in manifest.video_ids
                if (self.paths.corpus / f"{vid}.md").is_file()
            ]
            allowed, disallowed = self._tools(edit=True)
            request = AgentRequest(
                label="revise",
                prompt=reviser_prompt(
                    title=manifest.title,
                    comments=pending,
                    evidence_files=[f"evidence/{p.name}" for p in evidence],
                    corpus_files=corpus,
                    next_version=next_version,
                ),
                system_prompt=reviser_system_prompt(self.skill, allow_web=self.config.allow_web),
                model=self.config.compose_model,
                cwd=self.paths.dir,
                allowed_tools=allowed,
                disallowed_tools=disallowed,
                default_video_ids=list(manifest.video_ids),
            )
            await self._run(request)
            if not receipt_path.is_file():
                raise RuntimeError("revise: the reviser wrote no receipt.json")
            try:
                receipt = parse_receipt(
                    json.loads(receipt_path.read_text(encoding="utf-8")),
                    expected_ids=[c["id"] for c in pending],
                    version=next_version,
                )
            except (ValueError, ReceiptError) as exc:
                raise RuntimeError(f"revise: {exc}") from exc
            outcomes = {
                o: sum(1 for i in receipt.items if i.outcome == o)
                for o in ("addressed", "deferred", "rejected")
            }
            self.emit(
                "revise",
                "done",
                "receipt complete: " + ", ".join(f"{n} {k}" for k, n in outcomes.items() if n),
                **outcomes,
            )
            report = await self.stage_verify(list(manifest.video_ids))
            summary = load_export(self.paths) or ExportSummary()
            if not summary.videos:
                # A guide imported without an export: keep its manifest's sources.
                from src.guides.export import ExportedVideo

                summary.videos = [
                    ExportedVideo(
                        video_id=src["video_id"],
                        title=str(src.get("title") or src["video_id"]),
                        channel_name=str(src.get("channel") or src.get("channel_name") or ""),
                        path="",
                        chunk_count=0,
                    )
                    for src in manifest.sources
                ]
            published = self.stage_publish(
                topic=manifest.topic,
                title=manifest.title,
                summary=summary,
                report=report,
                compiled_at=manifest.compiled_at,
                version=next_version,
                manifest=manifest,
            )
            if manifest.chunk_count and not summary.chunk_count:
                published.chunk_count = manifest.chunk_count
                published.cluster_count = manifest.cluster_count
                write_manifest(self.paths, published)
            write_receipt(self.paths, receipt)
            resolve_comments(self.paths, receipt)
            receipt_path.unlink(missing_ok=True)
            self.emit(
                "publish", "progress", f"{len(receipt.items)} comments resolved in v{next_version}"
            )
            return published
        except Exception as exc:
            self.emit("run", "error", str(exc))
            raise

    def revise(self, **kwargs: Any) -> Manifest:
        return asyncio.run(self.revise_async(**kwargs))
