"""The live half of the held-out critique eval — stores, retrieval, one call.

:mod:`src.evals.critique` is pure arithmetic over findings. This module is what
produces those findings: it opens the stores with the held-out video excluded,
retrieves the criteria the artifact should be judged against, asks the answering
model for structured findings, scores them and writes the run.

Two things here are worth reading rather than skimming.

**Exclusion is set at construction, not per call.** Both stores are built with
``exclude_video_ids``, so every read path they own — the vector query, the BM25
``collection.get`` the context provider issues directly, the summary router that
decides which videos get searched at all, and neighbour expansion — is filtered
without any caller having to remember. The alternative, an argument threaded
through four call chains, is the version where one path forgets and the
experiment quietly measures nothing.

**The proof does not trust the filter.** After the run,
:func:`~src.evals.critique.held_out_leaks` re-scans every retrieved chunk id,
every video id and every citation for the held-out video by prefix, and the count
goes in the run file. A filter that stopped applying looks exactly like a filter
that is working, so the number a reviewer reads is produced by a check that would
still fail if every ``$nin`` in the stack were deleted.

Cost: **two LLM calls per setup** — one to produce the critique, one to pair its
findings against the held-out criteria. No RAGAS judge runs at all: three of the
four metrics are pure string and id arithmetic, and the fourth is the single
matching call. That is what makes it affordable for V5, V6 and V8 to re-run this
harness rather than treat it as a one-off number.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import Settings
from src.evals.critique import (
    DEFAULT_GROUNDING_GATE,
    DEFAULT_MATCH_REPEATS,
    GATE_EXCLUSIVE,
    EXCLUSION_VERSION,
    TIMESTAMP_TOLERANCE_SECONDS,
    ChunkTextFn,
    ContestedPair,
    Criterion,
    CriterionMatch,
    CritiqueDataset,
    EmbedFn,
    Finding,
    MatchFn,
    MatchResult,
    SetupCritique,
    attach_provenance,
    build_run,
    load_critique_dataset,
    parse_findings,
    repeated_matcher,
    score_critique,
    verify_dataset,
)

#: The setup the baseline row measures: the summary-filtered chunk dump. Named
#: here rather than passed in, because "better than chunk-dumping" is the claim
#: this whole harness exists to make falsifiable — the baseline is part of the
#: experiment's definition, not a parameter of it.
BASELINE_SETUP = "rag_llm_filtered"

#: The baseline plus both sides of any disagreement the corpus holds about this
#: kind of document.
#:
#: This exists because of what the baseline row measures. Its retrieval returned
#: ten chunks from four videos, and the corpus's résumé disagreement — one
#: creator saying 11-12 point body text, another saying 10 — is in two videos it
#: never reached. A top-k over one embedding neighbourhood converges: it finds
#: the *most typical* answer, and the most typical answer is the consensus,
#: which is exactly the material a disagreement is not. So the baseline could
#: not have named a conflict however willing it was, and its
#: ``contested_coverage`` is ``None`` rather than 0 for that reason.
#:
#: The fix is not a bigger k. It is to look the disagreement up: the conflict
#: layer already knows which two chunks contradict each other, so this setup
#: appends both sides of the most relevant ones to the retrieved context and
#: leaves everything else identical. What it then measures is behavioural — the
#: model has both sides in front of it, and either it names the axis or it
#: blends the two into one confident sentence.
CONFLICT_AWARE_SETUP = "rag_conflict_aware"

#: Disagreements appended to the conflict-aware context, most relevant first.
#: Small on purpose: this is a measurement of what the model does with a
#: conflict it can see, and burying the artifact's own material under a dozen
#: injected excerpts would measure something else.
CONFLICT_INJECT_LIMIT = 2

#: V6's arm: the rubric-driven reviewer, scored against the same held-out
#: expert as the chunk dump.
#:
#: The two arms differ in *where the criteria come from*, and in nothing else —
#: same held-out video, same corpus, same artifact, same answering model, same
#: matcher. :data:`BASELINE_SETUP` retrieves ten chunks at review time and asks
#: the model to work out what rules they imply. This one never retrieves: the
#: rules were distilled once at pack build time over the whole corpus, and the
#: review walks them one at a time and returns a verdict per rubric.
#:
#: Only the **failures** become findings (see
#: :func:`src.documents.rubric_review.verdicts_as_findings`). That is not a
#: filter chosen to flatter the arm — it is the only version of this comparison
#: that is not the padding attack in :mod:`src.evals.KNOWN_GAP_attack2`. Sixty-one
#: rubrics emitted as sixty-one findings, each with its own real pack citation,
#: would score recall far above the baseline while having applied nothing to the
#: document; a rubric earns a finding only when the reviewer judged this document
#: to fail it *and* pointed at a section of it.
#:
#: What the arm carries that the baseline does not is stated rather than hidden:
#: the packs read the corpus through a build the baseline never had. That is the
#: thing being measured, and the run records the pack provenance so it can be
#: argued with.
RUBRIC_PACK_SETUP = "rubric_packs"

CRITIQUE_SYSTEM_PROMPT = """You review a document against criteria drawn from \
expert video transcripts.

You are given (a) the document under review and (b) numbered excerpts from expert \
transcripts. Work out, from the excerpts alone, the general rules those experts \
apply to this kind of document, then judge the document against them.

Rules:
- Base every finding on the excerpts. Do not use criteria you happen to know but \
that no excerpt states.
- State each finding's `criterion` as a general, checkable rule that would apply \
to any document of this kind — not as a comment about this document. Put the \
comment about this document in `detail`.
- Cite the excerpt(s) the rule came from. `quote` must be words copied verbatim \
from the excerpt, `video_id` and `start_seconds` must be that excerpt's own.
- Set `contested` to true only when two or more excerpts from DIFFERENT videos \
disagree about the rule, and cite both sides. Do not average conflicting advice \
into one finding.
- Do not repeat the same rule as two findings.

Return only JSON in this shape:
{"findings": [{"id": "f01", "criterion": "...", "detail": "...", \
"contested": false, "citations": [{"video_id": "...", "start_seconds": 0.0, \
"quote": "..."}]}]}"""


# ── resolvers ─────────────────────────────────────────────────────────────


def chunk_text_lookup(
    settings: Settings,
    *,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> ChunkTextFn:
    """``(video_id, seconds) -> the (chunk_id, text) pairs spoken around then``.

    Every chunk whose span covers the timestamp is a candidate, because chunks
    overlap by design and a quote near a boundary lives in both. Deliberately
    built on a store with **no** exclusion: this resolver is the provenance
    check, and it has to be able to see the held-out video in order to notice
    that something cited it.
    """
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import TranscriptChunkStore

    store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=HuggingFaceEmbeddingModel(settings.embedding_model),
        collection_name=settings.chunk_collection,
    )
    cache: dict[str, list[tuple[float, float, str, str]]] = {}

    def spans(video_id: str) -> list[tuple[float, float, str, str]]:
        if video_id not in cache:
            cache[video_id] = [
                (
                    float(chunk.start_seconds or 0.0),
                    float(chunk.end_seconds or 0.0),
                    chunk.chunk_id,
                    chunk.text,
                )
                for chunk in store.chunks_for_videos([video_id])
            ]
        return cache[video_id]

    def lookup(video_id: str, seconds: float) -> list[tuple[str, str]]:
        return [
            (chunk_id, text)
            for start, end, chunk_id, text in spans(video_id)
            if start - tolerance <= seconds <= end + tolerance
        ]

    return lookup


def contested_pairs(
    conflict_path: Path,
    *,
    exclude_video_ids: Sequence[str] = (),
) -> list[ContestedPair]:
    """The committed conflict artifact, reduced to what the scorer needs.

    Conflicts touching an excluded video are dropped rather than filtered later:
    a held-out run must not be able to score itself on a disagreement one of
    whose sides is the video it is meant to be blind to, and dropping it here
    means no downstream caller has to remember.

    A missing artifact yields an empty list, which makes ``contested_coverage``
    ``None`` on every cell — the honest reading of "the disagreement layer was
    never built" is that this run did not measure it, not that it scored zero.
    """
    from src.rag.conflicts import ConflictStore

    index = ConflictStore(conflict_path).load()
    if index is None:
        return []
    blocked = set(exclude_video_ids)
    pairs: list[ContestedPair] = []
    for conflict in index.conflicts:
        videos = (conflict.left.video_id, conflict.right.video_id)
        if blocked & set(videos):
            continue
        pairs.append(
            ContestedPair(
                conflict_id=conflict.conflict_id,
                axis=conflict.axis,
                video_ids=videos,
                chunk_ids=(conflict.left.chunk_id, conflict.right.chunk_id),
            )
        )
    return pairs


def embedder(settings: Settings) -> EmbedFn:
    """The local sentence-transformers model, as a batch embed callable."""
    from src.rag.embeddings import HuggingFaceEmbeddingModel

    model = HuggingFaceEmbeddingModel(settings.embedding_model)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.embed_documents(list(texts))

    return embed


#: The prompt is deliberately left as it was when the committed run was scored.
#: An independent review put this matcher against an adversarial set — polarity
#: inversions ("always list every certification" vs "drop the ones that add
#: nothing"), homonyms ("number your sections" vs "put numbers on your claims"),
#: pairs sharing only a rationale ("recruiters skim"), and half-rules — and it
#: refused 9 of 9 without any of them being named here. Spelling them out would
#: be a change to the instrument that produced the committed score, and this
#: harness exists to make such changes visible: if it is ever edited, bump
#: :data:`MATCHER_VERSION` so cached pairings from the old wording are not
#: reused, and re-run rather than leaving the number and the prompt out of step.
MATCH_SYSTEM_PROMPT = """You decide which review criteria a critique applied.

You are given a numbered list of CRITERIA (rules an expert applies to a kind of
document) and a numbered list of FINDINGS (points a reviewer made about one
document). For each criterion, name the finding that applies that same underlying
rule, or null if none does.

A finding matches a criterion only when it asserts the same rule. Being about the
same subject is not enough: "link to the source code of each project" and "list
your profile links so people can contact you" are both about links and are
different rules. Judge the rule, not the wording, and not whether the finding is
correct about the document.

Each finding may be used for at most one criterion. If a finding is plausible for
two criteria, give it to the closer one and leave the other null.

Return only JSON in this shape:
{"matches": [{"criterion_id": "c01", "finding_id": "f03", "why": "both say ..."},
             {"criterion_id": "c02", "finding_id": null, "why": "not raised"}]}"""


#: Where resolved pairings are remembered. Beside the matrix cache and for the
#: same reason: gitignored derived state, one file per entry, safe to delete.
DEFAULT_MATCH_CACHE_DIR = Path(".yt-agent/critique_cache")

#: Bumped when the matcher prompt or the voting rule changes, so a stored
#: pairing produced under the old rules is not silently reused under the new.
MATCHER_VERSION = "v2-vote5"


def cached_matcher(
    inner: MatchFn,
    *,
    cache_dir: Path = DEFAULT_MATCH_CACHE_DIR,
    repeats: int = DEFAULT_MATCH_REPEATS,
) -> MatchFn:
    """Remember a resolved pairing, keyed by the exact matrix it resolved.

    The fingerprint is every criterion and every finding as text, plus the
    matcher version and the repeat count — so re-scoring an unchanged run is
    free *and* returns the identical number, while any change to what is being
    matched (or to how) recomputes. That is the second half of the fix for a
    non-deterministic scorer: voting narrows the noise, caching means a rerun
    does not re-roll it at all, and a score that moves is a score somebody
    changed something to move.
    """

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        material = json.dumps(
            {
                "version": MATCHER_VERSION,
                "repeats": repeats,
                "criteria": [[c.id, c.criterion] for c in criteria],
                "findings": [[f.id, f.criterion, f.detail] for f in findings],
            },
            sort_keys=True,
        )
        key = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        path = cache_dir / f"{key}.json"
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                return _result_from_cache(stored, criteria, findings)
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # A corrupt entry is a miss, never a wrong answer.
        result = inner(criteria, findings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "runs": result.runs,
                    "why": {m.criterion.id: m.why for m in result.matches},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    return match


def _result_from_cache(
    stored: dict[str, Any], criteria: Sequence[Criterion], findings: Sequence[Finding]
) -> MatchResult:
    """Rebuild a pairing from the stored per-run votes.

    The *votes* are cached, not the resolved consensus, so a change to the
    voting rule re-resolves from the same evidence instead of needing the calls
    to be paid for again.
    """
    from src.evals.critique import consensus, enforce_one_to_one

    runs = [dict(run) for run in stored["runs"]]
    matches = enforce_one_to_one(consensus(criteria, findings, runs))
    why = stored.get("why") or {}
    for row in matches:
        row.why = why.get(row.criterion.id)
    return MatchResult(matches=matches, runs=runs)


def llm_matcher(settings: Settings) -> MatchFn:
    """Pair criteria to findings with one LLM call, then commit the pairing.

    One call per setup, no per-pair scoring: the whole matrix is decided in a
    single judgement, which is both cheaper than a cross-encoder sweep and better
    at the thing being judged (see :data:`~src.evals.critique.MATCH_THRESHOLD`
    for the calibration that ruled the local models out).

    The model's own one-to-one instruction is not trusted —
    :func:`~src.evals.critique.enforce_one_to_one` re-applies it downstream —
    and the ``why`` it gives for every pairing is carried into the run file, so a
    reviewer reads the reason a criterion counted rather than a bare tick.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs

    llm = ChatOpenAI(**chat_model_kwargs(settings))

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        rows = [CriterionMatch(criterion=c) for c in criteria]
        if not criteria or not findings:
            return MatchResult(matches=rows, runs=[{c.id: None for c in criteria}])
        prompt = (
            "CRITERIA\n"
            + "\n".join(f"{c.id}: {c.criterion}" for c in criteria)
            + "\n\nFINDINGS\n"
            + "\n".join(f"{f.id}: {f.criterion} — {f.detail}" for f in findings)
        )
        response = llm.invoke(
            [SystemMessage(content=MATCH_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        payload = _json_object(str(getattr(response, "content", response)))
        by_id = {row.criterion.id: row for row in rows}
        finding_ids = {f.id for f in findings}
        by_finding = {f.id: f for f in findings}
        for entry in payload.get("matches") or []:
            if not isinstance(entry, dict):
                continue
            row = by_id.get(str(entry.get("criterion_id") or ""))
            if row is None:
                continue
            # Recorded before the null check: "not raised" is the reason a
            # reader most wants, and discarding it left every unmatched row in
            # the UI with no explanation at all.
            row.why = str(entry.get("why") or "").strip() or None
            finding_id = entry.get("finding_id")
            if not finding_id or str(finding_id) not in finding_ids:
                continue
            row.finding_id = str(finding_id)
            row.finding_criterion = by_finding[str(finding_id)].criterion
            # The matcher gives a verdict, not a similarity. 1.0 records "this
            # pairing was asserted"; the reason is what a reviewer actually
            # reads, and it is what ``enforce_one_to_one`` breaks ties on when
            # the model reuses a finding despite being told not to.
            row.score = 1.0
        return MatchResult(matches=rows, runs=[{row.criterion.id: row.finding_id for row in rows}])

    return match


# ── the critique itself ───────────────────────────────────────────────────

#: ``(dataset, setup) -> one critique``. The live implementation is
#: :func:`build_critique_fn`; the tests supply a fake so the run assembly and
#: scoring can be exercised with no models, no Chroma and no API key.
CritiqueFn = Callable[[CritiqueDataset, str], SetupCritique]


def load_artifact(dataset: CritiqueDataset) -> Any:
    """The stored document under review, or a clear failure.

    Read from the document store rather than re-fetched: the run has to be
    reproducible, and a page that changed between two runs would move the scores
    for reasons that are not about retrieval.
    """
    from src.documents.store import DocumentStore

    document = DocumentStore().get(dataset.artifact_id)
    if document is None:
        raise ValueError(
            f"No stored document {dataset.artifact_id!r}. Review "
            f"{dataset.artifact_url} once in the chat so it is fetched and stored."
        )
    return document


def conflict_excerpts(
    settings: Settings,
    query: str,
    conflicts: Sequence[ContestedPair],
    *,
    limit: int = CONFLICT_INJECT_LIMIT,
) -> list[Any]:
    """Both sides of the disagreements closest to ``query``, as chunk records.

    Ranked by cosine between the **axis** and the review query, which is the
    right text to rank on: the axis is one sentence naming the question the two
    creators answer differently, while the two chunks are seventy seconds of
    speech each and would rank on whatever else they happen to mention.

    Both sides always travel together. Appending one side of a conflict would be
    worse than appending neither — it puts a contested claim in the context
    wearing the same clothes as an uncontested one, which is precisely the state
    this whole slice exists to get the system out of.

    Ties break on ``conflict_id`` so the same corpus and query always inject the
    same excerpts in the same order.
    """
    if not conflicts:
        return []
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import TranscriptChunkStore

    embed = HuggingFaceEmbeddingModel(settings.embedding_model)
    vectors = embed.embed_documents([query] + [pair.axis for pair in conflicts])

    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        norms = (sum(a * a for a in left) ** 0.5) * (sum(b * b for b in right) ** 0.5)
        return dot / norms if norms else 0.0

    ranked = sorted(
        zip(conflicts, (cosine(vectors[0], row) for row in vectors[1:])),
        key=lambda item: (-item[1], item[0].conflict_id),
    )[: max(0, limit)]

    wanted = {chunk_id for pair, _ in ranked for chunk_id in pair.chunk_ids}
    videos = sorted({video_id for pair, _ in ranked for video_id in pair.video_ids})
    store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embed,
        collection_name=settings.chunk_collection,
    )
    by_id = {chunk.chunk_id: chunk for chunk in store.chunks_for_videos(videos)}
    return [by_id[chunk_id] for chunk_id in sorted(wanted) if chunk_id in by_id]


def pack_exposure(packs: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Every chunk and video the pack **build** put in front of a model.

    Not the chunks the shipped rubrics happen to quote. The leak check has to
    see what the system was *exposed* to — a held-out video that reached the
    build prompt and was merely not quoted has still contaminated the
    experiment — which is the same reading
    :func:`src.evals.pack_ablation.pack_as_critique` takes of the same packs.

    It is also the honest answer to "what was in this arm's context" for
    :func:`~src.evals.critique.contested_coverage`. The reviewer itself sees no
    transcript at all; whatever chance it had to notice a disagreement was
    spent at build time, on these chunks.
    """
    chunk_ids = sorted(
        {chunk_id for pack in packs for unit in pack.units for chunk_id in unit.chunk_ids}
    )
    video_ids = sorted(
        {video_id for pack in packs for unit in pack.units for video_id in unit.video_ids}
    )
    return chunk_ids, video_ids


def pack_finding_provenance(
    packs: Sequence[Any],
    *,
    qualify: bool = False,
) -> dict[str, list[str]]:
    """``finding id -> the chunks that finding's own distillation saw``.

    The per-finding half of :func:`pack_exposure`, and the reason the pack arms
    can be graded under :data:`~src.evals.critique.GATE_PROVENANCE` at all. A
    rubric is distilled from exactly one unit — a RAPTOR theme or a graph
    community — and that unit's ``chunk_ids`` are the entire corpus the
    distilling model had in front of it when it wrote that rule. So they are
    literally "what this finding retrieved", not an approximation of it.

    ``qualify`` matches the id scheme the reviewer uses
    (``{topic}:{rubric_id}``, because rubric ids are numbered per pack and
    ``r0101`` exists in all four); the D2 ablation scores one pack at a time and
    uses the bare id.

    A rubric whose ``unit_id`` names no unit in its own pack is left out of the
    map rather than given an empty list, so it comes back ``None`` and takes the
    cell to ungraded. A pack that cannot say where a rubric came from is exactly
    the case the gate must not wave through.
    """
    provenance: dict[str, list[str]] = {}
    for pack in packs:
        units = {unit.unit_id: sorted(dict.fromkeys(unit.chunk_ids)) for unit in pack.units}
        for rubric in pack.rubrics:
            chunk_ids = units.get(getattr(rubric, "unit_id", "") or "")
            if chunk_ids is None:
                continue
            key = f"{pack.topic}:{rubric.rubric_id}" if qualify else rubric.rubric_id
            provenance[key] = chunk_ids
    return provenance


def provenance_for_run(
    run: dict[str, Any],
    *,
    packs_dir: Path | str = Path("experts"),
) -> dict[str, dict[str, Sequence[str]]]:
    """Rebuild ``{setup: {finding_id: chunk_ids}}`` for a committed run's arms.

    A run written before the gate existed did not serialise per-finding
    provenance, but for a pack arm it was never lost: the pack is a committed
    artifact and still says which unit every rubric came out of. So re-scoring
    such a run under the gate reads the answer back rather than marking the arm
    ungraded over a field that did not exist yet.

    Only the arms that *have* an answer get one. A retrieval arm's setup name
    matches no pack, so it is absent from the map, stays ``None``, and is
    ungraded — which is the correct verdict and not a gap in this function.
    """
    from src.rag.packs import PackStore

    store = PackStore(Path(packs_dir))
    topic = str(run.get("topic") or "")
    out: dict[str, dict[str, Sequence[str]]] = {}
    for cell in run.get("cells", []):
        setup = str(cell.get("setup") or "")
        if setup == RUBRIC_PACK_SETUP:
            from src.documents.rubric_review import load_review_packs

            packs = load_review_packs(packs_dir)
            if packs:
                out[setup] = dict(pack_finding_provenance(packs, qualify=True))
            continue
        if not topic:
            continue
        try:
            pack = store.load_pack(topic, setup)
        except (OSError, ValueError):
            continue
        if pack is not None:
            out[setup] = dict(pack_finding_provenance([pack]))
    return out


def rubric_critique(
    llm: Any,
    dataset: CritiqueDataset,
    document: Any,
    query: str,
    *,
    setup: str = RUBRIC_PACK_SETUP,
    model_name: str = "",
    packs_dir: Path | str | None = None,
) -> SetupCritique:
    """V6's arm: judge the artifact rubric by rubric, and hand over the failures.

    No retrieval runs here — see :data:`RUBRIC_PACK_SETUP`. The document is read
    with the same :func:`~src.documents.review.select_sections` call the baseline
    makes, so the two arms are judging the same text and any gap between them is
    about the criteria, not about which half of the page each one saw.
    """
    from src.agents.rubric_review_agent import RubricReviewAgent
    from src.documents.review import select_sections
    from src.documents.rubric_review import load_review_packs, verdicts_as_findings
    from src.rag.eval import estimate_tokens

    started = time.monotonic()
    result = SetupCritique(setup=setup)
    packs = load_review_packs(packs_dir)
    if not packs:
        result.error = "No rubric packs are built. Run `uv run python -m src.cli build-packs`."
        result.elapsed_seconds = time.monotonic() - started
        return result

    chunk_ids, video_ids = pack_exposure(packs)
    result.retrieved_chunk_ids = chunk_ids
    result.retrieved_video_ids = video_ids

    selection = select_sections(document, query)
    try:
        review = RubricReviewAgent(llm, packs, model_name=model_name).review(document, selection)
    except Exception as exc:  # noqa: BLE001 - a failed setup is a reported cell
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_seconds = time.monotonic() - started
        return result

    # Stamped here rather than inside ``verdicts_as_findings``, which does not
    # know about scoring and should not have to: the pack is what knows which
    # unit each rubric came out of.
    result.findings = attach_provenance(
        verdicts_as_findings(review.review),
        pack_finding_provenance(packs, qualify=True),
    )
    result.answer = review.answer
    result.token_estimate = estimate_tokens(review.answer)
    stats = review.review.stats
    counts = stats["verdicts"]
    declared = sorted(
        {
            video_id
            for pack in packs
            for video_id in getattr(getattr(pack, "provenance", None), "held_out_video_ids", [])
        }
    )
    held_out = ", ".join(declared) or "nothing"
    if dataset.held_out_video_id not in declared:
        # Said out loud rather than left to the leak count. A pack that never
        # declared this video held out may still show zero leaks — the build
        # simply never routed to it — and "excluded on purpose" and "missed by
        # luck" are not the same experiment.
        held_out += f" — but NOT {dataset.held_out_video_id}, which this run holds out"
    result.trace = [
        {
            "phase": "route",
            "label": "Load rubric packs",
            # What the packs *say* they held out, not what this run wanted them
            # to: the two agreeing is the claim, and printing the dataset's own
            # id here would make the trace agree with itself by construction.
            # ``held_out_leaks`` re-derives the answer from the chunk ids above
            # either way.
            "detail": (
                f"{stats['packs_used']} packs, {stats['rubrics_total']} rubrics — distilled "
                f"once over the corpus, held out {held_out}; "
                "no retrieval runs at review time"
            ),
        },
        {
            "phase": "read",
            "label": "Read the document",
            "detail": selection.detail(),
        },
        {
            "phase": "answer",
            "label": "Judge each rubric",
            "detail": (
                f"{counts.get('fail', 0)} fail, {counts.get('pass', 0)} pass, "
                f"{counts.get('n-a', 0)} n/a, {counts.get('unjudged', 0)} unjudged — "
                "only the failures are scored as findings"
            ),
        },
    ]
    result.elapsed_seconds = time.monotonic() - started
    return result


def build_critique_fn(
    settings: Settings,
    dataset: CritiqueDataset,
    conflicts: Sequence[ContestedPair] = (),
) -> CritiqueFn:
    """A critique callable over the real stack, held-out video excluded.

    The stores are built once and shared across setups, exactly as
    :func:`src.evals.ablation.build_retrieve` does — so a multi-setup run loads
    the embedding model and the cross-encoder once between them.

    ``conflicts`` is only read by :data:`CONFLICT_AWARE_SETUP`; every other
    setup retrieves exactly as it did before this argument existed, so the
    baseline row stays comparable with the runs already committed.
    """
    from src.agents.llm import chat_model_kwargs
    from src.documents.review import REVIEW_INTENT_QUERIES, select_sections
    from src.rag.context import MultiTranscriptRagContextProvider
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.eval import estimate_tokens
    from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
    from src.rag.summaries import TranscriptSummaryStore

    excluded = [dataset.held_out_video_id]
    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    # The raw store has no exclusion of its own — it lives in src/rag, which this
    # module does not own — and it can read a full transcript by video id. Nothing
    # reaches it on the corpus-wide path this harness uses, but "unreachable
    # today" is how a leak gets built in, so it is wrapped rather than trusted.
    raw_store = _HeldOutBlocked(
        RawTranscriptStore(
            settings.chroma_path, collection_name=settings.raw_transcript_collection
        ),
        excluded,
    )
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
        exclude_video_ids=excluded,
    )
    summary_store = TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        collection_name=settings.transcript_summary_collection,
        exclude_video_ids=excluded,
    )
    reranker = None
    if settings.rerank_enabled:
        from src.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker.from_model_name(settings.rerank_model)

    provider = MultiTranscriptRagContextProvider(
        raw_store=raw_store,
        chunk_store=chunk_store,
        # No indexer. A held-out run must never be able to write to the corpus,
        # and an auto-index of the held-out video is the one write that would
        # destroy the experiment rather than merely dirty it.
        indexer=None,
        summary_store=summary_store,
        retrieval_mode=settings.retrieval_mode,
        retrieval_candidates=settings.retrieval_candidates,
        reranker=reranker,
        neighbor_span=settings.neighbor_span,
    )

    document = load_artifact(dataset)
    query = REVIEW_INTENT_QUERIES.get(dataset.artifact_kind, REVIEW_INTENT_QUERIES["document"])

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(**chat_model_kwargs(settings))

    def critique(_: CritiqueDataset, setup: str) -> SetupCritique:
        started = time.monotonic()
        # The rubric arm shares the document, the section selection and the
        # answering client with the retrieval arms, and shares nothing else:
        # it never touches the provider above, because having no retrieval at
        # review time is the thing it is being measured for.
        if setup == RUBRIC_PACK_SETUP:
            return rubric_critique(
                llm,
                dataset,
                document,
                query,
                setup=setup,
                model_name=settings.deepseek_model,
            )
        result = SetupCritique(setup=setup)
        try:
            context = provider.get_context(
                query,
                top_k=settings.rag_top_k,
                # The conflict-aware setup routes exactly as the baseline does.
                # Its only difference is what gets appended afterwards, so that
                # a gap between the two rows is attributable to the conflict
                # layer rather than to two different retrieval strategies.
                filter_transcripts=setup in {BASELINE_SETUP, CONFLICT_AWARE_SETUP},
                transcript_filter_top_k=settings.transcript_filter_top_k,
                transcript_filter_min_score=settings.transcript_filter_min_score,
                retrieval_mode=settings.retrieval_mode,
            )
        except Exception as exc:  # noqa: BLE001 - a failed setup is a reported cell
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_seconds = time.monotonic() - started
            return result

        chunks = list(context.retrieved_chunks or [])
        # Appended, not merged into the ranking: these are not retrieval hits
        # and pretending they competed for a top-k slot would misreport what the
        # setup did. Dedup on chunk id keeps a side that retrieval already found
        # from appearing twice.
        injected: list[Any] = []
        if setup == CONFLICT_AWARE_SETUP:
            seen = {chunk.chunk_id for chunk in chunks}
            injected = [
                chunk
                for chunk in conflict_excerpts(settings, query, conflicts)
                if chunk.chunk_id not in seen
            ]
            chunks = chunks + injected
        result.retrieved_chunk_ids = [
            f"chunk:{chunk.video_id}:{chunk.chunk_index}" for chunk in chunks
        ]
        result.retrieved_video_ids = sorted({chunk.video_id for chunk in chunks})
        excerpts = _format_excerpts(chunks)
        selection = select_sections(document, query)
        prompt = (
            f"DOCUMENT UNDER REVIEW ({dataset.artifact_kind}) — {dataset.artifact_url}\n"
            f"{_format_document(selection.sections)}\n\n"
            f"EXPERT TRANSCRIPT EXCERPTS\n{excerpts}"
        )
        result.token_estimate = estimate_tokens(prompt)
        result.trace = [
            {
                "phase": "filter" if setup == BASELINE_SETUP else "retrieve",
                "label": "Retrieve review criteria",
                "detail": (
                    f"{len(chunks)} chunks from {len(result.retrieved_video_ids)} videos "
                    f"— {dataset.held_out_video_id} excluded from every store"
                ),
            },
            *(
                [
                    {
                        "phase": "merge",
                        "label": "Add both sides of known disagreements",
                        "detail": (
                            f"{len(injected)} chunks appended from the conflict layer "
                            f"— {len(injected) // 2} disagreement(s), both sides each"
                        ),
                    }
                ]
                if setup == CONFLICT_AWARE_SETUP
                else []
            ),
            {
                "phase": "read",
                "label": "Read the document",
                "detail": selection.detail(),
            },
        ]
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            answer = str(getattr(response, "content", response))
            result.answer = answer
            # No ``trust_provenance`` and nothing stamped afterwards, on purpose.
            # Every finding here came out of one call over one shared pool, so
            # there is no per-finding retrieval to record and the honest state
            # is ``None`` — which makes the cell **ungraded** under
            # :data:`~src.evals.critique.GATE_PROVENANCE` rather than passed.
            # Declaring the pool per finding would satisfy the gate and mean
            # nothing; see :mod:`src.evals.KNOWN_GAP_attack2`.
            result.findings = parse_findings(_json_object(answer))
        except Exception as exc:  # noqa: BLE001 - a failed setup is a reported cell
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_seconds = time.monotonic() - started
        return result

    return critique


class _HeldOutBlocked:
    """A raw store that refuses the held-out video instead of returning it.

    Fails loud. A held-out run that silently reads the transcript it is meant to
    be blind to produces a number that looks perfect and means nothing, so the
    right behaviour on that path is to stop the run.
    """

    def __init__(self, inner: Any, excluded: Sequence[str]) -> None:
        self._inner = inner
        self._excluded = set(excluded)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_raw_document(self, video_id: str) -> Any:
        self._check(video_id)
        return self._inner.get_raw_document(video_id)

    def join_raw_text(self, video_id: str) -> str:
        self._check(video_id)
        return str(self._inner.join_raw_text(video_id))

    def ensure_raw_document(self, source_url: str, refresh: bool = False) -> Any:
        for video_id in self._excluded:
            if video_id in str(source_url):
                self._check(video_id)
        return self._inner.ensure_raw_document(source_url, refresh)

    def _check(self, video_id: str) -> None:
        if video_id in self._excluded:
            raise ValueError(
                f"{video_id} is held out of this run and must not be read. "
                "Reaching this means an exclusion was bypassed."
            )


def _format_excerpts(chunks: Sequence[Any]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        start = float(getattr(chunk, "start_seconds", 0.0) or 0.0)
        end = float(getattr(chunk, "end_seconds", 0.0) or 0.0)
        title = getattr(chunk, "title", None) or ""
        parts.append(
            f"[{index}] video_id={chunk.video_id} start_seconds={start:.1f} "
            f"end_seconds={end:.1f} title={title!r}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _format_document(sections: Sequence[Any]) -> str:
    return "\n\n".join(
        f"## {section.heading}\n{section.text}" if section.heading else section.text
        for section in sections
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply, fenced or not. ``{}`` if there is none."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(stripped)
        if match is None:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


# ── the run ───────────────────────────────────────────────────────────────


def run_critique_eval(
    dataset: CritiqueDataset,
    setups: Sequence[str],
    critique: CritiqueFn,
    match: MatchFn,
    chunk_text: ChunkTextFn,
    *,
    conflicts: Sequence[ContestedPair] = (),
    config: dict[str, Any] | None = None,
    gate: str = DEFAULT_GROUNDING_GATE,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Critique the artifact with each setup, score them, assemble the run.

    The dataset's own quotes are verified first. If the ground truth cannot
    resolve against the transcript it claims to come from, nothing measured
    against it means anything, so this refuses to run rather than producing a
    number that looks fine.
    """
    bad = [check for check in verify_dataset(dataset, chunk_text) if not check.resolved]
    if bad:
        raise ValueError(
            "The held-out criteria do not all resolve against the transcript: "
            + "; ".join(f"{check.citation.start_seconds}s {check.reason}" for check in bad[:5])
        )

    cells: list[dict[str, Any]] = []
    for setup in setups:
        if on_progress is not None:
            on_progress(f"[{setup}] critiquing {dataset.artifact_url}")
        result = critique(dataset, setup)
        cells.append(score_critique(result, dataset, match, chunk_text, conflicts, gate=gate))
    return build_run(
        dataset,
        cells,
        config={
            "exclusion_version": EXCLUSION_VERSION,
            **(config or {}),
        },
        baseline=setups[0] if setups else BASELINE_SETUP,
        gate=gate,
    )


def replay_matcher(run: dict[str, Any]) -> MatchFn:
    """Replay the pairings a committed run's matcher already voted, per setup.

    ``score_critique`` needs a :data:`~src.evals.critique.MatchFn`, and the only
    shipped one costs an LLM call. But every cell already stores ``match_runs``
    — the five per-repeat votes the matcher cast — so re-scoring a committed run
    under a new *arithmetic* rule needs no model at all: the pairing is data,
    and re-rolling it would change the number for a reason that has nothing to
    do with the rule being tested.

    Cells are consumed in file order, one per call, because the matcher
    signature carries no setup and two arms of the same run are matched over the
    same criteria list. A run whose cells are re-scored out of order would get
    another cell's votes, so this raises instead of guessing when it runs out.
    """
    from src.evals.critique import consensus, enforce_one_to_one

    pending = [
        (str(cell.get("setup") or ""), list(cell.get("match_runs") or []))
        for cell in run.get("cells", [])
    ]
    position = 0

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        nonlocal position
        if position >= len(pending):
            raise ValueError(
                f"{run.get('run_id')} stored votes for {len(pending)} cells and a "
                f"{position + 1}th was asked for — re-score its cells once each, in order"
            )
        _, runs = pending[position]
        position += 1
        merged = enforce_one_to_one(consensus(criteria, findings, [dict(row) for row in runs]))
        return MatchResult(matches=merged, runs=[dict(row) for row in runs])

    return match


def rescore_committed_run(
    run: dict[str, Any],
    dataset: CritiqueDataset,
    match: MatchFn,
    chunk_text: ChunkTextFn,
    *,
    conflicts: Sequence[ContestedPair] = (),
    config: dict[str, Any] | None = None,
    gate: str = DEFAULT_GROUNDING_GATE,
    provenance: dict[str, dict[str, Sequence[str]]] | None = None,
    now: Any = None,
) -> dict[str, Any]:
    """Re-score a committed run's stored findings under the current scorer.

    The same move :mod:`src.evals.rejudge` makes for the matrix, and needed for
    the same reason: when the *scoring rule* changes, the old number is not
    comparable and the old answers are still perfectly good. Re-running
    retrieval and the critique call to re-apply an arithmetic change would also
    change the findings, so the two effects could not be told apart.

    Retrieval is not repeated, so the held-out guarantee is inherited from the
    source run — and re-checked here, against the ids that run stored, rather
    than assumed.

    ``provenance`` is ``{setup: {finding_id: chunk_ids}}``, and exists because a
    run committed before the gate did not serialise what each finding retrieved
    even when the engine knew. For a pack arm the answer is still on disk — the
    pack is committed and :func:`pack_finding_provenance` reads it back — so
    supplying it re-scores that arm on the evidence it actually had rather than
    marking it ungraded over a missing field. Nothing is invented: a setup the
    map does not name keeps whatever the stored findings carried, which for a
    pre-gate run is nothing at all.
    """
    supplied = provenance or {}
    cells: list[dict[str, Any]] = []
    for stored in run.get("cells", []):
        setup = str(stored.get("setup") or "")
        findings = parse_findings(
            {"findings": stored.get("findings", [])},
            # A stored run is the engine's own record, not a model reply.
            trust_provenance=True,
        )
        if setup in supplied:
            findings = attach_provenance(findings, dict(supplied[setup]))
        critique = SetupCritique(
            setup=setup,
            findings=findings,
            retrieved_chunk_ids=list(stored.get("retrieved_chunk_ids") or []),
            retrieved_video_ids=list(stored.get("retrieved_video_ids") or []),
            elapsed_seconds=float(stored.get("elapsed_seconds") or 0.0),
            token_estimate=int(stored.get("token_estimate") or 0),
            answer=str(stored.get("answer") or ""),
            error=stored.get("error"),
            trace=list(stored.get("trace") or []),
        )
        cells.append(score_critique(critique, dataset, match, chunk_text, conflicts, gate=gate))
    rescored = build_run(
        dataset,
        cells,
        config={
            **(run.get("config") or {}),
            **(config or {}),
            "rescored_from": run.get("run_id"),
            "rescored_gate_from": (run.get("grounding_gate") or GATE_EXCLUSIVE),
        },
        baseline=str(run.get("baseline") or BASELINE_SETUP),
        now=now,
        gate=gate,
    )
    rescored["rescored_from"] = run.get("run_id")
    return rescored


def run_default_critique_eval(
    settings: Settings,
    *,
    setups: Sequence[str] | None = None,
    dataset_path: Path | None = None,
    repeats: int = DEFAULT_MATCH_REPEATS,
    gate: str = DEFAULT_GROUNDING_GATE,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load the held-out dataset and measure the named setups over the real stack."""
    dataset = load_critique_dataset(dataset_path)
    chosen = list(setups or [BASELINE_SETUP])
    # Stated in the run rather than left to be inferred from the arm's name.
    # The rubric arm's whole advantage is a build the baseline never had, so
    # which packs it used and what each of them held out is part of reading its
    # number — not a footnote.
    pack_config: dict[str, Any] = {}
    if RUBRIC_PACK_SETUP in chosen:
        from src.documents.rubric_review import load_review_packs

        packs = load_review_packs()
        pack_config = {
            "rubric_packs": [
                {
                    "topic": pack.topic,
                    "arm": pack.arm,
                    "rubrics": len(pack.rubrics),
                    "held_out_video_ids": list(pack.provenance.held_out_video_ids),
                    "excluded_video_ids": list(pack.provenance.excluded_video_ids),
                }
                for pack in packs
            ],
            "rubric_pack_note": (
                "the rubric arm runs no retrieval at review time; its criteria were "
                "distilled once over the corpus at pack build time, and only rubrics "
                "the reviewer failed against a named section of the document are "
                "scored as findings"
            ),
        }
    # Loaded with the held-out video already dropped, so a run can never score
    # itself on a disagreement one of whose sides it is meant to be blind to.
    conflicts = contested_pairs(
        settings.conflict_path, exclude_video_ids=[dataset.held_out_video_id]
    )
    return run_critique_eval(
        dataset,
        chosen,
        build_critique_fn(settings, dataset, conflicts),
        cached_matcher(
            repeated_matcher(llm_matcher(settings), repeats=repeats),
            repeats=repeats,
        ),
        chunk_text_lookup(settings),
        conflicts=conflicts,
        gate=gate,
        config={
            "answer_model": settings.deepseek_model,
            "grounding_gate_note": (
                "criteria_recall and evidence_precision are None for any arm that cannot "
                "record what each finding's own reasoning retrieved — the retrieval arms "
                "emit every finding from one shared pool, so the gate cannot grade them "
                "and criteria_recall_ungated is the only figure they have"
            ),
            "conflicts_available": len(conflicts),
            "conflict_inject_limit": CONFLICT_INJECT_LIMIT,
            "matcher": "llm",
            "matcher_model": settings.deepseek_model,
            "matcher_version": MATCHER_VERSION,
            "match_repeats": repeats,
            "embedding_model": settings.embedding_model,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": settings.rag_top_k,
            "rerank_enabled": settings.rerank_enabled,
            "transcript_filter_top_k": settings.transcript_filter_top_k,
            "transcript_filter_min_score": settings.transcript_filter_min_score,
            "corpus_note": (
                "3 of 53 videos have no stored summary, so the summary-filtered "
                "setup routes over 50 videos before the held-out one is removed"
            ),
            **pack_config,
        },
        on_progress=on_progress,
    )
