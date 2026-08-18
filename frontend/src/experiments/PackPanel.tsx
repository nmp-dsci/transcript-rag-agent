import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import { useDemo } from '../demo';
import type {
  PackAblation,
  PackAblationBasis,
  PackAblationVerdict,
  PackDetail,
  PackList,
  PackMember,
  PackRubric,
  PackStaleness,
  PackSummary,
  ResearchDiffRow,
  ResearchReport,
} from '../api/types';

function fmt(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' ? value.toFixed(digits) : '—';
}

function pct(value: number | null | undefined): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : '—';
}

function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/**
 * One rubric, expanded to its quotes.
 *
 * The criterion and its check are the claim; the quotes underneath are what
 * makes it falsifiable, so every one carries the creator who said it and a link
 * to the second they said it at. The build produces no severity and no
 * confidence score, and this panel does not invent one — what it shows instead
 * is the two things that *were* measured: how many distinct creators back the
 * rule, and how cleanly each quote snapped onto the transcript.
 */
function RubricRow({ rubric, open, onToggle }: {
  rubric: PackRubric;
  open: boolean;
  onToggle: () => void;
}) {
  const creators = rubric.creators.length;
  const worst = rubric.evidence.reduce(
    (low, item) => Math.min(low, item.ratio),
    rubric.evidence.length > 0 ? 1 : 0,
  );
  return (
    <li className={`pk-rubric${open ? ' open' : ''}`}>
      <button type="button" className="pk-rubhead" onClick={onToggle} aria-expanded={open}>
        <span className="pk-rubid">{rubric.rubric_id}</span>
        <span className="pk-rubtext">{rubric.criterion}</span>
        <span className="pk-rubtags">
          {rubric.contested && (
            <span className="pk-badge contested" title="Two or more creators disagree">
              contested
            </span>
          )}
          <span
            className={`pk-badge ${creators >= 2 ? 'ok' : 'thin'}`}
            title={rubric.creators.join(' · ')}
          >
            {creators} creator{creators === 1 ? '' : 's'}
          </span>
          <span className="pk-badge" title="Lowest quote-to-transcript match ratio">
            match {fmt(worst, 2)}
          </span>
        </span>
      </button>

      {open && (
        <div className="pk-rubbody">
          {rubric.check && (
            <p className="pk-check">
              <span className="microlabel">how to check</span>
              {rubric.check}
            </p>
          )}
          {rubric.why && (
            <p className="pk-why">
              <span className="microlabel">why</span>
              {rubric.why}
            </p>
          )}
          <p className="pk-unit">
            from <code>{rubric.unit_id}</code> ({rubric.unit_kind}) — {rubric.unit_title}
          </p>
          <ul className="pk-evlist">
            {rubric.evidence.map((item) => (
              <li key={`${item.chunk_id}-${item.quote_start_seconds}`} className="pk-ev">
                <div className="pk-evmeta">
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noreferrer"
                    className={`pk-ts${item.resolved ? '' : ' bad'}`}
                    title={`${item.chunk_id} — opens at ${clock(item.quote_start_seconds)}`}
                  >
                    {clock(item.quote_start_seconds)}
                  </a>
                  <span className="pk-creator">{item.channel_name ?? item.video_id}</span>
                  {item.title && <span className="pk-vidtitle">{item.title}</span>}
                </div>
                <blockquote className="pk-quote">“{item.quote}”</blockquote>
                {item.ratio < 0.999 && (
                  <div className="pk-drift">
                    model wrote “{item.model_quote}” — snapped at {fmt(item.ratio, 2)}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </li>
  );
}

/**
 * Which videos routed into this pack, with the score that put them there.
 *
 * Membership is a retrieval result, not a hand-kept list — the pack's topic
 * description is embedded and matched against the per-video summaries by the
 * same call a live question makes. The override buttons record a human's
 * disagreement with that result in the pack's manifest; they deliberately do
 * not change what is on screen, because the rendered pack is the artifact on
 * disk and only a rebuild can move it.
 */
function MembershipTable({ detail, onOverride }: {
  detail: PackDetail;
  onOverride: (videoId: string, included: boolean | null) => void;
}) {
  // Overrides POST a refused route in demo mode; membership stays readable.
  const demo = useDemo();
  const members: PackMember[] = detail.members ?? [];
  if (members.length === 0) return null;
  const silent = members.filter((member) => member.in_units === false).length;
  return (
    <section className="pk-section">
      <h4>
        Membership — {members.length} videos routed by topic description
        <span className="pk-hint">
          scored against each video's summary; overrides apply at the next build
          {silent > 0
            ? ` · ${silent} routed in but reached no source unit, so none of these rules can rest on them`
            : ''}
        </span>
      </h4>
      <div className="exp-scroll">
        <table className="exp-table pk-table">
          <thead>
            <tr>
              <th className="num">score</th>
              <th>video</th>
              <th>creator</th>
              <th className="num">chunks</th>
              <th>source</th>
              <th>pin</th>
            </tr>
          </thead>
          <tbody>
            {members.map((member) => {
              const override = detail.overrides[member.video_id];
              return (
                <tr key={member.video_id}>
                  <td className="num">{fmt(member.score)}</td>
                  <td className="pk-cellwrap">{member.title ?? member.video_id}</td>
                  <td className="pk-cellwrap">{member.channel_name ?? '—'}</td>
                  <td className="num">{member.chunk_count}</td>
                  <td className="pk-src">
                    {member.routed ? (
                      <span className="pk-badge ok">routed</span>
                    ) : (
                      <span className="pk-badge">hand-added</span>
                    )}
                    {typeof override === 'boolean' && (
                      <span className="pk-badge contested">
                        pinned {override ? 'in' : 'out'}
                      </span>
                    )}
                    {member.in_units === false ? (
                      <span
                        className="pk-badge thin"
                        title="No chunk of this video reached a source unit, so no rule here can rest on it — the theme layer and entity graph are derived state built at an earlier chunk count."
                      >
                        no units
                      </span>
                    ) : member.cited === false ? (
                      <span
                        className="pk-badge thin"
                        title="Reached a source unit but no rubric quoted it."
                      >
                        uncited
                      </span>
                    ) : null}
                  </td>
                  <td className="pk-pin">
                    {demo ? (
                      <span className="pk-badge">demo</span>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="btn sm"
                          onClick={() => onOverride(member.video_id, true)}
                        >
                          in
                        </button>
                        <button
                          type="button"
                          className="btn sm"
                          onClick={() => onOverride(member.video_id, false)}
                        >
                          out
                        </button>
                        <button
                          type="button"
                          className="btn sm"
                          onClick={() => onOverride(member.video_id, null)}
                        >
                          auto
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * D2: the three arms side by side on the held-out critique harness.
 *
 * The spread under each recall is the scorer's matcher disagreeing with itself
 * across repeats. A lead smaller than that range is not a result, which is what
 * the verdict line under the table says in words — the honest reading of two
 * arms inside each other's noise is "these are the same".
 */
function AblationRows({ ablation }: { ablation: PackAblation }) {
  // The run names its own metric columns; the fallback mirrors CRITIQUE_METRICS
  // only so an older run file without a `metrics` list still renders.
  const metrics =
    ablation.metrics.length > 0
      ? ablation.metrics
      : ['criteria_recall', 'evidence_precision', 'provenance', 'contested_coverage'];
  const best: Record<string, number> = {};
  for (const cell of ablation.cells) {
    for (const metric of metrics) {
      const value = cell.scores[metric];
      if (typeof value === 'number' && (best[metric] === undefined || value > best[metric])) {
        best[metric] = value;
      }
    }
  }
  return (
    <section className="pk-section">
      <h4>
        D2 — raptor vs communities vs merged, on the held-out expert
        <span className="pk-hint">
          {ablation.held_out_title ? `${ablation.held_out_title} · ` : ''}
          {ablation.criteria_applicable ?? '—'}/{ablation.criteria_total ?? '—'} criteria
          apply
          {ablation.match_repeats ? ` · ${ablation.match_repeats} matcher repeats` : ''}
        </span>
      </h4>
      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              {metrics.map((metric) => (
                <th key={metric} className="num">{metric}</th>
              ))}
              <th className="num" title="recall over every criterion, not just the applicable ones">
                recall_all
              </th>
              <th className="num" title="recall counted once per rule">recall_grouped</th>
              <th className="num">rubrics</th>
              <th className="num">cited</th>
              <th className="num">leaks</th>
            </tr>
          </thead>
          <tbody>
            {ablation.cells.map((cell) => (
              <tr key={cell.arm}>
                <td className="exp-cfg">
                  {cell.arm}
                  {cell.arm === ablation.baseline && <span className="exp-basetag">base</span>}
                </td>
                {metrics.map((metric) => {
                  const value = cell.scores[metric];
                  const low = cell.score_spread[`${metric}_min`];
                  const high = cell.score_spread[`${metric}_max`];
                  const isBest = typeof value === 'number' && value === best[metric];
                  return (
                    <td key={metric} className={`num${isBest ? ' best' : ''}`}>
                      {fmt(value)}
                      {typeof low === 'number' && typeof high === 'number' && low !== high && (
                        <span className="crit-spread">
                          {low.toFixed(3)}–{high.toFixed(3)}
                        </span>
                      )}
                    </td>
                  );
                })}
                <td className="num">{fmt(cell.criteria_recall_all)}</td>
                <td className="num">{fmt(cell.criteria_recall_grouped)}</td>
                <td className="num">{cell.rubrics ?? '—'}</td>
                <td className="num">
                  {cell.citations_resolved ?? '—'}/{cell.citations_total ?? '—'}
                </td>
                <td className={`num${(cell.held_out_leaks ?? 0) > 0 ? ' bad' : ''}`}>
                  {cell.held_out_leaks ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pk-verdicts">
        {Object.entries(ablation.verdicts).map(([metric, verdict]) => (
          <div key={metric} className="pk-verdict">
            <span className="microlabel">{metric}</span>
            <b>{verdict.leader ?? '—'}</b>
            <VerdictBadge verdict={verdict} />
            <span className="pk-verdictwhy">{verdict.reason}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

/**
 * Whether an added rule broke new ground or rediscovered the shipped pack's.
 *
 * V8's gate asks whether the loop beats the hand-built pack; it does not, and
 * this is the mechanical reason. An addition is new against *round one*, which
 * is all the gap critic can see — so the loop's novelty claim is honest — but
 * four of its ten additions cite chunks the shipped pack already cites. Without
 * this the diff reads as ten discoveries and the page is silent on why the arm
 * lost.
 *
 * Overlap is identity on cited chunk ids, not text similarity: two rules worded
 * differently off the same transcript second are the same finding, and a
 * similarity score would be a judgement where an identity check will do.
 * Rendering nothing when `ids` is absent keeps "not compared" distinct from
 * "compared, found new".
 */
export function RediscoveryMark({ ids }: { ids?: string[] }) {
  if (!ids) return null;
  if (!ids.length) {
    return (
      <span className="rs-newground">
        new ground — no shipped rule cites any chunk this one cites
      </span>
    );
  }
  return (
    <span className="rs-rediscovered">
      rediscovered — the shipped pack already covers this from the same chunk (
      {ids.map((id, i) => (
        <Fragment key={id}>
          {i > 0 && ', '}
          <code>{id}</code>
        </Fragment>
      ))}
      )
    </span>
  );
}

/**
 * Says out loud how far the pack is behind the corpus being browsed.
 *
 * V5's acceptance gate failed on exactly one check — "a reader is told when the
 * pack is behind the corpus they are browsing" — with the card printing 56
 * videos while the header above it said 67. The old UI did carry the fact, but
 * only inside a hover tooltip on a badge for videos that "reached no source
 * unit": a symptom, and only for a reader who already suspected something.
 *
 * Counts, not a warning colour. "Behind by 11 videos" tells a reader how much
 * to discount the rules below; a triangle tells them nothing they can act on.
 * Rendering nothing when the comparison is unavailable is deliberate — silence
 * is honest, whereas an unqualified "up to date" would not be.
 */
function StalenessNotice({ staleness }: { staleness?: PackStaleness | null }) {
  if (!staleness) return null;
  if (staleness.current) {
    return (
      <p className="pk-fresh">
        Built from the corpus as it stands now — {staleness.live_videos} videos,{' '}
        {staleness.live_chunks} chunks.
      </p>
    );
  }
  const videos = staleness.behind_videos;
  const chunks = staleness.behind_chunks;
  const gained = videos > 0 || chunks > 0;
  return (
    <p className="pk-stale">
      <b>These rules are out of date.</b> They were distilled from{' '}
      {staleness.build_videos} videos / {staleness.build_chunks} chunks (
      <code>{staleness.build_digest}</code>). The corpus you are browsing now holds{' '}
      {staleness.live_videos} videos / {staleness.live_chunks} chunks (
      <code>{staleness.live_digest}</code>) —{' '}
      {gained ? (
        <>
          {Math.abs(videos)} video{Math.abs(videos) === 1 ? '' : 's'} and {Math.abs(chunks)} chunk
          {Math.abs(chunks) === 1 ? '' : 's'} the build never saw
        </>
      ) : (
        <>a re-chunk since the build, so stored citations may have moved</>
      )}
      . Nothing below is wrong, but it is not everything the corpus now supports. Rebuild
      with <code>uv run python -m src.cli build-packs</code>.
    </p>
  );
}

/**
 * How much weight a metric's lead can carry, in one badge.
 *
 * Three states, not two. Only `criteria_recall` is repeated, so only it can
 * clear a runner-up's spread; the rest are single draws. Rendering those as
 * "decisive" told a reader the gap had survived a noise check that was never
 * run — the badge is the whole of what most readers take away, so it has to be
 * the honest part. `unrepeated` says the gap is real and its reliability
 * unmeasured, which is a different and weaker claim than either neighbour.
 */
export function VerdictBadge({ verdict }: { verdict: PackAblationVerdict }) {
  // Runs committed before `basis` existed carry only the boolean; fall back to
  // it rather than mislabelling an old run as unrepeated.
  const basis = verdict.basis ?? (verdict.decisive ? 'cleared-spread' : 'inside-spread');
  const label: Record<PackAblationBasis, string> = {
    'cleared-spread': 'decisive',
    'inside-spread': 'inside scorer noise',
    unrepeated: 'scored once — reliability unmeasured',
    tied: 'tied',
    'single-arm': 'no comparison',
    'no-scored-arm': 'not scored',
  };
  const tone: Record<PackAblationBasis, string> = {
    'cleared-spread': 'ok',
    'inside-spread': 'thin',
    unrepeated: 'thin',
    tied: 'thin',
    'single-arm': 'thin',
    'no-scored-arm': 'thin',
  };
  return <span className={`pk-badge ${tone[basis]}`}>{label[basis]}</span>;
}

/** The three arms' deterministic composition — available even without a D2 run. */
function ArmComposition({ detail }: { detail: PackDetail }) {
  const arms = Object.values(detail.arms);
  if (arms.length === 0) return null;
  return (
    <section className="pk-section">
      <h4>
        Arms as built
        <span className="pk-hint">
          every arm spends the same number of source units, so a bigger pack
          cannot win by being bigger
        </span>
      </h4>
      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              <th className="num">units</th>
              <th className="num">rubrics</th>
              <th className="num">quotes</th>
              <th className="num">quotes resolved</th>
              <th className="num">≥2 creators</th>
              <th className="num">contested</th>
              <th className="num">gaps</th>
              <th>units by kind</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((arm) => (
              <tr key={arm.arm}>
                <td className="exp-cfg">
                  {arm.arm}
                  {arm.arm === detail.arm && <span className="exp-basetag">shipped</span>}
                </td>
                <td className="num">{arm.unit_budget ?? '—'}</td>
                <td className="num">{arm.checks.rubrics}</td>
                <td className="num">{arm.checks.evidence_total}</td>
                <td className="num">{pct(arm.checks.quote_resolution)}</td>
                <td className="num">{pct(arm.checks.multi_creator_share)}</td>
                <td className="num">{arm.checks.contested_rubrics}</td>
                <td className="num">{arm.gaps ?? '—'}</td>
                <td className="pk-cellwrap">
                  {Object.entries(arm.units_by_kind)
                    .map(([kind, count]) => `${count} ${kind}`)
                    .join(' · ') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * The loop-built pack's v1 → v2 diff, in the browser where the rules live.
 *
 * The build report argues the loop's case; this section is the pack browser's
 * own answer to "what actually changed in the criteria", so a reader comparing
 * rules does not have to hold two tabs open. Three things are said plainly
 * because each of them is a way the diff could be read too generously:
 *
 * * It is keyed on ``rubric_id``, and round one's surviving rules are carried
 *   forward as the same objects — so there is no *changed* class to show, and a
 *   text diff would invite paraphrase drift to be read into rows where none
 *   exists. Kept means byte-identical.
 * * Every added rule names the probe it came from, and a probe whose id starts
 *   at 51 is one the gap critic wrote. That is the causal chain, on screen,
 *   rather than a claim that a second round happened.
 * * The control's own diff sits beside it. It added rules too, on the same
 *   executor budget and with no critic, and hiding that would make growth look
 *   like iteration.
 */
function ResearchDiffSection({ report }: { report: ResearchReport }) {
  const [open, setOpen] = useState(false);
  const diff = report.diff;
  // unit_id -> the gap that caused it, read off round 2's own probe records.
  const causeOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const outcome of report.rounds[1]?.probes ?? []) {
      if (outcome.probe.origin.startsWith('gap:')) {
        map[outcome.unit_id] = outcome.probe.origin.slice(4);
      }
    }
    return map;
  }, [report]);
  const gapText = useMemo(() => {
    const map: Record<string, string> = {};
    for (const gap of report.gaps) map[gap.gap_id] = gap.missing;
    return map;
  }, [report]);

  const rows: Array<{ kind: string; row: ResearchDiffRow }> = [
    ...diff.added.map((row) => ({ kind: 'added', row })),
    ...diff.removed.map((row) => ({ kind: 'removed', row })),
    ...diff.kept.map((row) => ({ kind: 'kept', row })),
  ];
  if (rows.length === 0) return null;
  return (
    <section className="pk-section">
      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} Loop-built pack: {diff.before_arm} → {diff.after_arm} —{' '}
          {diff.added.length} added, {diff.kept.length} kept, {diff.removed.length} removed
        </button>
        <span className="pk-hint">
          the one-shot control, same budget and no critic, added{' '}
          {report.control_diff.added.length} on the same v1 · full build report below
        </span>
      </h4>
      {open && (
        <ul className="rs-diff">
          {rows.map(({ kind, row }) => {
            const gapId = causeOf[row.unit_id];
            return (
              <li key={`${kind}-${row.rubric_id}`} className={`rs-diffrow ${kind}`}>
                <span className="rs-diffmark">
                  {kind === 'added' ? '+' : kind === 'removed' ? '−' : ' '}
                </span>
                <span className="rs-diffbody">
                  <span className="rs-crit">{row.criterion}</span>
                  <span className="rs-critmeta">
                    <code>{row.rubric_id}</code> · <code>{row.unit_id}</code> ·{' '}
                    {row.creators.join(' · ') || 'no creator'} · {row.citations} quote
                    {row.citations === 1 ? '' : 's'}
                    {gapId && (
                      <>
                        {' '}
                        · asked for by <code>{gapId}</code>: {gapText[gapId] ?? ''}
                      </>
                    )}
                  </span>
                  {kind === 'added' && <RediscoveryMark ids={row.already_in_shipped} />}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/** Source units that reached this pack's topic and produced no rule, and why. */
function GapList({ detail }: { detail: PackDetail }) {
  const gaps = detail.gaps ?? [];
  const [open, setOpen] = useState(false);
  if (gaps.length === 0) return null;
  return (
    <section className="pk-section">
      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} {gaps.length} source units produced no rule
        </button>
        <span className="pk-hint">
          "we never looked", "one creator was talking" and "we looked and found
          nothing" are three different admissions
        </span>
      </h4>
      {open && (
        <ul className="pk-gaps">
          {gaps.map((gap) => (
            <li key={gap.unit_id}>
              <code>{gap.unit_id}</code>
              <span className="pk-gaptitle">{gap.unit_title}</span>
              <span className="pk-gapreason">
                {gap.chunks} chunks · {gap.videos} videos
                {typeof gap.creator_count === 'number'
                  ? ` · ${gap.creator_count} creators`
                  : ''}{' '}
                — {gap.reason}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function PackPanel() {
  const [list, setList] = useState<PackList | null>(null);
  const [topic, setTopic] = useState<string | null>(null);
  const [detail, setDetail] = useState<PackDetail | null>(null);
  // Null for every topic no deep-research loop has been run for, which is three
  // of the four — the section it feeds renders nothing rather than an empty card.
  const [research, setResearch] = useState<ResearchReport | null>(null);
  const [openRubric, setOpenRubric] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const loaded = await api.packs();
        setList(loaded);
        const first = loaded.packs.find((p) => p.built) ?? loaded.packs[0];
        setTopic(first?.topic ?? null);
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    if (!topic) return;
    setDetail(null);
    setResearch(null);
    setOpenRubric(null);
    // An override notice names a video in the pack that was on screen when it
    // was recorded; carrying it into another pack would read as a claim about
    // that pack's membership instead.
    setNotice(null);
    void (async () => {
      try {
        setDetail(await api.pack(topic));
        setError(null);
      } catch (err) {
        setError((err as Error).message);
      }
      // Separate and swallowed: a topic with no build report is the ordinary
      // case, and it must not read as the pack having failed to load.
      setResearch(await api.packResearch(topic).catch(() => null));
    })();
  }, [topic]);

  const override = useCallback(
    async (videoId: string, included: boolean | null) => {
      if (!topic) return;
      try {
        const result = await api.setPackMember(topic, videoId, included);
        setNotice(`${videoId}: recorded — applies ${result.applies}`);
        setDetail(await api.pack(topic));
      } catch (err) {
        setNotice((err as Error).message);
      }
    },
    [topic],
  );

  const summary: PackSummary | undefined = useMemo(
    () => list?.packs.find((p) => p.topic === topic),
    [list, topic],
  );

  if (error && !list) {
    return (
      <section className="exp-card">
        <p className="exp-empty">Could not load expert packs: {error}</p>
      </section>
    );
  }
  if (!list) return null;

  const checks = detail?.checks;
  const provenance = detail?.provenance;

  return (
    <section className="exp-card pk-card">
      <div className="exp-cardhead">
        <div>
          <h3>Expert pack{summary ? `: ${summary.topic}` : 's'}</h3>
          <span className="exp-sub">
            {provenance ? (
              <>
                v{detail?.version} · shipped arm <code>{detail?.arm}</code> · built from corpus{' '}
                <code>{provenance.corpus_digest}</code> ({provenance.chunk_count} chunks,{' '}
                {provenance.video_count} videos) · {provenance.rubric_model}
              </>
            ) : (
              list.build_command
            )}
          </span>
        </div>
        <div className="exp-seg" role="tablist" aria-label="Expert pack">
          {list.packs.map((pack) => (
            <button
              key={pack.topic}
              type="button"
              role="tab"
              aria-selected={topic === pack.topic}
              className={topic === pack.topic ? 'on' : ''}
              onClick={() => setTopic(pack.topic)}
            >
              {pack.topic}
              {!pack.built && <span className="pk-unbuilt"> ·</span>}
            </button>
          ))}
        </div>
      </div>

      <StalenessNotice staleness={detail?.staleness} />

      <p className="pk-intro">
        A rubric pack is a standing list of review criteria drawn from the corpus{' '}
        <em>as it stood at build time</em>, each one carrying the transcript words it
        came from. Membership is
        routed — the topic description below is embedded and matched against every
        video's summary by the same call a live question makes — and the criteria are
        written from the <em>members</em> of a theme rather than its summary, because
        theme summaries read more cross-creator than the clusters underneath them are.
        The build assigns no severity and no confidence; the badges on each rule are
        the two things that were measured — how many distinct creators back it, and
        how cleanly its weakest quote snapped onto the transcript.
      </p>

      {summary && <p className="pk-blurb">{summary.blurb}</p>}
      {detail?.routing_text && (
        <p className="pk-routing">
          <span className="microlabel">routing text</span>
          {detail.routing_text}
        </p>
      )}

      {checks && (
        <div className="pk-checks">
          <span className="pk-check-item">
            <span className="microlabel">quotes resolving</span>
            <b className={checks.quote_resolution >= 0.95 ? 'good' : 'bad'}>
              {pct(checks.quote_resolution)}
            </b>
            <span className="pk-checksub">
              {checks.evidence_resolved}/{checks.evidence_total}
            </span>
          </span>
          <span className="pk-check-item">
            <span className="microlabel">rules citing ≥2 creators</span>
            <b className={checks.multi_creator_share >= 0.3 ? 'good' : 'bad'}>
              {pct(checks.multi_creator_share)}
            </b>
            <span className="pk-checksub">
              {checks.multi_creator_rubrics}/{checks.rubrics}
            </span>
          </span>
          <span className="pk-check-item">
            <span className="microlabel">distinct creators</span>
            <b>{checks.creators}</b>
          </span>
          <span className="pk-check-item">
            <span className="microlabel">contested</span>
            <b>{checks.contested_rubrics}</b>
          </span>
          <span className="pk-check-item">
            <span className="microlabel">excluded-video citations</span>
            <b className={checks.excluded_video_citations === 0 ? 'good' : 'bad'}>
              {checks.excluded_video_citations}
            </b>
            <span className="pk-checksub">
              {detail?.excluded_video_ids.length ?? 0} videos blocked
            </span>
          </span>
        </div>
      )}

      {notice && <p className="pk-notice">{notice}</p>}

      {detail && !detail.built && (
        <p className="exp-empty">
          <b>{detail.name}</b> is declared but not built. Run{' '}
          <code>{detail.build_command} --topic {detail.topic}</code>.
        </p>
      )}

      {detail?.rubrics && detail.rubrics.length > 0 && (
        <section className="pk-section">
          <h4>
            {detail.rubrics.length} criteria
            <span className="pk-hint">click a rule to read its evidence</span>
          </h4>
          <ul className="pk-rublist">
            {detail.rubrics.map((rubric) => (
              <RubricRow
                key={rubric.rubric_id}
                rubric={rubric}
                open={openRubric === rubric.rubric_id}
                onToggle={() =>
                  setOpenRubric(openRubric === rubric.rubric_id ? null : rubric.rubric_id)
                }
              />
            ))}
          </ul>
        </section>
      )}

      {detail && <ArmComposition detail={detail} />}
      {research && <ResearchDiffSection report={research} />}

      {detail?.ablation ? (
        <AblationRows ablation={detail.ablation} />
      ) : detail?.built ? (
        <p className="pk-nod2">
          No D2 score for this pack. The held-out critique harness reviews one
          artifact — a résumé — so only a pack whose criteria apply to it can be
          scored against it. The arms are still comparable above on the
          deterministic counts.
        </p>
      ) : null}

      {detail && <MembershipTable detail={detail} onOverride={override} />}
      {detail && <GapList detail={detail} />}
    </section>
  );
}
