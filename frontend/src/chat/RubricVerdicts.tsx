import { useMemo, useState } from 'react';

import type { RubricReview, RubricVerdict } from '../api/types';

/**
 * The rubric reviewer's verdicts, one row per rubric.
 *
 * This panel is the difference between this setup and the chunk-dump review
 * beside it. A prose review can be read and believed; it cannot be *checked*,
 * because nothing states what was checked and came back clean. Here the list is
 * the pack, every row carries the id of the rule it applied and a link to the
 * second a creator said it, and the rules that did not apply say so instead of
 * disappearing.
 *
 * Two things the filters are for. Severity, because a portfolio failing sixteen
 * rubrics is unreadable until the three that stop a recruiter are on top. And
 * verdict, because the default view hides `pass` and `n-a` — a reviewer wants
 * the failures first — while leaving both one click away, since "48 rules did
 * not apply to a website" is the honest headline of this document and hiding it
 * permanently would be the same silence the prose review has.
 */

type VerdictFilter = 'fail' | 'all' | 'pass' | 'n-a' | 'unjudged';
type SeverityFilter = 'all' | 'blocker' | 'major' | 'minor';

function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

const VERDICT_LABEL: Record<string, string> = {
  fail: 'fail',
  pass: 'pass',
  'n-a': 'n/a',
  unjudged: 'not decided',
};

/** One rubric row, collapsed to its verdict until the rubric itself is opened. */
function VerdictRow({
  verdict,
  onOpenSection,
}: {
  verdict: RubricVerdict;
  onOpenSection?: (index: number) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`rvrow rv-${verdict.verdict}`}>
      <div className="rvhead">
        <span className={`rvbadge rv-${verdict.verdict}`}>
          {VERDICT_LABEL[verdict.verdict] ?? verdict.verdict}
        </span>
        {verdict.verdict === 'fail' && verdict.severity !== 'none' ? (
          <span className={`rvsev sev-${verdict.severity}`}>{verdict.severity}</span>
        ) : null}
        {/*
          The rubric id, always, and it is a button rather than a label: it is
          the click-through the whole panel exists for. Opening it shows the
          check that decided this verdict and the creators behind it.
        */}
        <button
          type="button"
          className="rvid"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          title="Open this rubric — its check, why it exists, and who said it"
        >
          {verdict.rubric_id}
        </button>
        <span className="rvpack">{verdict.pack_name}</span>
        <span className="rvsecs">
          {verdict.sections.map((index) => (
            <button
              type="button"
              key={index}
              className="rvsec"
              title={`Open section ${index + 1} of the document above`}
              onClick={() => onOpenSection?.(index)}
            >
              §{index + 1}
            </button>
          ))}
        </span>
        {/*
          The timestamp is on the collapsed row, not behind the expander. It is
          half of what this panel claims — every verdict carries its rubric id
          and the moment a person said the rule out loud — and a claim you have
          to expand a row to check is a claim most readers never check.
        */}
        <span className="rvts">
          {verdict.evidence.map((item) => (
            <a
              key={`${item.video_id}-${item.start_seconds}`}
              href={item.url}
              target="_blank"
              rel="noreferrer noopener"
              title={`${item.channel_name ?? item.video_id} — “${item.quote}”`}
            >
              {clock(item.start_seconds)}
            </a>
          ))}
        </span>
      </div>

      <div className="rvcrit">{verdict.criterion}</div>
      {verdict.finding ? <div className="rvfind">↳ {verdict.finding}</div> : null}
      {verdict.note ? <div className="rvnote">↳ {verdict.note}</div> : null}

      {open ? (
        <div className="rvdetail">
          <div className="rvline">
            <span className="rvlabel">check</span>
            <span>{verdict.check || '—'}</span>
          </div>
          <div className="rvline">
            <span className="rvlabel">why</span>
            <span>{verdict.why || '—'}</span>
          </div>
          <div className="rvline">
            <span className="rvlabel">theme</span>
            <span>{verdict.unit_title || '—'}</span>
          </div>
          <ul className="rvquotes">
            {verdict.evidence.map((item) => (
              <li key={item.chunk_id}>
                <a href={item.url} target="_blank" rel="noreferrer noopener">
                  {item.channel_name ?? item.video_id} · {clock(item.start_seconds)} ↗
                </a>
                <span className="rvquote">“{item.quote}”</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </li>
  );
}

export function RubricVerdicts({
  review,
  onOpenSection,
}: {
  review: RubricReview;
  /** Expand a section of the document card above this answer. */
  onOpenSection?: (index: number) => void;
}) {
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('fail');
  const [severity, setSeverity] = useState<SeverityFilter>('all');

  const counts = review.stats.verdicts ?? {};
  const severities = review.stats.severities ?? {};

  const rows = useMemo(() => {
    return review.verdicts.filter((verdict) => {
      if (verdictFilter !== 'all' && verdict.verdict !== verdictFilter) return false;
      // Severity only ever describes a failure, so filtering by it on a view
      // that is showing passes would silently empty the list.
      if (severity !== 'all' && verdict.severity !== severity) return false;
      return true;
    });
  }, [review.verdicts, verdictFilter, severity]);

  const failed = counts.fail ?? 0;
  const share = Math.round((review.stats.id_and_timestamp_share ?? 0) * 100);

  return (
    <div className="rubrics">
      <div className="rvsum">
        <span className="rvsum-lead">
          {review.stats.rubrics_total} rubrics from {review.stats.packs_used} expert packs
        </span>
        <span className="rvsum-counts">
          <b className="rv-fail">{failed} fail</b> · {counts.pass ?? 0} pass ·{' '}
          {counts['n-a'] ?? 0} n/a to a {review.document_kind}
          {counts.unjudged ? ` · ${counts.unjudged} not decided` : ''}
        </span>
        {/*
          Stated on every review, not only the clean ones. "Each verdict carries
          its rubric id and a timestamp" is the claim this setup makes, and a
          claim nobody prints is a claim nobody can catch failing.
        */}
        <span className="rvsum-prov" title="Verdicts carrying both a rubric id and a corpus timestamp">
          {review.stats.with_id_and_timestamp}/{review.stats.rubrics_total} carry rubric id +
          timestamp ({share}%) · {review.stats.evidence_links} source links
        </span>
      </div>

      <div className="rvfilters">
        <span className="microlabel">verdict</span>
        {(['fail', 'pass', 'n-a', 'unjudged', 'all'] as VerdictFilter[]).map((value) => {
          const total = value === 'all' ? review.stats.rubrics_total : (counts[value] ?? 0);
          return (
            <button
              key={value}
              type="button"
              className={`pill${verdictFilter === value ? ' on' : ''}`}
              aria-pressed={verdictFilter === value}
              disabled={total === 0 && value !== 'all'}
              onClick={() => setVerdictFilter(value)}
            >
              {VERDICT_LABEL[value] ?? value} {total}
            </button>
          );
        })}

        <span className="microlabel">severity</span>
        {(['all', 'blocker', 'major', 'minor'] as SeverityFilter[]).map((value) => (
          <button
            key={value}
            type="button"
            className={`pill${severity === value ? ' on' : ''}`}
            aria-pressed={severity === value}
            disabled={value !== 'all' && (severities[value] ?? 0) === 0}
            onClick={() => setSeverity(value)}
          >
            {value}
            {value === 'all' ? '' : ` ${severities[value] ?? 0}`}
          </button>
        ))}
      </div>

      <ul className="rvlist">
        {rows.map((verdict) => (
          <VerdictRow
            key={`${verdict.topic}-${verdict.rubric_id}`}
            verdict={verdict}
            {...(onOpenSection ? { onOpenSection } : {})}
          />
        ))}
        {rows.length === 0 ? (
          <li className="rvempty">No rubric matches this filter.</li>
        ) : null}
      </ul>

      {review.packs.some((pack) => pack.error) ? (
        <div className="rvpackerr">
          {review.packs
            .filter((pack) => pack.error)
            .map((pack) => (
              <div key={pack.topic}>
                {pack.name} could not be applied — its {pack.rubrics} rubrics are
                unjudged. {pack.error}
              </div>
            ))}
        </div>
      ) : null}
    </div>
  );
}
