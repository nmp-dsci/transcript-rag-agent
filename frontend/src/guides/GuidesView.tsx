import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { captureEvent } from '../analytics';
import { api } from '../api/client';
import type { GuideActivity, GuideDetail, GuideJob, GuideSummary } from '../api/types';
import { useDemo } from '../demo';
import { CommentRail } from './CommentRail';
import { ComposePanel } from './ComposePanel';
import { EvidenceMap } from './EvidenceMap';
import { GuideReader, type GuideReaderHandle, type GuideSection, type GuideSelection } from './GuideReader';
import { ResearchMap } from './ResearchMap';
import { useGuidesStyles } from './styles';

const ACTIVITY_LIMIT = 200;

/** Fold a live activity event into the job the way the server does, so the
 *  map moves between full job snapshots. */
export function appendActivity(job: GuideJob | null, event: GuideActivity): GuideJob | null {
  if (!job) return job;
  const activity = [...job.activity, event].slice(-ACTIVITY_LIMIT);
  const counters: Record<string, number> = { ...job.counters, tool_calls: (job.counters.tool_calls ?? 0) + 1 };
  if (event.name.endsWith('retrieve_chunks') && event.question != null) {
    counters.retrieval_queries = (counters.retrieval_queries ?? 0) + 1;
  } else if (event.name === 'WebSearch' || event.name === 'WebFetch') {
    counters.web_fetches = (counters.web_fetches ?? 0) + 1;
  } else if (event.name === 'Read') {
    counters.files_read = (counters.files_read ?? 0) + 1;
  }
  return { ...job, activity, counters };
}

const GUIDE_PARAM = 'guide';

/** The selected guide lives in the query string, not the hash — the hash is
 *  the tab router, and a `#guides/slug` would be read as an unknown tab. */
export function guideFromLocation(search: string): string | null {
  const value = new URLSearchParams(search).get(GUIDE_PARAM);
  return value && /^[a-z0-9][a-z0-9-]{1,80}$/.test(value) ? value : null;
}

function rememberGuide(slug: string | null): void {
  const url = new URL(window.location.href);
  if (slug) url.searchParams.set(GUIDE_PARAM, slug);
  else url.searchParams.delete(GUIDE_PARAM);
  window.history.replaceState(null, '', url.toString());
}

export function citeRate(guide: Pick<GuideSummary, 'cite_total' | 'cite_valid'>): string {
  if (!guide.cite_total) return 'no cites';
  return `cites ${guide.cite_valid}/${guide.cite_total}`;
}

function citeTone(guide: Pick<GuideSummary, 'cite_total' | 'cite_valid'>): 'good' | 'warn' | 'bad' {
  if (!guide.cite_total) return 'warn';
  const rate = guide.cite_valid / guide.cite_total;
  return rate >= 0.95 ? 'good' : rate >= 0.8 ? 'warn' : 'bad';
}

function GuideEntry({
  guide,
  selected,
  onSelect,
}: {
  guide: GuideSummary;
  selected: boolean;
  onSelect: (slug: string) => void;
}) {
  return (
    <button
      type="button"
      className={`rentry ${selected ? 'on' : ''}`}
      aria-current={selected ? 'true' : undefined}
      onClick={() => onSelect(guide.slug)}
    >
      <div className="rq">{guide.title}</div>
      <div className="rmeta">
        <span>{guide.videos} videos</span>
        <span>{guide.chunk_count} chunks</span>
        <span>v{guide.current_version}</span>
        <span className={`badge ${citeTone(guide)}`}>{citeRate(guide)}</span>
        {guide.comments_open > 0 && <span>{guide.comments_open} open</span>}
      </div>
    </button>
  );
}

function Provenance({ guide }: { guide: GuideDetail }) {
  const composer = guide.model?.composer;
  const extractor = guide.model?.extractor;
  const imported = guide.provenance?.pipeline !== 'guides write';
  return (
    <div>
      <div className="guide-prov" aria-label="How this guide was made">
        <span className="badge plain">{guide.videos} videos</span>
        <span className="badge plain">{guide.chunk_count} chunks read</span>
        {guide.cluster_count > 0 && <span className="badge plain">{guide.cluster_count} passes</span>}
        <span className={`badge ${citeTone(guide)}`}>{citeRate(guide)}</span>
        {composer && <span className="badge plain">{composer}</span>}
        {guide.web_allowed ? (
          <span className="badge warn">web on</span>
        ) : (
          <span className="badge plain">corpus only</span>
        )}
        <span className="badge plain">compiled {guide.compiled_at || '—'}</span>
      </div>
      <details className="note-more guide-how">
        <summary>How was this guide written?</summary>
        <p>
          {imported ? (
            <>
              Hand-run before the pipeline existed and imported as v1: every YouTube link in its Sources
              became a video-level cite, so the {citeRate(guide)} above resolve to videos, not chunks.
            </>
          ) : (
            <>
              Every chunk of the {guide.videos} confirmed videos ({guide.chunk_count} in all) was exported
              to files and read in full by {guide.cluster_count} parallel extraction {guide.cluster_count === 1 ? 'pass' : 'passes'}
              {extractor ? ` (${extractor})` : ''}, each writing claims with a chunk id and a verbatim quote. One
              composer{composer ? ` (${composer})` : ''} wrote the page from that evidence; a verifier with no LLM then
              resolved every cite against the corpus ({citeRate(guide)}). Evidence came from the corpus and a
              retrieval tool only{guide.web_allowed ? ', plus web search enabled for this run' : ''}.
            </>
          )}
          {guide.gaps.length > 0 && ` The corpus left ${guide.gaps.length} question${guide.gaps.length === 1 ? '' : 's'} unanswered — see Evidence.`}
        </p>
      </details>
    </div>
  );
}

export function GuidesView() {
  useGuidesStyles();
  const demo = useDemo();
  const [guides, setGuides] = useState<GuideSummary[] | null>(null);
  const [writeCommand, setWriteCommand] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [slug, setSlug] = useState<string | null>(() => guideFromLocation(window.location.search));
  const [detail, setDetail] = useState<GuideDetail | null>(null);
  const [sections, setSections] = useState<GuideSection[]>([]);
  const reader = useRef<GuideReaderHandle | null>(null);
  const [job, setJob] = useState<GuideJob | null>(null);
  const [sdkProblem, setSdkProblem] = useState<string | null>(null);
  /** 'job' shows the research map for the current run; null shows the reader. */
  const [view, setView] = useState<'job' | null>(null);
  const lastJobStatus = useRef<string | null>(null);
  const [selection, setSelection] = useState<GuideSelection | null>(null);
  const [version, setVersion] = useState<number | null>(null);
  const [side, setSide] = useState<'comments' | 'evidence'>('comments');

  const load = useCallback(async () => {
    try {
      const list = await api.guides();
      setGuides(list.guides);
      setWriteCommand(list.write_command);
      setError(null);
      return list.guides;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setGuides([]);
      return [];
    }
  }, []);

  useEffect(() => {
    void load().then((list) => {
      // A deep link wins; otherwise open the most recent guide so the tab is
      // never an empty frame next to a list of things to click.
      if (slug && list.some((g) => g.slug === slug)) return;
      const first = list[0]?.slug ?? null;
      setSlug(first);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const reloadDetail = useCallback(async (which: string) => {
    const found = await api.guide(which);
    setDetail(found);
    return found;
  }, []);

  useEffect(() => {
    rememberGuide(slug);
    setSections([]);
    setSelection(null);
    setVersion(null);
    if (!slug) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    api
      .guide(slug)
      .then((found) => {
        if (cancelled) return;
        setDetail(found);
        captureEvent('guide_open', { slug: found.slug, version: found.current_version });
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  // The job stream: seeded with the current job, then stage and activity
  // events. Demo mode never opens it — the server refuses the stream there
  // and nothing in demo can start a job.
  useEffect(() => {
    if (demo) return;
    const controller = new AbortController();
    api.guideJob().then((state) => setSdkProblem(state.sdk)).catch(() => undefined);
    api
      .subscribeGuideJob(
        {
          snapshot: (data) => setJob(data.job),
          job: (data) => setJob(data.job),
          activity: (data) => setJob((prev) => (prev && prev.id === data.job_id ? appendActivity(prev, data.event) : prev)),
        },
        controller.signal,
      )
      .catch(() => undefined);
    return () => controller.abort();
  }, [demo]);

  // When a run finishes, the catalog changes: reload it and open the result.
  useEffect(() => {
    const status = job?.status ?? null;
    const changed = status !== lastJobStatus.current;
    lastJobStatus.current = status;
    if (!changed || !job || status !== 'done') return;
    void load().then((list) => {
      if (list.some((g) => g.slug === job.slug)) {
        setSlug((current) => {
          if (current === job.slug) void reloadDetail(job.slug);
          return job.slug;
        });
        setView(null);
      }
    });
  }, [job, load, reloadDetail]);

  const onReady = useCallback((found: GuideSection[]) => setSections(found), []);
  const onSelection = useCallback((found: GuideSelection) => setSelection(found), []);

  const pageUrl = useMemo(() => {
    if (!detail) return null;
    if (version !== null && version !== detail.current_version && detail.version_urls[String(version)]) {
      return `${detail.version_urls[String(version)]}?v=${version}`;
    }
    // The version is part of the URL so a republished page is never served
    // from a stale iframe cache.
    return `${detail.html_url}?v=${detail.current_version}`;
  }, [detail, version]);

  const jump = useCallback((anchor: string | null) => {
    if (!anchor) return;
    reader.current?.scrollTo(anchor);
    reader.current?.highlight(anchor);
  }, []);

  return (
    <div className="stage guides">
      <aside className="rail guides-rail" aria-label="Field guides">
        <div className="rail-head">
          <strong>Field guides</strong>
          {guides && (
            <span className="rmeta">
              {guides.length} guide{guides.length === 1 ? '' : 's'} · written from the corpus
            </span>
          )}
        </div>
        <div className="rail-list">
          {job && (job.status === 'running' || view === 'job') && (
            <button
              type="button"
              className={`rentry gj-entry ${view === 'job' ? 'on' : ''}`}
              aria-current={view === 'job' ? 'true' : undefined}
              onClick={() => setView('job')}
            >
              <div className="rq">{job.title || job.slug}</div>
              <div className="rmeta">
                <span className={`badge ${job.status === 'running' ? 'acc' : job.status === 'error' ? 'bad' : 'good'}`}>
                  {job.status === 'running' ? `● ${job.stage ?? 'starting'}` : job.status === 'error' ? '✕ failed' : '✓ done'}
                </span>
                <span>{job.video_ids.length} videos</span>
              </div>
            </button>
          )}
          {guides === null && <div className="rail-empty">loading…</div>}
          {guides && guides.length === 0 && !job && <div className="rail-empty">No guides yet.</div>}
          {guides?.map((guide) => (
            <GuideEntry
              key={guide.slug}
              guide={guide}
              selected={view === null && guide.slug === slug}
              onSelect={(next) => {
                setView(null);
                setSlug(next);
              }}
            />
          ))}
        </div>
        {!demo && (
          <ComposePanel
            running={job}
            sdkProblem={sdkProblem}
            onStarted={(started) => {
              setJob(started);
              setView('job');
            }}
          />
        )}
        {demo && writeCommand && (
          <div className="rail-foot">
            Guides are written by an agent from the corpus; the demo is read-only.
          </div>
        )}
      </aside>

      <div className="guides-main">
        {error && <div className="guide-error">{error}</div>}
        {view === 'job' && job && <ResearchMap job={job} />}
        {view !== 'job' && detail && (
          <>
            <header className="guide-head">
              <div className="guide-title">
                <h2>{detail.title}</h2>
                {detail.subtitle && <p className="guide-sub">{detail.subtitle}</p>}
              </div>
              <div>
                <Provenance guide={detail} />
              </div>
              <div className="guide-links">
                {detail.versions.length > 1 && (
                  <select
                    aria-label="Version"
                    value={version ?? detail.current_version}
                    onChange={(event) => setVersion(Number(event.target.value))}
                  >
                    {detail.versions.map((v) => (
                      <option key={v} value={v}>
                        v{v}
                        {v === detail.current_version ? ' (current)' : ''}
                      </option>
                    ))}
                  </select>
                )}
                <a href={detail.html_url} target="_blank" rel="noopener">
                  open page ↗
                </a>
                {detail.markdown_url && (
                  <a href={detail.markdown_url} target="_blank" rel="noopener">
                    for agents (.md) ↗
                  </a>
                )}
              </div>
            </header>
            {sections.length > 0 && (
              <nav className="guide-toc" aria-label="Guide sections">
                {sections.map((section) => (
                  <button key={section.id} type="button" onClick={() => reader.current?.scrollTo(section.id)}>
                    {section.label}
                  </button>
                ))}
              </nav>
            )}
            <div className="guide-split">
              {pageUrl && (
                <GuideReader ref={reader} url={pageUrl} title={detail.title} onReady={onReady} onSelection={onSelection} />
              )}
              <div className="guide-side">
                <div className="guide-side-tabs" role="tablist">
                  <button type="button" role="tab" aria-selected={side === 'comments'} className={side === 'comments' ? 'on' : ''} onClick={() => setSide('comments')}>
                    Commentary{detail.comments_open > 0 ? ` · ${detail.comments_open}` : ''}
                  </button>
                  <button type="button" role="tab" aria-selected={side === 'evidence'} className={side === 'evidence' ? 'on' : ''} onClick={() => setSide('evidence')}>
                    Evidence · {citeRate(detail).replace('cites ', '')}
                  </button>
                </div>
                {side === 'comments' ? (
                  <CommentRail
                    guide={detail}
                    selection={selection}
                    running={job}
                    sdkProblem={sdkProblem}
                    demo={demo}
                    onAdded={(comment) =>
                      setDetail((prev) =>
                        prev ? { ...prev, comments: [...prev.comments, comment], comments_open: prev.comments_open + 1, comments_total: prev.comments_total + 1 } : prev,
                      )
                    }
                    onRevisionStarted={(started) => {
                      setJob(started);
                      setView('job');
                    }}
                    onJump={jump}
                  />
                ) : (
                  <EvidenceMap guide={detail} onJump={(id) => jump(id)} />
                )}
              </div>
            </div>
          </>
        )}
        {view !== 'job' && !detail && guides && guides.length === 0 && !error && (
          <div className="guide-empty">
            <h2>No field guides yet</h2>
            <p>
              A field guide is a long-form, cited write-up the agent composes from the transcript
              corpus. Import a hand-made page or write one:
            </p>
            <p>
              <code>python -m src.cli guides import page.html --slug my-guide</code>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
