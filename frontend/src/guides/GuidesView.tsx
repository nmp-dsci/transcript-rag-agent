import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../api/client';
import type { GuideDetail, GuideSummary } from '../api/types';
import { useDemo } from '../demo';
import { GuideReader, type GuideReaderHandle, type GuideSection } from './GuideReader';
import { useGuidesStyles } from './styles';

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
  return (
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

  useEffect(() => {
    rememberGuide(slug);
    setSections([]);
    if (!slug) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    api
      .guide(slug)
      .then((found) => {
        if (!cancelled) setDetail(found);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  const onReady = useCallback((found: GuideSection[]) => setSections(found), []);

  const pageUrl = useMemo(() => {
    if (!detail) return null;
    // The version is part of the URL so a republished page is never served
    // from a stale iframe cache.
    return `${detail.html_url}?v=${detail.current_version}`;
  }, [detail]);

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
          {guides === null && <div className="rail-empty">loading…</div>}
          {guides && guides.length === 0 && <div className="rail-empty">No guides yet.</div>}
          {guides?.map((guide) => (
            <GuideEntry key={guide.slug} guide={guide} selected={guide.slug === slug} onSelect={setSlug} />
          ))}
        </div>
        {!demo && writeCommand && (
          <div className="rail-foot">
            New guides are written from the terminal for now:
            <br />
            <code>{writeCommand}</code>
          </div>
        )}
      </aside>

      <div className="guides-main">
        {error && <div className="guide-error">{error}</div>}
        {detail && (
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
            {pageUrl && <GuideReader ref={reader} url={pageUrl} title={detail.title} onReady={onReady} />}
          </>
        )}
        {!detail && guides && guides.length === 0 && !error && (
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
