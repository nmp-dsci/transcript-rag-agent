import { useEffect, useRef } from 'react';

import type { WebChunk, WebSourceSummary } from '../api/types';

interface Props {
  source: WebSourceSummary | null;
  chunks: WebChunk[] | undefined;
  selectedChunk: number | null;
  /** Absent in demo mode — asking is disabled there, so no button renders. */
  onAskAbout?: (url: string) => void;
}

/** What each recorded state means, in the terms a reader can act on.
 *
 * These are recorded by the refresh loop, never inferred, so the pane can
 * state them plainly rather than hedging. */
const STATE_NOTE: Record<string, string> = {
  changed: 'The page changed at the last check, and these chunks are the new text.',
  moved: 'The page moved. The document kept its identity; the link points at where it lives now.',
  gone: 'The page no longer resolves. The chunks are kept — an answer built on them is still real — but a citation into it cannot be re-checked.',
  blocked: 'robots.txt now disallows this source, so it is no longer fetched. The stored copy is what it was when access was last permitted.',
  truncated: 'The body hit the fetch byte cap, so this is part of a document rather than all of it.',
};

/** Where a chunk's heading links to, when the page offers an anchor for it. */
function anchorUrl(source: WebSourceSummary, chunk: WebChunk): string | null {
  const base = chunk.url || source.url;
  if (!base) return null;
  return chunk.anchor ? `${base}#${chunk.anchor}` : base;
}

export function WebSourceDetail({ source, chunks, selectedChunk, onAskAbout }: Props) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [selectedChunk, chunks]);

  if (!source) return null;

  const meta = [
    `revision ${source.revision}`,
    `${source.chunk_count} chunks`,
    source.section_count ? `${source.section_count} sections` : null,
    source.word_count ? `${source.word_count.toLocaleString()} words` : null,
    source.from_feed ? 'from the feed' : null,
    source.published_at ? source.published_at.slice(0, 10) : null,
  ]
    .filter(Boolean)
    .join(' · ');

  const note = STATE_NOTE[source.state];

  return (
    <div className="detail">
      <div className="vhead">
        <span className="t">{source.title || source.url}</span>
        <span className="m">{meta}</span>
        {source.verifiable && onAskAbout ? (
          <button
            type="button"
            className="btn sm"
            style={{ marginLeft: 'auto' }}
            onClick={() => onAskAbout(source.url)}
          >
            Ask about this →
          </button>
        ) : null}
      </div>

      <div className="websrc-meta">
        {source.verifiable ? (
          <a href={source.url} target="_blank" rel="noreferrer">
            {source.url}
          </a>
        ) : (
          <span className="websrc-dead" title={source.state_reason ?? undefined}>
            {source.url}
          </span>
        )}
      </div>

      {note ? (
        <p className={`websrc-note${source.verifiable ? '' : ' bad'}`}>
          {note}
          {source.state_reason ? ` (${source.state_reason})` : ''}
        </p>
      ) : null}

      {chunks === undefined ? (
        <div className="waiting" style={{ marginTop: 14 }}>
          <span className="pulse" />
          loading chunks…
        </div>
      ) : chunks.length === 0 ? (
        <div className="rankempty">No chunks stored for this document.</div>
      ) : (
        chunks.map((chunk) => {
          const selected = chunk.chunk_index === selectedChunk;
          const link = anchorUrl(source, chunk);
          return (
            <div
              className={`chunkcard${selected ? ' on' : ''}`}
              key={chunk.chunk_index}
              ref={selected ? selectedRef : undefined}
            >
              <div className="cbody">
                <div className="h">
                  <span className="id">#c{chunk.chunk_index}</span>
                  {/* The section heading is this corpus half's citation unit,
                      the way mm:ss is the transcript half's. */}
                  <span>{chunk.heading ? `§ ${chunk.heading}` : 'no heading'}</span>
                  {chunk.part_index ? <span>part {chunk.part_index + 1}</span> : null}
                  {link && source.verifiable ? (
                    <a href={link} target="_blank" rel="noreferrer">
                      ▸ open{chunk.anchor ? ' at this section' : ' the page'}
                    </a>
                  ) : null}
                </div>
                <p>{chunk.text}</p>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
