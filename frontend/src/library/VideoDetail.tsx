import { useEffect, useRef } from 'react';

import type { Chunk, Video } from '../api/types';
import { fmtSeconds } from '../answers/render';

interface Props {
  video: Video | null;
  chunks: Chunk[] | undefined;
  selectedChunk: number | null;
  onAskAbout: (url: string) => void;
}

function timestampUrl(video: Video, chunk: Chunk): string | null {
  const base = chunk.source_url || video.source_url;
  if (!base) return null;
  const seconds = Math.floor(chunk.start_seconds ?? 0);
  return `${base}${base.includes('?') ? '&' : '?'}t=${seconds}`;
}

export function VideoDetail({ video, chunks, selectedChunk, onAskAbout }: Props) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [selectedChunk, chunks]);

  if (!video) {
    return (
      <div className="detail">
        <div className="empty">
          <h2>Explore the corpus</h2>
          <p>
            Pick a video in the tree to read its chunks — the exact units retrieval returns —
            or run a query in the Retrieval Lab above to compare BM25 against semantic ranking.
          </p>
        </div>
      </div>
    );
  }

  const meta = [
    video.channel_name,
    video.duration_seconds ? fmtSeconds(video.duration_seconds) : null,
    video.upload_date ? String(video.upload_date) : null,
    `${video.chunk_count} chunks`,
    video.view_count ? `${video.view_count.toLocaleString()} views` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div className="detail">
      <div className="vhead">
        <span className="t">{video.title || video.video_id}</span>
        <span className="m">{meta}</span>
        {video.source_url ? (
          <button
            type="button"
            className="btn sm"
            style={{ marginLeft: 'auto' }}
            onClick={() => onAskAbout(video.source_url!)}
          >
            Ask about this →
          </button>
        ) : null}
      </div>

      {video.summary ? (
        <details style={{ marginTop: 10 }}>
          <summary style={{ cursor: 'pointer', color: 'var(--muted)', fontSize: 11.5 }}>
            transcript summary
          </summary>
          <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>{video.summary}</p>
        </details>
      ) : null}

      {chunks === undefined ? (
        <div className="waiting" style={{ marginTop: 14 }}>
          <span className="pulse" />
          loading chunks…
        </div>
      ) : chunks.length === 0 ? (
        <div className="rankempty">No chunks stored for this video.</div>
      ) : (
        chunks.map((chunk) => {
          const selected = chunk.chunk_index === selectedChunk;
          const link = timestampUrl(video, chunk);
          return (
            <div
              className={`chunkcard${selected ? ' on' : ''}`}
              key={chunk.chunk_index}
              ref={selected ? selectedRef : undefined}
            >
              <div className="h">
                <span className="id">#c{chunk.chunk_index}</span>
                <span>
                  {fmtSeconds(chunk.start_seconds)}–{fmtSeconds(chunk.end_seconds)}
                </span>
                {chunk.segment_count ? <span>{chunk.segment_count} segments</span> : null}
                {link ? (
                  <a href={link} target="_blank" rel="noreferrer">
                    ▸ open at {fmtSeconds(chunk.start_seconds)}
                  </a>
                ) : null}
              </div>
              <p>{chunk.text}</p>
            </div>
          );
        })
      )}
    </div>
  );
}
