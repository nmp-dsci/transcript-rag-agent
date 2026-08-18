import { useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type { Chunk, VideoChunkEnrichment, Video } from '../api/types';
import { fmtSeconds, videoTimestampUrl } from '../answers/render';

interface Props {
  video: Video | null;
  chunks: Chunk[] | undefined;
  selectedChunk: number | null;
  /** Absent in demo mode — asking is disabled there, so no button renders. */
  onAskAbout?: (url: string) => void;
}

function timestampUrl(video: Video, chunk: Chunk): string | null {
  return videoTimestampUrl(chunk.source_url || video.source_url, chunk.start_seconds);
}

export function VideoDetail({ video, chunks, selectedChunk, onAskAbout }: Props) {
  const selectedRef = useRef<HTMLDivElement>(null);
  const [enrichment, setEnrichment] = useState<VideoChunkEnrichment | null>(null);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [selectedChunk, chunks]);

  useEffect(() => {
    setEnrichment(null);
    if (!video) return;
    let live = true;
    void api
      .chunkEnrichment(video.video_id)
      .then((data) => {
        if (live) setEnrichment(data);
      })
      .catch(() => {
        if (live) setEnrichment({ chunks: {} });
      });
    return () => {
      live = false;
    };
  }, [video?.video_id]);

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
        {video.thumbnail_url ? (
          <img className="vthumb" src={video.thumbnail_url} alt="" loading="lazy" />
        ) : null}
        <span className="t">{video.title || video.video_id}</span>
        <span className="m">{meta}</span>
        {video.source_url && onAskAbout ? (
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
          const chunkEnrichment = enrichment?.chunks[String(chunk.chunk_index)];
          return (
            <div
              className={`chunkcard${selected ? ' on' : ''}`}
              key={chunk.chunk_index}
              ref={selected ? selectedRef : undefined}
            >
              <div className="cbody">
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
              <div className="cgraph">
                <span className="microlabel">graph rag</span>
                {enrichment === null ? (
                  <span className="cgraph-empty">loading…</span>
                ) : !chunkEnrichment ? (
                  <span className="cgraph-empty">not extracted</span>
                ) : (
                  <>
                    <div className="cgraph-entities">
                      {chunkEnrichment.entities.map((name) => (
                        <span className="badge acc" key={name}>
                          {name}
                        </span>
                      ))}
                    </div>
                    <ul className="cgraph-claims">
                      {chunkEnrichment.claims.map((claim) => (
                        <li key={claim.id} className={`polarity-${claim.polarity}`}>
                          {claim.text}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
