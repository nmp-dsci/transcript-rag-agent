import { useState } from 'react';

import { api } from '../api/client';
import type { RankMode, RankRow, Rankings } from '../api/types';
import { fmtSeconds } from '../answers/render';

const MODE_CHOICES: { label: string; modes: RankMode[] }[] = [
  { label: 'Semantic', modes: ['semantic'] },
  { label: 'BM25', modes: ['bm25'] },
  { label: 'Both', modes: ['semantic', 'bm25'] },
];

const MODE_META: Record<RankMode, { title: string; note: string }> = {
  semantic: { title: 'SEMANTIC', note: 'cosine · embeddings' },
  bm25: { title: 'BM25', note: 'keyword · Okapi' },
};

/** How a chunk's rank here compares with its rank in the other mode. */
export function movement(row: RankRow): { className: string; label: string } | null {
  if (row.other_rank == null) return { className: 'only', label: 'only here' };
  const delta = row.other_rank - row.rank;
  if (delta === 0) return { className: 'same', label: '=' };
  return delta > 0
    ? { className: 'up', label: `↑${delta}` }
    : { className: 'dn', label: `↓${Math.abs(delta)}` };
}

interface Props {
  scopeVideoId: string | null;
  scopeLabel: string;
  onSelectChunk: (videoId: string, chunkIndex: number) => void;
  selectedChunk: string | null;
}

export function RetrievalLab({
  scopeVideoId,
  scopeLabel,
  onSelectChunk,
  selectedChunk,
}: Props) {
  const [query, setQuery] = useState('');
  const [choice, setChoice] = useState(2);
  const [topK, setTopK] = useState(10);
  const [result, setResult] = useState<Rankings | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async () => {
    const trimmed = query.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.rank(trimmed, MODE_CHOICES[choice]!.modes, topK, scopeVideoId));
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const modes = result ? (Object.keys(result.modes) as RankMode[]) : [];
  // Chunk indices restart per video, so "#c11" alone is ambiguous whenever the
  // results span more than one video.
  const multiVideo =
    new Set(
      modes.flatMap((mode) => (result?.modes[mode] ?? []).map((row) => row.video_id)),
    ).size > 1;

  return (
    <div className="lab">
      <div className="labrow">
        <span className="microlabel" style={{ color: 'var(--accent2)' }}>
          retrieval lab
        </span>
        <input
          type="search"
          value={query}
          placeholder="Rank the corpus for a query — see where keyword and semantic disagree…"
          aria-label="Retrieval query"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void search();
          }}
        />
        <span className="chipselect">Scope: {scopeLabel}</span>
        <label className="toggle">
          top_k
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            style={{ width: 58 }}
            onChange={(event) => setTopK(Number(event.target.value) || 10)}
          />
        </label>
        <div className="modes" role="group" aria-label="Ranking mode">
          {MODE_CHOICES.map((option, index) => (
            <button
              key={option.label}
              type="button"
              className={index === choice ? 'on' : ''}
              onClick={() => setChoice(index)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <button type="button" className="btn pri" onClick={() => void search()} disabled={busy || !query.trim()}>
          {busy ? 'Ranking…' : 'Rank'}
        </button>
        {result && modes.length === 2 ? (
          <span className="badge acc" title="Chunks both modes agreed on">
            overlap {result.overlap.count}/{result.overlap.of}
          </span>
        ) : null}
      </div>

      {error ? <div className="errtext" style={{ marginTop: 8 }}>{error}</div> : null}

      {result ? (
        <div className="ranks">
          {modes.map((mode) => {
            const rows = result.modes[mode] ?? [];
            return (
              <div className="rankcol" key={mode}>
                <div className="h">
                  {MODE_META[mode].title}
                  <span className="badge plain">{MODE_META[mode].note}</span>
                </div>
                {rows.length === 0 ? (
                  <div className="rankempty">No matches for this query.</div>
                ) : (
                  rows.map((row) => {
                    const move = modes.length === 2 ? movement(row) : null;
                    return (
                      <button
                        type="button"
                        key={row.chunk_id}
                        className={`rrow${row.chunk_id === selectedChunk ? ' on' : ''}`}
                        title={row.preview}
                        onClick={() =>
                          row.video_id != null &&
                          row.chunk_index != null &&
                          onSelectChunk(row.video_id, row.chunk_index)
                        }
                      >
                        <span className="rk">{row.rank}</span>
                        <span className="cid">
                          {multiVideo && row.video_id ? `${row.video_id.slice(0, 4)}·` : ''}#c
                          {row.chunk_index}
                        </span>
                        <span className="tx">{row.preview}</span>
                        {row.start_seconds != null ? (
                          <span className="rk" style={{ width: 'auto' }}>
                            {fmtSeconds(row.start_seconds)}
                          </span>
                        ) : null}
                        <span className="sc">{row.score?.toFixed(2) ?? '—'}</span>
                        {move ? <span className={`mv ${move.className}`}>{move.label}</span> : null}
                      </button>
                    );
                  })
                )}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
