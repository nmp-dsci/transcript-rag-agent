import { useMemo, useState } from 'react';

import type { Entry } from '../api/types';
import { fmtTime } from '../answers/render';

interface Props {
  history: Entry[];
  selectedId: string | null;
  disabled: boolean;
  onSelect: (id: string) => void;
}

function bestComposite(entry: Entry): number | null {
  const values = entry.answers
    .map((answer) => answer.evaluation?.composite)
    .filter((value): value is number => value != null);
  return values.length ? Math.max(...values) : null;
}

export function HistoryRail({ history, selectedId, disabled, onSelect }: Props) {
  const [query, setQuery] = useState('');

  const entries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return history
      .slice()
      .sort((a, b) => (a.asked_at < b.asked_at ? 1 : -1))
      .filter((entry) => !needle || entry.question.toLowerCase().includes(needle));
  }, [history, query]);

  return (
    <aside className="rail">
      <div className="rail-head">
        <input
          type="search"
          value={query}
          placeholder="Search conversations…"
          autoComplete="off"
          aria-label="Search conversations"
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="rail-list">
        {entries.length === 0 ? (
          <div className="rail-empty">
            {history.length ? 'No matches.' : 'No questions yet.\nAsk your first one below.'}
          </div>
        ) : (
          entries.map((entry) => {
            const composite = bestComposite(entry);
            return (
              <button
                type="button"
                key={entry.id}
                className={`rentry${entry.id === selectedId ? ' on' : ''}`}
                disabled={disabled}
                onClick={() => onSelect(entry.id)}
              >
                <div className="rq">{entry.question}</div>
                <div className="rmeta">
                  <span className={`badge ${composite != null ? 'good' : 'plain'}`}>
                    {composite != null ? `✓ ${composite.toFixed(2)}` : 'unjudged'}
                  </span>
                  <span>
                    {entry.answers.length} ans
                  </span>
                  <span>{fmtTime(entry.asked_at)}</span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
