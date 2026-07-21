import type { ReactNode } from 'react';

import type { Corpus } from '../api/types';
import {
  type TreeFilter,
  applyFilter,
  insightBadgeClass,
  insightFilter,
  insightKey,
  summarisedCount,
} from './insights';

interface Props {
  corpus: Corpus | null;
  embeddingModel: string | null;
  filter: TreeFilter | null;
  onFilterChange: (filter: TreeFilter | null) => void;
  /** Rendered at the end of the header row — the pipeline's sub-tab switch. */
  children?: ReactNode;
}

export function CorpusSummary({
  corpus,
  embeddingModel,
  filter,
  onFilterChange,
  children,
}: Props) {
  const videos = corpus?.videos ?? [];
  const totals = corpus?.totals ?? { videos: 0, chunks: 0, channels: 0 };
  const insights = corpus?.insights ?? [];
  const summarised = summarisedCount(videos);

  const stats: { value: string; label: string; wide?: boolean }[] = [
    { value: String(totals.videos), label: 'videos' },
    { value: String(totals.chunks), label: 'chunks' },
    { value: String(totals.channels), label: 'channels' },
    { value: `${summarised}/${totals.videos}`, label: 'with summaries' },
    { value: embeddingModel ?? '—', label: 'embedding model', wide: true },
  ];

  return (
    <header className="pipe-head">
      <div className="pipe-stats">
        {stats.map((stat) => (
          <div className={`pipe-stat${stat.wide ? ' wide' : ''}`} key={stat.label}>
            <b>{stat.value}</b>
            <span>{stat.label}</span>
          </div>
        ))}
        <div className="spacer" />
        {children}
      </div>

      {insights.length > 0 || filter ? (
        <div className="pipe-insights">
          <span className="microlabel">corpus health</span>
          {insights.map((insight, index) => {
            const key = insightKey(insight, index);
            const target = insightFilter(insight, index);
            const active = filter?.key === key;
            const badge = `badge ${insightBadgeClass(insight.level)}`;
            if (!target) {
              return (
                <span className={badge} key={key}>
                  {insight.message}
                </span>
              );
            }
            const count = applyFilter(videos, target).length;
            return (
              <button
                type="button"
                key={key}
                className={`${badge} pipe-chip${active ? ' on' : ''}`}
                aria-pressed={active}
                title={
                  active
                    ? 'Clear this filter'
                    : `Show the ${count} video${count === 1 ? '' : 's'} this affects`
                }
                onClick={() => onFilterChange(active ? null : target)}
              >
                {insight.message}
                <span className="pipe-chip-go" aria-hidden="true">
                  {active ? '×' : '→'}
                </span>
              </button>
            );
          })}
          {filter ? (
            <button
              type="button"
              className="pill on pipe-clear"
              onClick={() => onFilterChange(null)}
            >
              filtered: {filter.label} · clear
            </button>
          ) : null}
        </div>
      ) : null}
    </header>
  );
}
