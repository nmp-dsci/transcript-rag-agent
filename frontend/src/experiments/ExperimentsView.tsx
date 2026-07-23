import { useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import type { AblationRun, Experiments, GoldenRunSummary } from '../api/types';
import { useExperimentStyles } from './styles';

/** 3-decimal fixed, or an em dash for a metric a run did not report. */
function fmt(value: number | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}

function signed(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

/** The averages for one config under the chosen domain view ("overall" or a domain). */
function averagesFor(
  cell: AblationRun['cells'][number],
  domain: string,
): Record<string, number> {
  return domain === 'overall' ? cell.averages : (cell.by_domain[domain] ?? {});
}

function AblationTable({ run }: { run: AblationRun }) {
  const domains = useMemo(() => {
    const found = new Set<string>();
    for (const cell of run.cells) for (const key of Object.keys(cell.by_domain)) found.add(key);
    return ['overall', ...Array.from(found).sort()];
  }, [run]);
  const [domain, setDomain] = useState('overall');
  const view = domains.includes(domain) ? domain : 'overall';

  // Higher is better for every metric here, so the column best is its max.
  const bestByMetric = useMemo(() => {
    const best: Record<string, number> = {};
    for (const cell of run.cells) {
      const averages = averagesFor(cell, view);
      for (const metric of run.metrics) {
        const value = averages[metric];
        if (typeof value === 'number' && (best[metric] === undefined || value > best[metric])) {
          best[metric] = value;
        }
      }
    }
    return best;
  }, [run, view]);

  return (
    <section className="exp-card">
      <div className="exp-cardhead">
        <div>
          <h3>Retrieval ablation</h3>
          <span className="exp-sub">
            {run.run_id} · {run.entries} golden questions · baseline{' '}
            <code>{run.baseline}</code>
          </span>
        </div>
        {domains.length > 1 && (
          <div className="exp-seg" role="tablist" aria-label="Domain">
            {domains.map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={view === name}
                className={view === name ? 'on' : ''}
                onClick={() => setDomain(name)}
              >
                {name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>config</th>
              {run.metrics.map((metric) => (
                <th key={metric} className="num">
                  {metric}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {run.cells.map((cell) => {
              const averages = averagesFor(cell, view);
              return (
                <tr key={cell.label}>
                  <td className="exp-cfg">
                    {cell.label}
                    {cell.label === run.baseline && <span className="exp-basetag">base</span>}
                  </td>
                  {run.metrics.map((metric) => {
                    const value = averages[metric];
                    const isBest =
                      typeof value === 'number' && value === bestByMetric[metric];
                    return (
                      <td key={metric} className={`num${isBest ? ' best' : ''}`}>
                        {fmt(value)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {run.deltas.length > 0 && (
        <div className="exp-deltas">
          <span className="microlabel">deltas vs {run.baseline}</span>
          {run.deltas.map((delta) => (
            <div key={delta.label} className="exp-deltarow">
              <span className="exp-cfg">{delta.label}</span>
              <span className="exp-chips">
                {run.metrics.map((metric) => {
                  const value = delta.vs_baseline[metric];
                  if (value === undefined) return null;
                  const cls = value > 0.0005 ? 'pos' : value < -0.0005 ? 'neg' : 'flat';
                  return (
                    <span key={metric} className={`exp-delta ${cls}`}>
                      {metric} {signed(value)}
                    </span>
                  );
                })}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function GoldenRuns({ runs }: { runs: GoldenRunSummary[] }) {
  return (
    <section className="exp-card">
      <div className="exp-cardhead">
        <h3>End-to-end golden runs</h3>
      </div>
      <div className="exp-goldlist">
        {runs.map((run) => {
          const averages = run.summary.averages ?? {};
          const rerank = (run.config as { rerank_enabled?: boolean }).rerank_enabled;
          const mode = (run.config as { retrieval_mode?: string }).retrieval_mode;
          const judge = (run.config as { judge_model?: string }).judge_model;
          return (
            <div key={run.run_id} className="exp-gold">
              <div className="exp-goldhead">
                <b>{run.setup}</b>
                <span className="exp-tags">
                  <span className="exp-tag">{mode ?? '—'}</span>
                  {rerank && <span className="exp-tag">+rerank</span>}
                  <span className="exp-tag">judge {judge ?? '—'}</span>
                </span>
                <span className="exp-sub">
                  {run.run_id} · {run.summary.scored ?? '—'}/{run.summary.entries ?? '—'} scored
                </span>
              </div>
              <div className="exp-goldmetrics">
                {['context_recall', 'faithfulness', 'answer_relevancy', 'composite'].map(
                  (metric) => (
                    <span key={metric} className="exp-gm">
                      <span className="microlabel">{metric}</span>
                      <b>{fmt(averages[metric])}</b>
                    </span>
                  ),
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function ExperimentsView() {
  useExperimentStyles();
  const [data, setData] = useState<Experiments | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        setData(await api.experiments());
        setError(null);
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  const ablations = data?.ablations ?? [];
  const goldenRuns = data?.golden_runs ?? [];
  const nothing = data !== null && ablations.length === 0 && goldenRuns.length === 0;

  return (
    <div className="scrollview">
      <div className="pagewrap">
        <p className="exp-intro">
          Committed retrieval experiments — the ablation sweeps and end-to-end golden runs
          under <code>evals/runs/</code>. Every number here is reproducible from a snapshot a
          reviewer can open in the repo.
        </p>

        {error && <p className="exp-empty">Could not load experiments: {error}</p>}

        {nothing && (
          <p className="exp-empty">
            No committed experiment runs yet. Generate them with{' '}
            <code>uv run python -m src.cli eval-ablation</code> and{' '}
            <code>uv run python -m src.cli eval-golden</code>.
          </p>
        )}

        {ablations.map((run) => (
          <AblationTable key={run.run_id} run={run} />
        ))}

        {goldenRuns.length > 0 && <GoldenRuns runs={goldenRuns} />}
      </div>
    </div>
  );
}
