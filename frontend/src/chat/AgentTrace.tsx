import type { AgentStep } from '../api/types';

/**
 * The agentic setup's research loop, rendered as it happens.
 *
 * Each iteration emits `retrieval_start` (the query it chose) and then
 * `retrieval_complete` (how many chunks came back), so a step is "live" until
 * its completion event arrives. Without this the agentic default would look
 * like a 30-second stall.
 */
interface Props {
  steps: AgentStep[];
  running: boolean;
}

interface Iteration {
  iteration: number;
  query: string;
  chunkCount: number | null;
}

function toIterations(steps: AgentStep[]): Iteration[] {
  const byIteration = new Map<number, Iteration>();
  for (const step of steps) {
    if (step.event_type === 'answer_start') continue;
    const existing = byIteration.get(step.iteration);
    byIteration.set(step.iteration, {
      iteration: step.iteration,
      query: step.query ?? existing?.query ?? '',
      chunkCount:
        step.event_type === 'retrieval_complete'
          ? (step.chunk_count ?? 0)
          : (existing?.chunkCount ?? null),
    });
  }
  return [...byIteration.values()].sort((a, b) => a.iteration - b.iteration);
}

export function AgentTrace({ steps, running }: Props) {
  const iterations = toIterations(steps);
  if (!iterations.length) return null;

  const writing = steps.some((step) => step.event_type === 'answer_start');
  const body = (
    <>
      {iterations.map((item) => (
        <div
          className={`trace-line${item.chunkCount == null ? ' live' : ''}`}
          key={item.iteration}
        >
          <span className="n">[{item.iteration}]</span>
          <span className="q">{item.query || 'retrieving…'}</span>
          <span className="c">
            {item.chunkCount == null ? 'searching…' : `${item.chunkCount} chunks`}
          </span>
        </div>
      ))}
      {writing && running ? (
        <div className="trace-line">
          <span className="n">✓</span>
          <span className="q">writing the answer…</span>
        </div>
      ) : null}
    </>
  );

  // Live: always visible. Finished: collapsed, because the answer is the point.
  if (running) return <div className="trace">{body}</div>;
  return (
    <details className="trace">
      <summary>
        research trace — {iterations.length} retrieval
        {iterations.length === 1 ? '' : 's'}
      </summary>
      <div style={{ marginTop: 6 }}>{body}</div>
    </details>
  );
}
