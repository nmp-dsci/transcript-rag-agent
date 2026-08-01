import type { TraceStep } from '../api/types';

/**
 * The persisted execution trace of a finished answer — every step its path
 * actually ran (route decisions, retrievals with chunk counts, rerank passes,
 * LLM calls), rehydrated from history so it survives a reload.
 *
 * Collapsed by default: the answer is the point; the workings are one click
 * away. The live-streaming view stays `AgentTrace` — this renders after the
 * fact, for every setup rather than only the agentic one.
 */
export function AnswerTrace({ steps }: { steps: TraceStep[] }) {
  if (!steps.length) return null;
  return (
    <details className="trace">
      <summary>
        trace — {steps.length} step{steps.length === 1 ? '' : 's'}
      </summary>
      <div style={{ marginTop: 6 }}>
        {steps.map((step, index) => (
          <div className="trace-line" key={`${step.label}-${index}`}>
            <span className={`ph ph-${step.phase}`}>{step.phase}</span>
            <span className="n">{step.label}</span>
            <span className="q" title={step.detail}>
              {step.detail}
            </span>
            <span className="c">
              {[
                step.chunk_ids.length ? `${step.chunk_ids.length} chunks` : '',
                step.elapsed_ms != null ? `${step.elapsed_ms}ms` : '',
              ]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
        ))}
      </div>
    </details>
  );
}
