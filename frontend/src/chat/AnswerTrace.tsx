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
/**
 * What the wrapping note under a step is, named by the step that carries it.
 *
 * A label of "note" would be true of every one of them and useful for none;
 * "videos" tells a reader the line below is the routing decision they came to
 * check, and "warning" tells them the line below is a caveat on everything
 * under it rather than more description of the step.
 */
function noteLabel(phase: TraceStep['phase']): string {
  if (phase === 'filter') return 'videos';
  if (phase === 'route') return 'warning';
  return 'note';
}

export function AnswerTrace({ steps }: { steps: TraceStep[] }) {
  if (!steps.length) return null;
  return (
    <details className="trace">
      <summary>
        trace — {steps.length} step{steps.length === 1 ? '' : 's'}
      </summary>
      <div style={{ marginTop: 6 }}>
        {steps.map((step, index) => (
          <div key={`${step.label}-${index}`}>
            <div className="trace-line">
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
            {/*
              On its own wrapping line rather than inside the truncated detail:
              a query is what makes a retrieval checkable, and a query you can
              only see the first eight words of is one you cannot check.
            */}
            {step.query ? (
              <div className="trace-query">
                <span className="trace-query-label">query</span>
                <span>{step.query}</span>
              </div>
            ) : null}
            {/*
              Same treatment, same reason, one field over: the detail line
              clips at about 66 characters, so a 273-character coverage caveat
              shows its problem and hides its instruction, and a list of matched
              videos shows the first two of five. Both are the point of their
              step, so both wrap here instead.
            */}
            {step.note ? (
              <div className="trace-query trace-note">
                <span className="trace-query-label">{noteLabel(step.phase)}</span>
                <span>{step.note}</span>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </details>
  );
}
