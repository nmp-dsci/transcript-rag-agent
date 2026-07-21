import { useState } from 'react';

import type { Answer } from '../api/types';
import { cleanAnswer, escapeHtml, fmtSeconds, renderAnswer } from '../answers/render';

const CLAMP_AT = 1600;

function metaChips(answer: Answer): string[] {
  const chips = [`~${answer.token_estimate} tok`, `${answer.chunk_count} chunks`];
  if (answer.llm_calls != null) chips.push(`${answer.llm_calls} LLM calls`);
  if (answer.iterations != null) chips.push(`${answer.iterations} iterations`);
  if (answer.elapsed_seconds) chips.push(`${answer.elapsed_seconds}s`);
  if (answer.top_k != null) chips.push(`top_k ${answer.top_k}`);
  if (answer.model) chips.push(answer.model);
  if (answer.terminated_reason) chips.push(answer.terminated_reason);
  return chips;
}

export function AnswerBody({ answer }: { answer: Answer }) {
  const [expanded, setExpanded] = useState(false);

  if (answer.error) {
    return <div className="errtext">Error: {answer.error}</div>;
  }

  const cleaned = cleanAnswer(answer.answer);
  const long = cleaned.length > CLAMP_AT;
  const references = answer.references ?? [];

  return (
    <>
      <div
        className={`body${long && !expanded ? ' clamp' : ''}`}
        // renderAnswer escapes all agent text before building this markup.
        dangerouslySetInnerHTML={{ __html: renderAnswer(cleaned, references) }}
      />
      {long ? (
        <button type="button" className="linkbtn" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less ▴' : 'Show full answer ▾'}
        </button>
      ) : null}

      <div className="chips">
        {metaChips(answer).map((chip) => (
          <span className="chip" key={chip}>
            {chip}
          </span>
        ))}
      </div>

      {references.length ? (
        <details className="refs">
          <summary>Sources ({references.length})</summary>
          <ul>
            {references.map((reference, index) => (
              <li key={`${reference.label ?? index}-${index}`}>
                <span className="rnum">{reference.label ?? '[?]'}</span>
                <a
                  href={reference.timestamp_url || reference.source_url || '#'}
                  target="_blank"
                  rel="noreferrer"
                >
                  open
                  {reference.start_seconds != null ? ` at ${fmtSeconds(reference.start_seconds)}` : ''}
                </a>
                <span className="vid">{reference.video_id ?? ''}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <details className="cmd">
        <summary>command</summary>
        <pre dangerouslySetInnerHTML={{ __html: escapeHtml(answer.command) }} />
      </details>
    </>
  );
}
