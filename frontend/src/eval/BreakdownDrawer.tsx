import { useEffect, useRef } from 'react';

import type { Evaluation, EvaluationDetails } from '../api/types';
import {
  faithfulnessArithmetic,
  precisionArithmetic,
  relevancyArithmetic,
  spreadRange,
} from './breakdown';
import { explainerFor } from './explainers';
import { MetricExplainers } from './MetricExplainer';
import { useEvalStyles } from './styles';

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

function num(value: number, digits = 2): string {
  return value.toFixed(digits);
}

/** ✓ / ✗ with the verdict spelled out, so the mark never carries meaning alone. */
function Verdict({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return (
    <span className="v">
      <span aria-hidden="true">{ok ? '✓' : '✗'}</span>
      {ok ? yes : no}
    </span>
  );
}

/** How the derived arithmetic lines up with the score on the strip. */
function Reconciliation({ derived, reported }: { derived: number; reported?: number }) {
  if (reported == null) return null;
  const agrees = Math.abs(derived - reported) < 0.005;
  return (
    <span className="recon">
      {agrees
        ? `matches the ${num(reported)} reported for this metric`
        : `reported ${num(reported)} — the stored score was produced by a different run`}
    </span>
  );
}

function FaithfulnessBody({
  detail,
  reported,
}: {
  detail: NonNullable<EvaluationDetails['faithfulness']>;
  reported?: number;
}) {
  const { claims, supported, total } = detail;
  const derived = faithfulnessArithmetic(supported, total);
  const failing = claims.filter((claim) => !claim.verdict).length;
  return (
    <>
      <p className="bd-formula">
        <b>faithfulness = supported claims ÷ total claims</b>
      </p>
      <p className="bd-sub">
        The judge broke the answer into {total} standalone claim{total === 1 ? '' : 's'} and
        checked each one against the retrieved chunks.{' '}
        {failing > 0
          ? `${failing} of them ${failing === 1 ? 'is' : 'are'} not backed by the chunks — those are the lines to fix.`
          : 'Every claim is backed by the chunks.'}
      </p>
      {claims.map((claim, index) => (
        <div className={`bd-claim ${claim.verdict ? 'ok' : 'no'}`} key={`${index}-${claim.claim}`}>
          <Verdict ok={Boolean(claim.verdict)} yes="supported" no="not supported" />
          <div className="txt">{claim.claim}</div>
          {claim.reason ? <div className="why">{claim.reason}</div> : null}
        </div>
      ))}
      <div className="bd-arith">
        <span className="eq">
          {supported} ÷ {total} = {derived == null ? '—' : num(derived)}
        </span>
        {derived == null ? null : <Reconciliation derived={derived} reported={reported} />}
      </div>
    </>
  );
}

function RelevancyBody({
  detail,
  question,
  reported,
}: {
  detail: NonNullable<EvaluationDetails['answer_relevancy']>;
  question?: string;
  reported?: number;
}) {
  const { generated_questions: generated, noncommittal, similarities } = detail;
  const { mean, multiplier, score } = relevancyArithmetic(similarities, noncommittal);
  return (
    <>
      <p className="bd-formula">
        <b>
          answer relevancy = mean cosine(original question, generated questions) × (0 if
          noncommittal else 1)
        </b>
      </p>
      <p className="bd-sub">
        The judge read only the answer and wrote the question it thinks was being answered. The
        closer that reconstruction is to the real question, the more on-topic the answer.
      </p>
      {question ? (
        <div className="bd-qa">
          <div className="k">original question</div>
          <div className="q">{question}</div>
        </div>
      ) : null}
      <div className="tblwrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Question the judge reconstructed</th>
              <th>Cosine</th>
            </tr>
          </thead>
          <tbody>
            {generated.map((text, index) => (
              <tr key={`${index}-${text}`}>
                <td className="num">{index + 1}</td>
                <td className="bd-prev">{text || <em>(empty)</em>}</td>
                <td className="num">
                  {similarities[index] == null ? '—' : num(similarities[index] as number, 3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="eval-flags">
        <span className={`badge ${noncommittal ? 'bad' : 'good'}`}>
          {noncommittal ? 'noncommittal → ×0' : 'committal → ×1'}
        </span>
        <span className="microlabel">
          {noncommittal
            ? 'the judge read the answer as a hedge or refusal, which zeroes the metric'
            : 'the answer commits to a position, so the cosine mean stands'}
        </span>
      </div>
      <div className="bd-arith">
        <span className="eq">
          mean({similarities.map((value) => num(value, 3)).join(', ') || '—'}) = {num(mean, 3)} ×{' '}
          {multiplier} = {num(score)}
        </span>
        <Reconciliation derived={score} reported={reported} />
      </div>
    </>
  );
}

function PrecisionBody({
  detail,
  reported,
}: {
  detail: NonNullable<EvaluationDetails['context_precision']>;
  reported?: number;
}) {
  const { verdicts } = detail;
  const { steps, usefulCount, sum, score } = precisionArithmetic(
    verdicts.map((entry) => entry.verdict),
  );
  const terms = steps
    .filter((step) => step.precisionAtK != null)
    .map((step) => `${step.usefulSoFar}/${step.rank}`);
  return (
    <>
      <p className="bd-formula">
        <b>context precision = average precision = mean of precision@k over the useful ranks</b>
      </p>
      <p className="bd-sub">
        Each retrieved chunk gets a useful / not-useful verdict in the order retrieval returned
        them. A useful chunk only earns useful-so-far ÷ its rank, so the same chunk is worth 1.00
        at rank 1 and 0.20 at rank 5 — rank is the whole point of this metric.
      </p>
      <div className="tblwrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Verdict</th>
              <th>precision@k</th>
              <th>Chunk</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step, index) => {
              const entry = verdicts[index];
              return (
                <tr key={step.rank}>
                  <td className="num">{step.rank}</td>
                  <td>
                    <span className={`badge ${step.verdict ? 'good' : 'plain'}`}>
                      <span aria-hidden="true">{step.verdict ? '✓ ' : '✗ '}</span>
                      {step.verdict ? 'useful' : 'not useful'}
                    </span>
                  </td>
                  <td>
                    <span className={`bd-pk ${step.precisionAtK == null ? 'off' : 'on'}`}>
                      {step.precisionAtK == null
                        ? '— adds nothing'
                        : `${step.usefulSoFar}/${step.rank} = ${num(step.precisionAtK)}`}
                    </span>
                  </td>
                  <td>
                    <div className="bd-prev">{entry?.chunk_preview ?? ''}</div>
                    {entry?.reason ? <div className="bd-why">{entry.reason}</div> : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="bd-arith">
        <span className="eq">
          {usefulCount === 0
            ? 'no chunk was judged useful = 0.00'
            : `(${terms.join(' + ')}) ÷ ${usefulCount} = ${num(sum, 3)} ÷ ${usefulCount} = ${num(score)}`}
        </span>
        <Reconciliation derived={score} reported={reported} />
      </div>
    </>
  );
}

interface Props {
  metric: string;
  label: string;
  evaluation: Evaluation;
  question?: string;
  onClose: () => void;
}

/**
 * The judge's workings for one metric, in a dismissible drawer.
 *
 * Falls back to the static explainer when an evaluation predates stored
 * derivations, so an old score still teaches the metric instead of dead-ending.
 */
export function BreakdownDrawer({ metric, label, evaluation, question, onClose }: Props) {
  useEvalStyles();
  const dialog = useRef<HTMLDivElement>(null);
  const closer = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closer.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== 'Tab' || !dialog.current) return;
      // Keep Tab inside the drawer: it covers the whole screen, so tabbing out
      // to the thread behind it would strand focus somewhere invisible.
      const targets = [...dialog.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
      if (targets.length === 0) return;
      const first = targets[0] as HTMLElement;
      const last = targets[targets.length - 1] as HTMLElement;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('keydown', onKeyDown, true);
      previous?.focus?.();
    };
  }, [onClose]);

  const detail = evaluation.details?.[metric as keyof EvaluationDetails] ?? null;
  const reported = evaluation.scores?.[metric];
  const range = spreadRange(evaluation, metric);
  const explainer = explainerFor(metric);

  return (
    <>
      <div className="bd-backdrop" onClick={onClose} aria-hidden="true" />
      <div
        className="bd-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`${label} breakdown`}
        ref={dialog}
      >
        <div className="bd-head">
          <span className="microlabel">how this was scored</span>
          <h2>{label}</h2>
          <span className="composite">{reported == null ? '—' : num(reported)}</span>
          <button type="button" className="bd-close" onClick={onClose} ref={closer}>
            Close
          </button>
        </div>
        <div className="bd-body">
          {range ? (
            <div className="eval-flags" style={{ marginBottom: 10 }}>
              <span className="badge warn">
                ±{num(range.spread, 2)} across {range.samples} judge samples
              </span>
              <span className="microlabel">
                samples ranged {num(range.min)} – {num(range.max)}
              </span>
            </div>
          ) : null}
          {evaluation.self_graded === true ? (
            <div className="eval-flags" style={{ marginBottom: 10 }}>
              <span className="badge bad">self-graded</span>
              <span className="microlabel">
                {evaluation.judge_model} judged its own answer
              </span>
            </div>
          ) : null}

          {metric === 'faithfulness' && evaluation.details?.faithfulness ? (
            <FaithfulnessBody detail={evaluation.details.faithfulness} reported={reported} />
          ) : null}
          {metric === 'answer_relevancy' && evaluation.details?.answer_relevancy ? (
            <RelevancyBody
              detail={evaluation.details.answer_relevancy}
              question={question}
              reported={reported}
            />
          ) : null}
          {metric === 'context_precision' && evaluation.details?.context_precision ? (
            <PrecisionBody detail={evaluation.details.context_precision} reported={reported} />
          ) : null}

          {detail ? null : (
            <>
              <p className="bd-sub">
                This evaluation stored no workings for {label.toLowerCase()} — it was judged
                before derivations were recorded, or capture failed for this metric. Re-judge the
                question to get the claim-by-claim breakdown. In the meantime, here is what the
                metric measures.
              </p>
              {explainer ? <MetricExplainers only={metric} /> : null}
            </>
          )}
        </div>
      </div>
    </>
  );
}
