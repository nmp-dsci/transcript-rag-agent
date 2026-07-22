import { useEffect, useState } from 'react';

import type { Answer, AgentStep, Followup } from '../api/types';
import { AgentTrace } from './AgentTrace';
import { AnswerBody } from './AnswerBody';
import { ScoreStrip, fmtScore } from './ScoreStrip';

export interface RunningSetup {
  key: string;
  title: string;
  startedAt: number;
  steps: AgentStep[];
  error?: string;
}

interface Props {
  answers: Answer[];
  running: RunningSetup[];
  judging: boolean;
  busy?: boolean;
  onJudge?: () => void;
  onCompare?: () => void;
  remainingSetups: number;
  /**
   * Research steps for finished answers, by setup key. Held in session state
   * rather than the history file, so the trace survives until reload only.
   */
  traces?: Record<string, AgentStep[]>;
  /**
   * The question these answers reply to. Passed down to the score breakdown so
   * answer relevancy can show it beside the question the judge reconstructed.
   */
  question?: string;
  /** Ask a proposed follow-up, reusing this answer's retrieval scope. */
  onAskFollowup?: (query: string) => void;
}

/** Re-render on an interval so running timers advance. */
function useTick(active: boolean): void {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const id = window.setInterval(() => setTick((value) => value + 1), 250);
    return () => window.clearInterval(id);
  }, [active]);
}

function bestKey(answers: Answer[]): string | null {
  const scored = answers.filter((a) => a.evaluation?.composite != null);
  if (scored.length < 2) return null;
  return scored.reduce((best, current) =>
    (current.evaluation!.composite ?? 0) > (best.evaluation!.composite ?? 0) ? current : best,
  ).key;
}

/**
 * One assistant turn.
 *
 * Several setups answering the same question produce a single bubble with tabs
 * rather than one bubble each — the answers are alternatives to compare, not a
 * sequence of separate replies.
 */
export function MessageBubble({
  answers,
  running,
  judging,
  busy,
  onJudge,
  onCompare,
  remainingSetups,
  traces,
  question,
  onAskFollowup,
}: Props) {
  useTick(running.length > 0);

  const [activeKey, setActiveKey] = useState<string | null>(null);
  const winner = bestKey(answers);

  // Follow the run: show the first answer that lands, then settle on the winner
  // once judging has ranked them. An explicit tab click wins over both.
  const [pinned, setPinned] = useState(false);
  useEffect(() => {
    if (pinned) return;
    const preferred = winner ?? answers[0]?.key ?? null;
    if (preferred) setActiveKey(preferred);
  }, [answers, winner, pinned]);

  const active = answers.find((a) => a.key === activeKey) ?? answers[0];
  const showTabs = answers.length + running.length > 1;
  const unjudged = answers.some((a) => !a.evaluation && !a.error);

  return (
    <div className={`msg-bot${winner && winner === active?.key ? ' top' : ''}`}>
      {showTabs ? (
        <div className="tabs" role="tablist">
          {answers.map((answer) => (
            <button
              key={answer.key}
              type="button"
              role="tab"
              data-key={answer.key}
              aria-selected={answer.key === active?.key}
              className={`tab${answer.key === active?.key ? ' on' : ''}`}
              onClick={() => {
                setPinned(true);
                setActiveKey(answer.key);
              }}
            >
              <span className="sw" />
              {answer.title}
              {answer.evaluation?.composite != null ? (
                <span className={answer.key === winner ? 'win' : ''}>
                  {fmtScore(answer.evaluation.composite)}
                  {answer.key === winner ? ' TOP' : ''}
                </span>
              ) : null}
            </button>
          ))}
          {running.map((setup) => (
            <span className="tab" key={setup.key} data-key={setup.key}>
              <span className="sw" />
              {setup.title}
              <span style={{ color: 'var(--accent2)' }}>
                {((Date.now() - setup.startedAt) / 1000).toFixed(0)}s
              </span>
            </span>
          ))}
        </div>
      ) : null}

      {active ? (
        <>
          {!showTabs ? (
            <div className="bothead">
              <span className="setupchip">{active.title}</span>
              <span>
                {active.elapsed_seconds}s · ~{active.token_estimate} tok ·{' '}
                {active.chunk_count} chunks
              </span>
              {active.evaluation?.composite != null ? (
                <span className="badge good">RAGAS {fmtScore(active.evaluation.composite)}</span>
              ) : null}
            </div>
          ) : null}
          {traces?.[active.key]?.length ? (
            <AgentTrace steps={traces[active.key]!} running={false} />
          ) : null}
          <AnswerBody answer={active} />
          <ScoreStrip
            evaluation={active.evaluation}
            judging={judging && !active.evaluation}
            {...(question ? { question } : {})}
          />
        </>
      ) : null}

      {running.map((setup) => (
        <div key={setup.key}>
          {answers.length ? null : (
            <div className="bothead">
              <span className="setupchip">{setup.title}</span>
              <span style={{ color: 'var(--accent2)' }}>
                answering… {((Date.now() - setup.startedAt) / 1000).toFixed(0)}s
              </span>
            </div>
          )}
          <AgentTrace steps={setup.steps} running />
          {setup.steps.length === 0 ? (
            <div className="waiting">
              <span className="pulse" />
              {setup.key === 'rag_agent' ? 'researching…' : 'retrieving and answering…'}
            </div>
          ) : null}
        </div>
      ))}

      {answers.length > 1 ? (
        <div className="compare">
          <span className="microlabel">compare — same question, every setup</span>
          <div className="compare-cols">
            {answers.map((answer) => (
              <div
                className={`ccol${answer.key === winner ? ' top' : ''}`}
                key={answer.key}
                data-key={answer.key}
              >
                <div className="t">
                  <span className="sw" />
                  {answer.title.replace(/^rag_llm |^rag_agent ?/, '').replace(/[()]/g, '') ||
                    answer.key}
                </div>
                <div className={`comp${answer.evaluation?.composite == null ? ' na' : ''}`}>
                  {answer.evaluation?.composite == null
                    ? 'unjudged'
                    : fmtScore(answer.evaluation.composite)}
                  {answer.key === winner ? ' · TOP' : ''}
                </div>
                <div className="sub">
                  {answer.elapsed_seconds}s · ~{answer.token_estimate} tok
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {running.length === 0 && answers.length ? (
        <div className="msg-actions">
          {remainingSetups > 0 && onCompare ? (
            <button type="button" className="linkbtn" onClick={onCompare} disabled={busy}>
              Compare {remainingSetups} more setup{remainingSetups === 1 ? '' : 's'} ▸
            </button>
          ) : null}
          {onJudge ? (
            <button type="button" className="linkbtn" onClick={onJudge} disabled={judging}>
              {judging ? 'judging…' : unjudged ? 'Judge with RAGAS' : 'Re-judge'}
            </button>
          ) : null}
        </div>
      ) : null}
      <Followups
        followups={active?.followups ?? []}
        busy={busy}
        onAsk={onAskFollowup}
      />
    </div>
  );
}

/**
 * The follow-ups the answering LLM proposed, as one-click next questions.
 *
 * The agent contract returns these on every answer; offering them keeps a
 * research thread going without the user having to phrase the next question.
 */
function Followups({
  followups,
  busy,
  onAsk,
}: {
  followups: Followup[];
  busy?: boolean;
  onAsk?: (query: string) => void;
}) {
  const usable = followups.filter((f) => f.followup_query?.trim());
  if (!usable.length || !onAsk) return null;
  return (
    <div className="followups">
      <span className="microlabel">Follow up</span>
      <div className="suggest">
        {usable.map((followup) => (
          <button
            key={followup.followup_query}
            type="button"
            disabled={busy}
            title={followup.rationale || undefined}
            onClick={() => onAsk(followup.followup_query)}
          >
            {followup.topic || followup.followup_query}
          </button>
        ))}
      </div>
    </div>
  );
}
