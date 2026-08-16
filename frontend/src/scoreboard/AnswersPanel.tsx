import { useMemo, useState } from 'react';

import type { ScoreboardQuestion, ScoreboardQuestionSetup } from '../api/types';
import { cleanAnswer } from '../answers/render';
import { fmtScore } from '../chat/ScoreStrip';
import { metricLabel } from '../eval/metrics';

/** One graded answer, flattened out of the question-major payload. */
interface AnswerRow extends ScoreboardQuestionSetup {
  questionId: string;
  question: string;
  domain: string | null;
  questionType: string | null;
}

const ALL = '';

/**
 * True when the judge saw fewer chunks than retrieval returned.
 *
 * Row-level rather than buried in the expanded panel: it changes what the
 * score means (coverage and evidence breadth were assessed against less than
 * the answer saw), and a caveat you can only find by expanding all eighty
 * cells one at a time is a caveat nobody finds.
 */
function isPartialContext(row: AnswerRow): boolean {
  return (
    row.contexts_expected != null &&
    row.contexts_resolved != null &&
    row.contexts_resolved < row.contexts_expected
  );
}

function flatten(questions: ScoreboardQuestion[]): AnswerRow[] {
  return questions.flatMap((question) =>
    question.setups.map((setup) => ({
      ...setup,
      questionId: question.id,
      question: question.question,
      domain: question.domain,
      questionType: question.question_type,
    })),
  );
}

/**
 * The answers behind the scores: every (question x setup) cell of the selected
 * run, filterable by question, setup, and answering model.
 *
 * The leaderboard says which setup won and `QuestionsPanel` says on which
 * questions; neither shows what was actually said. Reading the text is how you
 * tell a genuinely better answer from one that merely scored well, so this
 * table exists to make a composite auditable rather than taken on trust.
 *
 * The run itself is chosen by the picker at the top of the tab — these filters
 * narrow *within* one run, because answers from different runs were produced
 * under different configs, and often over different corpora, and are not
 * comparable side by side or through the leaderboard.
 */
export function AnswersPanel({ questions }: { questions: ScoreboardQuestion[] }) {
  const rows = useMemo(() => flatten(questions), [questions]);
  const [questionId, setQuestionId] = useState(ALL);
  const [setupKey, setSetupKey] = useState(ALL);
  const [model, setModel] = useState(ALL);
  const [judgedOnly, setJudgedOnly] = useState(false);
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);

  const setups = useMemo(() => {
    const seen = new Map<string, string>();
    for (const row of rows) if (!seen.has(row.key)) seen.set(row.key, row.title);
    return Array.from(seen, ([key, title]) => ({ key, title }));
  }, [rows]);

  const models = useMemo(
    () => Array.from(new Set(rows.map((row) => row.model).filter((m): m is string => !!m))),
    [rows],
  );

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return rows.filter((row) => {
      if (questionId !== ALL && row.questionId !== questionId) return false;
      if (setupKey !== ALL && row.key !== setupKey) return false;
      if (model !== ALL && row.model !== model) return false;
      if (judgedOnly && !row.judged) return false;
      if (needle && !`${row.question} ${row.answer}`.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [rows, questionId, setupKey, model, judgedOnly, search]);

  if (rows.length === 0) return null;

  return (
    <details className="panel qpanel">
      <summary>
        <h2 style={{ display: 'inline' }}>Answers ({rows.length})</h2>
        <span className="sub"> — the text each setup produced, and what the judge scored it</span>
      </summary>

      <div className="formrow" style={{ margin: '10px 0' }}>
        <span className="microlabel">question</span>
        <select
          value={questionId}
          onChange={(event) => setQuestionId(event.target.value)}
          aria-label="Filter by question"
        >
          <option value={ALL}>all questions</option>
          {questions.map((question) => (
            <option key={question.id} value={question.id}>
              {question.question.length > 70
                ? `${question.question.slice(0, 70)}…`
                : question.question}
            </option>
          ))}
        </select>

        <span className="microlabel">setup</span>
        <select
          value={setupKey}
          onChange={(event) => setSetupKey(event.target.value)}
          aria-label="Filter by setup"
        >
          <option value={ALL}>all setups</option>
          {setups.map((setup) => (
            <option key={setup.key} value={setup.key}>
              {setup.title}
            </option>
          ))}
        </select>

        {models.length > 1 ? (
          <>
            <span className="microlabel">answer model</span>
            <select
              value={model}
              onChange={(event) => setModel(event.target.value)}
              aria-label="Filter by answer model"
            >
              <option value={ALL}>all models</option>
              {models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </>
        ) : null}

        <input
          type="search"
          value={search}
          placeholder="search answer text"
          aria-label="Search answer text"
          onChange={(event) => setSearch(event.target.value)}
        />

        <label className="microlabel" style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <input
            type="checkbox"
            checked={judgedOnly}
            onChange={(event) => setJudgedOnly(event.target.checked)}
          />
          judged only
        </label>

        <span className="nchip">
          {filtered.length} of {rows.length}
        </span>
      </div>

      {filtered.length === 0 ? (
        <div className="rankempty" style={{ padding: 20 }}>
          No answers match these filters.
        </div>
      ) : (
        <div className="tblwrap">
          <table>
            <thead>
              <tr>
                <th>Question</th>
                <th>Setup</th>
                {models.length > 1 ? <th>Model</th> : null}
                <th>Composite</th>
                <th>Latency</th>
                <th>~Tokens</th>
                <th>Chunks</th>
                <th>Answer</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => {
                const rowId = `${row.questionId}:${row.key}`;
                const isOpen = expanded === rowId;
                const text = cleanAnswer(row.answer);
                return (
                  <tr key={rowId}>
                    <td>
                      <span className="qtext" title={row.question}>
                        {row.question}
                      </span>
                    </td>
                    <td>{row.title}</td>
                    {models.length > 1 ? (
                      <td>
                        <span className="badge acc">{row.model ?? '—'}</span>
                      </td>
                    ) : null}
                    <td className="num">
                      {row.error ? (
                        <span className="badge bad" title={row.error}>
                          error
                        </span>
                      ) : row.judged ? (
                        <>
                          <b style={{ color: 'var(--accent2)' }}>{fmtScore(row.composite)}</b>
                          {row.cap_applied ? (
                            <>
                              {' '}
                              <span className="badge warn">capped</span>
                            </>
                          ) : null}
                          {row.grounding_floor_breached && !row.cap_applied ? (
                            <>
                              {' '}
                              <span className="badge bad">ungrounded</span>
                            </>
                          ) : null}
                        </>
                      ) : row.rejudged === false ? (
                        <span className="badge plain" title={row.rejudge_skipped_reason ?? ''}>
                          not rescored
                        </span>
                      ) : (
                        <span className="badge plain">unjudged</span>
                      )}
                      {isPartialContext(row) ? (
                        <>
                          {' '}
                          <span
                            className="badge warn"
                            title={`judged on ${row.contexts_resolved} of ${row.contexts_expected} retrieved chunks`}
                          >
                            partial context
                          </span>
                        </>
                      ) : null}
                    </td>
                    <td className="num">
                      {row.elapsed_seconds == null ? '—' : `${row.elapsed_seconds}s`}
                    </td>
                    <td className="num">{row.token_estimate ?? '—'}</td>
                    <td className="num">{row.chunk_count}</td>
                    <td>
                      {text ? (
                        <>
                          <span className={isOpen ? 'ans-full' : 'qtext'} title={isOpen ? '' : text}>
                            {text}
                          </span>
                          {isOpen ? <CellJudgement row={row} /> : null}
                          <button
                            type="button"
                            className="btn sm"
                            style={{ marginTop: 4 }}
                            onClick={() => setExpanded(isOpen ? null : rowId)}
                          >
                            {isOpen ? 'Collapse' : 'Expand'}
                          </button>
                        </>
                      ) : (
                        <>
                          <span className="nchip">no answer recorded</span>
                          {/*
                            Here rather than only in CellJudgement: that panel
                            renders on expand and the Expand button renders only
                            when there is answer text — so on the one skip reason
                            that actually occurs ("no answer to grade") the
                            explanation had no path to the page at all.
                          */}
                          <NotRescored row={row} />
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/*
        This note used to end "so they are compared through the leaderboard,
        not read side by side here" — which sent a reader who noticed two runs
        were incomparable into the leaderboard to compare them anyway. The
        leaderboard ranks setups *within* one run; it is not a cross-run
        instrument, and two runs can differ by corpus (visible, at the top of
        the tab) or by engine behaviour (recorded nowhere at all).
      */}
      <p className="board-note" style={{ marginTop: 10 }}>
        These are the answers from the run selected above. Switching runs changes the whole set:
        another run&apos;s answers were produced under a different configuration, and often over a
        different corpus, so they are <b>not</b> a comparison set — neither read side by side here
        nor carried across through the leaderboard, which ranks setups against each other{' '}
        <em>within</em> one run. To compare two runs, check first that they name the same corpus at
        the top of this tab; even then, nothing in a run file records the engine version that wrote
        its answers.
      </p>
    </details>
  );
}

/**
 * A judge-written reason, terminated so the sentence after it is a new one.
 *
 * The reasons are written as clauses ("faithfulness 0.42 below 0.6 — depth
 * cannot rescue an ungrounded answer") and are followed here by a sentence of
 * our own, which ran straight on from them.
 */
function sentence(text: string | null | undefined): string {
  const trimmed = (text ?? '').trim();
  if (!trimmed) return '';
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

/**
 * Why a cell carries no score under this run's rubric.
 *
 * Its own component because it has to render in two places: inside the
 * expanded judgement panel, and — for the skip reason that actually occurs, a
 * cell with no answer — in the row itself, which is the only place a cell
 * without answer text can show anything at all.
 */
function NotRescored({ row }: { row: AnswerRow }) {
  if (row.rejudged !== false) return null;
  return (
    <p className="capwhy">
      <span className="badge plain">not rescored</span> This cell could not be scored under this
      run&apos;s rubric ({row.rejudge_skipped_reason ?? 'no answer to grade'}), so it is excluded
      from every average rather than carried over at its previous rubric&apos;s score.
    </p>
  );
}

/**
 * Why this cell scored what it did, as page text rather than a tooltip.
 *
 * A capped 0.50 is the one score on this tab that cannot be read off its parts
 * — the composite deliberately disagrees with the weighted sum — so the reason
 * has to be readable, and readable means selectable text under the answer, not
 * a hover. The per-metric rationales sit beside it for the same reason: a depth
 * score is a judgement, and a judgement with no stated reason is a number to be
 * taken on trust.
 */
function CellJudgement({ row }: { row: AnswerRow }) {
  const rationales = Object.entries(row.rationales ?? {});
  const partialContext = isPartialContext(row);
  const notes =
    row.cap_applied ||
    row.grounding_floor_breached ||
    row.rejudged === false ||
    row.depth_error ||
    partialContext;
  if (!notes && rationales.length === 0) return null;

  return (
    <div className="cellwhy">
      {row.cap_applied ? (
        <p className="capwhy">
          <span className="badge warn">capped</span> Composite capped at{' '}
          {fmtScore(row.composite)}
          {row.composite_uncapped != null ? ` from ${fmtScore(row.composite_uncapped)}` : ''}:{' '}
          {sentence(row.cap_reason) || 'the rubric cap applied.'}
        </p>
      ) : row.grounding_floor_breached ? (
        <p className="capwhy">
          <span className="badge bad">ungrounded</span> {sentence(row.grounding_reason)} The cap
          changed nothing here — this answer scored below it on its own.
        </p>
      ) : null}

      <NotRescored row={row} />

      {partialContext ? (
        <p className="whyline partial">
          <span className="badge warn">partial context</span> Judged on {row.contexts_resolved} of{' '}
          {row.contexts_expected} retrieved chunks — the rest could not be resolved from the chunk
          store, so coverage and evidence breadth were assessed against less than this answer
          actually saw.
        </p>
      ) : null}

      {row.depth_error ? (
        <p className="whyline partial">
          <span className="badge bad">depth judging failed</span> {row.depth_error}. The composite
          above is the grounding half only, renormalised.
        </p>
      ) : null}

      {rationales.map(([metric, rationale]) => (
        <p key={metric} className="whyline">
          <b>{metricLabel(metric)}</b>
          {row.scores?.[metric] != null ? ` ${fmtScore(row.scores[metric])}` : ''} — {rationale}
        </p>
      ))}
    </div>
  );
}
