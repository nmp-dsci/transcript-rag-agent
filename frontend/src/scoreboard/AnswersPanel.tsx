import { useMemo, useState } from 'react';

import type { ScoreboardQuestion, ScoreboardQuestionSetup } from '../api/types';
import { cleanAnswer } from '../answers/render';
import { fmtScore } from '../chat/ScoreStrip';

/** One graded answer, flattened out of the question-major payload. */
interface AnswerRow extends ScoreboardQuestionSetup {
  questionId: string;
  question: string;
  domain: string | null;
  questionType: string | null;
}

const ALL = '';

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
 * under different configs and are not comparable side by side.
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
                        <b style={{ color: 'var(--accent2)' }}>{fmtScore(row.composite)}</b>
                      ) : (
                        <span className="badge plain">unjudged</span>
                      )}
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
                        <span className="nchip">no answer recorded</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="board-note" style={{ marginTop: 10 }}>
        These are the answers from the run selected above. Switching runs changes the whole set —
        answers from two runs were produced under different configurations, so they are compared
        through the leaderboard, not read side by side here.
      </p>
    </details>
  );
}
