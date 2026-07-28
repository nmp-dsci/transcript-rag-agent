import type { ScoreboardQuestion } from '../api/types';
import { fmtScore } from '../chat/ScoreStrip';

/** Every golden question in the selected matrix run, and how each setup scored on it.
 * The leaderboard above only shows the average; this is what that average is built from. */
export function QuestionsPanel({ questions }: { questions: ScoreboardQuestion[] }) {
  if (questions.length === 0) return null;
  const setupKeys = Array.from(
    new Set(questions.flatMap((question) => question.setups.map((setup) => setup.key))),
  );
  const setupTitles = new Map(
    questions.flatMap((question) => question.setups.map((setup) => [setup.key, setup.title] as const)),
  );
  // The run carries every golden question whether or not it was judged — a run
  // committed with --no-judge has none — so the header counts the two separately
  // rather than calling all of them judged.
  const judgedCount = questions.filter((question) =>
    question.setups.some((setup) => setup.judged),
  ).length;

  return (
    <details className="panel qpanel">
      <summary>
        <h2 style={{ display: 'inline' }}>
          Questions ({questions.length}, {judgedCount} judged)
        </h2>
        <span className="sub"> — every golden question in this run, per-setup composite</span>
      </summary>
      <div className="tblwrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Domain</th>
              <th>Type</th>
              {setupKeys.map((key) => (
                <th key={key}>{setupTitles.get(key) ?? key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {questions.map((question) => {
              const byKey = new Map(question.setups.map((setup) => [setup.key, setup]));
              return (
                <tr key={question.id}>
                  <td>
                    <span className="qtext" title={question.question}>
                      {question.question}
                    </span>
                  </td>
                  <td>
                    {question.domain ? <span className="badge plain">{question.domain}</span> : '—'}
                  </td>
                  <td>
                    {question.question_type ? (
                      <span className="badge plain">{question.question_type}</span>
                    ) : (
                      '—'
                    )}
                  </td>
                  {setupKeys.map((key) => {
                    const setup = byKey.get(key);
                    if (!setup) return <td key={key} className="num">—</td>;
                    if (setup.error) {
                      return (
                        <td key={key} className="num">
                          <span className="badge bad" title={setup.error}>
                            error
                          </span>
                        </td>
                      );
                    }
                    if (!setup.judged) {
                      return (
                        <td key={key} className="num">
                          <span className="badge plain">unjudged</span>
                        </td>
                      );
                    }
                    return (
                      <td key={key} className="num">
                        {fmtScore(setup.composite)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}
