import { useCallback, useEffect, useState } from 'react';

import { api } from '../api/client';
import type { MatrixRunOption, Scoreboard } from '../api/types';
import { METRICS, fmtScore } from '../chat/ScoreStrip';
import { LOW_N } from '../eval/breakdown';
import { MetricExplainers } from '../eval/MetricExplainer';
import { useEvalStyles } from '../eval/styles';
import { EfficiencyPanel } from './EfficiencyPanel';
import { ProvenanceBar } from './ProvenanceBar';
import { QuestionsPanel } from './QuestionsPanel';

type GroupBy = 'setup' | 'setup_model';

/** "matrix-20260727-015519 · 14 questions · 4 setups" */
function runLabel(run: MatrixRunOption): string {
  const parts = [run.run_id];
  if (run.entry_count != null) parts.push(`${run.entry_count} questions`);
  parts.push(`${run.setups.length} setup${run.setups.length === 1 ? '' : 's'}`);
  if (!run.judged) parts.push('unjudged');
  return parts.join(' · ');
}

export function ScoreboardView() {
  useEvalStyles();
  const [board, setBoard] = useState<Scoreboard | null>(null);
  const [groupBy, setGroupBy] = useState<GroupBy>('setup_model');
  const [judgeFilter, setJudgeFilter] = useState('');
  // '' means "whatever the server picks", i.e. the newest committed run.
  const [runId, setRunId] = useState('');
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setBoard(await api.scoreboard(groupBy, judgeFilter || null, runId || null));
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [groupBy, judgeFilter, runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const rows = (board?.setups ?? []).filter((row) => row.answers > 0);
  const bestKey = rows[0]?.avg_composite != null ? rows[0] : null;
  const judgeOptions = board?.provenance.judge_models ?? [];
  const runOptions = board?.runs ?? [];

  return (
    <div className="scrollview">
      <div className="pagewrap">
        <div className="statusstrip">
          <div className="stat">
            <div className="microlabel">eval set</div>
            <b>{board?.entries_judged ?? '—'}</b>
            <span>of {board?.entries_total ?? '—'} golden questions judged</span>
          </div>
          <div className="stat">
            <div className="microlabel">judge model</div>
            <b style={{ fontSize: 13 }}>{board?.judge_model ?? '—'}</b>
            <span>
              {judgeOptions.length > 1 ? `${judgeOptions.length} judges in history` : 'single judge'}
            </span>
          </div>
          <div className="stat">
            <div className="microlabel">grouping</div>
            <b style={{ fontSize: 13 }}>
              {groupBy === 'setup_model' ? 'setup × model' : 'setup'}
            </b>
            <span>{rows.length} rows</span>
          </div>
        </div>

        <div className="formrow" style={{ marginBottom: 12 }}>
          <span className="microlabel">matrix run</span>
          <select
            value={runId}
            onChange={(event) => setRunId(event.target.value)}
            aria-label="Matrix run"
            disabled={runOptions.length === 0}
          >
            <option value="">
              {runOptions.length === 0 ? 'no committed runs' : 'newest run'}
            </option>
            {runOptions.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {runLabel(run)}
              </option>
            ))}
          </select>
          <span className="microlabel">group by</span>
          <select
            value={groupBy}
            onChange={(event) => setGroupBy(event.target.value as GroupBy)}
            aria-label="Group by"
          >
            <option value="setup_model">setup × answer model</option>
            <option value="setup">setup only</option>
          </select>
          <span className="microlabel">judge</span>
          <select
            value={judgeFilter}
            onChange={(event) => setJudgeFilter(event.target.value)}
            aria-label="Filter by judge model"
          >
            <option value="">all judges</option>
            {judgeOptions.map((judge) => (
              <option key={judge} value={judge}>
                {judge}
              </option>
            ))}
          </select>
          <button type="button" className="btn sm" onClick={() => void load()}>
            Refresh
          </button>
        </div>

        {error ? <div className="errtext">{error}</div> : null}

        <div className="tblwrap">
          {rows.length === 0 || !board?.entries_judged ? (
            <div className="rankempty" style={{ padding: 30 }}>
              {runOptions.length === 0
                ? 'No committed matrix run yet — run one from the Experiments tab (or ' +
                  '`eval-matrix` on the CLI) to score every setup over the golden set.'
                : 'This run has no judged answers — re-run it with judging enabled.'}
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Setup</th>
                  {groupBy === 'setup_model' ? <th>Answer model</th> : null}
                  <th title={`n = questions judged for this row. Fewer than ${LOW_N} is dimmed — too thin to rank on.`}>
                    n judged
                  </th>
                  {METRICS.map(([name, label, title]) => (
                    <th key={name} title={title}>
                      {label}
                    </th>
                  ))}
                  <th>Composite</th>
                  <th>Win rate</th>
                  <th>Latency</th>
                  <th>~Tokens</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const lowN = row.judged < LOW_N;
                  return (
                  <tr
                    key={`${row.key}:${row.model ?? 'legacy'}`}
                    className={`${row === bestKey ? 'bestrow' : ''}${row.legacy ? ' legacyrow' : ''}${lowN ? ' lown' : ''}`}
                  >
                    <td>{row.title}</td>
                    {groupBy === 'setup_model' ? (
                      <td>
                        <span className={`badge ${row.legacy ? 'plain' : 'acc'}`}>
                          {row.model ?? '— pre-provenance'}
                        </span>
                      </td>
                    ) : null}
                    <td className="num">
                      n={row.judged}{' '}
                      <span className="nchip">of {row.answers} answers</span>
                      {lowN ? (
                        <>
                          {' '}
                          <span
                            className="badge warn"
                            title={`Averaged over only ${row.judged} judged question${row.judged === 1 ? '' : 's'} — treat this row as a hint, not a ranking. ${LOW_N}+ before comparing.`}
                          >
                            thin
                          </span>
                        </>
                      ) : null}
                    </td>
                    {METRICS.map(([name]) => {
                      const value = row.avg_scores[name];
                      return (
                        <td key={name}>
                          <span className="cellbar">
                            <span className="mbar">
                              <i
                                style={{
                                  width: `${value != null ? Math.round(value * 100) : 0}%`,
                                }}
                              />
                            </span>
                            <span className="num">{fmtScore(value)}</span>
                          </span>
                        </td>
                      );
                    })}
                    <td className="num">
                      <b style={{ color: 'var(--accent2)' }}>{fmtScore(row.avg_composite)}</b>
                    </td>
                    <td className="num">
                      {row.win_rate == null
                        ? '—'
                        : `${Math.round(row.win_rate * 100)}% (${row.wins}/${row.contests})`}
                      {row.contests > 0 && row.contests < LOW_N ? (
                        <>
                          {' '}
                          <span
                            className="nchip"
                            title={`Only ${row.contests} head-to-head contest${row.contests === 1 ? '' : 's'} — a win rate over this few questions is noise.`}
                          >
                            n={row.contests}
                          </span>
                        </>
                      ) : null}
                    </td>
                    <td className="num">
                      {row.avg_latency_seconds == null ? '—' : `${row.avg_latency_seconds}s`}
                    </td>
                    <td className="num">{row.avg_token_estimate ?? '—'}</td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        <EfficiencyPanel rows={rows} />

        <QuestionsPanel questions={board?.questions ?? []} />

        <div className="panel">
          <h2>What the metrics mean</h2>
          <p className="sub">
            Every column above comes out of one of these three. Open a metric on any answer in the
            chat to see the judge&apos;s claim-by-claim workings for that question.
          </p>
          <MetricExplainers />
        </div>

        {board ? <ProvenanceBar provenance={board.provenance} /> : null}

        <p className="board-note">
          Every row comes from one <b>committed matrix run</b>: each setup answering the same
          golden questions under one recorded configuration, graded by one judge. That is what
          makes these rows comparable — unlike the Chat tab, where whichever questions happened
          to be asked would decide the ranking. Chat and its history are the <em>live</em> set
          for exploring the corpus; this tab is the <em>eval</em> set. All answers are graded
          under one eval process — <b>RAGAS</b>: faithfulness (is the answer supported by the
          retrieved chunks?), answer relevancy (does it address the question?), context precision
          (were the retrieved chunks useful?). Composite is their mean. A win counts a question
          where a setup scored highest <em>among answers graded by the same judge</em>. Rows
          judged on fewer than {LOW_N} questions are dimmed and marked{' '}
          <span className="badge warn">thin</span>: an average over a handful of questions moves
          several points on one bad answer, so read those as a hint rather than a ranking.
        </p>
      </div>
    </div>
  );
}
