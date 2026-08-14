import { useCallback, useEffect, useState } from 'react';

import { api } from '../api/client';
import type { MatrixRunOption, Scoreboard } from '../api/types';
import { fmtScore } from '../chat/ScoreStrip';
import { LOW_N, isLowN } from '../eval/breakdown';
import { metricLabel, metricTitle, weightLabel } from '../eval/metrics';
import { MetricExplainers } from '../eval/MetricExplainer';
import { useEvalStyles } from '../eval/styles';
import { AnswersPanel } from './AnswersPanel';
import { corpusDetail, corpusLabel, corpusValue } from './corpus';
import { CorpusNote } from './CorpusNote';
import { EfficiencyPanel } from './EfficiencyPanel';
import { ProvenanceBar } from './ProvenanceBar';
import { QuestionsPanel } from './QuestionsPanel';
import { RubricPanel } from './RubricPanel';

type GroupBy = 'setup' | 'setup_model';

/**
 * "matrix-20260727-015519 · 14 questions · 4 setups · ragas-v1 · corpus 1bdb1971 (size not recorded)"
 *
 * The rubric is part of the label because two runs over identical questions
 * and setups can rank them differently purely by rubric — without it, switching
 * runs to compare rankings would look like the engines had changed.
 *
 * The corpus is part of it for the stronger version of the same reason. The
 * picker is the only surface a reader flipping between runs actually reads, and
 * two committed runs whose numbers came from different corpora rendered here as
 * the identical string `20 questions · 2 setups · ragas-v1` — so moving between
 * two incomparable measurements looked like watching one number change.
 */
function runLabel(run: MatrixRunOption): string {
  const parts = [run.run_id];
  if (run.entry_count != null) parts.push(`${run.entry_count} questions`);
  parts.push(`${run.setups.length} setup${run.setups.length === 1 ? '' : 's'}`);
  if (!run.judged) parts.push('unjudged');
  if (run.rubric_version) parts.push(run.rubric_version);
  parts.push(corpusLabel(run));
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
  // The rubric decides which metrics exist, so the columns come from the
  // payload rather than from a list hard-coded against the original three.
  const metrics = board?.provenance.metrics ?? [];
  const weights = board?.provenance.metric_weights ?? {};
  const groups = board?.provenance.metric_groups ?? [];
  const rubricVersion = board?.provenance.rubric_version ?? 'ragas-v1';
  // More than one rubric among the records means the composites in one column
  // were not produced on one scale, so the header naming a single rubric would
  // be a claim the data does not support.
  const mixedRubrics = (board?.provenance.rubric_versions ?? []).length > 1;
  const selectedRun = board?.run ?? null;
  const skipped = selectedRun?.skipped_cells ?? 0;

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
          {/*
            The corpus these numbers were retrieved from, beside the numbers.
            The page header states the corpus indexed *right now*, which is a
            different corpus from the one a committed run was scored on as soon
            as anything is ingested — so without this tile the page asserts a
            corpus the table below does not belong to.
          */}
          <div className="stat">
            <div className="microlabel">corpus scored on</div>
            <b style={{ fontSize: 13 }} title={selectedRun?.corpus ?? undefined}>
              {selectedRun ? corpusValue(selectedRun) : '—'}
            </b>
            <span>{selectedRun ? corpusDetail(selectedRun) : 'no run selected'}</span>
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

        {/*
          Directly under the picker, because the picker is where a reader
          decides that two runs are comparable. See CorpusNote for what it can
          and cannot honestly claim.
        */}
        <CorpusNote run={selectedRun} runs={runOptions} />

        {mixedRubrics ? (
          <div className="rubricwarn">
            <span className="badge bad">mixed rubrics</span> This run's answers were composited
            under more than one rubric ({(board?.provenance.rubric_versions ?? []).join(', ')}).
            The columns below are labelled with the original three-metric rubric because that is
            the only one every record here definitely has — the composites are not on one scale,
            so treat the ranking as unusable rather than close.
          </div>
        ) : null}

        {board?.provenance.self_graded ? (
          <div className="rubricwarn">
            <span className="badge bad">self-graded</span> The model that wrote these answers also
            graded them ({board.provenance.judge_models.join(', ')}). These scores are
            self-assessment, not an independent verdict — set{' '}
            <code>YT_AGENT_JUDGE_MODEL</code> to a different provider and re-run before reading
            this ranking as a result.
          </div>
        ) : null}

        <div className="tblwrap">
          {rows.length === 0 || !board?.entries_judged ? (
            <div className="rankempty" style={{ padding: 30 }}>
              {runOptions.length === 0
                ? 'No committed matrix run yet — run one from the Experiments tab (or ' +
                  '`eval-matrix` on the CLI) to score every setup over the golden set.'
                : 'This run has no judged answers — re-run it with judging enabled.'}
            </div>
          ) : (
            <table aria-label="Leaderboard">
              <thead>
                <tr>
                  <th>Setup</th>
                  {groupBy === 'setup_model' ? <th>Answer model</th> : null}
                  <th title={`n = questions judged for this row. ${LOW_N} or fewer is dimmed — too thin to rank on.`}>
                    n judged
                  </th>
                  {metrics.map((name) => {
                    const weight = weightLabel(weights[name]);
                    return (
                      <th key={name} title={metricTitle(name)}>
                        {metricLabel(name)}
                        {weight ? <span className="wchip">{weight}</span> : null}
                      </th>
                    );
                  })}
                  <th>Composite</th>
                  <th>Win rate</th>
                  <th>Latency</th>
                  <th>~Tokens</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const lowN = isLowN(row.judged);
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
                            title={`Averaged over only ${row.judged} judged question${row.judged === 1 ? '' : 's'} — treat this row as a hint, not a ranking. More than ${LOW_N} before comparing.`}
                          >
                            thin
                          </span>
                        </>
                      ) : null}
                    </td>
                    {metrics.map((name) => {
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
                      {row.capped ? (
                        <>
                          {' '}
                          <span
                            className="badge warn"
                            title={`${row.capped} of this row's ${row.judged} judged answers scored above the cap but were pulled down to it. Open the Answers panel and expand a capped cell to read why.`}
                          >
                            capped {row.capped}
                          </span>
                        </>
                      ) : null}
                      {row.ungrounded ? (
                        <>
                          {' '}
                          <span
                            className="badge bad"
                            title={`${row.ungrounded} of this row's ${row.judged} judged answers breached the grounding floor (faithfulness below 0.6, or not scorable at all). That is a superset of the capped ones: an answer can be ungrounded and have scored below the cap on its own.`}
                          >
                            ungrounded {row.ungrounded}
                          </span>
                        </>
                      ) : null}
                    </td>
                    <td className="num">
                      {row.win_rate == null
                        ? '—'
                        : `${Math.round(row.win_rate * 100)}% (${row.wins}/${row.contests})`}
                      {row.contests > 0 && isLowN(row.contests) ? (
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

        {skipped > 0 ? (
          <p className="board-note" style={{ marginTop: 0 }}>
            {skipped} of {(selectedRun?.rejudged_cells ?? 0) + skipped} cells in this run could not
            be scored under <b>{rubricVersion}</b> (no answer to grade, or the cell errored). They
            are excluded from every average above rather than carried over at their previous
            rubric&apos;s score, which is why some rows have a lower <em>n</em> than the run&apos;s
            question count.
          </p>
        ) : null}

        <RubricPanel
          metrics={metrics}
          weights={weights}
          groups={groups}
          rubricVersion={rubricVersion}
          rows={rows}
          composite={board?.provenance.composite ?? ''}
        />

        <EfficiencyPanel rows={rows} />

        <QuestionsPanel questions={board?.questions ?? []} />

        <AnswersPanel questions={board?.questions ?? []} />

        <div className="panel">
          <h2>What the metrics mean</h2>
          <p className="sub">
            Every column above comes out of one of these. Open a metric on any answer in the chat
            to see the judge&apos;s claim-by-claim workings for that question.
          </p>
          <MetricExplainers names={metrics} />
        </div>

        {board ? <ProvenanceBar provenance={board.provenance} run={selectedRun} /> : null}

        <p className="board-note">
          Every row comes from one <b>committed matrix run</b>: each setup answering the same
          golden questions under one recorded configuration, graded by one judge. That is what
          makes these rows comparable — unlike the Chat tab, where whichever questions happened
          to be asked would decide the ranking. Chat and its history are the <em>live</em> set
          for exploring the corpus; this tab is the <em>eval</em> set. All answers in a run are
          graded under one rubric — this one is <b>{rubricVersion}</b>, whose composite is{' '}
          {board?.provenance.composite ?? 'the mean of the metric scores'}. What makes rows
          comparable is that they sit <em>inside</em> one run: picking a different run above
          replaces the whole table, and the new numbers are only comparable with these ones when
          both runs name the same corpus — read the corpus line at the top of the tab before
          carrying a figure from one run to another. A win counts a question
          where a setup scored highest <em>among answers graded by the same judge</em>. Rows
          judged on {LOW_N} questions or fewer are dimmed and marked{' '}
          <span className="badge warn">thin</span>: an average over a handful of questions moves
          several points on one bad answer, so read those as a hint rather than a ranking.
        </p>
      </div>
    </div>
  );
}
