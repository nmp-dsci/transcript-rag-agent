import type { MatrixRunOption } from '../api/types';
import { corpusPeers, corpusSize, corpusState, digestList, hasCorpusHazard } from './corpus';

function plural(count: number, one: string, many: string): string {
  return count === 1 ? one : many;
}

/**
 * What corpus — and what engine — produced the numbers below, stated on the page.
 *
 * The Scoreboard used to show a run's scores under a header quoting the corpus
 * indexed *right now*, with nothing tying a run's numbers to the corpus they
 * were actually retrieved from. Two committed runs both carry a
 * `rag_llm_filtered` column scored on different corpora; flipping between them
 * moved the numbers with nothing on the page explaining why, which reads as a
 * result rather than as two incomparable measurements.
 *
 * Three claims live here, and they are deliberately distinct:
 *
 * 1. **This run's corpus** — digest, plus size when the run recorded it. A run
 *    that recorded only the digest says so rather than borrowing counts from
 *    anywhere else.
 * 2. **How the other runs relate to it** — same digest (comparable), a
 *    different digest (not comparable), or unrecorded (*unknown*, which is not
 *    a claim of equality and must not render like one).
 * 3. **The engine** — no run file records the version of the engine that
 *    answered its questions, so the page says that plainly and points at the
 *    README rather than inventing a version that was never captured.
 */
export function CorpusNote({
  run,
  runs,
}: {
  run: MatrixRunOption | null;
  runs: MatrixRunOption[];
}) {
  if (!run) return null;
  const state = corpusState(run);
  const peers = corpusPeers(run, runs);
  const size = corpusSize(run);
  const hazard = hasCorpusHazard(peers, state);

  return (
    <div className={hazard ? 'rubricwarn' : 'board-note'} style={{ marginBottom: 12 }}>
      {state === 'unrecorded' ? (
        <>
          <span className="badge bad">no corpus recorded</span> This run predates corpus identity
          in the run fingerprint, so which videos and chunks produced the numbers below is{' '}
          <b>unknown</b>. Unknown is not the same claim as <em>the same corpus</em>: nothing here
          shows these rows share evidence with any other run, or with the corpus indexed right now
          (the videos and chunks counted in the page header).
        </>
      ) : (
        <>
          <span className="badge acc" title={run.corpus ?? undefined}>
            corpus {run.corpus}
          </span>{' '}
          The rows below were scored on{' '}
          {size ? (
            <b>{size}</b>
          ) : (
            <>
              this corpus, whose <b>video and chunk counts this run did not record</b>
            </>
          )}
          . The videos and chunks counted in the page header are the corpus indexed{' '}
          <em>right now</em> — a different question from which corpus produced these numbers.
        </>
      )}{' '}
      {peers.different.length > 0 ? (
        <>
          {peers.different.length} other committed{' '}
          {plural(peers.different.length, 'run', 'runs')} here{' '}
          {plural(peers.different.length, 'was', 'were')} scored on a different corpus (
          {digestList(peers.different)}). A column that appears in both is not a like-for-like
          comparison — the two sets of numbers came from different evidence.{' '}
        </>
      ) : null}
      {peers.unknown.length > 0 ? (
        <>
          {peers.unknown.length} other {plural(peers.unknown.length, 'run', 'runs')}{' '}
          {plural(peers.unknown.length, 'has', 'have')} no corpus relationship on record: unknown,
          not equal.{' '}
        </>
      ) : null}
      {peers.same.length > 0 ? (
        <>
          {peers.same.length} other {plural(peers.same.length, 'run', 'runs')}{' '}
          {plural(peers.same.length, 'records', 'record')} this same digest and so retrieved from
          the same corpus.{' '}
        </>
      ) : null}
      No run records the <b>engine version</b> behind its answers either. The cell fingerprint is
      built from configuration, not code, so a change in how an engine behaves that moves no config
      field leaves no trace in any run file — two runs&apos; identically named columns can be
      different engines. <code>evals/runs/README.md</code> (&ldquo;What the fingerprint cannot
      see&rdquo;) records the case where that has already happened.
    </div>
  );
}
