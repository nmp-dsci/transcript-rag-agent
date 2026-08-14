import type { MatrixRunOption } from '../api/types';

/**
 * How a committed run describes the corpus its numbers were scored on.
 *
 * Three states, and the difference between the last two is the whole point of
 * this module:
 *
 * - `sized`     — digest and counts recorded. The corpus is identified *and*
 *                 its size is readable.
 * - `digest`    — digest recorded, counts absent. Two runs can still be told
 *                 apart (or shown to match), but nothing says how big it was.
 * - `unrecorded`— nothing recorded. Runs committed before corpus identity
 *                 entered the run fingerprint. **Unknown, not equal**: these
 *                 must never render the same as "the same corpus", because a
 *                 reader who reads absence as agreement compares numbers that
 *                 have no shown relationship at all.
 */
export type CorpusState = 'sized' | 'digest' | 'unrecorded';

export function corpusState(run: MatrixRunOption | null | undefined): CorpusState {
  if (!run?.corpus) return 'unrecorded';
  return run.corpus_videos != null && run.corpus_chunks != null ? 'sized' : 'digest';
}

/** The first 8 hex characters — enough to tell two corpora apart in a label. */
export function shortDigest(digest: string): string {
  return digest.slice(0, 8);
}

/** "71 videos · 1792 chunks", or null when the run recorded no counts. */
export function corpusSize(run: MatrixRunOption | null | undefined): string | null {
  if (run?.corpus_videos == null || run?.corpus_chunks == null) return null;
  return `${run.corpus_videos} videos · ${run.corpus_chunks} chunks`;
}

/**
 * The corpus clause for the run picker.
 *
 * The picker is the comparison surface: a reader flipping between two runs sees
 * only these strings, so if two runs were scored on different corpora the label
 * is where that has to show. Before this, both of the 20-question runs read
 * `20 questions · 2 setups · ragas-v1` and were indistinguishable.
 */
export function corpusLabel(run: MatrixRunOption): string {
  const size = corpusSize(run);
  if (!run.corpus) return 'no corpus recorded';
  if (!size) return `corpus ${shortDigest(run.corpus)} (size not recorded)`;
  return `corpus ${shortDigest(run.corpus)} (${size})`;
}

/** The one-line value for the status strip: the digest, or its absence. */
export function corpusValue(run: MatrixRunOption | null | undefined): string {
  return run?.corpus ? shortDigest(run.corpus) : 'not recorded';
}

/** The line under it — the size, or why there is none. */
export function corpusDetail(run: MatrixRunOption | null | undefined): string {
  switch (corpusState(run)) {
    case 'sized':
      return corpusSize(run) as string;
    case 'digest':
      return 'size not recorded';
    default:
      return 'run predates corpus identity';
  }
}

/** How every other committed run's corpus relates to the selected one's. */
export interface CorpusPeers {
  /** Same digest — comparable with the selected run on corpus grounds. */
  same: MatrixRunOption[];
  /** A digest that is definitely not this one's. */
  different: MatrixRunOption[];
  /** No corpus recorded, or none recorded here — relationship unknown. */
  unknown: MatrixRunOption[];
}

/**
 * Partition the other runs against the selected one.
 *
 * A run whose own corpus is unrecorded has an *unknown* relationship to every
 * other run, including runs that do record one — so `same` stays empty rather
 * than defaulting to agreement.
 */
export function corpusPeers(
  selected: MatrixRunOption | null | undefined,
  runs: MatrixRunOption[],
): CorpusPeers {
  const peers: CorpusPeers = { same: [], different: [], unknown: [] };
  for (const run of runs) {
    if (selected && run.run_id === selected.run_id) continue;
    if (!run.corpus || !selected?.corpus) peers.unknown.push(run);
    else if (run.corpus === selected.corpus) peers.same.push(run);
    else peers.different.push(run);
  }
  return peers;
}

/** True when flipping runs in the picker cannot be read as like-for-like. */
export function hasCorpusHazard(peers: CorpusPeers, state: CorpusState): boolean {
  return state === 'unrecorded' || peers.different.length > 0 || peers.unknown.length > 0;
}

/** "1bdb1971 and 3 others", for naming the offending digests without a wall. */
export function digestList(runs: MatrixRunOption[]): string {
  const digests = Array.from(
    new Set(runs.map((run) => (run.corpus ? shortDigest(run.corpus) : '')).filter(Boolean)),
  );
  if (digests.length === 0) return '';
  if (digests.length <= 3) return digests.join(', ');
  return `${digests.slice(0, 3).join(', ')} and ${digests.length - 3} more`;
}
