import type { Provenance } from '../api/types';
import { fmtTime } from '../answers/render';

/** What produced these scores — the methodology, stated rather than assumed. */
export function ProvenanceBar({ provenance }: { provenance: Provenance }) {
  const depthJudges = provenance.depth_judge_models ?? [];
  const entries: [string, string][] = [
    ['judge', provenance.judge_models.join(', ') || '—'],
    // Only under a rubric that has one. Naming the grounding judge alone would
    // credit an independent verdict to 40% of the composite while the other
    // 60% came from somewhere the reader was never told about.
    ...(depthJudges.length > 0
      ? ([['depth judge', depthJudges.join(', ')]] as [string, string][])
      : []),
    ['ragas', provenance.ragas_versions.join(', ') || '—'],
    ['embeddings', provenance.embedding_models.join(', ') || '—'],
    ['metrics', provenance.metrics.join(' · ')],
    ['composite', provenance.composite],
    ['last judged', provenance.last_judged ? fmtTime(provenance.last_judged) : 'never'],
  ];
  return (
    <div className="provbar">
      {entries.map(([label, value]) => (
        <span key={label}>
          <b>{label}</b> {value}
        </span>
      ))}
      {provenance.self_graded ? (
        <span
          className="provwarn"
          title="The answering model is also a judge here, so these scores are self-assessment rather than an independent check. Judge with a different model before trusting the ranking."
        >
          <span className="badge bad">self-graded</span> the answering model graded its own
          answers
        </span>
      ) : null}
    </div>
  );
}
