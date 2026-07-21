import type { Provenance } from '../api/types';
import { fmtTime } from '../answers/render';

/** What produced these scores — the methodology, stated rather than assumed. */
export function ProvenanceBar({ provenance }: { provenance: Provenance }) {
  const entries: [string, string][] = [
    ['judge', provenance.judge_models.join(', ') || '—'],
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
    </div>
  );
}
