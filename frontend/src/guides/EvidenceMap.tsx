import type { GuideClaims, GuideDetail } from '../api/types';

export interface SectionEvidence {
  id: string;
  claims: number;
  valid: number;
  videos: string[];
}

/** Claims per section, in page order — the "consolidated view" (screen C):
 *  what each part of the guide rests on. Pure, from claims.json alone. */
export function sectionEvidence(claims: GuideClaims | null): SectionEvidence[] {
  if (!claims) return [];
  const bySection = new Map<string, SectionEvidence>();
  for (const id of claims.sections) bySection.set(id, { id, claims: 0, valid: 0, videos: [] });
  for (const claim of claims.claims) {
    const key = claim.section_id ?? '(unsectioned)';
    const entry = bySection.get(key) ?? { id: key, claims: 0, valid: 0, videos: [] };
    entry.claims += 1;
    if (claim.valid) entry.valid += 1;
    if (!entry.videos.includes(claim.video_id)) entry.videos.push(claim.video_id);
    bySection.set(key, entry);
  }
  return [...bySection.values()];
}

/** A stable colour index per video, so a source reads the same in every row. */
export function videoPalette(videoIds: string[]): Map<string, number> {
  const map = new Map<string, number>();
  videoIds.forEach((id, index) => map.set(id, index % 8));
  return map;
}

export function EvidenceMap({ guide, onJump }: { guide: GuideDetail; onJump: (id: string) => void }) {
  const rows = sectionEvidence(guide.claims);
  const palette = videoPalette(guide.video_ids);
  const titles = new Map(guide.sources.map((s) => [s.video_id, s.title]));
  const total = guide.claims?.total ?? 0;
  const valid = guide.claims?.valid ?? 0;
  return (
    <div className="ev" aria-label="Evidence map">
      <div className="gj-kpis ev-kpis">
        <div className="kpi">
          <div className="v">{rows.filter((r) => r.claims > 0).length}</div>
          <div className="l">sections with cites</div>
        </div>
        <div className="kpi">
          <div className="v">{total}</div>
          <div className="l">claims cited</div>
        </div>
        <div className="kpi">
          <div className="v">{total ? `${Math.round((valid / total) * 100)}%` : '—'}</div>
          <div className="l">cites resolve</div>
        </div>
        <div className="kpi">
          <div className="v">{guide.gaps.length}</div>
          <div className="l">gaps the corpus doesn't cover</div>
        </div>
      </div>
      {rows.length === 0 && <div className="rail-empty">No claims file for this guide.</div>}
      <ul className="ev-rows">
        {rows.map((row) => (
          <li key={row.id}>
            <button type="button" className="ev-row" onClick={() => onJump(row.id)} title={`jump to ${row.id}`}>
              <span className="ev-id">{row.id}</span>
              <span className="ev-src" aria-label={`${row.videos.length} sources`}>
                {row.videos.map((video) => (
                  <i key={video} className={`ev-sq c${palette.get(video) ?? 0}`} title={titles.get(video) ?? video} />
                ))}
              </span>
              <span className="ev-n">
                {row.claims} claim{row.claims === 1 ? '' : 's'} · {row.videos.length} video{row.videos.length === 1 ? '' : 's'}
                {row.valid < row.claims && <span className="ev-bad"> · {row.claims - row.valid} unresolved</span>}
              </span>
            </button>
          </li>
        ))}
      </ul>
      {guide.gaps.length > 0 && (
        <details className="ev-gaps">
          <summary>What the corpus does not cover ({guide.gaps.length})</summary>
          <ul>
            {guide.gaps.map((gap) => (
              <li key={gap}>{gap}</li>
            ))}
          </ul>
        </details>
      )}
      {guide.claims && guide.claims.structure_errors.length > 0 && (
        <p className="gc-err">structure: {guide.claims.structure_errors.join('; ')}</p>
      )}
    </div>
  );
}
