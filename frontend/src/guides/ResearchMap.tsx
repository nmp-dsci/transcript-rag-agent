import type { GuideActivity, GuideClusterState, GuideJob } from '../api/types';

interface VideoMeta {
  video_id: string;
  title: string;
  channel_name: string;
  chunk_count: number;
}

/** What the map draws, derived from a job snapshot alone — pure, so the
 *  component is testable with a fixture and never needs the stream itself. */
export interface ResearchView {
  clusters: { name: string; state: GuideClusterState; videos: VideoMeta[]; read: Set<string> }[];
  chunksTotal: number;
  chunksRead: number;
  claims: number;
  retrievalQueries: number;
  webFetches: number;
  filesRead: number;
}

function isClusterState(value: unknown): value is GuideClusterState {
  return !!value && typeof value === 'object' && 'status' in (value as object);
}

export function deriveResearch(job: GuideJob, clusterVideos: Record<string, string[]> = {}): ResearchView {
  const videos = (job.clusters.videos ?? {}) as Record<string, VideoMeta>;
  const readIds = new Set<string>();
  for (const entry of job.activity) {
    if (entry.name !== 'Read') continue;
    const match = /corpus\/([A-Za-z0-9_-]+)\.md/.exec(entry.message);
    if (match?.[1]) readIds.add(match[1]);
  }
  const clusterNames = Object.keys(job.clusters)
    .filter((key) => key !== 'videos' && isClusterState(job.clusters[key]))
    .sort((a, b) => Number(a.split('-')[1] ?? 0) - Number(b.split('-')[1] ?? 0));
  const assigned = new Set<string>();
  const clusters = clusterNames.map((name) => {
    const ids = clusterVideos[name] ?? [];
    ids.forEach((id) => assigned.add(id));
    return {
      name,
      state: job.clusters[name] as GuideClusterState,
      videos: ids.map((id) => videos[id] ?? { video_id: id, title: id, channel_name: '', chunk_count: 0 }),
      read: new Set(ids.filter((id) => readIds.has(id))),
    };
  });
  // Videos the export listed but no cluster claims yet (before extract starts)
  // sit in an unnamed group so the map is never blank while chunks exist.
  const unassigned = Object.values(videos).filter((v) => !assigned.has(v.video_id));
  if (unassigned.length && clusters.length === 0) {
    clusters.push({
      name: 'all',
      state: { status: 'pending', claims: 0 },
      videos: unassigned,
      read: new Set(unassigned.filter((v) => readIds.has(v.video_id)).map((v) => v.video_id)),
    });
  }
  const all = Object.values(videos);
  const chunksTotal = job.counters.chunk_count ?? all.reduce((n, v) => n + v.chunk_count, 0);
  const chunksRead = all.filter((v) => readIds.has(v.video_id)).reduce((n, v) => n + v.chunk_count, 0);
  return {
    clusters,
    chunksTotal,
    chunksRead,
    claims: job.counters.claims ?? clusterNames.reduce((n, k) => n + ((job.clusters[k] as GuideClusterState).claims ?? 0), 0),
    retrievalQueries: job.counters.retrieval_queries ?? 0,
    webFetches: job.counters.web_fetches ?? 0,
    filesRead: job.counters.files_read ?? 0,
  };
}

/** Cluster → video ids, recovered from the activity log's file reads per
 *  extractor label until the export summary carries it explicitly. */
export function clusterMembership(job: GuideJob): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const entry of job.activity) {
    const cluster = /^extract:(cluster-\d+)$/.exec(entry.label)?.[1];
    const video = /corpus\/([A-Za-z0-9_-]+)\.md/.exec(entry.message)?.[1];
    if (!cluster || !video) continue;
    out[cluster] ??= [];
    if (!out[cluster].includes(video)) out[cluster].push(video);
  }
  return out;
}

export function stageLabel(stage: string, state?: { status: string; message?: string }): string {
  if (!state) return stage;
  if (state.status === 'done' || state.status === 'skip') return `✓ ${stage}`;
  if (state.status === 'error') return `✕ ${stage}`;
  if (state.status === 'start' || state.status === 'progress') return `● ${stage}`;
  return stage;
}

export function StageRail({ job }: { job: GuideJob }) {
  return (
    <ol className="gj-stages" aria-label="Stages">
      {job.stage_order.map((stage) => {
        const state = job.stages[stage];
        const status = state?.status ?? 'pending';
        const live = status === 'start' || status === 'progress';
        return (
          <li key={stage} className={`gj-stage ${status} ${live ? 'live' : ''}`} title={state?.message ?? ''}>
            {stageLabel(stage, state)}
          </li>
        );
      })}
    </ol>
  );
}

export function ActivityLog({ activity, limit = 14 }: { activity: GuideActivity[]; limit?: number }) {
  const recent = activity.slice(-limit).reverse();
  return (
    <div className="gj-log" aria-label="Agent activity">
      {recent.length === 0 && <div className="gj-log-empty">waiting for the agent…</div>}
      {recent.map((entry, index) => (
        // eslint-disable-next-line react/no-array-index-key
        <div key={`${entry.at}-${index}`} className={`gj-log-row ${entry.name.endsWith('retrieve_chunks') ? 'q' : ''}`}>
          <span className="gj-log-at">{entry.at.slice(11, 19)}</span>
          <span className="gj-log-label">{entry.label.replace('extract:', '')}</span>
          <span className="gj-log-msg">
            {entry.question ? `retrieve_chunks "${entry.question}"` : entry.message}
            {entry.results != null && <span className="gj-log-dim"> → {entry.results} chunks</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

const BOX_W = 170;
const BOX_GAP = 12;
const ROW_H = 26;
const HEAD_H = 34;

/** The map: one box per cluster, one bar per video (full when the extractor
 *  has read its file), the claim count once the pass is done. Inline SVG on
 *  theme tokens, no fixed pixel sizes beyond the viewBox. */
function Map({ view }: { view: ResearchView }) {
  const cols = Math.max(1, view.clusters.length);
  const rows = Math.max(1, ...view.clusters.map((c) => c.videos.length));
  const width = cols * (BOX_W + BOX_GAP) - BOX_GAP + 8;
  const boxH = HEAD_H + rows * ROW_H + 8;
  const height = boxH + 8;
  return (
    <svg className="gj-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Research map: clusters, videos read, claims found">
      {view.clusters.map((cluster, index) => {
        const x = 4 + index * (BOX_W + BOX_GAP);
        const done = cluster.state.status === 'done';
        const live = cluster.state.status === 'reading';
        return (
          <g key={cluster.name} transform={`translate(${x} 4)`}>
            <rect className={`gj-box ${cluster.state.status}`} width={BOX_W} height={boxH} rx="8" />
            <text className="gj-box-title" x="10" y="18">
              {cluster.name}
            </text>
            <text className={`gj-box-state ${cluster.state.status}`} x={BOX_W - 10} y="18" textAnchor="end">
              {done ? `${cluster.state.claims} claims` : live ? 'reading' : cluster.state.status}
            </text>
            {cluster.videos.map((video, row) => {
              const y = HEAD_H + row * ROW_H;
              const read = cluster.read.has(video.video_id);
              return (
                <g key={video.video_id} transform={`translate(10 ${y})`}>
                  <title>
                    {video.title} · {video.chunk_count} chunks{read ? ' · read' : ''}
                  </title>
                  <rect className="gj-bar-bg" width={BOX_W - 20} height="6" rx="2" />
                  {read && <rect className="gj-bar" width={BOX_W - 20} height="6" rx="2" />}
                  <text className={`gj-bar-label ${read ? 'read' : ''}`} y="17">
                    {video.title.length > 26 ? `${video.title.slice(0, 25)}…` : video.title} · {video.chunk_count}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

/** The videos the run reads, named once the export has listed them. Shown in
 *  the header because an asked-for guide skips the checklist: this is where
 *  the choice is visible. */
function ChosenVideos({ job }: { job: GuideJob }) {
  const videos = (job.clusters.videos ?? {}) as Record<string, VideoMeta>;
  return (
    <details className="note-more gj-chosen">
      <summary>
        {job.video_ids.length} video{job.video_ids.length === 1 ? '' : 's'} chosen from the corpus
      </summary>
      <ul>
        {job.video_ids.map((id) => {
          const meta = videos[id];
          return (
            <li key={id}>
              <a href={`https://www.youtube.com/watch?v=${id}`} target="_blank" rel="noopener">
                {meta?.title ?? id}
              </a>
              {meta && (
                <span className="gj-log-dim">
                  {' '}
                  · {meta.channel_name} · {meta.chunk_count} chunks
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </details>
  );
}

export function ResearchMap({ job }: { job: GuideJob }) {
  const view = deriveResearch(job, clusterMembership(job));
  return (
    <div className="gj">
      <div className="gj-head">
        <div>
          {job.question ? (
            <>
              <p className="gj-eyebrow">answering</p>
              <h2>{job.question}</h2>
            </>
          ) : (
            <h2>{job.title || job.slug}</h2>
          )}
          <p className="gj-sub">
            {job.kind === 'write' ? 'Writing from' : job.kind === 'revise' ? 'Revising with' : 'Backfilling for'}{' '}
            {job.video_ids.length} videos · {job.allow_web ? 'web on' : 'corpus only'} · started {job.started_at.slice(11, 16)}
            {job.status === 'error' && <span className="gj-err"> · failed: {job.error}</span>}
            {job.status === 'done' && (
              <span className="gj-ok">
                {' '}
                · published v{job.version ?? '?'}
                {job.question && job.title && job.title !== job.question ? ` as “${job.title}”` : ''}
              </span>
            )}
          </p>
          {job.kind === 'write' && <ChosenVideos job={job} />}
        </div>
        <StageRail job={job} />
      </div>
      <div className="gj-body">
        <div className="gj-left">
          {job.kind === 'write' &&
            (view.clusters.length > 0 ? <Map view={view} /> : <div className="gj-map-empty">exporting the corpus…</div>)}
          {job.kind === 'revise' && (
            <div className="gj-revise">
              <div className="gj-revise-h">Applying {job.comment_ids.length} comment{job.comment_ids.length === 1 ? '' : 's'} as one batch</div>
              <div className="gj-revise-ids">
                {job.comment_ids.map((id) => (
                  <code key={id}>{id}</code>
                ))}
              </div>
              <p className="gj-sub">
                The reviser edits the page in place, cites anything new from the corpus, and must return a receipt naming every id
                exactly once — addressed, deferred or rejected.
              </p>
            </div>
          )}
          {job.kind === 'markdown' && <div className="gj-map-empty">rendering the agent-facing copy from the published page…</div>}
          <div className="gj-kpis">
            {job.kind === 'write' ? (
              <>
                <div className="kpi">
                  <div className="v">
                    {view.chunksRead} / {view.chunksTotal}
                  </div>
                  <div className="l">chunks read in full</div>
                </div>
                <div className="kpi">
                  <div className="v">{view.claims}</div>
                  <div className="l">claims with a chunk id</div>
                </div>
              </>
            ) : (
              <>
                <div className="kpi">
                  <div className="v">{view.filesRead}</div>
                  <div className="l">files read</div>
                </div>
                <div className="kpi">
                  <div className="v">{job.counters.tool_calls ?? 0}</div>
                  <div className="l">tool calls</div>
                </div>
              </>
            )}
            <div className="kpi">
              <div className="v">{view.retrievalQueries}</div>
              <div className="l">retrieval queries</div>
            </div>
            <div className="kpi">
              <div className="v">{view.webFetches}</div>
              <div className="l">web fetches{job.allow_web ? '' : ' (off)'}</div>
            </div>
          </div>
          {job.message && job.status === 'running' && <p className="gj-msg">{job.message}</p>}
          {job.cite_total != null && (
            <p className="gj-msg">
              cites {job.cite_valid}/{job.cite_total} resolve
            </p>
          )}
        </div>
        <ActivityLog activity={job.activity} />
      </div>
    </div>
  );
}
