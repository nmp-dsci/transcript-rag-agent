import { useState } from 'react';

import { api } from '../api/client';
import type { GuideCandidate, GuideJob, GuideScope } from '../api/types';

interface Props {
  /** A job already running blocks a new one; the panel says so instead of 409ing. */
  running: GuideJob | null;
  /** Why the SDK cannot run here (from /api/guides/job), or null when it can. */
  sdkProblem: string | null;
  onStarted: (job: GuideJob) => void;
}

const DEFAULT_MIN_SCORE = 2;

export function slugFromTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
}

/** Minutes, from the chunk count: the judge run read ~700 chunks in ~12 min. */
export function estimateMinutes(chunks: number): number {
  return Math.max(4, Math.round(chunks / 60) + 3);
}

/** Screen A of the flow: topic → candidate checklist → start. State is local
 *  until "Write guide"; nothing is sent while the user can still change it. */
export function ComposePanel({ running, sdkProblem, onStarted }: Props) {
  const [topic, setTopic] = useState('');
  const [title, setTitle] = useState('');
  const [scope, setScope] = useState<GuideScope | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [allowWeb, setAllowWeb] = useState(false);
  const [busy, setBusy] = useState<'scope' | 'start' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [addId, setAddId] = useState('');

  const findSources = async () => {
    const clean = topic.trim();
    if (!clean) return;
    setBusy('scope');
    setError(null);
    try {
      const found = await api.guideScope(clean, 30);
      setScope(found);
      setSelected(new Set(found.candidates.filter((c) => c.score >= DEFAULT_MIN_SCORE).map((c) => c.video_id)));
      if (!title) setTitle(clean.replace(/\b\w/g, (m) => m.toUpperCase()));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const addVideo = () => {
    const id = addId.trim();
    if (!id) return;
    setSelected((prev) => new Set(prev).add(id));
    setScope((prev) =>
      prev && !prev.candidates.some((c) => c.video_id === id)
        ? { ...prev, candidates: [...prev.candidates, { video_id: id, title: id, channel_name: '', chunk_count: 0, probe_hits: 0, probes: [], title_match: false, score: 0 }] }
        : prev,
    );
    setAddId('');
  };

  const start = async () => {
    if (!scope || selected.size === 0) return;
    setBusy('start');
    setError(null);
    try {
      const job = await api.startGuide({
        topic: scope.topic,
        title: title.trim() || undefined,
        slug: slugFromTitle(title.trim() || scope.topic) || undefined,
        video_ids: scope.candidates.filter((c) => selected.has(c.video_id)).map((c) => c.video_id),
        allow_web: allowWeb,
      });
      onStarted(job);
      setScope(null);
      setTopic('');
      setTitle('');
      setSelected(new Set());
      setAllowWeb(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const chosen = scope ? scope.candidates.filter((c) => selected.has(c.video_id)) : [];
  const chunks = chosen.reduce((n, c) => n + c.chunk_count, 0);
  const clusters = Math.min(6, Math.max(1, Math.ceil(chosen.length / 3)));
  const blocked = !!running && running.status === 'running';

  return (
    <div className="gc" aria-label="New guide">
      <div className="rail-head">
        <strong>New guide</strong>
        <span className="rmeta">the agent reads every chunk of the videos you confirm</span>
      </div>
      {sdkProblem && <p className="gc-warn">{sdkProblem}</p>}
      {blocked && <p className="gc-warn">A guide job is running — wait for it to finish.</p>}
      <form
        className="gc-form"
        onSubmit={(event) => {
          event.preventDefault();
          void findSources();
        }}
      >
        <input
          type="text"
          value={topic}
          placeholder='topic, e.g. "context engineering for coding agents"'
          aria-label="Topic"
          onChange={(event) => setTopic(event.target.value)}
          disabled={busy !== null}
        />
        <button type="submit" className="btn" disabled={busy !== null || !topic.trim()}>
          {busy === 'scope' ? 'scanning…' : 'Find sources'}
        </button>
      </form>
      {error && <p className="gc-err">{error}</p>}
      {scope && (
        <div className="gc-scope">
          <p className="gc-note">
            {scope.probes.length} probe questions through hybrid retrieval · {scope.total_videos} videos scanned ·{' '}
            {scope.candidates.length} candidates. Untick what does not belong; * marks a title match.
          </p>
          <ul className="gc-list">
            {scope.candidates.map((candidate) => (
              <CandidateRow key={candidate.video_id} candidate={candidate} checked={selected.has(candidate.video_id)} onToggle={toggle} />
            ))}
          </ul>
          <div className="gc-add">
            <input
              type="text"
              value={addId}
              placeholder="add a video id"
              aria-label="Add a video id"
              onChange={(event) => setAddId(event.target.value)}
            />
            <button type="button" className="btn ghost" onClick={addVideo} disabled={!addId.trim()}>
              add
            </button>
          </div>
          <label className="gc-opt">
            <input type="text" value={title} aria-label="Title" placeholder="title" onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="gc-opt gc-check">
            <input type="checkbox" checked={allowWeb} onChange={(event) => setAllowWeb(event.target.checked)} /> allow web search this
            run <span className="gc-dim">(off: evidence comes only from the corpus)</span>
          </label>
          <p className="gc-note">
            {chosen.length} videos · {chunks} chunks · {clusters} extraction {clusters === 1 ? 'pass' : 'passes'} · Sonnet 5
            extract → Opus 5 compose · ~{estimateMinutes(chunks)} min
          </p>
          <button type="button" className="btn" onClick={() => void start()} disabled={busy !== null || blocked || chosen.length === 0 || !!sdkProblem}>
            {busy === 'start' ? 'starting…' : 'Write guide'}
          </button>
        </div>
      )}
    </div>
  );
}

function CandidateRow({ candidate, checked, onToggle }: { candidate: GuideCandidate; checked: boolean; onToggle: (id: string) => void }) {
  const hits = candidate.probes.length;
  return (
    <li className={`gc-row ${checked ? 'on' : ''}`}>
      <label>
        <input type="checkbox" checked={checked} onChange={() => onToggle(candidate.video_id)} />
        <span className="gc-title">
          {candidate.title_match && <span className="gc-star">*</span>}
          {candidate.title}
        </span>
        <span className="gc-meta">
          {candidate.channel_name && <span>{candidate.channel_name}</span>}
          <span>{candidate.chunk_count} chunks</span>
        </span>
        <span className="gc-hits" title={candidate.probes.join('\n')}>
          <span className="gc-hitbar" style={{ width: `${Math.min(100, hits * 12.5)}%` }} />
          <span className="gc-hitn">{hits}/8</span>
        </span>
      </label>
    </li>
  );
}
