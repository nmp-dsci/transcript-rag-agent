import { useCallback, useEffect, useState } from 'react';

import { api } from '../api/client';
import type { GuideComment, GuideDetail, GuideJob } from '../api/types';
import type { GuideSelection } from './GuideReader';
import { MicButton } from '../speech/MicButton';
import { useDictation } from '../speech/useDictation';

interface Props {
  guide: GuideDetail;
  selection: GuideSelection | null;
  running: GuideJob | null;
  sdkProblem: string | null;
  demo: boolean;
  /** Server-decided (health `stt`): whether the voice mic renders in the box. */
  stt?: boolean;
  onAdded: (comment: GuideComment) => void;
  onRevisionStarted: (job: GuideJob) => void;
  onJump: (anchor: string | null) => void;
}

const TONE: Record<GuideComment['status'], string> = {
  open: 'warn',
  addressed: 'good',
  deferred: 'plain',
  rejected: 'bad',
};

export function commentCounts(comments: GuideComment[]): Record<GuideComment['status'], number> {
  const counts = { open: 0, addressed: 0, deferred: 0, rejected: 0 };
  for (const comment of comments) counts[comment.status] += 1;
  return counts;
}

/** Screen D: the commentary rail. Comments are local until "Add"; a
 *  revision sends every open comment as one batch and the receipt that comes
 *  back gives each id exactly one outcome. */
export function CommentRail({ guide, selection, running, sdkProblem, demo, stt = false, onAdded, onRevisionStarted, onJump }: Props) {
  const [body, setBody] = useState('');
  const [anchor, setAnchor] = useState<string | null>(null);
  const [sectionId, setSectionId] = useState<string | null>(null);
  const [quote, setQuote] = useState('');
  const [busy, setBusy] = useState<'add' | 'revise' | null>(null);
  const [error, setError] = useState<string | null>(null);

  // A fresh selection in the page becomes the draft's anchor and quote; an
  // empty selection leaves a draft the reader is still typing untouched.
  useEffect(() => {
    if (!selection || !selection.quote) return;
    setAnchor(selection.anchor);
    setSectionId(selection.sectionId);
    setQuote(selection.quote);
  }, [selection]);

  const counts = commentCounts(guide.comments);
  const blocked = !!running && running.status === 'running';

  const add = useCallback(async (text: string) => {
    if (!text.trim()) return;
    setBusy('add');
    setError(null);
    try {
      const created = await api.addGuideComment(guide.slug, {
        body: text.trim(),
        anchor,
        section_id: sectionId,
        quote,
      });
      onAdded(created);
      setBody('');
      setQuote('');
      setAnchor(null);
      setSectionId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, [guide.slug, anchor, sectionId, quote, onAdded]);

  // Stop = Add: a spoken comment is on the record the moment the mic stops,
  // with whatever quote was selected when it started. Typed text still
  // needs the button.
  const dictation = useDictation(body, setBody, (folded) => void add(folded));

  const revise = async () => {
    setBusy('revise');
    setError(null);
    try {
      onRevisionStarted(await api.reviseGuide(guide.slug, {}));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <aside className="rail cr" aria-label="Commentary">
      <div className="rail-head">
        <strong>Commentary</strong>
        <span className="rmeta">
          {counts.open} open · {counts.addressed} addressed
          {counts.rejected > 0 && ` · ${counts.rejected} rejected`}
          {counts.deferred > 0 && ` · ${counts.deferred} deferred`}
        </span>
      </div>
      <div className="rail-list cr-list">
        {guide.comments.length === 0 && (
          <div className="rail-empty">
            {demo ? 'No commentary on this guide yet.' : 'Select text in the guide, then write a comment.'}
          </div>
        )}
        {guide.comments.map((comment) => (
          <button
            key={comment.id}
            type="button"
            className="rentry cr-item"
            onClick={() => onJump(comment.anchor)}
            title={comment.anchor ? `jump to ${comment.anchor}` : undefined}
          >
            {comment.quote && <div className="cr-quote">“{comment.quote}”</div>}
            <div className="cr-body">{comment.body}</div>
            <div className="rmeta">
              <span className={`badge ${TONE[comment.status]}`}>
                {comment.status}
                {comment.resolved_in_version ? ` in v${comment.resolved_in_version}` : ''}
              </span>
              <code>{comment.id}</code>
              {comment.anchor && <span>{comment.anchor}</span>}
            </div>
            {comment.reason && <div className="cr-reason">↳ {comment.reason}</div>}
          </button>
        ))}
      </div>
      {!demo && (
        <form
          className="cr-compose"
          onSubmit={(event) => {
            event.preventDefault();
            void add(body);
          }}
        >
          {quote && (
            <div className="cr-draft-quote">
              “{quote.length > 140 ? `${quote.slice(0, 139)}…` : quote}”
              <button type="button" className="cr-clear" onClick={() => { setQuote(''); setAnchor(null); setSectionId(null); }} aria-label="Clear selection">
                ×
              </button>
            </div>
          )}
          <div className={`cr-box ${dictation.listening ? 'rec' : ''}`}>
            <textarea
              value={dictation.display}
              rows={3}
              readOnly={dictation.listening}
              placeholder={
                dictation.listening
                  ? 'listening… stop to add the comment'
                  : anchor
                    ? `comment on ${anchor}`
                    : stt
                      ? 'comment on the whole guide, or press the mic'
                      : 'comment on the whole guide'
              }
              aria-label="Comment"
              onChange={(event) => setBody(event.target.value)}
            />
            {stt && (
              <MicButton
                size="sm"
                listening={dictation.listening}
                supported={dictation.speech.supported}
                disabled={busy !== null}
                onToggle={dictation.toggle}
                idleTitle="Comment by voice — stop adds it"
              />
            )}
          </div>
          {dictation.speech.error && <p className="gc-err">{dictation.speech.error}</p>}
          <div className="cr-actions">
            <button type="submit" className="btn ghost" disabled={busy !== null || dictation.listening || !body.trim()}>
              {busy === 'add' ? 'adding…' : 'Add comment'}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => void revise()}
              disabled={busy !== null || blocked || counts.open === 0 || !!sdkProblem}
              title={sdkProblem ?? (blocked ? 'a guide job is running' : undefined)}
            >
              {busy === 'revise' ? 'starting…' : `Send ${counts.open} to the agent`}
            </button>
          </div>
          {error && <p className="gc-err">{error}</p>}
          {sdkProblem && counts.open > 0 && <p className="gc-warn">{sdkProblem}</p>}
        </form>
      )}
    </aside>
  );
}
