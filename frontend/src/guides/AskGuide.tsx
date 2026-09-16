import { type KeyboardEvent, useState } from 'react';

import { api } from '../api/client';
import type { GuideJob } from '../api/types';
import { GUIDE_QUESTIONS } from '../questions';
import { MicButton } from '../speech/MicButton';
import { useDictation } from '../speech/useDictation';

interface Props {
  /** A job already running blocks a new one; the surface says so instead of 409ing. */
  running: GuideJob | null;
  /** Why the SDK cannot run here (from /api/guides/job), or null when it can. */
  sdkProblem: string | null;
  /** Server-decided (health `stt`): whether the mic renders. */
  stt: boolean;
  onStarted: (job: GuideJob) => void;
}

/** The ask surface: one box, typed or spoken, and Build. The server scopes
 *  the corpus and starts at once; the research map then shows which videos
 *  it chose, so nothing is decided out of sight. */
export function AskGuide({ running, sdkProblem, stt, onStarted }: Props) {
  const [question, setQuestion] = useState('');
  const [allowWeb, setAllowWeb] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dictation = useDictation(question, setQuestion);
  const blocked = !!running && running.status === 'running';
  const ready = question.trim().length > 0 && !busy && !blocked && !dictation.listening && !sdkProblem;

  const build = async () => {
    const clean = question.trim();
    if (!clean || !ready) return;
    setBusy(true);
    setError(null);
    try {
      onStarted(await api.askGuide({ question: clean, allow_web: allowWeb }));
      setQuestion('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void build();
    }
  };

  return (
    <div className="ask" aria-label="New guide">
      <div className="ask-eyebrow">new field guide</div>
      <h2>What do you want a guide on?</h2>
      <p className="ask-lead">
        Ask like you would in Chat. The agent finds the videos that answer it, reads every chunk of them, and
        writes a cited guide you can comment on. It names the guide itself; the question stands in until then.
      </p>
      <div className={`ask-box ${dictation.listening ? 'rec' : ''}`}>
        <textarea
          value={dictation.display}
          rows={3}
          readOnly={dictation.listening}
          aria-label="Question"
          placeholder={
            dictation.listening
              ? 'listening…'
              : 'e.g. How do teams calibrate an LLM judge against human labels, and where does it drift?'
          }
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onKey}
        />
        <div className="ask-row">
          <span className={`ask-hint ${dictation.listening ? 'rec' : ''}`}>
            {dictation.listening ? '● listening — stop to finish' : 'Enter to build · Shift+Enter for a newline'}
          </span>
          <span className="spacer" />
          {stt && (
            <MicButton
              listening={dictation.listening}
              supported={dictation.speech.supported}
              disabled={busy || blocked}
              onToggle={dictation.toggle}
              idleTitle="Ask by voice"
            />
          )}
          <button type="button" className="btn pri" onClick={() => void build()} disabled={!ready} title={sdkProblem ?? (blocked ? 'a guide job is running' : undefined)}>
            {busy ? 'starting…' : 'Build guide'}
          </button>
        </div>
      </div>
      <label className="ask-opt">
        <input type="checkbox" checked={allowWeb} onChange={(event) => setAllowWeb(event.target.checked)} /> allow web search this
        run <span className="gc-dim">(recorded in the manifest; off keeps it corpus-only)</span>
      </label>
      {dictation.speech.error && <p className="gc-err">{dictation.speech.error}</p>}
      {error && <p className="gc-err">{error}</p>}
      {sdkProblem && <p className="gc-warn">{sdkProblem}</p>}
      {blocked && running && (
        <p className="gc-warn">
          A guide job is running ({running.title || running.slug}) — one at a time; ask again when it finishes.
        </p>
      )}
      <div className="suggest ask-suggest" aria-label="Example questions">
        {GUIDE_QUESTIONS.map((example) => (
          <button key={example} type="button" onClick={() => setQuestion(example)} disabled={dictation.listening}>
            {example}
          </button>
        ))}
      </div>
      <p className="ask-note">
        Every run bills your Claude subscription: Sonnet 5 reads the chunks in parallel passes, Opus 5 writes the page.
        A guide of ~700 chunks takes about fifteen minutes.
      </p>
    </div>
  );
}
