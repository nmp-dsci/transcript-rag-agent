import { useState } from 'react';

import { api } from '../api/client';

interface Props {
  onIndexed: () => void;
}

export function IndexPanel({ onIndexed }: Props) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<'video' | 'channel'>('video');
  const [url, setUrl] = useState('');
  const [channel, setChannel] = useState('');
  const [latest, setLatest] = useState(5);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ text: string; kind: '' | 'ok' | 'err' }>({
    text: '',
    kind: '',
  });

  const start = async () => {
    if (mode === 'video' && !url.trim()) {
      setResult({ text: 'Enter a video URL.', kind: 'err' });
      return;
    }
    if (mode === 'channel' && !channel.trim()) {
      setResult({ text: 'Enter a channel.', kind: 'err' });
      return;
    }
    setBusy(true);
    setResult({ text: 'Indexing… this can take a while.', kind: '' });
    try {
      const payload =
        mode === 'video'
          ? { mode: 'video' as const, url: url.trim() }
          : { mode: 'channel' as const, channel: channel.trim(), latest };
      const data = await api.index(payload);
      if (data.ok) {
        setResult({ text: `Indexed ${data.target} — ready to ask about it.`, kind: 'ok' });
        setUrl('');
        onIndexed();
      } else {
        setResult({
          text: `Indexing failed${data.detail ? `: ${data.detail}` : ` (exit code ${data.exit_code})`}`,
          kind: 'err',
        });
      }
    } catch (err) {
      setResult({ text: `Request failed: ${(err as Error).message}`, kind: 'err' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel" style={{ marginBottom: 0, borderRadius: 0, border: 'none', borderBottom: '1px solid var(--border)' }}>
      <div className="formrow" style={{ margin: 0 }}>
        <button
          type="button"
          className={`pill${open ? ' on' : ''}`}
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          + Index new content
        </button>
        {!open && result.text ? (
          <span className={`result ${result.kind}`}>{result.text}</span>
        ) : null}
      </div>

      {open ? (
        <>
          <p className="sub" style={{ marginTop: 10 }}>
            Fetch transcripts and add them to the RAG index. Channel runs can take several
            minutes and report when finished.
          </p>
          <div className="formrow">
            <button
              type="button"
              className={`pill${mode === 'video' ? ' on' : ''}`}
              onClick={() => setMode('video')}
            >
              Single video
            </button>
            <button
              type="button"
              className={`pill${mode === 'channel' ? ' on' : ''}`}
              onClick={() => setMode('channel')}
            >
              Channel · latest N
            </button>
          </div>
          {mode === 'video' ? (
            <div className="formrow">
              <input
                type="text"
                value={url}
                spellCheck={false}
                placeholder="https://www.youtube.com/watch?v=…"
                aria-label="Video URL"
                onChange={(event) => setUrl(event.target.value)}
              />
            </div>
          ) : (
            <div className="formrow">
              <input
                type="text"
                value={channel}
                spellCheck={false}
                placeholder="Channel URL or @handle"
                aria-label="Channel"
                onChange={(event) => setChannel(event.target.value)}
              />
              <input
                type="number"
                min={1}
                max={50}
                value={latest}
                title="How many latest videos"
                aria-label="How many latest videos"
                onChange={(event) => setLatest(Number(event.target.value) || 5)}
              />
            </div>
          )}
          <div className="formrow">
            <button type="button" className="btn pri" onClick={() => void start()} disabled={busy}>
              Start indexing
            </button>
            <span className={`result ${result.kind}`}>{result.text}</span>
          </div>
        </>
      ) : null}
    </div>
  );
}
