import { useEffect, useRef, useState } from 'react';

import type { Corpus, SetupSpec } from '../api/types';

export interface AskOptions {
  question: string;
  setups: string[];
  url: string | null;
  topK: number | null;
  autoJudge: boolean;
}

interface Props {
  setups: SetupSpec[];
  corpus: Corpus | null;
  busy: boolean;
  scope: string;
  onScopeChange: (scope: string) => void;
  defaultSetup: string;
  onDefaultSetupChange: (key: string) => void;
  onAsk: (options: Omit<AskOptions, 'question'> & { question: string }) => void;
  onCancel: () => void;
}

export function Composer({
  setups,
  corpus,
  busy,
  scope,
  onScopeChange,
  defaultSetup,
  onDefaultSetupChange,
  onAsk,
  onCancel,
}: Props) {
  const [question, setQuestion] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [topK, setTopK] = useState('');
  const [extraSetups, setExtraSetups] = useState<string[]>([]);
  const [autoJudge, setAutoJudge] = useState(
    () => localStorage.getItem('tlab.autojudge') !== '0',
  );
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    localStorage.setItem('tlab.autojudge', autoJudge ? '1' : '0');
  }, [autoJudge]);

  useEffect(() => {
    const node = textarea.current;
    if (!node) return;
    node.style.height = 'auto';
    node.style.height = `${Math.min(node.scrollHeight, 140)}px`;
  }, [question]);

  const submit = () => {
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    // The default setup always runs; advanced selections add to it.
    const keys = [defaultSetup, ...extraSetups.filter((key) => key !== defaultSetup)];
    onAsk({
      question: trimmed,
      setups: keys,
      url: scope || null,
      topK: topK ? Number(topK) : null,
      autoJudge,
    });
    setQuestion('');
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        <textarea
          ref={textarea}
          rows={1}
          value={question}
          disabled={busy}
          placeholder="Ask the indexed transcripts anything…  (Enter to send, Shift+Enter for a newline)"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          aria-label="Question"
        />
        <div className="crow">
          <label className="chipselect">
            Scope:{' '}
            <select
              value={scope}
              onChange={(event) => onScopeChange(event.target.value)}
              style={{ border: 'none', background: 'none', padding: 0 }}
              aria-label="Retrieval scope"
            >
              <option value="">Whole corpus</option>
              {(corpus?.videos ?? [])
                .filter((video) => video.source_url)
                .map((video) => (
                  <option key={video.video_id} value={video.source_url ?? ''}>
                    {(video.title || video.video_id).slice(0, 60)}
                  </option>
                ))}
            </select>
          </label>

          <label className="chipselect">
            Agent:{' '}
            <select
              value={defaultSetup}
              onChange={(event) => onDefaultSetupChange(event.target.value)}
              style={{ border: 'none', background: 'none', padding: 0 }}
              aria-label="Answering agent"
            >
              {setups.map((setup) => (
                <option key={setup.key} value={setup.key}>
                  {setup.title}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className={`pill${showAdvanced ? ' on' : ''}`}
            onClick={() => setShowAdvanced(!showAdvanced)}
            aria-expanded={showAdvanced}
          >
            ⚙ advanced
          </button>

          <span className="spacer" />

          {busy ? (
            <button type="button" className="btn danger" onClick={onCancel}>
              Cancel (Esc)
            </button>
          ) : null}
          <button
            type="button"
            className="btn pri"
            onClick={submit}
            disabled={busy || !question.trim()}
          >
            Send
          </button>
        </div>

        {showAdvanced ? (
          <div className="advanced">
            <label className="toggle">
              top_k
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                placeholder="10"
                onChange={(event) => setTopK(event.target.value)}
                style={{ width: 64 }}
              />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={autoJudge}
                onChange={(event) => setAutoJudge(event.target.checked)}
              />
              auto-judge with RAGAS
            </label>
            <span className="microlabel">also run</span>
            {setups
              .filter((setup) => setup.key !== defaultSetup)
              .map((setup) => (
                <button
                  key={setup.key}
                  type="button"
                  title={setup.description}
                  className={`pill${extraSetups.includes(setup.key) ? ' on' : ''}`}
                  onClick={() =>
                    setExtraSetups((current) =>
                      current.includes(setup.key)
                        ? current.filter((key) => key !== setup.key)
                        : [...current, setup.key],
                    )
                  }
                >
                  {setup.title}
                </button>
              ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
