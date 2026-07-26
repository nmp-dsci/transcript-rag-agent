import { useEffect, useState } from 'react';

import { api } from '../api/client';
import type { PromptEntry, Prompts } from '../api/types';
import { usePromptStyles } from './styles';

/** Highlight {placeholder} template variables inside a prompt body. */
function PromptText({ text }: { text: string }) {
  const parts = text.split(/(\{[a-z_]+\})/g);
  return (
    <pre className="pr-text">
      {parts.map((part, index) =>
        /^\{[a-z_]+\}$/.test(part) ? (
          // eslint-disable-next-line react/no-array-index-key
          <span key={index} className="pr-var">
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </pre>
  );
}

function PromptItem({ prompt }: { prompt: PromptEntry }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(prompt.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard unavailable (http, permissions) — the button just stays quiet.
    }
  };
  return (
    <div className="pr-item">
      <div className="pr-itemhead">
        <span className="pr-name">{prompt.name}</span>
        <span className={`pr-role ${prompt.role === 'system' ? 'system' : ''}`}>
          {prompt.role.replace('_', ' ')}
        </span>
        <span className="pr-module">{prompt.module}</span>
        <button type="button" className="pr-copy" onClick={() => void copy()}>
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      {prompt.template_vars.length > 0 && (
        <div className="pr-vars">
          {prompt.template_vars.map((name) => (
            <span key={name} className="pr-var">{`{${name}}`}</span>
          ))}
        </div>
      )}
      <PromptText text={prompt.text} />
    </div>
  );
}

export function PromptsView() {
  usePromptStyles();
  const [data, setData] = useState<Prompts | null>(null);
  const [error, setError] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  useEffect(() => {
    void api
      .prompts()
      .then(setData)
      .catch(() => setError(true));
  }, []);

  if (error) {
    return <div className="exp-empty">Could not load prompts from the server.</div>;
  }
  if (!data) {
    return <div className="exp-empty">Loading prompts…</div>;
  }

  return (
    <div>
      <p className="pr-intro">
        Every LLM prompt in the app, grouped by system. The server returns the
        live constants the agents import — this view shows literally what runs,
        so it can never drift from the code.
      </p>
      {data.systems.map((system) => {
        const isOpen = open[system.key] ?? false;
        return (
          <section key={system.key} className="pr-group">
            <button
              type="button"
              className="pr-grouphead"
              aria-expanded={isOpen}
              onClick={() => setOpen({ ...open, [system.key]: !isOpen })}
            >
              <span className="pr-caret">{isOpen ? '▾' : '▸'}</span>
              <h3>{system.title}</h3>
              <span className="pr-groupdesc">{system.description}</span>
              <span className="pr-count">
                {system.count} prompt{system.count === 1 ? '' : 's'}
              </span>
            </button>
            {isOpen &&
              system.prompts.map((prompt) => (
                <PromptItem key={prompt.name} prompt={prompt} />
              ))}
          </section>
        );
      })}
      {data.notes.map((note) => (
        <p key={note} className="pr-note">
          {note}
        </p>
      ))}
    </div>
  );
}
