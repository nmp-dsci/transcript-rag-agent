import { useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import type { PromptEntry, SystemDesign, SystemDesignNode } from '../api/types';
import { useDesignStyles } from './styles';

const NODE_WIDTH = 180;
const NODE_HEIGHT = 46;
const VIEWBOX = '0 0 1160 600';

/** Highlight {placeholder} template variables inside a prompt body. */
function PromptText({ text }: { text: string }) {
  const parts = text.split(/(\{[a-z_]+\})/g);
  return (
    <pre className="ds-text">
      {parts.map((part, index) =>
        /^\{[a-z_]+\}$/.test(part) ? (
          // eslint-disable-next-line react/no-array-index-key
          <span key={index} className="ds-var">
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
    <div className="ds-item">
      <div className="ds-itemhead">
        <span className="ds-name">{prompt.name}</span>
        <span className={`ds-role ${prompt.role === 'system' ? 'system' : ''}`}>
          {prompt.role.replace('_', ' ')}
        </span>
        <button type="button" className="ds-copy" onClick={() => void copy()}>
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <PromptText text={prompt.text} />
    </div>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

function DetailPanel({ node }: { node: SystemDesignNode | null }) {
  if (!node) {
    return (
      <aside className="ds-panel">
        <p className="ds-empty">
          Click a node on the left to see its system prompts and live configuration.
        </p>
      </aside>
    );
  }
  const configEntries = Object.entries(node.config);
  return (
    <aside className="ds-panel">
      <div className="ds-panelhead">
        <h3>{node.label}</h3>
        <span className="ds-kindtag">{node.kind}</span>
      </div>
      <p className="ds-desc">{node.description}</p>

      {configEntries.length > 0 ? (
        <div className="ds-cfg">
          <table>
            <tbody>
              {configEntries.map(([key, value]) => (
                <tr key={key}>
                  <td className="k">{key}</td>
                  <td className="v">{formatValue(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {node.prompts.length > 0 ? (
        <div className="ds-prompts">
          {node.prompts.map((prompt) => (
            <PromptItem key={prompt.name} prompt={prompt} />
          ))}
        </div>
      ) : null}
    </aside>
  );
}

function Graph({
  design,
  selectedId,
  onSelect,
}: {
  design: SystemDesign;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const byId = useMemo(
    () => new Map(design.nodes.map((node) => [node.id, node])),
    [design.nodes],
  );

  return (
    <div className="ds-graphwrap">
      <svg
        className="ds-graph"
        viewBox={VIEWBOX}
        role="img"
        aria-label="System design graph — click a node for details"
      >
        <defs>
          <marker
            id="ds-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L10,5 L0,10 z" fill="var(--border2)" />
          </marker>
        </defs>

        {design.edges.map((edge) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          const highlighted = selectedId === edge.source || selectedId === edge.target;
          return (
            <line
              key={`${edge.source}-${edge.target}`}
              className={`ds-edge${highlighted ? ' hi' : ''}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              markerEnd="url(#ds-arrow)"
            />
          );
        })}

        {design.nodes.map((node) => (
          <g
            key={node.id}
            className={`ds-node kind-${node.kind}${node.id === selectedId ? ' sel' : ''}`}
            transform={`translate(${node.x - NODE_WIDTH / 2}, ${node.y - NODE_HEIGHT / 2})`}
            role="button"
            tabIndex={0}
            aria-label={`${node.label} (${node.kind})`}
            aria-pressed={node.id === selectedId}
            onClick={() => onSelect(node.id)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onSelect(node.id);
              }
            }}
          >
            <rect width={NODE_WIDTH} height={NODE_HEIGHT} />
            <text x={NODE_WIDTH / 2} y={19} textAnchor="middle">
              {node.label}
            </text>
            <text className="ds-kind" x={NODE_WIDTH / 2} y={34} textAnchor="middle">
              {node.kind}
              {node.prompts.length > 0 ? ` · ${node.prompts.length} prompt${node.prompts.length === 1 ? '' : 's'}` : ''}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function SystemDesignView() {
  useDesignStyles();
  const [design, setDesign] = useState<SystemDesign | null>(null);
  const [error, setError] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    void api
      .systemDesign()
      .then(setDesign)
      .catch(() => setError(true));
  }, []);

  if (error) {
    return <div className="ds-toplevel-empty">Could not load the system design graph from the server.</div>;
  }
  if (!design) {
    return <div className="ds-toplevel-empty">Loading system design…</div>;
  }

  const selected = design.nodes.find((node) => node.id === selectedId) ?? null;

  return (
    <div>
      <p className="ds-intro">
        How transcript·lab is actually built — every answer path, the shared models, and the
        stores each one reads from. Click a node to see its live system prompts and the exact
        configuration it runs with; the server reads these straight off the running config, so
        this view can never drift from what actually executes.
      </p>

      <div className="ds-layout">
        <div>
          <Graph design={design} selectedId={selectedId} onSelect={setSelectedId} />
          <div className="ds-legend">
            <span>
              <i className="ds-swatch agent" /> answer path
            </span>
            <span>
              <i className="ds-swatch stage" /> pipeline stage
            </span>
            <span>
              <i className="ds-swatch model" /> shared model
            </span>
            <span>
              <i className="ds-swatch store" /> store
            </span>
          </div>
        </div>
        <DetailPanel node={selected} />
      </div>
    </div>
  );
}
