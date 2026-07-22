import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../api/client';
import type { ChunkGraph, GraphEdge, GraphNode } from '../api/types';
import { fmtSeconds } from '../answers/render';
import {
  type Point,
  VIEW_H,
  VIEW_W,
  channelLegend,
  chunkTimestampUrl,
  colourMap,
  edgeOpacity,
  nodeRadius,
  projectNode,
} from './graph';

const DIMMED = 0.09;
const K_MIN = 1;
const K_MAX = 20;
const MAX_RENDERED_NODES = 500;

interface EdgeLayerProps {
  edges: GraphEdge[];
  positions: Map<string, Point>;
  minSimilarity: number;
  /** Ids in the query neighbourhood; empty means "no query, show everything". */
  highlight: Set<string>;
}

/**
 * Edges are the bulk of the DOM at real corpus size, and they never depend on
 * hover — memoising them keeps pointer movement from touching ~1000 nodes.
 */
const EdgeLayer = memo(function EdgeLayer({
  edges,
  positions,
  minSimilarity,
  highlight,
}: EdgeLayerProps) {
  const querying = highlight.size > 0;
  return (
    <g className="graph-edges" stroke="var(--dim)">
      {edges.map((edge) => {
        const from = positions.get(edge.source);
        const to = positions.get(edge.target);
        if (!from || !to) return null;
        const touched = !querying || highlight.has(edge.source) || highlight.has(edge.target);
        return (
          <line
            key={`${edge.source}|${edge.target}`}
            x1={from.cx}
            y1={from.cy}
            x2={to.cx}
            y2={to.cy}
            strokeWidth={touched && querying ? 1.1 : 0.7}
            opacity={touched ? edgeOpacity(edge.similarity, minSimilarity) : 0.03}
          />
        );
      })}
    </g>
  );
});

interface NodeLayerProps {
  nodes: GraphNode[];
  positions: Map<string, Point>;
  colours: Map<string, string>;
  maxDegree: number;
  highlight: Set<string>;
  onHover: (id: string | null) => void;
  onPick: (id: string) => void;
}

/** Also memoised: only a new graph or a new query changes what a node looks like. */
const NodeLayer = memo(function NodeLayer({
  nodes,
  positions,
  colours,
  maxDegree,
  highlight,
  onHover,
  onPick,
}: NodeLayerProps) {
  const querying = highlight.size > 0;
  return (
    <g className="graph-nodes">
      {nodes.map((node) => {
        const point = positions.get(node.id);
        if (!point) return null;
        const radius = nodeRadius(node.degree, maxDegree);
        const inQuery = highlight.has(node.id);
        const colour = colours.get(node.channel_id ?? '') ?? 'var(--dim)';
        return (
          <g
            key={node.id}
            className="graph-node"
            opacity={querying && !inQuery ? DIMMED : 1}
            onMouseEnter={() => onHover(node.id)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onPick(node.id)}
          >
            {/* Transparent hit target: the smallest nodes are only a few px wide. */}
            <circle cx={point.cx} cy={point.cy} r={radius + 5} fill="transparent" />
            <circle
              cx={point.cx}
              cy={point.cy}
              r={inQuery ? radius + 1.5 : radius}
              fill={colour}
              stroke={inQuery ? 'var(--text)' : 'var(--panel)'}
              strokeWidth={inQuery ? 1.6 : 0.7}
            >
              <title>{`${node.channel_name ?? 'Unknown'} · ${node.title ?? node.video_id} #c${node.chunk_index}\n${node.preview}`}</title>
            </circle>
          </g>
        );
      })}
    </g>
  );
});

export function ChunkGraphView() {
  const [k, setK] = useState(6);
  const [minSimilarity, setMinSimilarity] = useState(0.55);
  const [query, setQuery] = useState('');
  const [graph, setGraph] = useState<ChunkGraph | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);

  // Guards state updates against a slow request outliving the component, e.g.
  // the user switching away from this sub-tab before it resolves.
  const liveRef = useRef(true);
  useEffect(
    () => () => {
      liveRef.current = false;
    },
    [],
  );

  const load = useCallback(
    async (queryText: string) => {
      setBusy(true);
      setError(null);
      try {
        const next = await api.chunkGraph({
          k,
          min_similarity: minSimilarity,
          query: queryText.trim() || null,
          top_k: 10,
        });
        if (!liveRef.current) return;
        setGraph(next);
        setPinned(null);
        setHovered(null);
      } catch (err) {
        if (!liveRef.current) return;
        setError((err as Error).message);
        setGraph(null);
      } finally {
        if (liveRef.current) setBusy(false);
      }
    },
    [k, minSimilarity],
  );

  // Only the first render fetches: dragging the similarity slider changes
  // `load`, and rebuilding the graph on every step of it would be unusable.
  const loaded = useRef(false);
  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    void load('');
  }, [load]);

  const totalNodeCount = graph?.nodes.length ?? 0;
  // Rendering scales directly with the DOM this produces (two circles and two
  // handlers per node, a line per edge), so a huge corpus is capped to the
  // highest-degree chunks rather than drawn in full.
  const nodes = useMemo(() => {
    const allNodes = graph?.nodes ?? [];
    if (allNodes.length <= MAX_RENDERED_NODES) return allNodes;
    return [...allNodes].sort((a, b) => b.degree - a.degree).slice(0, MAX_RENDERED_NODES);
  }, [graph]);
  const truncated = nodes.length < totalNodeCount;
  const edges = useMemo(() => {
    const allEdges = graph?.edges ?? [];
    if (!truncated) return allEdges;
    const ids = new Set(nodes.map((node) => node.id));
    return allEdges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  }, [graph, nodes, truncated]);
  const positions = useMemo(
    () => new Map(nodes.map((node) => [node.id, projectNode(node)] as const)),
    [nodes],
  );
  const colours = useMemo(() => colourMap(nodes), [nodes]);
  const legend = useMemo(() => channelLegend(nodes), [nodes]);
  const maxDegree = useMemo(
    () => nodes.reduce((most, node) => Math.max(most, node.degree), 0),
    [nodes],
  );
  const nearest = graph?.query?.nearest ?? [];
  // Keyed on the graph, not on `nearest`: the `?? []` fallback would be a fresh
  // array every render, and a fresh Set would defeat the layer memoisation.
  const highlight = useMemo(
    () => new Set((graph?.query?.nearest ?? []).map((hit) => hit.chunk_id)),
    [graph],
  );
  const byId = useMemo(() => new Map(nodes.map((node) => [node.id, node] as const)), [nodes]);

  const focusId = pinned ?? hovered;
  const focus = focusId ? (byId.get(focusId) ?? null) : null;
  const focusPoint = focusId ? positions.get(focusId) : undefined;
  const pick = useCallback((id: string) => setPinned((current) => (current === id ? null : id)), []);

  const stats = graph?.stats;
  const link = focus ? chunkTimestampUrl(focus.source_url, focus.start_seconds) : null;

  return (
    <div className="graph">
      <div className="graph-controls">
        <span className="microlabel" style={{ color: 'var(--accent2)' }}>
          chunk graph
        </span>
        <input
          type="search"
          value={query}
          placeholder="Highlight a query's retrieval neighbourhood…"
          aria-label="Graph query"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void load(query);
          }}
        />
        <label className="toggle">
          k
          <input
            type="number"
            min={K_MIN}
            max={K_MAX}
            value={k}
            style={{ width: 54 }}
            aria-label="Neighbours per chunk"
            onChange={(event) =>
              setK(Math.min(K_MAX, Math.max(K_MIN, Number(event.target.value) || K_MIN)))
            }
          />
        </label>
        <label className="toggle">
          min sim
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={minSimilarity}
            aria-label="Minimum similarity"
            onChange={(event) => setMinSimilarity(Number(event.target.value))}
          />
          <span className="graph-num">{minSimilarity.toFixed(2)}</span>
        </label>
        <button type="button" className="btn pri" onClick={() => void load(query)} disabled={busy}>
          {busy ? 'Building…' : query.trim() ? 'Trace query' : 'Rebuild'}
        </button>
        {query.trim() && graph?.query ? (
          <button
            type="button"
            className="btn sm"
            onClick={() => {
              setQuery('');
              void load('');
            }}
          >
            clear query
          </button>
        ) : null}
        {stats ? (
          <>
            <span className="badge plain">{stats.nodes} nodes</span>
            <span className="badge plain">{stats.edges} edges</span>
            <span className={`badge ${stats.isolated_nodes > 0 ? 'warn' : 'plain'}`}>
              {stats.isolated_nodes} isolated
            </span>
            <span className="badge acc">mean sim {stats.mean_similarity.toFixed(3)}</span>
          </>
        ) : null}
        {truncated ? (
          <span className="badge warn">
            Showing the {MAX_RENDERED_NODES} highest-degree of {totalNodeCount} chunks — raise min
            similarity or lower k to see the full graph
          </span>
        ) : null}
      </div>

      {error ? <div className="errtext" style={{ padding: '10px 16px' }}>{error}</div> : null}

      <div className="graph-body">
        <div className="graph-canvas">
          {busy && !graph ? (
            <div className="waiting" style={{ padding: 18 }}>
              <span className="pulse" />
              building the similarity graph…
            </div>
          ) : nodes.length === 0 && !error ? (
            <div className="empty" style={{ margin: 18 }}>
              <h2>Nothing to plot yet</h2>
              <p>
                Index a video or a channel, then this shows every chunk positioned by embedding
                similarity — the map retrieval searches over.
              </p>
            </div>
          ) : (
            <svg
              className="graph-svg"
              viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
              preserveAspectRatio="xMidYMid meet"
              role="img"
              aria-label={`Similarity graph of ${nodes.length} chunks`}
            >
              <EdgeLayer
                edges={edges}
                positions={positions}
                minSimilarity={minSimilarity}
                highlight={highlight}
              />
              <NodeLayer
                nodes={nodes}
                positions={positions}
                colours={colours}
                maxDegree={maxDegree}
                highlight={highlight}
                onHover={setHovered}
                onPick={pick}
              />
              {focusPoint ? (
                <circle
                  className="graph-focus"
                  cx={focusPoint.cx}
                  cy={focusPoint.cy}
                  r={nodeRadius(focus?.degree ?? 0, maxDegree) + 7}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth={2}
                />
              ) : null}
            </svg>
          )}
          {graph && nodes.length > 0 && edges.length === 0 ? (
            <p className="graph-note">
              No edges clear a similarity of {minSimilarity.toFixed(2)} — lower the threshold to
              see how these chunks relate.
            </p>
          ) : null}
        </div>

        <aside className="graph-side">
          {legend.length > 0 ? (
            <div className="graph-block">
              <span className="microlabel">channels</span>
              {legend.map((item) => (
                <div className="graph-legend" key={item.id || 'unknown'}>
                  <span className="sw" style={{ background: item.colour }} />
                  <span className="nm">{item.name}</span>
                  <span className="ct">{item.count}</span>
                </div>
              ))}
            </div>
          ) : null}

          {graph?.query ? (
            <div className="graph-block">
              <span className="microlabel">nearest to “{graph.query.text}”</span>
              {nearest.length === 0 ? (
                <p className="graph-note">Nothing matched this query.</p>
              ) : (
                nearest.map((hit, index) => {
                  const node = byId.get(hit.chunk_id);
                  return (
                    <button
                      type="button"
                      key={hit.chunk_id}
                      className={`graph-hit${hit.chunk_id === focusId ? ' on' : ''}`}
                      onClick={() => pick(hit.chunk_id)}
                    >
                      <span className="rk">{index + 1}</span>
                      <span className="tx">{node?.preview ?? hit.chunk_id}</span>
                      <span className="sc">{hit.similarity.toFixed(2)}</span>
                    </button>
                  );
                })
              )}
            </div>
          ) : null}

          <div className="graph-block">
            <span className="microlabel">chunk</span>
            {focus ? (
              <div className="chunkcard on" style={{ marginTop: 6 }}>
                <div className="h">
                  <span className="id">#c{focus.chunk_index}</span>
                  <span>{fmtSeconds(focus.start_seconds)}</span>
                  <span>{focus.degree} links</span>
                </div>
                <p style={{ fontWeight: 600, color: 'var(--text)' }}>
                  {focus.title ?? focus.video_id}
                </p>
                <p style={{ color: 'var(--muted)', fontSize: 11 }}>
                  {focus.channel_name ?? 'Unknown channel'}
                </p>
                <p style={{ marginTop: 6 }}>{focus.preview}</p>
                {link ? (
                  <p style={{ marginTop: 6 }}>
                    <a href={link} target="_blank" rel="noreferrer" style={{ color: 'var(--accent2)' }}>
                      ▸ open at {fmtSeconds(focus.start_seconds)}
                    </a>
                  </p>
                ) : null}
              </div>
            ) : (
              <p className="graph-note">
                Hover a node to preview its chunk; click to pin it here.
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
