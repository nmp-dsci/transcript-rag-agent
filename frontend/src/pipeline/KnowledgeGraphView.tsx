import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type {
  EntityDetail,
  EntityNode,
  GraphCommunity,
  KnowledgeGraph,
} from "../api/types";
import { fmtSeconds } from "../answers/render";
import { VIEW_H, VIEW_W, nodeRadius, projectNode } from "./graph";
import { communityLegend, communityColourMap } from "./entityGraph";

const MAX_RENDERED_NODES = 400;

function chunkTimestampUrl(
  sourceUrl: string,
  startSeconds: number | null,
): string {
  const seconds = Math.max(0, Math.floor(startSeconds ?? 0));
  return `${sourceUrl}${sourceUrl.includes("?") ? "&" : "?"}t=${seconds}s`;
}

function DetailPanel({
  node,
  detail,
  loading,
  community,
}: {
  node: EntityNode | null;
  detail: EntityDetail | null;
  loading: boolean;
  community: GraphCommunity | undefined;
}) {
  if (!node) {
    return (
      <aside className="kg-panel">
        <p className="kg-empty">
          Click an entity to see its mentions, community, and every dated claim
          the graph extracted about it.
        </p>
      </aside>
    );
  }
  return (
    <aside className="kg-panel">
      <div className="kg-panelhead">
        <h3>{node.name}</h3>
        <span className="kg-kindtag">{node.type}</span>
      </div>
      <p className="kg-desc">
        {node.mentions} mention{node.mentions === 1 ? "" : "s"}
        {community
          ? ` · community ${community.id} (${community.entity_count} entities)`
          : ""}
      </p>
      {community?.summary ? (
        <p className="kg-community-summary">{community.summary}</p>
      ) : null}
      {detail?.entity?.aliases.length ? (
        <p className="kg-aliases">
          also: {detail.entity.aliases.slice(0, 6).join(" · ")}
        </p>
      ) : null}

      <div className="kg-claims">
        <h4>Claims{detail ? ` (${detail.claims.length})` : ""}</h4>
        {loading ? <p className="kg-empty">Loading…</p> : null}
        {detail && detail.claims.length === 0 ? (
          <p className="kg-empty">No claims recorded for this entity.</p>
        ) : null}
        {detail?.claims.map((claim) => (
          <div key={claim.id} className="kg-claim">
            <div className="kg-claimhead">
              <span className="kg-date">
                {claim.upload_date?.slice(0, 10) ?? "undated"}
              </span>
              {claim.video_title ? (
                <span className="kg-video">{claim.video_title}</span>
              ) : null}
            </div>
            <p className="kg-claimtext">{claim.text}</p>
            <a
              className="kg-claimlink"
              href={chunkTimestampUrl(claim.source_url, claim.start_seconds)}
              target="_blank"
              rel="noreferrer"
            >
              watch at {fmtSeconds(claim.start_seconds)} ↗
            </a>
          </div>
        ))}
      </div>
    </aside>
  );
}

export function KnowledgeGraphView() {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EntityDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    void api
      .knowledgeGraph()
      .then(setGraph)
      .catch((err) => setError((err as Error).message));
  }, []);

  const rendered = useMemo(() => {
    if (!graph) return [];
    return [...graph.nodes]
      .sort((a, b) => b.mentions - a.mentions)
      .slice(0, MAX_RENDERED_NODES);
  }, [graph]);

  const renderedIds = useMemo(
    () => new Set(rendered.map((node) => node.id)),
    [rendered],
  );
  const positions = useMemo(
    () => new Map(rendered.map((node) => [node.id, projectNode(node)])),
    [rendered],
  );
  const colours = useMemo(() => communityColourMap(rendered), [rendered]);
  const legend = useMemo(
    () => communityLegend(rendered).slice(0, 8),
    [rendered],
  );
  const maxMentions = useMemo(
    () => rendered.reduce((max, node) => Math.max(max, node.mentions), 1),
    [rendered],
  );

  const matches = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return null;
    return new Set(
      rendered
        .filter((node) => node.name.toLowerCase().includes(term))
        .map((n) => n.id),
    );
  }, [rendered, search]);

  const selectedNode =
    graph?.nodes.find((node) => node.id === selectedId) ?? null;
  const selectedCommunity = graph?.communities.find(
    (community) => community.id === selectedNode?.community_id,
  );

  const pick = (id: string) => {
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    void api
      .knowledgeGraphEntity(id)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setDetailLoading(false));
  };

  if (error) {
    return <p className="kg-toplevel-empty">{error}</p>;
  }
  if (!graph) {
    return <p className="kg-toplevel-empty">Loading knowledge graph…</p>;
  }
  if (graph.nodes.length === 0) {
    return (
      <p className="kg-toplevel-empty">
        No knowledge graph built yet. Run{" "}
        <code>uv run python -m src.cli index-graph</code> to extract entities
        and claims from the indexed chunks.
      </p>
    );
  }

  return (
    <div className="kg-layout">
      <div>
        <div className="kg-toolbar">
          <input
            type="text"
            placeholder="Find an entity by name…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search entities"
          />
          <span className="kg-count">
            {graph.nodes.length} entities · {graph.edges.length} relations ·{" "}
            {graph.communities.length} communities
            {rendered.length < graph.nodes.length
              ? ` (showing top ${rendered.length} by mentions)`
              : ""}
          </span>
        </div>

        <div className="kg-graphwrap">
          <svg
            className="kg-graph"
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            role="img"
            aria-label="Knowledge graph — click an entity for details"
          >
            <g className="kg-edges" stroke="var(--dim)">
              {graph.edges.map((edge) => {
                if (
                  !renderedIds.has(edge.source) ||
                  !renderedIds.has(edge.target)
                )
                  return null;
                const from = positions.get(edge.source);
                const to = positions.get(edge.target);
                if (!from || !to) return null;
                return (
                  <line
                    key={`${edge.source}|${edge.target}`}
                    x1={from.cx}
                    y1={from.cy}
                    x2={to.cx}
                    y2={to.cy}
                    strokeWidth={0.6}
                    opacity={0.12}
                  />
                );
              })}
            </g>
            <g className="kg-nodes">
              {rendered.map((node) => {
                const point = positions.get(node.id);
                if (!point) return null;
                const radius = nodeRadius(node.mentions, maxMentions);
                const dimmed = matches !== null && !matches.has(node.id);
                const selected = node.id === selectedId;
                return (
                  <g
                    key={node.id}
                    className="kg-node"
                    opacity={dimmed ? 0.12 : 1}
                    onClick={() => pick(node.id)}
                  >
                    <circle
                      cx={point.cx}
                      cy={point.cy}
                      r={radius + 5}
                      fill="transparent"
                    />
                    <circle
                      cx={point.cx}
                      cy={point.cy}
                      r={selected ? radius + 1.5 : radius}
                      fill={
                        colours.get(node.community_id ?? -1) ?? "var(--dim)"
                      }
                      stroke={selected ? "var(--text)" : "var(--panel)"}
                      strokeWidth={selected ? 1.6 : 0.7}
                    >
                      <title>{`${node.name} (${node.mentions} mentions)`}</title>
                    </circle>
                  </g>
                );
              })}
            </g>
          </svg>
        </div>

        <div className="kg-legend">
          {legend.map((item) => (
            <span key={item.id}>
              <i className="kg-swatch" style={{ background: item.colour }} />
              community {item.id >= 0 ? item.id : "—"} · {item.count}
            </span>
          ))}
        </div>
      </div>

      <DetailPanel
        node={selectedNode}
        detail={detail}
        loading={detailLoading}
        community={selectedCommunity}
      />
    </div>
  );
}
