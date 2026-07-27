/**
 * Knowledge-graph geometry and encoding — the GraphRAG counterpart to graph.ts.
 *
 * Positions come from the server (Fruchterman-Reingold over the entity graph,
 * normalised to [-1, 1] exactly like the chunk graph's PCA projection), so
 * this module only maps them into viewBox space and picks colours/radii —
 * the same split of responsibility as graph.ts, reusing its constants where
 * the encoding is identical (viewBox size, node radius, channel palette).
 */

import type { EntityNode } from "../api/types";
import { CHANNEL_COLOURS, UNKNOWN_CHANNEL_COLOUR } from "./graph";

export interface CommunityLegendItem {
  id: number;
  colour: string;
  count: number;
}

/**
 * Colour per community, ordered by entity count so the dominant cluster keeps
 * the same colour as the graph is refreshed.
 */
export function communityLegend(nodes: EntityNode[]): CommunityLegendItem[] {
  const counts = new Map<number | null, number>();
  for (const node of nodes) {
    counts.set(node.community_id, (counts.get(node.community_id) ?? 0) + 1);
  }
  const ordered = [...counts.entries()].sort((a, b) => {
    if ((a[0] === null) !== (b[0] === null)) return a[0] === null ? 1 : -1;
    return b[1] - a[1];
  });
  return ordered.map(([id, count], index) => ({
    id: id ?? -1,
    count,
    colour:
      id === null
        ? UNKNOWN_CHANNEL_COLOUR
        : CHANNEL_COLOURS[index % CHANNEL_COLOURS.length]!,
  }));
}

export function communityColourMap(nodes: EntityNode[]): Map<number, string> {
  return new Map(communityLegend(nodes).map((item) => [item.id, item.colour]));
}
