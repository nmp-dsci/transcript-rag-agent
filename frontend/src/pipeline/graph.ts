/**
 * Chunk-graph geometry and encoding.
 *
 * The server ships PCA coordinates already normalised to [-1, 1], so layout
 * here is a pure affine map into the SVG viewBox — no simulation, and the same
 * corpus always draws the same picture. Everything in this module is pure so
 * the visual encoding can be tested without rendering.
 */

import type { GraphNode } from '../api/types';

/** Fixed viewBox: the SVG scales to its container via preserveAspectRatio. */
export const VIEW_W = 1000;
export const VIEW_H = 620;
const PAD = 28;

export interface Point {
  cx: number;
  cy: number;
}

/**
 * Map a node's normalised coordinates into viewBox space.
 *
 * The y axis is flipped because SVG grows downward while the projection is
 * cartesian; PCA sign is arbitrary, so this only fixes an orientation.
 */
export function projectNode(node: Pick<GraphNode, 'x' | 'y'>): Point {
  const clampedX = Math.max(-1, Math.min(1, node.x));
  const clampedY = Math.max(-1, Math.min(1, node.y));
  return {
    cx: PAD + ((clampedX + 1) / 2) * (VIEW_W - PAD * 2),
    cy: PAD + ((1 - clampedY) / 2) * (VIEW_H - PAD * 2),
  };
}

const MIN_R = 3.5;
const MAX_R = 13;

/** Area-proportional sizing, so a hub does not swamp the plot. */
export function nodeRadius(degree: number, maxDegree: number): number {
  if (maxDegree <= 0) return MIN_R;
  const ratio = Math.max(0, Math.min(1, degree / maxDegree));
  return MIN_R + (MAX_R - MIN_R) * Math.sqrt(ratio);
}

/**
 * Edge opacity across the visible similarity band.
 *
 * Every edge already clears `minSimilarity`, so stretching the band from there
 * to 1 is what makes the strong links stand out against the merely eligible.
 */
export function edgeOpacity(similarity: number, minSimilarity: number): number {
  const floor = Math.max(0, Math.min(0.99, minSimilarity));
  const ratio = Math.max(0, Math.min(1, (similarity - floor) / (1 - floor)));
  return Number((0.07 + ratio * 0.38).toFixed(3));
}

/**
 * Categorical channel colours.
 *
 * Deliberately fixed hexes rather than theme tokens, matching the setup
 * swatches in theme.css: they label which channel a chunk came from, so they
 * must not shift when the palette does.
 */
export const CHANNEL_COLOURS = [
  '#2f81f7',
  '#d29922',
  '#3fb950',
  '#a371f7',
  '#e5534b',
  '#39c5cf',
  '#db6d28',
  '#bf8700',
];

export const UNKNOWN_CHANNEL_COLOUR = '#8b949e';

export interface ChannelLegendItem {
  id: string;
  name: string;
  colour: string;
  count: number;
}

/**
 * Assign a colour per channel, ordered by chunk count so the dominant channel
 * keeps the same colour as nodes come and go.
 */
export function channelLegend(nodes: GraphNode[]): ChannelLegendItem[] {
  const counts = new Map<string, { name: string; count: number }>();
  for (const node of nodes) {
    const id = node.channel_id ?? '';
    const existing = counts.get(id);
    if (existing) existing.count += 1;
    else counts.set(id, { name: node.channel_name ?? 'Unknown channel', count: 1 });
  }
  // Chunks with no channel are a leftover rather than a category, so they sit
  // last whatever their count.
  const ordered = [...counts.entries()].sort((a, b) => {
    if ((a[0] === '') !== (b[0] === '')) return a[0] === '' ? 1 : -1;
    return b[1].count - a[1].count || a[0].localeCompare(b[0]);
  });
  return ordered.map(([id, value], index) => ({
    id,
    name: value.name,
    count: value.count,
    colour: id === '' ? UNKNOWN_CHANNEL_COLOUR : CHANNEL_COLOURS[index % CHANNEL_COLOURS.length]!,
  }));
}

export function colourMap(nodes: GraphNode[]): Map<string, string> {
  return new Map(channelLegend(nodes).map((item) => [item.id, item.colour]));
}

/** Deep-link into the video at the moment the chunk starts. */
export function chunkTimestampUrl(
  sourceUrl: string | null,
  startSeconds: number | null,
): string | null {
  if (!sourceUrl) return null;
  const seconds = Math.max(0, Math.floor(startSeconds ?? 0));
  return `${sourceUrl}${sourceUrl.includes('?') ? '&' : '?'}t=${seconds}s`;
}
