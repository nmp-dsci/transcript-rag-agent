import { describe, expect, it } from 'vitest';

import { graphNode } from './fixtures';
import {
  CHANNEL_COLOURS,
  UNKNOWN_CHANNEL_COLOUR,
  VIEW_H,
  VIEW_W,
  channelLegend,
  chunkTimestampUrl,
  colourMap,
  edgeOpacity,
  nodeRadius,
  projectNode,
} from './graph';

describe('projectNode', () => {
  it('puts the origin at the centre of the viewBox', () => {
    const { cx, cy } = projectNode({ x: 0, y: 0 });
    expect(cx).toBeCloseTo(VIEW_W / 2);
    expect(cy).toBeCloseTo(VIEW_H / 2);
  });

  it('keeps the extremes inside the padded box', () => {
    const min = projectNode({ x: -1, y: -1 });
    const max = projectNode({ x: 1, y: 1 });
    expect(min.cx).toBeGreaterThan(0);
    expect(max.cx).toBeLessThan(VIEW_W);
    expect(min.cy).toBeLessThan(VIEW_H);
    expect(max.cy).toBeGreaterThan(0);
  });

  it('flips y so the projection reads as cartesian', () => {
    expect(projectNode({ x: 0, y: 1 }).cy).toBeLessThan(projectNode({ x: 0, y: -1 }).cy);
  });

  it('clamps coordinates that overshoot the normalised range', () => {
    expect(projectNode({ x: 5, y: 0 })).toEqual(projectNode({ x: 1, y: 0 }));
  });
});

describe('nodeRadius', () => {
  it('grows with degree', () => {
    expect(nodeRadius(10, 10)).toBeGreaterThan(nodeRadius(1, 10));
  });

  it('falls back to the minimum when nothing has edges', () => {
    expect(nodeRadius(0, 0)).toBe(nodeRadius(0, 10));
  });

  it('stays bounded for the biggest hub', () => {
    expect(nodeRadius(999, 999)).toBeLessThanOrEqual(13);
  });
});

describe('edgeOpacity', () => {
  it('stretches the visible band from the threshold to 1', () => {
    expect(edgeOpacity(0.5, 0.5)).toBeLessThan(edgeOpacity(0.9, 0.5));
  });

  it('never renders an edge fully invisible or opaque', () => {
    expect(edgeOpacity(0.5, 0.5)).toBeGreaterThan(0);
    expect(edgeOpacity(1, 0)).toBeLessThan(1);
  });
});

describe('channelLegend', () => {
  const NODES = [
    graphNode({ id: '1', channel_id: 'a', channel_name: 'Alpha' }),
    graphNode({ id: '2', channel_id: 'b', channel_name: 'Beta' }),
    graphNode({ id: '3', channel_id: 'b', channel_name: 'Beta' }),
    graphNode({ id: '4', channel_id: null }),
  ];

  it('orders channels by chunk count', () => {
    expect(channelLegend(NODES).map((item) => item.name)).toEqual([
      'Beta',
      'Alpha',
      'Unknown channel',
    ]);
  });

  it('gives the dominant channel the first palette colour', () => {
    expect(channelLegend(NODES)[0]?.colour).toBe(CHANNEL_COLOURS[0]);
  });

  it('reserves a neutral colour for chunks with no channel', () => {
    const unknown = channelLegend(NODES).find((item) => item.id === '');
    expect(unknown?.colour).toBe(UNKNOWN_CHANNEL_COLOUR);
  });

  it('maps every channel id to a colour', () => {
    const colours = colourMap(NODES);
    expect(colours.get('a')).toBeTruthy();
    expect(colours.get('b')).toBeTruthy();
    expect(colours.get('a')).not.toBe(colours.get('b'));
  });

  it('handles an empty graph', () => {
    expect(channelLegend([])).toEqual([]);
  });
});

describe('chunkTimestampUrl', () => {
  it('appends the start second to a watch URL', () => {
    expect(chunkTimestampUrl('https://www.youtube.com/watch?v=x', 91.7)).toBe(
      'https://www.youtube.com/watch?v=x&t=91s',
    );
  });

  it('starts the query string when the URL has none', () => {
    expect(chunkTimestampUrl('https://youtu.be/x', 5)).toBe('https://youtu.be/x?t=5s');
  });

  it('treats a missing timestamp as the start of the video', () => {
    expect(chunkTimestampUrl('https://youtu.be/x', null)).toBe('https://youtu.be/x?t=0s');
  });

  it('has no link without a source URL', () => {
    expect(chunkTimestampUrl(null, 10)).toBeNull();
  });
});
