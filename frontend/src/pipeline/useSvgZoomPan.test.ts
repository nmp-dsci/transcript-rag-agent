import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useSvgZoomPan } from './useSvgZoomPan';

/** A fake <svg> whose CTM is a pure uniform scale with no translation, so
 * svgPoint(client) === client / scale — enough to exercise the hook's math
 * without a real browser layout engine (jsdom implements neither
 * getScreenCTM nor createSVGPoint). */
function fakeSvg(scale = 1): SVGSVGElement {
  const inverse = { a: 1 / scale, b: 0, c: 0, d: 1 / scale, e: 0, f: 0 };
  const ctm = { a: scale, b: 0, c: 0, d: scale, e: 0, f: 0, inverse: () => inverse };
  const svg = {
    getScreenCTM: () => ctm,
    createSVGPoint: () => {
      let px = 0;
      let py = 0;
      return {
        get x() {
          return px;
        },
        set x(value: number) {
          px = value;
        },
        get y() {
          return py;
        },
        set y(value: number) {
          py = value;
        },
        matrixTransform: (m: typeof inverse) => ({ x: px * m.a + m.e, y: py * m.d + m.f }),
      };
    },
    getBoundingClientRect: () => ({
      left: 0,
      top: 0,
      right: 1000,
      bottom: 620,
      width: 1000,
      height: 620,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }),
  };
  return svg as unknown as SVGSVGElement;
}

function pointerEvent(overrides: Record<string, unknown> = {}) {
  return {
    button: 0,
    clientX: 0,
    clientY: 0,
    pointerId: 1,
    currentTarget: { setPointerCapture: vi.fn(), releasePointerCapture: vi.fn() },
    ...overrides,
  } as unknown as React.PointerEvent<SVGSVGElement>;
}

function setup(scale = 1) {
  const svg = fakeSvg(scale);
  const ref = { current: svg };
  return renderHook(() => useSvgZoomPan(ref));
}

describe('useSvgZoomPan', () => {
  it('starts untransformed', () => {
    const { result } = setup();
    expect(result.current.transform).toEqual({ x: 0, y: 0, k: 1 });
    expect(result.current.isZoomed).toBe(false);
  });

  it('zoomIn scales up around the viewport centre and keeps that point fixed', () => {
    const { result } = setup();
    act(() => result.current.zoomIn());
    // rect centre is (500, 310); zooming there keeps it visually stationary,
    // which pins x/y to centre - k*centre for the new k.
    expect(result.current.transform.k).toBeCloseTo(1.4);
    expect(result.current.transform.x).toBeCloseTo(500 - 1.4 * 500);
    expect(result.current.transform.y).toBeCloseTo(310 - 1.4 * 310);
  });

  it('zoomOut shrinks and reset returns to identity', () => {
    const { result } = setup();
    act(() => result.current.zoomIn());
    act(() => result.current.zoomOut());
    expect(result.current.transform.k).toBeCloseTo(1);
    act(() => result.current.reset());
    expect(result.current.transform).toEqual({ x: 0, y: 0, k: 1 });
    expect(result.current.isZoomed).toBe(false);
  });

  it('clamps scale so zooming in forever does not grow without bound', () => {
    const { result } = setup();
    for (let i = 0; i < 20; i += 1) act(() => result.current.zoomIn());
    expect(result.current.transform.k).toBeLessThanOrEqual(8);
  });

  it('clamps scale so zooming out forever does not shrink to zero', () => {
    const { result } = setup();
    for (let i = 0; i < 20; i += 1) act(() => result.current.zoomOut());
    expect(result.current.transform.k).toBeGreaterThanOrEqual(0.4);
  });

  it('wheel zooms toward the cursor and prevents the page from scrolling', () => {
    const { result } = setup();
    const preventDefault = vi.fn();
    act(() =>
      result.current.svgProps.onWheel({
        clientX: 200,
        clientY: 100,
        deltaY: -100,
        preventDefault,
      } as unknown as React.WheelEvent<SVGSVGElement>),
    );
    expect(preventDefault).toHaveBeenCalled();
    // Negative deltaY (scroll up / pinch out) zooms in.
    expect(result.current.transform.k).toBeGreaterThan(1);
  });

  it('drags to pan by the screen-pixel delta', () => {
    const { result } = setup();
    act(() => result.current.svgProps.onPointerDown(pointerEvent({ clientX: 100, clientY: 100 })));
    act(() => result.current.svgProps.onPointerMove(pointerEvent({ clientX: 130, clientY: 115 })));
    expect(result.current.transform.x).toBeCloseTo(30);
    expect(result.current.transform.y).toBeCloseTo(15);
  });

  it('divides the pan delta by the CTM scale so drag tracks the cursor 1:1 on screen', () => {
    const { result } = setup(2);
    act(() => result.current.svgProps.onPointerDown(pointerEvent({ clientX: 0, clientY: 0 })));
    act(() => result.current.svgProps.onPointerMove(pointerEvent({ clientX: 20, clientY: 0 })));
    expect(result.current.transform.x).toBeCloseTo(10);
  });

  it('suppresses the click that ends a real drag, but not a plain click', () => {
    const { result } = setup();

    // A drag past the threshold...
    act(() => result.current.svgProps.onPointerDown(pointerEvent({ clientX: 0, clientY: 0 })));
    act(() => result.current.svgProps.onPointerMove(pointerEvent({ clientX: 50, clientY: 0 })));
    act(() => result.current.svgProps.onPointerUp(pointerEvent()));
    const dragStop = vi.fn();
    act(() =>
      result.current.svgProps.onClickCapture({
        stopPropagation: dragStop,
      } as unknown as React.MouseEvent<SVGSVGElement>),
    );
    expect(dragStop).toHaveBeenCalled();

    // ...but a down/up with no meaningful movement is a click, not a drag.
    act(() => result.current.svgProps.onPointerDown(pointerEvent({ clientX: 0, clientY: 0 })));
    act(() => result.current.svgProps.onPointerMove(pointerEvent({ clientX: 1, clientY: 0 })));
    act(() => result.current.svgProps.onPointerUp(pointerEvent()));
    const clickStop = vi.fn();
    act(() =>
      result.current.svgProps.onClickCapture({
        stopPropagation: clickStop,
      } as unknown as React.MouseEvent<SVGSVGElement>),
    );
    expect(clickStop).not.toHaveBeenCalled();
  });

  it('clears the stale drag flag on a timer when a drag ends without a click reaching the svg (e.g. released over the zoom-control buttons)', () => {
    vi.useFakeTimers();
    const { result } = setup();

    act(() => result.current.svgProps.onPointerDown(pointerEvent({ clientX: 0, clientY: 0 })));
    act(() => result.current.svgProps.onPointerMove(pointerEvent({ clientX: 50, clientY: 0 })));
    // Drag ends outside the svg subtree (e.g. over a zoom button) -
    // onPointerLeave/onPointerUp fires but no click ever reaches onClickCapture.
    act(() => result.current.svgProps.onPointerUp(pointerEvent()));

    // Before the timer fires, a genuine unrelated click on the svg must still
    // not be swallowed once the timer has had a chance to clear the flag.
    act(() => vi.runAllTimers());

    const laterClickStop = vi.fn();
    act(() =>
      result.current.svgProps.onClickCapture({
        stopPropagation: laterClickStop,
      } as unknown as React.MouseEvent<SVGSVGElement>),
    );
    expect(laterClickStop).not.toHaveBeenCalled();

    vi.useRealTimers();
  });

  it('double-click zooms in at the click point', () => {
    const { result } = setup();
    act(() =>
      result.current.svgProps.onDoubleClick({
        clientX: 400,
        clientY: 300,
      } as unknown as React.MouseEvent<SVGSVGElement>),
    );
    expect(result.current.transform.k).toBeCloseTo(1.4);
  });
});
