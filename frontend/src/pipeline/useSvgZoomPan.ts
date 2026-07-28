/**
 * Wheel-to-zoom (toward the cursor), drag-to-pan, and +/-/reset controls for
 * an SVG diagram.
 *
 * Applies as a `transform` on a <g> wrapping the diagram content; the outer
 * <svg viewBox> is never touched, so the responsive scaling it already
 * provides is untouched by panning or zooming.
 */

import { useCallback, useRef, useState } from 'react';

export interface ZoomPanTransform {
  x: number;
  y: number;
  k: number;
}

const MIN_SCALE = 0.4;
const MAX_SCALE = 8;
const WHEEL_SENSITIVITY = 0.0015;
const BUTTON_STEP = 1.4;
/** Screen-pixel drag distance below which a pointer-up still counts as a
 * click on whatever's underneath, rather than a pan. */
const DRAG_THRESHOLD = 3;

function clampScale(k: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, k));
}

/** Pointer position in the <svg>'s own viewBox coordinate space — independent
 * of CSS/responsive scaling and of the content group's own transform.
 *
 * `getScreenCTM`/`createSVGPoint` are unimplemented in jsdom (it throws
 * rather than returning null), so a test environment falls back to the raw
 * client coordinates instead of crashing the interaction.
 */
function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): { x: number; y: number } {
  try {
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: clientX, y: clientY };
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const transformed = point.matrixTransform(ctm.inverse());
    return { x: transformed.x, y: transformed.y };
  } catch {
    return { x: clientX, y: clientY };
  }
}

export function useSvgZoomPan(svgRef: React.RefObject<SVGSVGElement | null>) {
  const [transform, setTransform] = useState<ZoomPanTransform>({ x: 0, y: 0, k: 1 });
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);
  const dragged = useRef(false);

  const zoomAt = useCallback(
    (clientX: number, clientY: number, factor: number) => {
      const svg = svgRef.current;
      if (!svg) return;
      const pointer = svgPoint(svg, clientX, clientY);
      setTransform((current) => {
        const nextK = clampScale(current.k * factor);
        if (nextK === current.k) return current;
        // Keep the world point under the pointer fixed on screen while k changes.
        const worldX = (pointer.x - current.x) / current.k;
        const worldY = (pointer.y - current.y) / current.k;
        return { k: nextK, x: pointer.x - nextK * worldX, y: pointer.y - nextK * worldY };
      });
    },
    [svgRef],
  );

  const zoomAtCenter = useCallback(
    (factor: number) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, factor);
    },
    [svgRef, zoomAt],
  );

  const onWheel = useCallback(
    (event: React.WheelEvent<SVGSVGElement>) => {
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, Math.exp(-event.deltaY * WHEEL_SENSITIVITY));
    },
    [zoomAt],
  );

  const onPointerDown = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    dragOrigin.current = { x: event.clientX, y: event.clientY };
    dragged.current = false;
    // Not implemented in every environment (jsdom included) — pan still
    // works without capture, it just won't survive the pointer leaving the
    // element mid-drag.
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }, []);

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!dragOrigin.current) return;
      const svg = svgRef.current;
      if (!svg) return;
      const dx = event.clientX - dragOrigin.current.x;
      const dy = event.clientY - dragOrigin.current.y;
      if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) dragged.current = true;
      // A pure delta needs only the CTM's uniform scale, not its translation.
      let scale = 1;
      try {
        scale = svg.getScreenCTM()?.a || 1;
      } catch {
        // Unimplemented in some test environments (jsdom) — fall back to 1:1.
      }
      dragOrigin.current = { x: event.clientX, y: event.clientY };
      setTransform((current) => ({ ...current, x: current.x + dx / scale, y: current.y + dy / scale }));
    },
    [svgRef],
  );

  const endDrag = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    dragOrigin.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  }, []);

  // A capture-phase listener runs before the event reaches the node it was
  // dispatched on, so stopping it here suppresses the node's own onClick
  // instead of racing it — the drag that just ended must not also select
  // whatever was under the pointer when it lifted.
  const onClickCapture = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (dragged.current) {
      event.stopPropagation();
      dragged.current = false;
    }
  }, []);

  const onDoubleClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => zoomAt(event.clientX, event.clientY, BUTTON_STEP),
    [zoomAt],
  );

  const zoomIn = useCallback(() => zoomAtCenter(BUTTON_STEP), [zoomAtCenter]);
  const zoomOut = useCallback(() => zoomAtCenter(1 / BUTTON_STEP), [zoomAtCenter]);
  const reset = useCallback(() => setTransform({ x: 0, y: 0, k: 1 }), []);

  return {
    transform,
    svgProps: {
      onWheel,
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerLeave: endDrag,
      onClickCapture,
      onDoubleClick,
    },
    zoomIn,
    zoomOut,
    reset,
    isZoomed: transform.k !== 1 || transform.x !== 0 || transform.y !== 0,
  };
}
