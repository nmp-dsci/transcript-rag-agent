/**
 * The transcript·lab mark: four transcript spans on a timeline, one retrieved.
 *
 * The bars are deliberately not zero-anchored like a bar chart. A transcript
 * chunk has a start and an end, so each span begins at a different offset and
 * they overlap — which is literally how `build_chunks` cuts a transcript, with
 * `overlap_chars` shared between neighbours. The accent span is the one
 * retrieval picked out, which is the whole job of the product.
 *
 * Neutral bars use `currentColor` so the mark inherits whatever the surrounding
 * text is, and stays correct in both themes without branching.
 */
export function Logo({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      // The wordmark beside it already names the app, so the mark is decorative.
      aria-hidden="true"
      focusable="false"
    >
      <rect x="4" y="3" width="9" height="3" rx="1.5" fill="currentColor" opacity="0.55" />
      <rect x="3" y="8" width="15" height="3" rx="1.5" fill="currentColor" opacity="0.55" />
      <rect x="6" y="13" width="8" height="3" rx="1.5" fill="var(--accent2)" />
      <rect x="7" y="18" width="14" height="3" rx="1.5" fill="currentColor" opacity="0.55" />
    </svg>
  );
}
