/** Engagement analytics for the public demo — and only the demo.
 *
 * Three gates, all of which must open before a single event leaves the
 * browser: the server must report `mode: "demo"` (so dev browsing never
 * pollutes visitor numbers), a PostHog key must have been baked in at build
 * time (`VITE_POSTHOG_KEY`), and init must not have already run.
 *
 * The app's tabs are hash routes, so a server log sees one URL for a whole
 * visit; pageviews are captured client-side on load and on every
 * `hashchange`, with the tab name as the path. Autocapture stays off — the
 * questions being answered are "who landed, what did they explore, did they
 * come back", not per-click telemetry.
 */

import posthog from 'posthog-js';

let started = false;

function tabFromHash(): string {
  return window.location.hash.replace('#', '') || 'chat';
}

function pageview(): void {
  posthog.capture('$pageview', { tab: tabFromHash() });
}

export function startAnalytics(mode: string | undefined): void {
  const key = import.meta.env.VITE_POSTHOG_KEY as string | undefined;
  if (started || mode !== 'demo' || !key) return;
  started = true;

  posthog.init(key, {
    api_host: (import.meta.env.VITE_POSTHOG_HOST as string | undefined) || 'https://us.i.posthog.com',
    // localStorage identity, no cookies: enough for retention cohorts on a
    // portfolio demo without consent-banner territory.
    persistence: 'localStorage',
    capture_pageview: false, // hash routes — captured manually below
    autocapture: false,
    disable_session_recording: true,
  });

  pageview();
  window.addEventListener('hashchange', pageview);
}

/** Test seam: reset the once-only latch. */
export function resetAnalyticsForTest(): void {
  started = false;
  window.removeEventListener('hashchange', pageview);
}
