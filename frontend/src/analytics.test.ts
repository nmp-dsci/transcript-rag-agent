/** The three gates in front of analytics: demo mode, a baked-in key, once.
 *
 * The property that matters most is the negative one — a dev serve (mode
 * "full") must never initialise PostHog no matter what key is present —
 * because polluted visitor numbers are silent and unrecoverable.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import posthog from 'posthog-js';
import { resetAnalyticsForTest, startAnalytics } from './analytics';

vi.mock('posthog-js', () => ({
  default: { init: vi.fn(), capture: vi.fn() },
}));

describe('startAnalytics', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_POSTHOG_KEY', 'phc_test');
    window.location.hash = '';
  });

  afterEach(() => {
    resetAnalyticsForTest();
    vi.unstubAllEnvs();
    vi.clearAllMocks();
  });

  it('never initialises outside demo mode, whatever the key says', () => {
    startAnalytics('full');
    startAnalytics(undefined);
    expect(posthog.init).not.toHaveBeenCalled();
    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it('never initialises without a baked-in key', () => {
    vi.stubEnv('VITE_POSTHOG_KEY', '');
    startAnalytics('demo');
    expect(posthog.init).not.toHaveBeenCalled();
  });

  it('initialises once in demo mode and captures the landing pageview', () => {
    startAnalytics('demo');
    startAnalytics('demo'); // health refreshes re-invoke; init must not
    expect(posthog.init).toHaveBeenCalledTimes(1);
    expect(posthog.capture).toHaveBeenCalledWith('$pageview', { tab: 'chat' });
  });

  it('captures a pageview with the tab name on every hash change', () => {
    startAnalytics('demo');
    window.location.hash = '#board';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(posthog.capture).toHaveBeenLastCalledWith('$pageview', { tab: 'board' });
  });

  it('keeps cookieless, no-autocapture settings', () => {
    startAnalytics('demo');
    const options = vi.mocked(posthog.init).mock.calls[0]?.[1];
    expect(options).toMatchObject({
      persistence: 'localStorage',
      capture_pageview: false,
      autocapture: false,
      disable_session_recording: true,
    });
  });
});
