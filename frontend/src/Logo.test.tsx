import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Logo } from './Logo';

describe('Logo', () => {
  it('is decorative — the wordmark beside it already names the app', () => {
    const { container } = render(<Logo />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('aria-hidden')).toBe('true');
  });

  it('renders at the requested size', () => {
    const { container } = render(<Logo size={32} />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('width')).toBe('32');
    // The viewBox is fixed, so any size scales the same geometry.
    expect(svg?.getAttribute('viewBox')).toBe('0 0 24 24');
  });

  it('marks exactly one span as the retrieved chunk', () => {
    const { container } = render(<Logo />);
    const accent = container.querySelectorAll('rect[fill="var(--accent2)"]');
    expect(accent).toHaveLength(1);
  });
});
