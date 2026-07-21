import { beforeEach, describe, expect, it } from 'vitest';

import { applyTheme, resolveTheme, setTheme, storedTheme } from './theme';

describe('resolveTheme', () => {
  it('prefers an explicit choice over the system setting', () => {
    expect(resolveTheme('light', 'dark')).toBe('light');
    expect(resolveTheme('dark', 'light')).toBe('dark');
  });

  it('falls back to the system setting when nothing is stored', () => {
    expect(resolveTheme(null, 'light')).toBe('light');
    expect(resolveTheme(null, 'dark')).toBe('dark');
  });
});

describe('storedTheme', () => {
  beforeEach(() => localStorage.clear());

  it('returns null when the user has never chosen', () => {
    expect(storedTheme()).toBeNull();
  });

  it('ignores a corrupted stored value', () => {
    localStorage.setItem('tlab.theme', 'chartreuse');
    expect(storedTheme()).toBeNull();
  });

  it('reads back both valid values', () => {
    setTheme('light');
    expect(storedTheme()).toBe('light');
    setTheme('dark');
    expect(storedTheme()).toBe('dark');
  });
});

describe('setTheme', () => {
  beforeEach(() => localStorage.clear());

  it('persists the choice and applies it to the document', () => {
    setTheme('light');
    expect(storedTheme()).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });
});

describe('applyTheme', () => {
  beforeEach(() => localStorage.clear());

  it('stamps the theme onto the root element without persisting it', () => {
    applyTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(storedTheme()).toBeNull();
  });
});
