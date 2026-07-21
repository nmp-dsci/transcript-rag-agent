/** Light/dark theme selection, persisted per browser. */

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'tlab.theme';

/** The user's stored choice, or null when they have never picked one. */
export function storedTheme(): Theme | null {
  const value = localStorage.getItem(STORAGE_KEY);
  return value === 'light' || value === 'dark' ? value : null;
}

export function systemTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/** An explicit choice wins; otherwise follow the OS setting. */
export function resolveTheme(stored: Theme | null, system: Theme): Theme {
  return stored ?? system;
}

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme);
}

export function setTheme(theme: Theme): void {
  localStorage.setItem(STORAGE_KEY, theme);
  applyTheme(theme);
}

export function initialTheme(): Theme {
  return resolveTheme(storedTheme(), systemTheme());
}
