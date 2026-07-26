/**
 * Scoped styles for the Prompts tab.
 *
 * Written against the existing theme custom properties, so light and dark both
 * come out right without the component branching on theme — the same contract
 * as src/experiments/styles.ts.
 */

const STYLE_ID = 'tlab-prompts';

const CSS = `
.pr-intro { color: var(--text2); max-width: 74ch; margin: 4px 0 18px; line-height: 1.55; }
.pr-note { color: var(--muted); font-size: 12px; margin: 14px 0 0; }
.pr-note a { color: var(--accent2); }

.pr-group { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  margin-bottom: 14px; overflow: hidden; }
.pr-grouphead { display: flex; align-items: baseline; gap: 10px; width: 100%;
  background: none; border: none; cursor: pointer; text-align: left;
  padding: 12px 16px; color: var(--text); }
.pr-grouphead:hover { background: var(--panel2); }
.pr-grouphead h3 { margin: 0; font-size: 13.5px; }
.pr-groupdesc { color: var(--muted); font-size: 12px; }
.pr-count { margin-left: auto; font: 600 10px var(--mono); color: var(--accent2);
  background: var(--accent-dim); border: 1px solid var(--accent-border);
  border-radius: 9px; padding: 1px 7px; white-space: nowrap; }
.pr-caret { font: 700 11px var(--mono); color: var(--dim); }

.pr-item { border-top: 1px solid var(--border); padding: 12px 16px; }
.pr-itemhead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pr-name { font: 700 12px var(--mono); color: var(--text); }
.pr-role { font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 8px; padding: 1px 6px; }
.pr-role.system { color: var(--accent2); background: var(--accent-dim);
  border-color: var(--accent-border); }
.pr-module { font: 10.5px var(--mono); color: var(--dim); }
.pr-copy { margin-left: auto; background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 6px; color: var(--muted); cursor: pointer; font: 600 10.5px var(--mono);
  padding: 3px 9px; }
.pr-copy:hover { color: var(--text); }
.pr-vars { margin-top: 6px; display: flex; gap: 5px; flex-wrap: wrap; }
.pr-var { font: 600 10px var(--mono); color: var(--warn, #d29922);
  background: var(--panel3); border: 1px solid var(--border2); border-radius: 8px;
  padding: 1px 6px; }
.pr-text { margin: 10px 0 0; background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; overflow-x: auto;
  font: 11.5px/1.55 var(--mono); color: var(--text2); white-space: pre-wrap; }
`;

/** Install the stylesheet once per document, keyed by id (see useEvalStyles). */
export function usePromptStyles(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}
