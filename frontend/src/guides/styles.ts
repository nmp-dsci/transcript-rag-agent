/**
 * Scoped styles for the Field Guides tab.
 *
 * Written entirely against the theme custom properties, so both themes come
 * out right without the component ever branching on the theme — the same
 * contract as src/experiments/styles.ts. The guide *document* is not styled
 * here: it renders inside a sandboxed iframe with its own editorial sheet
 * (guides/guide.css), and only the chrome around it is workbench.
 */

const STYLE_ID = 'tlab-guides';

const CSS = `
.guides { flex-direction: row; }
.guides-rail .rentry { display: block; }
.guides-rail .rq { font-weight: 500; color: var(--text); }
.guides-rail .rentry.on .rq { color: var(--accent2); }
.guides-rail .rmeta { flex-wrap: wrap; row-gap: 2px; }
.guides-rail .rail-foot { padding: 10px 12px; border-top: 1px solid var(--border);
  font: var(--t-2) var(--mono); color: var(--dim); line-height: 1.5; }
.guides-rail .rail-foot code { color: var(--muted); }

.guides-main { flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.guide-head { display: flex; align-items: flex-start; gap: 14px; flex-wrap: wrap;
  padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--panel); }
.guide-head > div { min-width: 0; }
.guide-head .guide-title { flex: 1 1 320px; }
.guide-head h2 { margin: 0; font-size: var(--t2); color: var(--text); line-height: 1.25; }
.guide-head .guide-sub { margin: 2px 0 0; font-size: var(--t-1); color: var(--muted);
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.guide-prov { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.guide-prov .badge { font-weight: 500; }
.guide-links { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.guide-links a, .guide-links button { font: var(--t-2) var(--mono); color: var(--accent2);
  text-decoration: none; padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border);
  background: none; cursor: pointer;
  transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease); }
.guide-links a:hover, .guide-links button:hover { background: var(--accent-dim); border-color: var(--accent-border); }

.guide-toc { display: flex; gap: 2px; flex-wrap: wrap; padding: 6px 12px; border-bottom: 1px solid var(--border);
  background: var(--panel3); }
.guide-toc button { font: var(--t-2) var(--mono); letter-spacing: 0.04em; color: var(--muted);
  background: none; border: 0; padding: 3px 8px; border-radius: 6px; cursor: pointer;
  transition: background var(--dur) var(--ease), color var(--dur) var(--ease); }
.guide-toc button:hover { color: var(--accent2); background: var(--accent-dim); }

.guide-frame { flex: 1; min-height: 0; position: relative; background: var(--bg); }
.guide-frame iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0;
  background: transparent; }
.guide-empty { padding: 40px 24px; color: var(--muted); max-width: 64ch; margin: 0 auto; }
.guide-empty h2 { color: var(--text); font-size: var(--t2); margin: 0 0 8px; }
.guide-empty code { font: var(--t-1) var(--mono); color: var(--text2); background: var(--panel2);
  padding: 1px 5px; border-radius: 4px; }
.guide-error { padding: 12px 16px; color: var(--bad); font-size: var(--t0); }

@media (max-width: 760px) {
  .guides { flex-direction: column; }
  .guides-rail { width: auto; min-width: 0; max-height: 34vh; border-right: 0; border-bottom: 1px solid var(--border); }
}
`;

export function useGuidesStyles(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}
