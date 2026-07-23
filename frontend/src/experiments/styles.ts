/**
 * Scoped styles for the Experiments tab.
 *
 * Written entirely against the existing theme custom properties, so both light
 * and dark come out right without the component ever branching on the theme —
 * the same contract as src/eval/styles.ts.
 */

const STYLE_ID = 'tlab-experiments';

const CSS = `
.exp-intro { color: var(--text2); max-width: 74ch; margin: 4px 0 18px; line-height: 1.55; }
.exp-empty { color: var(--muted); background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; }

.exp-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; margin-bottom: 16px; }
.exp-cardhead { display: flex; align-items: flex-start; gap: 12px; flex-wrap: wrap;
  margin-bottom: 12px; }
.exp-cardhead h3 { margin: 0; font-size: 14px; color: var(--text); }
.exp-sub { display: block; margin-top: 2px; font: 11px var(--mono); color: var(--muted); }

.exp-seg { margin-left: auto; display: inline-flex; border: 1px solid var(--border2);
  border-radius: 7px; overflow: hidden; }
.exp-seg button { background: var(--panel2); border: none; color: var(--muted); cursor: pointer;
  font: 600 11px var(--mono); padding: 4px 10px; border-left: 1px solid var(--border2); }
.exp-seg button:first-child { border-left: none; }
.exp-seg button.on { background: var(--accent-dim); color: var(--accent2); }

.exp-scroll { overflow-x: auto; }
.exp-table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.exp-table th { text-align: left; font: 600 10px var(--mono); letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--dim); padding: 7px 10px; border-bottom: 1px solid var(--border);
  white-space: nowrap; }
.exp-table th.num, .exp-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.exp-table td { padding: 7px 10px; border-bottom: 1px solid var(--border); color: var(--text2);
  white-space: nowrap; }
.exp-table tbody tr:last-child td { border-bottom: none; }
.exp-table td.num { font-family: var(--mono); }
.exp-table td.num.best { color: var(--good); font-weight: 700; }
.exp-cfg { color: var(--text); font-weight: 600; }
.exp-basetag { margin-left: 6px; font: 600 9px var(--mono); letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--muted); background: var(--panel2);
  border: 1px solid var(--border2); border-radius: 8px; padding: 1px 5px; }

.exp-deltas { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.exp-deltarow { display: flex; align-items: baseline; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.exp-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.exp-delta { font: 600 10px var(--mono); border-radius: 8px; padding: 2px 6px;
  border: 1px solid var(--border2); color: var(--muted); white-space: nowrap; }
.exp-delta.pos { color: var(--good); background: var(--good-dim); border-color: var(--good-border); }
.exp-delta.neg { color: var(--bad); background: var(--bad-dim); border-color: var(--bad-border); }

.exp-goldlist { display: flex; flex-direction: column; gap: 10px; }
.exp-gold { background: var(--panel3); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; }
.exp-goldhead { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.exp-goldhead b { color: var(--text); font-size: 13px; }
.exp-tags { display: flex; gap: 5px; flex-wrap: wrap; }
.exp-tag { font: 600 9.5px var(--mono); letter-spacing: 0.04em; color: var(--accent2);
  background: var(--accent-dim); border: 1px solid var(--accent-border); border-radius: 8px;
  padding: 1px 6px; }
.exp-goldmetrics { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 8px; }
.exp-gm { display: flex; flex-direction: column; gap: 1px; }
.exp-gm b { font: 700 13px var(--mono); color: var(--text); font-variant-numeric: tabular-nums; }
`;

/** Install the stylesheet once per document, keyed by id (see useEvalStyles). */
export function useExperimentStyles(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}
