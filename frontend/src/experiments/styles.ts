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

.mrun { margin-bottom: 18px; }
.mrun-head { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.mrun-head > div { flex: 1; min-width: 260px; }
.mrun-head > button { flex: 0 0 auto; }
.mrun-job { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
.mrun-status { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.mrun-setups { font: 11px var(--mono); color: var(--muted); }
.mrun-bar { width: 100%; height: 8px; margin-top: 10px; }
.mrun-message { margin: 8px 0 0; }

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

/* ── held-out critique eval ──────────────────────────────────────────────
   Everything below has to escape .exp-table's white-space: nowrap. This tab
   has clipped long text before — 66 of 273 characters — and a criterion is a
   whole sentence, so every block that carries prose re-declares its own
   wrapping rather than inheriting the table's. */
.exp-tag.ok { color: var(--good); background: var(--good-dim); border-color: var(--good-border); }
.exp-tag.bad { color: var(--bad); background: var(--bad-dim); border-color: var(--bad-border); }
.exp-table td.num.bad { color: var(--bad); font-weight: 700; }
.crit-intro { color: var(--text2); font-size: 12.5px; line-height: 1.55; max-width: 82ch;
  margin: 0 0 12px; white-space: normal; }
.crit-empty { color: var(--muted); font-size: 12px; margin: 8px 0 0; white-space: normal; }
.crit-detailrow > td { white-space: normal; padding: 0 10px 14px; }
.crit-detail { padding-top: 4px; white-space: normal; }
.crit-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px; align-items: start; }
.crit-list { list-style: none; margin: 8px 0 0; padding: 0; display: flex;
  flex-direction: column; gap: 8px; }
.crit-item { border-left: 2px solid var(--border2); padding: 4px 0 4px 10px;
  white-space: normal; }
.crit-item.hit { border-left-color: var(--good); }
.crit-item.miss { border-left-color: var(--border2); }
.crit-rule { color: var(--text); font-size: 12.5px; line-height: 1.5; }
.crit-rule b { font: 700 10.5px var(--mono); color: var(--muted); margin-right: 6px; }
.crit-na { margin-left: 6px; font: 600 9px var(--mono); letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--muted); background: var(--panel2);
  border: 1px solid var(--border2); border-radius: 8px; padding: 1px 5px;
  white-space: nowrap; }
.crit-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin-top: 3px; }
.crit-ts { font: 600 10.5px var(--mono); color: var(--accent2); background: var(--accent-dim);
  border: 1px solid var(--accent-border); border-radius: 8px; padding: 1px 6px;
  text-decoration: none; white-space: nowrap; }
.crit-ts.bad { color: var(--bad); background: var(--bad-dim); border-color: var(--bad-border); }
.crit-quote { color: var(--muted); font-size: 11.5px; font-style: italic; line-height: 1.45; }
.crit-found { color: var(--text2); font-size: 12px; line-height: 1.45; margin-top: 4px; }
.crit-why { color: var(--muted); }
.crit-spread { display: block; font: 500 9.5px var(--mono); color: var(--muted);
  letter-spacing: 0; }
.crit-agree { margin-left: 6px; font: 600 9px var(--mono); letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--muted); background: var(--panel2);
  border: 1px solid var(--border2); border-radius: 8px; padding: 1px 5px;
  white-space: nowrap; }
.crit-extra { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border); }
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
