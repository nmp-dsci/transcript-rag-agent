/**
 * Scoped styles for the eval-breakdown surfaces.
 *
 * theme.css is owned elsewhere, so these live here instead — every rule is
 * written against the existing custom properties so both themes come out right
 * without the components ever branching on the theme.
 */

const STYLE_ID = 'tlab-eval-breakdown';

const CSS = `
/* metric row promoted to a button */
.metricbtn {
  width: 100%; background: none; border: none; font: inherit; color: inherit;
  cursor: pointer; text-align: left; border-radius: 5px; padding: 2px 4px; margin: 2px -4px;
}
.metricbtn:hover { background: var(--hover); }
.metricbtn[aria-expanded='true'] { background: var(--hover); }
.metric .mval.wide { width: 82px; }

/* sample spread, drawn as a whisker across the bar */
.mbar.hasw { position: relative; overflow: visible; }
.mbar .whisk {
  position: absolute; top: -3px; height: 12px; box-sizing: border-box;
  border-left: 1px solid var(--text2); border-right: 1px solid var(--text2);
}
.mbar .whisk::after {
  content: ''; position: absolute; top: 50%; left: 0; right: 0; border-top: 1px solid var(--text2);
}

/* breakdown drawer */
.bd-backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.42); z-index: 40; border: none; padding: 0; }
.bd-drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: min(580px, 100%); z-index: 41;
  background: var(--panel); border-left: 1px solid var(--border);
  display: flex; flex-direction: column; box-shadow: -10px 0 30px rgba(0, 0, 0, 0.28);
}
.bd-head {
  display: flex; align-items: baseline; gap: 10px; flex: 0 0 auto;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
}
.bd-head h2 { margin: 0; font-size: 14px; color: var(--text); }
.bd-close {
  margin-left: auto; background: none; border: 1px solid var(--border2); color: var(--muted);
  border-radius: 6px; padding: 2px 10px; cursor: pointer; font-size: 12px;
}
.bd-close:hover { background: var(--btn-hover); color: var(--text); }
.bd-body { flex: 1; min-height: 0; overflow-y: auto; padding: 14px 16px 30px; }
.bd-body table { min-width: 0; }
.bd-body td { padding: 7px 9px; vertical-align: top; }
.bd-body th { padding: 7px 9px; }
.bd-formula {
  background: var(--bg); border: 1px solid var(--border2); border-radius: 6px;
  padding: 7px 10px; margin: 0 0 12px; font: 11px/1.5 var(--mono); color: var(--text2);
}
.bd-formula b { color: var(--accent2); font-weight: 600; }
.bd-arith {
  margin-top: 14px; background: var(--accent-dim); border: 1px solid var(--accent-border);
  border-radius: 7px; padding: 10px 12px; font: 12px/1.6 var(--mono); color: var(--accent2);
}
.bd-arith .eq { font-size: 14px; font-weight: 700; }
.bd-arith .recon { display: block; margin-top: 5px; font-size: 10.5px; color: var(--muted); }
.bd-sub { font-size: 11.5px; color: var(--muted); margin: 0 0 10px; line-height: 1.55; }

/* claim / verdict cards */
.bd-claim {
  border: 1px solid var(--border); border-left-width: 3px; border-radius: 7px;
  padding: 8px 11px; margin-top: 8px; background: var(--panel3);
}
.bd-claim.ok { border-left-color: var(--good); }
.bd-claim.no { border-left-color: var(--bad); background: var(--bad-dim); }
.bd-claim .v {
  display: flex; align-items: center; gap: 7px; margin-bottom: 4px;
  font: 600 10px var(--mono); letter-spacing: 0.06em; text-transform: uppercase;
}
.bd-claim.ok .v { color: var(--good); }
.bd-claim.no .v { color: var(--bad); }
.bd-claim .txt { font-size: 12.5px; color: var(--text); line-height: 1.5; }
.bd-claim .why { font-size: 11.5px; color: var(--muted); margin-top: 5px; line-height: 1.5; }
.bd-claim .rk { margin-left: auto; color: var(--dim); font-weight: 400; }

.bd-qa { background: var(--panel3); border: 1px solid var(--border); border-radius: 7px; padding: 9px 12px; margin-bottom: 10px; }
.bd-qa .k { font: 600 10px var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--dim); }
.bd-qa .q { font-size: 12.5px; color: var(--text); margin-top: 3px; line-height: 1.5; }

.bd-prev { font-size: 11.5px; color: var(--text2); line-height: 1.5; }
.bd-why { font-size: 11px; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.bd-pk { font: 11px var(--mono); white-space: nowrap; }
.bd-pk.on { color: var(--good); }
.bd-pk.off { color: var(--dim); }

/* metric explainer cards */
.explainers { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
.explainers > * { min-width: 0; }
.explainer { background: var(--panel3); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.explainer h3 { margin: 0 0 5px; font-size: 12.5px; color: var(--text); }
.explainer p { margin: 0; font-size: 11.5px; color: var(--muted); line-height: 1.55; }
.explainer .f {
  display: block; margin: 7px 0; padding: 5px 8px; border-radius: 5px;
  background: var(--bg); border: 1px solid var(--border2);
  font: 10.5px/1.45 var(--mono); color: var(--accent2);
}
.explainer .low { margin-top: 6px; font-size: 11px; color: var(--dim); line-height: 1.5; }

/* honest-uncertainty signals */
.eval-flags { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 7px; }

/* efficiency panel */
.effrow {
  display: grid; grid-template-columns: minmax(110px, 1.3fr) minmax(80px, 1fr) auto;
  gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border);
}
.effrow:last-child { border-bottom: none; }
.effrow > * { min-width: 0; }
.effname { font-size: 12px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.effname .effsub { display: block; font: 9.5px var(--mono); color: var(--dim); white-space: nowrap; }
.effbar { height: 8px; background: var(--panel2); border-radius: 4px; overflow: hidden; }
.effbar i { display: block; height: 100%; background: var(--accent); border-radius: 4px; }
.effbar.best i { background: var(--good); }
.effval { font: 700 12.5px var(--mono); color: var(--text); text-align: right; white-space: nowrap; }
.effval .u { display: block; font-weight: 400; font-size: 9.5px; color: var(--dim); }

/* low-n de-emphasis on the scoreboard */
tr.lown td { opacity: 0.62; }
.nchip { font: 9.5px var(--mono); color: var(--dim); }
`;

/**
 * Install the stylesheet once per document.
 *
 * Called from component render rather than an effect so the styles are present
 * before first paint, and keyed by id so repeated score strips share one tag.
 */
export function useEvalStyles(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}
