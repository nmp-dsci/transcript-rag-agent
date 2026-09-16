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
.guides-rail { width: 300px; }
.guides-rail .rentry { display: block; }
.guides-rail .rq { font-weight: 500; color: var(--text); }
.guides-rail .rentry.on .rq { color: var(--accent2); }
.guides-rail .rmeta { flex-wrap: wrap; row-gap: 2px; }
.guides-rail .rail-foot { padding: 10px 12px; border-top: 1px solid var(--border);
  font: var(--t-2) var(--mono); color: var(--dim); line-height: 1.5; }
.guides-rail .rail-foot code { color: var(--muted); }
.guides-rail .gj-entry { border-left: 2px solid var(--accent); }

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
.guide-how { max-width: 64ch; }
.guide-how p { margin: 4px 0 0; font-size: var(--t-1); color: var(--muted); line-height: 1.5; }
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

/* ── ask surface ── */
.guides-rail .gnew { display: block; text-align: center; color: var(--accent2); background: var(--accent-dim);
  border: 1px dashed var(--accent-border); margin-bottom: 6px; font-weight: 500; }
.guides-rail .gnew.on { border-style: solid; }
.ask { max-width: 660px; margin: 40px auto 0; padding: 0 20px 24px; }
.ask-eyebrow { font: var(--t-2) var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent2); }
.ask h2 { font-family: var(--display); font-size: var(--t3); letter-spacing: -0.015em; color: var(--text); margin: 6px 0 6px; line-height: 1.15; }
.ask-lead { color: var(--muted); font-size: var(--t0); line-height: 1.5; margin: 0 0 14px; }
.ask-box { background: var(--panel); border: 1px solid var(--border2); border-radius: 12px; padding: 10px 12px;
  box-shadow: 0 0 0 3px var(--accent-dim); transition: box-shadow var(--dur) var(--ease), border-color var(--dur) var(--ease); }
.ask-box:focus-within { border-color: var(--accent-border); }
.ask-box.rec { border-color: var(--bad-border); box-shadow: 0 0 0 3px var(--bad-dim); }
.ask-box textarea { width: 100%; resize: vertical; font: inherit; font-size: var(--t1); line-height: 1.5; color: var(--text);
  background: none; border: 0; padding: 4px 2px; min-height: 64px; }
.ask-box textarea:focus { outline: none; }
.ask-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; flex-wrap: wrap; }
.ask-row .spacer { flex: 1; }
.ask-hint { font: var(--t-2) var(--mono); color: var(--dim); }
.ask-hint.rec { color: var(--bad); }
.ask-opt { display: block; margin: 10px 2px 0; font-size: var(--t-1); color: var(--text2); }
.ask-opt input { margin-right: 6px; }
.ask .gc-err, .ask .gc-warn { margin: 8px 0 0; }
.ask-suggest { margin-top: 14px; }
.ask-note { margin: 16px 0 0; font: var(--t-2) var(--mono); color: var(--dim); line-height: 1.5; }

/* ── compose panel (screen A) ── */
.gc { border-top: 1px solid var(--border); display: flex; flex-direction: column; min-height: 0; }
.gc .rail-head { border-bottom: 0; }
.gc-form { display: flex; gap: 6px; padding: 0 12px 10px; }
.gc-form input { flex: 1; min-width: 0; }
.gc-warn { margin: 0 12px 8px; font-size: var(--t-1); color: var(--warn); }
.gc-err { margin: 0 12px 8px; font-size: var(--t-1); color: var(--bad); }
.gc-note { margin: 0 12px 8px; font: var(--t-2) var(--mono); color: var(--muted); line-height: 1.5; }
.gc-scope { display: flex; flex-direction: column; min-height: 0; }
.gc-list { list-style: none; margin: 0; padding: 0 6px; max-height: 38vh; overflow-y: auto; }
.gc-row label { display: grid; grid-template-columns: 16px minmax(0, 1fr) 64px; gap: 4px 8px; align-items: center;
  padding: 6px; border-radius: 6px; cursor: pointer; transition: background var(--dur) var(--ease); }
.gc-row label:hover { background: var(--hover); }
.gc-row.on label { background: var(--panel2); }
.gc-row input { margin: 0; grid-row: span 2; }
.gc-title { font-size: var(--t0); color: var(--text); line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden; }
.gc-star { color: var(--accent2); margin-right: 3px; }
.gc-meta { grid-column: 2; display: flex; gap: 8px; font: var(--t-2) var(--mono); color: var(--dim); }
.gc-hits { grid-row: span 2; display: flex; align-items: center; gap: 5px; }
.gc-hitbar { display: block; height: 6px; background: var(--accent); border-radius: 2px; }
.gc-hitn { font: var(--t-2) var(--mono); color: var(--muted); }
.gc-add { display: flex; gap: 6px; padding: 6px 12px; }
.gc-add input { flex: 1; min-width: 0; }
.gc-opt { display: block; padding: 4px 12px; font-size: var(--t0); color: var(--text2); }
.gc-opt input[type=text] { width: 100%; }
.gc-check input { margin-right: 6px; }
.gc-dim { color: var(--dim); font-size: var(--t-1); }
.gc .btn { margin: 4px 12px 12px; }

/* ── research map (screen B) ── */
.gj { flex: 1; min-height: 0; overflow-y: auto; padding: 14px 16px; }
.gj-head { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 12px; }
.gj-head > div { flex: 1 1 320px; min-width: 0; }
.gj-head h2 { margin: 0; font-size: var(--t2); color: var(--text); }
.gj-sub { margin: 2px 0 0; font: var(--t-1) var(--mono); color: var(--muted); }
.gj-eyebrow { margin: 0; font: var(--t-2) var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent2); }
.gj-chosen { margin-top: 6px; font-size: var(--t-1); color: var(--text2); }
.gj-chosen summary { cursor: pointer; color: var(--muted); }
.gj-chosen ul { margin: 6px 0 0; padding-left: 18px; columns: 2; column-gap: 18px; }
@media (max-width: 700px) { .gj-chosen ul { columns: 1; } }
.gj-chosen li { break-inside: avoid; margin: 2px 0; }
.gj-chosen a { color: var(--text); text-decoration: none; }
.gj-chosen a:hover { color: var(--accent2); }
.gj-err { color: var(--bad); }
.gj-ok { color: var(--good); }
.gj-stages { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px; }
.gj-stage { font: var(--t-2) var(--mono); padding: 3px 9px; border-radius: 99px; border: 1px solid var(--border); color: var(--dim); }
.gj-stage.done, .gj-stage.skip { color: var(--good); border-color: var(--good-border); }
.gj-stage.live { color: var(--text); border-color: var(--accent-border); background: var(--accent-dim); }
.gj-stage.error { color: var(--bad); border-color: var(--bad-border); }
.gj-body { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 1fr); gap: 14px; align-items: start; }
@media (max-width: 900px) { .gj-body { grid-template-columns: minmax(0, 1fr); } }
.gj-map { display: block; width: 100%; height: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
.gj-map-empty { padding: 30px; text-align: center; color: var(--dim); font-size: var(--t0); background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
.gj-box { fill: var(--panel3); stroke: var(--border); }
.gj-box.reading { stroke: var(--accent-border); }
.gj-box.done { stroke: var(--good-border); }
.gj-box-title { font: 600 11px var(--sans); fill: var(--text); }
.gj-box-state { font: 10px var(--mono); fill: var(--dim); }
.gj-box-state.reading { fill: var(--warn); }
.gj-box-state.done { fill: var(--good); }
.gj-bar-bg { fill: var(--panel2); }
.gj-bar { fill: var(--accent); }
.gj-bar-label { font: 10px var(--sans); fill: var(--dim); }
.gj-bar-label.read { fill: var(--text2); }
.gj-revise { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.gj-revise-h { color: var(--text); font-size: var(--t1); font-weight: 500; }
.gj-revise-ids { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.gj-revise-ids code { font: var(--t-2) var(--mono); color: var(--accent2); background: var(--accent-dim); border: 1px solid var(--accent-border); border-radius: 4px; padding: 1px 6px; }
.gj-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
@media (max-width: 700px) { .gj-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.gj-kpis .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.gj-kpis .v { font: var(--t2) var(--mono); color: var(--text); }
.gj-kpis .l { font-size: var(--t-1); color: var(--muted); }
.gj-msg { margin: 10px 0 0; font: var(--t-1) var(--mono); color: var(--muted); }
.gj-log { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px;
  font: var(--t-2) var(--mono); line-height: 1.6; color: var(--text2); max-height: 60vh; overflow-y: auto; }
.gj-log-empty { color: var(--dim); }
.gj-log-row { display: grid; grid-template-columns: 58px 74px minmax(0, 1fr); gap: 8px; }
.gj-log-at { color: var(--dim); }
.gj-log-label { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gj-log-msg { overflow-wrap: anywhere; }
.gj-log-row.q .gj-log-msg { color: var(--accent2); }
.gj-log-dim { color: var(--dim); }

/* ── reader + side rail (screens C and D) ── */
.guide-split { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 300px; }
@media (max-width: 900px) { .guide-split { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto; } }
.guide-side { min-height: 0; display: flex; flex-direction: column; border-left: 1px solid var(--border); background: var(--panel3); }
@media (max-width: 900px) { .guide-side { border-left: 0; border-top: 1px solid var(--border); max-height: 42vh; } }
.guide-side-tabs { display: flex; border-bottom: 1px solid var(--border); }
.guide-side-tabs button { flex: 1; background: none; border: 0; padding: 8px 4px; font: var(--t-2) var(--mono);
  letter-spacing: 0.02em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent;
  transition: color var(--dur) var(--ease), border-color var(--dur) var(--ease); }
.guide-side-tabs button.on { color: var(--text); border-bottom-color: var(--accent); }
.guide-links select { font: var(--t-2) var(--mono); color: var(--text2); background: var(--panel2); border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; }

.cr { width: auto; min-width: 0; border-right: 0; flex: 1; min-height: 0; }
.cr-list { padding-bottom: 6px; }
.cr-item { display: block; }
.cr-quote { font-size: var(--t-1); color: var(--muted); font-style: italic; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.cr-body { font-size: var(--t0); color: var(--text); margin-top: 2px; }
.cr-reason { font-size: var(--t-1); color: var(--muted); margin-top: 3px; }
.cr-item code, .cr-receipt code { font: var(--t-2) var(--mono); color: var(--dim); }
.cr-receipt { margin: 8px 10px; padding: 8px 10px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; font-size: var(--t-1); color: var(--text2); }
.cr-receipt summary { cursor: pointer; color: var(--text); font-weight: 500; }
.cr-receipt ul { list-style: none; margin: 6px 0 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.cr-receipt p { margin: 6px 0 0; }
.cr-dim { color: var(--dim); font: var(--t-2) var(--mono); }
.cr-compose { border-top: 1px solid var(--border); padding: 8px 10px; display: flex; flex-direction: column; gap: 6px; }
.cr-compose textarea { width: 100%; resize: vertical; font: inherit; font-size: var(--t0); }
.cr-box { position: relative; }
.cr-box textarea { display: block; padding-right: 36px; }
.cr-box.rec textarea { border-color: var(--bad-border); }
.cr-box .micbtn { position: absolute; right: 6px; bottom: 8px; }
.cr-draft-quote { position: relative; padding: 6px 26px 6px 8px; background: var(--accent-dim); border: 1px solid var(--accent-border); border-radius: 6px; font-size: var(--t-1); color: var(--text2); font-style: italic; }
.cr-clear { position: absolute; top: 2px; right: 4px; background: none; border: 0; color: var(--muted); cursor: pointer; font-size: var(--t1); }
.cr-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.cr-actions .btn { flex: 1; }

/* ── revisions rail ── */
.rv { width: auto; min-width: 0; border-right: 0; flex: 1; min-height: 0; }
.rv-list { padding-bottom: 6px; }
.rv-live { margin: 8px 10px; padding: 8px 10px; background: var(--panel); border: 1px solid var(--accent-border); border-radius: 8px;
  display: flex; flex-direction: column; gap: 8px; }
.rv-live.done { border-color: var(--good-border); }
.rv-live.error { border-color: var(--bad-border); }
.rv-live-h { display: flex; flex-direction: column; gap: 4px; align-items: flex-start; }
.rv-live .gj-revise-ids { margin: 0; }
.rv-live .gj-log { max-height: 34vh; }
/* The rail is narrow and every row is the reviser's, so drop the label column. */
.rv-live .gj-log-row { grid-template-columns: 58px minmax(0, 1fr); }
.rv-live .gj-log-label { display: none; }
.rv-kpis { display: flex; flex-wrap: wrap; gap: 4px 10px; font: var(--t-2) var(--mono); color: var(--muted); }
.rv-msg { margin: 0; font: var(--t-2) var(--mono); color: var(--muted); }
.rv-table { width: calc(100% - 20px); margin: 8px 10px 0; border-collapse: collapse; font-size: var(--t-1); }
.rv-table th { text-align: left; font: var(--t-2) var(--mono); letter-spacing: 0.04em; color: var(--dim); padding: 4px 6px;
  border-bottom: 1px solid var(--border); }
.rv-table td { padding: 5px 6px; color: var(--text2); border-bottom: 1px solid var(--border); vertical-align: top; }
.rv-table tbody tr { cursor: pointer; transition: background var(--dur) var(--ease); }
.rv-table tbody tr:hover { background: var(--hover); }
.rv-table tbody tr.on td { background: var(--panel2); }
.rv-table code { font: var(--t-1) var(--mono); color: var(--text); }
.rv-table td:last-child { display: flex; flex-wrap: wrap; gap: 3px; }
.rv-cur { font: var(--t-2) var(--mono); color: var(--accent2); }

.ev { flex: 1; min-height: 0; overflow-y: auto; padding: 10px; }
.ev-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 0 0 10px; }
.ev-rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
.ev-row { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; width: 100%; text-align: left; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; cursor: pointer; transition: border-color var(--dur) var(--ease); }
.ev-row:hover { border-color: var(--accent-border); }
.ev-id { font: var(--t-1) var(--mono); color: var(--text); }
.ev-src { display: flex; flex-wrap: wrap; gap: 3px; }
.ev-sq { display: inline-block; width: 9px; height: 9px; border-radius: 2px; background: var(--accent); }
.ev-sq.c1 { background: var(--good); } .ev-sq.c2 { background: var(--warn); } .ev-sq.c3 { background: var(--bad); }
.ev-sq.c4 { background: var(--accent2); } .ev-sq.c5 { background: var(--muted); } .ev-sq.c6 { background: var(--good-border); } .ev-sq.c7 { background: var(--warn-border); }
.ev-n { font: var(--t-2) var(--mono); color: var(--muted); }
.ev-bad { color: var(--bad); }
.ev-gaps { margin-top: 10px; font-size: var(--t-1); color: var(--text2); }
.ev-gaps summary { cursor: pointer; color: var(--text); }
.ev-gaps ul { padding-left: 18px; margin: 6px 0 0; }

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
