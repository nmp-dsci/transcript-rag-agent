/**
 * Styles owned by the RAG Pipeline view.
 *
 * Injected by PipelineView rather than added to theme.css, which this module
 * does not own. Everything here is built from the same tokens as theme.css, so
 * both themes are covered and the block can be lifted into theme.css verbatim.
 */

export const PIPELINE_STYLES = `
/* ── Corpus summary strip ── */
.pipe-head {
  flex: 0 0 auto; background: var(--panel); border-bottom: 1px solid var(--border);
  padding: 9px 16px;
}
.pipe-stats { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; }
.pipe-stat { display: flex; flex-direction: column; line-height: 1.25; min-width: 0; }
.pipe-stat b { font: 600 15px var(--mono); color: var(--text); }
.pipe-stat span { font-size: 10.5px; color: var(--muted); }
.pipe-stat.wide b {
  font-size: 11.5px; color: var(--text2); max-width: 190px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.pipe-insights { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px; }
.pipe-chip {
  font-size: 10.5px; padding: 2px 8px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 6px;
}
.pipe-chip:hover { filter: brightness(1.18); }
.pipe-chip.on { box-shadow: 0 0 0 1px var(--accent); }
.pipe-chip-go { opacity: 0.7; }
.pipe-clear { margin-left: 4px; }

/* ── Staged indexing panel ── */
.pipe-index {
  flex: 0 0 auto; background: var(--panel3); border-bottom: 1px solid var(--border);
  padding: 10px 16px; max-height: 52%; overflow-y: auto;
}
.pipe-index-body { max-width: 1100px; }
.idx-stages { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; margin: 10px 0 0; padding: 0; }
.idx-stage {
  display: flex; align-items: baseline; gap: 7px; flex: 1 1 148px; min-width: 148px;
  padding: 5px 10px; border: 1px solid var(--border); border-radius: 7px; background: var(--panel);
}
.idx-stage .idx-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--border2); flex: 0 0 auto; }
.idx-stage .idx-name { font: 600 11px var(--mono); color: var(--dim); }
.idx-stage .idx-hint { font-size: 10px; color: var(--dim); margin-left: auto; }
.idx-stage.done { border-color: var(--good-border); background: var(--good-dim); }
.idx-stage.done .idx-dot { background: var(--good); }
.idx-stage.done .idx-name { color: var(--good); }
.idx-stage.active { border-color: var(--accent-border); background: var(--accent-dim); }
.idx-stage.active .idx-dot { background: var(--accent); animation: pulse 1.1s infinite; }
.idx-stage.active .idx-name { color: var(--accent2); }
.idx-log {
  margin-top: 8px; max-height: 110px; overflow-y: auto; background: var(--bg);
  border: 1px solid var(--border2); border-radius: 7px; padding: 7px 10px;
  font: 10.5px var(--mono); color: var(--muted);
}
.idx-log div { padding: 1px 0; white-space: pre-wrap; word-break: break-word; }
.idx-result {
  margin-top: 10px; background: var(--panel); border: 1px solid var(--accent-border);
  border-radius: 8px; padding: 10px 12px;
}
.idx-result-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12.5px; color: var(--text); }
.idx-added { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }

/* ── Ingestion queue: one row per queued/running/done/error job ── */
.result.acc { color: var(--accent2); }
.idxq-list { list-style: none; margin: 12px 0 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
.idxq-row {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 12px;
}
.idxq-row.running { border-color: var(--accent-border); }
.idxq-row.error { border-color: var(--bad-border); }
.idxq-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.idxq-target { font: 12px var(--mono); color: var(--text); word-break: break-all; }
.idxq-message { margin: 4px 0 0; }

/* ── Sub-tab panes ── */
.pipe-pane { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.pipe-pane[hidden] { display: none; }

/* ── Chunk graph ── */
.graph { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.graph-controls {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px; flex: 0 0 auto;
  padding: 10px 16px; background: var(--panel3); border-bottom: 1px solid var(--border);
}
.graph-controls input[type='search'] { flex: 1; min-width: 200px; padding: 6px 10px; font-size: 12px; }
.graph-controls input[type='range'] { accent-color: var(--accent); width: 92px; }
.graph-num { font: 10.5px var(--mono); color: var(--text2); width: 26px; }
.graph-body { flex: 1; min-height: 0; display: flex; }
.graph-canvas {
  flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column;
  overflow: hidden; padding: 6px;
}
.graph-svg { flex: 1; min-height: 0; width: 100%; display: block; }
.graph-node { cursor: pointer; }
.graph-focus { pointer-events: none; }
.graph-note { margin: 6px 2px; font-size: 11.5px; color: var(--dim); line-height: 1.5; }
.graph-side {
  width: 300px; min-width: 240px; flex: 0 0 auto; overflow-y: auto;
  border-left: 1px solid var(--border); background: var(--panel3); padding: 10px 12px;
}
.graph-block { margin-bottom: 14px; }
.graph-legend { display: flex; align-items: center; gap: 8px; margin-top: 5px; font-size: 11.5px; color: var(--text2); }
.graph-legend .sw { width: 9px; height: 9px; border-radius: 3px; flex: 0 0 auto; }
.graph-legend .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.graph-legend .ct { margin-left: auto; font: 9.5px var(--mono); color: var(--dim); }
.graph-hit {
  display: flex; align-items: center; gap: 7px; width: 100%; margin-top: 4px; padding: 4px 7px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--panel);
  font: 10.5px var(--mono); color: var(--muted); cursor: pointer; text-align: left;
}
.graph-hit:hover { background: var(--hover); }
.graph-hit.on { border-color: var(--accent-border); background: var(--accent-dim); }
.graph-hit .rk { color: var(--dim); flex: 0 0 auto; }
.graph-hit .tx {
  flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: 'Segoe UI', Helvetica, sans-serif;
}
.graph-hit .sc { color: var(--accent2); flex: 0 0 auto; }

/* ── Thumbnails ── */
.tree .thumb { width: 32px; height: 18px; border-radius: 3px; object-fit: cover; flex: 0 0 auto; background: var(--panel2); }
.vthumb { width: 120px; height: 68px; border-radius: 6px; object-fit: cover; background: var(--panel2); flex: 0 0 auto; }

@media (max-width: 1000px) {
  .graph-body { flex-direction: column; }
  .graph-side { width: auto; min-width: 0; max-height: 40%; border-left: none; border-top: 1px solid var(--border); }
}

/* ── Themes (RAPTOR level 2) ──
   Everything that can hold model-written text wraps. Theme titles and summaries
   are LLM output of unbounded length, and this app has repeatedly shipped a
   nowrap field that silently truncated one to a third of itself. The only
   ellipsis here is on the collapsed chunk preview, which is a deliberate
   fixed-length excerpt, not a title. */
.th-toplevel { color: var(--muted); background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; margin: 16px; line-height: 1.6; }
.th-toplevel code { font-size: 11.5px; word-break: break-all; }
.th-layout { flex: 1; min-height: 0; display: grid;
  grid-template-columns: minmax(280px, 380px) minmax(0, 1fr); gap: 12px;
  padding: 12px 16px; box-sizing: border-box; overflow: hidden; }
@media (max-width: 900px) { .th-layout { grid-template-columns: 1fr; overflow-y: auto; } }

.th-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 6px;
  padding-right: 4px; }
.th-listhead { padding: 2px 2px 6px; }
.th-blurb { margin: 6px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.55; }
.th-statline { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }

.th-row { display: flex; flex-direction: column; align-items: stretch; gap: 6px; width: 100%;
  text-align: left; padding: 9px 11px; border: 1px solid var(--border); border-radius: 9px;
  background: var(--panel); color: var(--text2); cursor: pointer; }
.th-row:hover { background: var(--hover); }
.th-row.on { border-color: var(--accent-border); background: var(--accent-dim); }
.th-row-title { font-size: 12.5px; font-weight: 600; color: var(--text); line-height: 1.45;
  overflow-wrap: anywhere; }
.th-row-meta { display: flex; flex-wrap: wrap; gap: 5px; }

.th-tag { font: 600 9.5px var(--mono); letter-spacing: 0.03em; border-radius: 8px;
  padding: 2px 7px; border: 1px solid var(--border2); color: var(--muted);
  background: var(--panel2); white-space: normal; overflow-wrap: anywhere; }
.th-tag.cross { color: var(--good); border-color: var(--good-border); background: var(--good-dim); }
.th-tag.single { color: var(--dim); }
.th-tag.warn { color: var(--bad); border-color: var(--bad-border); background: var(--bad-dim); }

.th-detail { min-height: 0; overflow-y: auto; background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; }
.th-empty { color: var(--muted); font-size: 12.5px; }
.th-title { margin: 0; font-size: 15px; color: var(--text); line-height: 1.4; overflow-wrap: anywhere; }
.th-summary { margin: 8px 0 0; font-size: 12.5px; color: var(--text2); line-height: 1.6;
  overflow-wrap: anywhere; }
.th-note { margin: 10px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.55;
  background: var(--panel3); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }

.th-group { margin-top: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
.th-grouphead { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
.th-groupname { font-size: 12.5px; font-weight: 600; color: var(--text); line-height: 1.4;
  overflow-wrap: anywhere; }
.th-groupmeta { font: 10.5px var(--mono); color: var(--accent2); overflow-wrap: anywhere; }

.th-chunk { border: 1px solid var(--border); border-radius: 8px; background: var(--panel3);
  margin-top: 5px; }
.th-chunk.on { border-color: var(--accent-border); }
.th-chunkhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; width: 100%;
  text-align: left; padding: 6px 9px; background: none; border: none; cursor: pointer; }
.th-chunkhead:hover { background: var(--hover); border-radius: 8px; }
.th-chunkid { font: 600 10px var(--mono); color: var(--dim); flex: 0 0 auto; }
.th-chunktime { font: 10px var(--mono); color: var(--accent2); flex: 0 0 auto; }
.th-chunkpreview { flex: 1 1 160px; min-width: 0; font-size: 11.5px; color: var(--muted);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.th-chunkbody { padding: 2px 10px 10px; }
.th-chunkbody p { margin: 0 0 6px; font-size: 12.5px; color: var(--text2); line-height: 1.6;
  overflow-wrap: anywhere; }
.th-chunkbody a { font-size: 11px; color: var(--accent2); }

/* ── Disagreements (the conflict layer) ──
   Two rules hold this block together. Nothing that can carry model-written text
   or a transcript quote is allowed to clip — no nowrap, no fixed height, no
   line-clamp anywhere below — because a truncated quote is a misquote and this
   app has shipped that bug more than once. And the two sides are a symmetric
   grid: equal columns, identical styling, no accent on either, because any
   asymmetry reads as the app taking a side in a disagreement that has none. */
.dis-wrap { flex: 1; min-height: 0; overflow-y: auto; padding: 12px 16px 20px;
  box-sizing: border-box; }
.dis-head { padding: 2px 2px 10px; }
.dis-blurb { margin: 6px 0 0; font-size: 12px; color: var(--muted); line-height: 1.6;
  max-width: 78ch; overflow-wrap: anywhere; }
.dis-blurb b { color: var(--text2); }
.dis-statline { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.dis-rejected { margin: 8px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.55;
  max-width: 78ch; overflow-wrap: anywhere; }
/* The count's error bar when the build never recorded one. Boxed and warn-toned
   rather than set as a fourth paragraph of muted grey, because the thing it has
   to beat is a reader skimming the headline chip and stopping — and the caveat
   is only worth writing if it is read. Warn and not bad: an unrecorded spread
   is a limit on what this build can say, not a failure of the layer. */
.dis-caveat { margin: 10px 0 0; padding: 9px 11px; font-size: 11.5px; color: var(--text2);
  line-height: 1.6; max-width: 78ch; overflow-wrap: anywhere; background: var(--warn-dim);
  border: 1px solid var(--warn-border); border-radius: 8px; }
.dis-caveat b, .dis-caveat i { color: var(--text); }

.dis-probes { margin-top: 10px; border: 1px solid var(--border); border-radius: 9px;
  background: var(--panel3); }
.dis-probehead { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; width: 100%;
  text-align: left; padding: 7px 10px; background: none; border: none; cursor: pointer; }
.dis-probehead:hover { background: var(--hover); border-radius: 9px; }
.dis-probesum { flex: 1 1 220px; min-width: 0; font-size: 11.5px; color: var(--muted);
  line-height: 1.5; overflow-wrap: anywhere; }
.dis-chev { font-size: 10px; color: var(--dim); flex: 0 0 auto; }
.dis-probelist { list-style: none; margin: 0; padding: 0 10px 10px; }
.dis-probe { border-top: 1px solid var(--border); padding: 8px 0 2px; }
.dis-probeline { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  font-size: 11.5px; color: var(--text2); }
.dis-probeverdicts { font: 10.5px var(--mono); color: var(--dim); overflow-wrap: anywhere; }
.dis-probewhy { margin: 4px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.55;
  overflow-wrap: anywhere; }
.dis-probeaxis { margin: 4px 0 0; font-size: 11.5px; color: var(--text2); line-height: 1.55;
  overflow-wrap: anywhere; }

.dis-list { display: flex; flex-direction: column; gap: 12px; margin-top: 4px; }
.dis-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; }
/* A factual contradiction gets a different edge from an axis on purpose. Two
   defensible answers and one wrong number are not the same object, and drawing
   them identically is the even-handedness that would mislead. */
.dis-card.factual { border-color: var(--bad-border); }
.dis-factual { margin: 6px 0 0; font-size: 11.5px; color: var(--text2); line-height: 1.6;
  background: var(--bad-dim); border: 1px solid var(--bad-border); border-radius: 8px;
  padding: 8px 10px; overflow-wrap: anywhere; }
.dis-cardhead { display: flex; flex-direction: column; gap: 4px; }
.dis-axis { margin: 0; font-size: 14.5px; font-weight: 600; color: var(--text);
  line-height: 1.45; overflow-wrap: anywhere; }
.dis-taglist { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }

.dis-sides { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
  margin-top: 12px; align-items: start; }
@media (max-width: 820px) { .dis-sides { grid-template-columns: 1fr; } }
.dis-side { min-width: 0; background: var(--panel3); border: 1px solid var(--border);
  border-radius: 9px; padding: 10px 12px; }
.dis-sidehead { display: flex; flex-direction: column; gap: 2px; }
.dis-sidelabel { font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--dim); }
.dis-creator { font-size: 12.5px; font-weight: 600; color: var(--text); line-height: 1.4;
  overflow-wrap: anywhere; }
.dis-position { margin: 6px 0 0; font-size: 12.5px; color: var(--text2); line-height: 1.6;
  overflow-wrap: anywhere; }
.dis-quote { margin: 8px 0 0; padding: 8px 10px; border-left: 2px solid var(--border2);
  background: var(--panel2); border-radius: 0 8px 8px 0; font-size: 12px; color: var(--text2);
  line-height: 1.65; overflow-wrap: anywhere; }
.dis-prov { display: flex; flex-direction: column; gap: 3px; margin-top: 8px; }
.dis-ts { font-size: 11px; color: var(--accent2); overflow-wrap: anywhere; }
.dis-ratio { font: 10px var(--mono); color: var(--dim); overflow-wrap: anywhere; }

.dis-why { margin-top: 12px; border-top: 1px solid var(--border); padding-top: 9px; }
.dis-why p { margin: 4px 0 0; font-size: 12px; color: var(--text2); line-height: 1.6;
  overflow-wrap: anywhere; }
.dis-foot { margin: 16px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.6;
  max-width: 82ch; overflow-wrap: anywhere; }

/* ── Knowledge graph (GraphRAG entities) ── */
.kg-toplevel-empty { color: var(--muted); background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; margin: 16px; }
.kg-toplevel-empty code { font-size: 11.5px; }
.kg-layout { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(280px, 1fr);
  gap: 12px; align-items: start; padding: 12px 16px; height: 100%; box-sizing: border-box; }
@media (max-width: 900px) { .kg-layout { grid-template-columns: 1fr; } }

.kg-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.kg-toolbar input { flex: 0 1 260px; }
.kg-count { font-size: 11px; color: var(--muted); }

.kg-graphwrap { position: relative; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.kg-graph { display: block; width: 100%; height: auto; cursor: grab; touch-action: none; }
.kg-graph:active { cursor: grabbing; }
.kg-node { cursor: pointer; }

.kg-zoomctl { position: absolute; right: 10px; bottom: 10px; display: flex; flex-direction: column;
  gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.kg-zoomctl button { width: 28px; height: 26px; border: none; background: var(--panel);
  color: var(--text2); font: 600 14px var(--mono); cursor: pointer; line-height: 1; }
.kg-zoomctl button:hover:not(:disabled) { background: var(--panel2); color: var(--text); }
.kg-zoomctl button:disabled { color: var(--dim); cursor: default; }
.kg-zoomctl button:last-child { font-size: 9px; letter-spacing: 0.02em; }

.kg-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 11px; color: var(--muted); }
.kg-legend span { display: inline-flex; align-items: center; gap: 5px; }
.kg-swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.kg-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; max-height: 70vh; overflow-y: auto; }
.kg-empty { color: var(--muted); font-size: 12.5px; }
.kg-panelhead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.kg-panelhead h3 { margin: 0; font-size: 14px; color: var(--text); }
.kg-kindtag { font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 8px; padding: 1px 7px; }
.kg-desc { color: var(--text2); font-size: 12px; margin: 6px 0 0; }
.kg-community-summary { color: var(--text2); font-size: 12px; margin: 8px 0 0; background: var(--panel3);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.kg-aliases { color: var(--dim); font-size: 11px; margin: 6px 0 0; }

.kg-claims { margin-top: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
.kg-claims h4 { margin: 0 0 8px; font-size: 11.5px; color: var(--dim); text-transform: uppercase;
  letter-spacing: 0.04em; }
.kg-claim { margin-top: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.kg-claim:last-child { border-bottom: none; }
.kg-claimhead { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.kg-date { font: 600 10.5px var(--mono); color: var(--accent2); }
.kg-video { font-size: 10.5px; color: var(--dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kg-claimtext { margin: 4px 0; font-size: 12.5px; color: var(--text2); line-height: 1.45; }
.kg-claimlink { font-size: 11px; }
`;
