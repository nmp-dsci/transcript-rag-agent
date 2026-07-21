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
`;
