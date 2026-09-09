/**
 * Scoped styles for the System Design tab.
 *
 * Written against the existing theme custom properties, so light and dark
 * both come out right without the component branching on theme — the same
 * contract as src/experiments/styles.ts.
 */

const STYLE_ID = 'tlab-design';

const CSS = `
.ds-intro { color: var(--text2); max-width: 74ch; margin: 4px 0 14px; line-height: 1.55; }
.ds-toplevel-empty { color: var(--muted); background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; }
/* One column until a node is selected — the graph is the page, and a
   permanently reserved empty panel made it look like the smaller half. */
.ds-layout { display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; align-items: start; }
.ds-layout.split { grid-template-columns: minmax(0, 1.35fr) minmax(300px, 1fr); }
@media (max-width: 900px) { .ds-layout.split { grid-template-columns: minmax(0, 1fr); } }

.ds-graphwrap { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  overflow: auto; padding: 4px; }
.ds-graph { display: block; width: 100%; height: auto; min-width: 720px; }

.ds-edge { stroke: var(--border2); stroke-width: 1.4; fill: none;
  transition: stroke var(--dur) var(--ease), opacity var(--dur) var(--ease); }
.ds-edge.hi { stroke: var(--accent2); stroke-width: 2; }
/* Trace: everything off the path recedes rather than disappearing, so the
   shape of the whole system stays legible behind the path being read. */
.ds-edge.dim { opacity: 0.18; }
.ds-node { transition: opacity var(--dur) var(--ease); }
.ds-node.dim { opacity: 0.3; }
.ds-node.lit rect { stroke: var(--accent2); }
.ds-node.lit text { fill: var(--text); }

.ds-node { cursor: pointer; outline: none; }
.ds-node rect { fill: var(--panel2); stroke: var(--border2); stroke-width: 1.3; rx: 9;
  transition: stroke var(--dur) var(--ease), fill var(--dur) var(--ease); }
/* 11.5px is the default; long labels step down to a 10px floor inline (see
   labelSize) rather than overflowing their node. */
.ds-node text { fill: var(--text2); font: 600 11.5px var(--mono); pointer-events: none; }
.ds-node .ds-kind { fill: var(--dim); font: 600 10px var(--mono); letter-spacing: 0.04em;
  text-transform: uppercase; }
.ds-node:hover rect { stroke: var(--accent2); }
.ds-node:focus-visible rect { stroke: var(--accent); stroke-width: 2; }
.ds-node.sel rect { fill: var(--accent-dim); stroke: var(--accent2); stroke-width: 2; }
.ds-node.sel text { fill: var(--text); }

.ds-node.kind-agent rect { stroke: var(--accent-border); }
.ds-node.kind-store rect { stroke: var(--good-border); }
.ds-node.kind-model rect { stroke: var(--warn-border); }
.ds-node.kind-stage rect { stroke-dasharray: 4 3; }

.ds-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: var(--t-1);
  color: var(--muted); }
.ds-legend-hint { margin-left: auto; color: var(--dim); font-style: italic; }
.ds-legend span { display: inline-flex; align-items: center; gap: 5px; }
.ds-swatch { width: 10px; height: 10px; border-radius: 3px; border: 1.3px solid; display: inline-block; }
.ds-swatch.agent { border-color: var(--accent-border); }
.ds-swatch.store { border-color: var(--good-border); }
.ds-swatch.model { border-color: var(--warn-border); }
.ds-swatch.stage { border-color: var(--border2); border-style: dashed; }

.ds-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; position: sticky; top: 8px; max-height: calc(100vh - 140px); overflow-y: auto; }
.ds-empty { color: var(--muted); font-size: 12.5px; }
.ds-panelhead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ds-panelhead h3 { margin: 0; font-size: 14px; color: var(--text); }
.ds-kindtag { font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 8px; padding: 1px 7px; }
.ds-desc { color: var(--text2); font-size: 12.5px; margin: 8px 0 0; }

.ds-cfg { margin: 12px 0 0; border-top: 1px solid var(--border); padding-top: 10px; }
.ds-cfg table { width: 100%; border-collapse: collapse; font-size: 12px; }
.ds-cfg td { padding: 3px 0; vertical-align: top; }
.ds-cfg td.k { color: var(--dim); font: 10.5px var(--mono); white-space: nowrap; padding-right: 10px; }
.ds-cfg td.v { color: var(--text2); font: 11.5px var(--mono); word-break: break-all; }

.ds-sectionhead { margin: 14px 0 6px; font: 600 9.5px var(--mono); letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--dim); border-top: 1px solid var(--border); padding-top: 10px; }

.ds-flow { display: flex; flex-direction: column; }
.ds-flow-branch { margin: 10px 0 4px; font: 700 9.5px var(--mono); letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--accent2); }
.ds-flow-branch:first-child { margin-top: 0; }
.ds-flow-converge { display: flex; align-items: center; gap: 8px; margin: 10px 0 4px;
  font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); }
.ds-flow-converge::after { content: ''; flex: 1; height: 1px; background: var(--border2); }
.ds-flow-step { display: flex; gap: 8px; padding: 5px 0; }
.ds-flow-num { flex: none; width: 16px; height: 16px; border-radius: 50%; background: var(--panel2);
  border: 1px solid var(--border2); color: var(--muted); font: 700 9px var(--mono);
  display: flex; align-items: center; justify-content: center; margin-top: 1px; }
.ds-flow-label { font: 600 11.5px var(--mono); color: var(--text); }
.ds-flow-detail { color: var(--text2); font-size: 11.5px; line-height: 1.4; margin-top: 1px; }

.ds-prompts { margin: 14px 0 0; border-top: 1px solid var(--border); padding-top: 10px; }
.ds-item { margin-top: 10px; }
.ds-item:first-child { margin-top: 0; }
.ds-itemhead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ds-name { font: 700 11.5px var(--mono); color: var(--text); }
.ds-role { font: 600 9.5px var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 8px; padding: 1px 6px; }
.ds-role.system { color: var(--accent2); background: var(--accent-dim); border-color: var(--accent-border); }
.ds-copy { margin-left: auto; background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 6px; color: var(--muted); cursor: pointer; font: 600 10px var(--mono); padding: 2px 8px; }
.ds-copy:hover { color: var(--text); }
.ds-text { margin: 6px 0 0; background: var(--panel3); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; overflow-x: auto;
  font: 11px/1.5 var(--mono); color: var(--text2); white-space: pre-wrap; max-height: 240px; }
.ds-var { font: 600 10px var(--mono); color: var(--warn, #d29922); background: var(--panel3);
  border: 1px solid var(--border2); border-radius: 8px; padding: 1px 6px; }
`;

/** Install the stylesheet once per document, keyed by id (see useEvalStyles). */
export function useDesignStyles(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}
