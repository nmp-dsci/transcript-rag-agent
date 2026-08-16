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
/* Disagreements in a cell's context. Full width rather than a column in
   .crit-cols: an axis is a whole question and reads as one line of prose, and
   the two-column grid would break it over four. Nothing here clips — the axis
   is model-written text of unbounded length. */
.crit-conflicts { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border);
  white-space: normal; }
.crit-missnote { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.crit-chips { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.crit-agree { margin-left: 6px; font: 600 9px var(--mono); letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--muted); background: var(--panel2);
  border: 1px solid var(--border2); border-radius: 8px; padding: 1px 5px;
  white-space: nowrap; }
.crit-extra { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border); }

/* ── matcher ballots ─────────────────────────────────────────────────────
   The per-repeat votes behind one consensus pairing. Full width like the
   conflicts block: a criterion is a whole sentence and the two-column grid
   would break the ballot chips away from the rule they belong to. */
.crit-ballots { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border);
  white-space: normal; }
.crit-draws { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.crit-draw { display: inline-flex; align-items: baseline; gap: 5px; font: 600 10.5px var(--mono);
  color: var(--muted); background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 8px; padding: 1px 6px; white-space: nowrap; }
.crit-draw.on { color: var(--accent2); background: var(--accent-dim);
  border-color: var(--accent-border); }
.crit-draw em { font-style: normal; font-size: 9px; color: var(--dim); }
/* A pairing the vote discarded, wherever it is named. Warn rather than bad:
   consensus doing its job is not an error, but a zero that came out of it is
   not the same zero as one nothing reached, and the reader has to see which. */
.crit-agree.lost, .crit-spread.lost { color: var(--warn, #c9922b); }

/* ── an arm the grounding gate could not grade ───────────────────────────
   Deliberately *not* styled like a zero or like a bad score. The cell holds no
   number at all, and the two ways to get this wrong are rendering a dash (which
   a reader fills in with the figure they last saw) and rendering red (which
   reads as "scored badly"). So: words, in muted text, with the published figure
   underneath labelled as uncertified, and a full-width row saying why. */
.exp-table td.num .crit-nomeasure { display: block; font: 600 9px var(--mono);
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }
/* .exp-table sets white-space: nowrap, and "no comparison — rag_llm_filtered is
   ungraded" is half again as long as the delta it replaced: left to itself it
   widens the recall column enough to push the whole table into a horizontal
   scroll. It wraps instead, right-aligned with the column it annotates. */
.crit-spread.nocmp { white-space: normal; max-width: 24ch; margin-left: auto; }
.crit-ungradedrow > td { white-space: normal; padding: 0 10px 10px;
  border-bottom: 1px solid var(--border); }
.crit-ungraded { color: var(--text2); font-size: 12px; line-height: 1.55;
  max-width: 82ch; border-left: 2px solid var(--warn, #c9922b); padding-left: 10px; }
.crit-ungraded b { color: var(--text); }
.crit-ungraded code { font: 600 11px var(--mono); color: var(--muted); }

/* ── expert rubric packs ─────────────────────────────────────────────────
   Same discipline as the block above: .exp-table sets white-space: nowrap and
   every rule in a pack is a whole sentence, so each block carrying prose
   re-declares its own wrapping. The clipping bug this guards against showed 66
   of 273 characters and looked like a short criterion rather than a broken one. */
.pk-intro { color: var(--text2); font-size: 12.5px; line-height: 1.55; max-width: 84ch;
  margin: 0 0 10px; white-space: normal; }
/* Reads before the rules it qualifies, so nobody trusts the list first and
   discovers its build date afterwards. Left border rather than a fill: it must
   be impossible to miss without looking like an error the reader caused. */
.pk-stale { color: var(--text2); font-size: 12.5px; line-height: 1.55; max-width: 84ch;
  margin: 0 0 12px; padding: 8px 12px; white-space: normal;
  border-left: 3px solid var(--warn, #c9922b); background: var(--surface2, rgba(201,146,43,0.07));
  border-radius: 0 6px 6px 0; }
.pk-stale b { color: var(--text); }
.pk-stale code { font-size: 11.5px; }
.pk-fresh { color: var(--muted); font-size: 11.5px; margin: 0 0 12px; white-space: normal; }
/* Why the loop lost, on the row where it happened. An addition that is new
   against round one but already covered by the pack it is measured against is
   rediscovery, and the diff read as ten discoveries until this said otherwise. */
.rs-rediscovered { display: block; font-size: 11px; margin-top: 3px; color: var(--warn, #c9922b);
  white-space: normal; }
.rs-newground { display: block; font-size: 11px; margin-top: 3px; color: var(--ok, #4a9d6b);
  white-space: normal; }
.rs-rediscovered code, .rs-newground code { font-size: 10.5px; }
.pk-blurb { color: var(--text); font-size: 13px; line-height: 1.5; margin: 0 0 8px;
  white-space: normal; }
.pk-routing { color: var(--muted); font-size: 11.5px; line-height: 1.5; margin: 0 0 12px;
  max-width: 90ch; white-space: normal; }
.pk-routing .microlabel, .pk-check .microlabel, .pk-why .microlabel { display: block; }
.pk-unbuilt { color: var(--bad); }

.pk-checks { display: flex; flex-wrap: wrap; gap: 20px; padding: 10px 12px;
  background: var(--panel3); border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 12px; }
.pk-check-item { display: flex; flex-direction: column; gap: 1px; }
.pk-check-item b { font: 700 15px var(--mono); color: var(--text);
  font-variant-numeric: tabular-nums; }
.pk-check-item b.good { color: var(--good); }
.pk-check-item b.bad { color: var(--bad); }
.pk-checksub { font: 10px var(--mono); color: var(--muted); }

.pk-section { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border); }
.pk-section h4 { margin: 0 0 8px; font-size: 12.5px; color: var(--text);
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; white-space: normal; }
.pk-hint { font: 400 11px var(--mono); color: var(--muted); white-space: normal; }
.pk-notice { margin: 8px 0 0; font: 11px var(--mono); color: var(--accent2);
  white-space: normal; }
.pk-nod2 { margin: 16px 0 0; padding: 10px 12px; background: var(--panel3);
  border: 1px solid var(--border); border-radius: 8px; color: var(--muted);
  font-size: 12px; line-height: 1.5; white-space: normal; }
.pk-linkbtn { background: none; border: none; padding: 0; cursor: pointer;
  color: var(--text); font: 600 12.5px inherit; }

.pk-rublist { list-style: none; margin: 0; padding: 0; display: flex;
  flex-direction: column; gap: 6px; }
.pk-rubric { border: 1px solid var(--border); border-radius: 8px; background: var(--panel3);
  overflow: hidden; }
.pk-rubric.open { border-color: var(--accent-border); }
.pk-rubhead { display: flex; align-items: flex-start; gap: 10px; width: 100%;
  background: none; border: none; cursor: pointer; text-align: left;
  padding: 9px 11px; color: inherit; font: inherit; }
.pk-rubhead:hover { background: var(--panel2); }
.pk-rubid { flex: 0 0 auto; font: 700 10.5px var(--mono); color: var(--muted);
  padding-top: 2px; }
.pk-rubtext { flex: 1 1 auto; min-width: 0; color: var(--text); font-size: 12.5px;
  line-height: 1.5; white-space: normal; overflow-wrap: anywhere; }
.pk-rubtags { flex: 0 0 auto; display: flex; gap: 5px; flex-wrap: wrap;
  justify-content: flex-end; }
.pk-badge { font: 600 9.5px var(--mono); letter-spacing: 0.04em; color: var(--muted);
  background: var(--panel2); border: 1px solid var(--border2); border-radius: 8px;
  padding: 1px 6px; white-space: nowrap; }
.pk-badge.ok { color: var(--good); background: var(--good-dim); border-color: var(--good-border); }
.pk-badge.thin { color: var(--muted); }
.pk-badge.contested { color: var(--bad); background: var(--bad-dim);
  border-color: var(--bad-border); }

.pk-rubbody { padding: 2px 11px 12px 11px; border-top: 1px solid var(--border);
  white-space: normal; }
.pk-check, .pk-why { margin: 10px 0 0; color: var(--text2); font-size: 12px;
  line-height: 1.5; max-width: 84ch; white-space: normal; }
.pk-why { color: var(--muted); }
.pk-unit { margin: 10px 0 0; font: 10.5px var(--mono); color: var(--muted);
  white-space: normal; overflow-wrap: anywhere; }
.pk-evlist { list-style: none; margin: 10px 0 0; padding: 0; display: flex;
  flex-direction: column; gap: 10px; }
.pk-ev { border-left: 2px solid var(--accent-border); padding-left: 10px;
  white-space: normal; }
.pk-evmeta { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }
.pk-ts { font: 600 10.5px var(--mono); color: var(--accent2); background: var(--accent-dim);
  border: 1px solid var(--accent-border); border-radius: 8px; padding: 1px 6px;
  text-decoration: none; white-space: nowrap; }
.pk-ts.bad { color: var(--bad); background: var(--bad-dim); border-color: var(--bad-border); }
.pk-creator { font: 600 11px var(--mono); color: var(--text); }
.pk-vidtitle { font-size: 11px; color: var(--muted); white-space: normal;
  overflow-wrap: anywhere; }
.pk-quote { margin: 4px 0 0; padding: 0; color: var(--text2); font-size: 12.5px;
  font-style: italic; line-height: 1.5; max-width: 84ch; white-space: normal;
  overflow-wrap: anywhere; }
.pk-drift { margin-top: 3px; font: 10px var(--mono); color: var(--muted);
  white-space: normal; overflow-wrap: anywhere; }

.pk-table td.pk-cellwrap { white-space: normal; min-width: 180px; max-width: 420px;
  overflow-wrap: anywhere; }
.exp-table td.pk-cellwrap { white-space: normal; overflow-wrap: anywhere; }
.pk-pin { display: flex; gap: 4px; }
.exp-table td.pk-src { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }

.pk-verdicts { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.pk-verdict { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
  white-space: normal; }
.pk-verdict b { font: 700 12px var(--mono); color: var(--text); }
.pk-verdictwhy { font-size: 11.5px; color: var(--muted); white-space: normal; }

.pk-gaps { list-style: none; margin: 8px 0 0; padding: 0; display: flex;
  flex-direction: column; gap: 7px; }
.pk-gaps li { display: flex; flex-direction: column; gap: 1px; white-space: normal;
  border-left: 2px solid var(--border2); padding-left: 10px; }
.pk-gaps code { font: 10.5px var(--mono); color: var(--muted); }
.pk-gaptitle { color: var(--text2); font-size: 12px; line-height: 1.45;
  overflow-wrap: anywhere; }
.pk-gapreason { color: var(--muted); font-size: 11px; line-height: 1.45; }

/* Deep-research build report. Every text cell wraps: the critic's findings and
   the criteria they produced are the evidence this panel exists to show, and a
   truncated one is a claim a reader cannot check. */
.rs-note { margin: 10px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.55;
  max-width: 92ch; white-space: normal; }
.exp-table tr.rs-loop td { background: var(--accent-dim); }

.rs-gaps { list-style: none; margin: 0; padding: 0; display: flex;
  flex-direction: column; gap: 12px; }
.rs-gap { border: 1px solid var(--border); border-radius: 8px; background: var(--panel3);
  padding: 10px 12px; white-space: normal; }
.rs-gaphead { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.rs-gapid { font: 700 10.5px var(--mono); color: var(--muted); }
.rs-gapmissing { flex: 1 1 320px; min-width: 0; color: var(--text); font-size: 12.5px;
  line-height: 1.5; overflow-wrap: anywhere; }
.rs-gapwhy { margin: 6px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.5;
  max-width: 92ch; overflow-wrap: anywhere; }
.rs-probe { margin: 8px 0 0; color: var(--text2); font-size: 12px; line-height: 1.5;
  overflow-wrap: anywhere; }
.rs-probe .microlabel { display: block; }
.rs-probemeta { display: block; margin-top: 2px; font: 10.5px var(--mono);
  color: var(--muted); overflow-wrap: anywhere; }
.rs-caused { list-style: none; margin: 9px 0 0; padding: 0; display: flex;
  flex-direction: column; gap: 7px; }
.rs-caused li { border-left: 2px solid var(--good-border); padding-left: 10px;
  display: flex; flex-direction: column; gap: 2px; }
.rs-caused code { font: 10.5px var(--mono); color: var(--muted); }
.rs-crit { color: var(--text); font-size: 12.5px; line-height: 1.5;
  overflow-wrap: anywhere; }
.rs-critmeta { font: 10.5px var(--mono); color: var(--muted); line-height: 1.5;
  overflow-wrap: anywhere; }

.rs-diff { list-style: none; margin: 8px 0 0; padding: 0; display: flex;
  flex-direction: column; gap: 6px; }
.rs-diffrow { display: flex; gap: 9px; align-items: baseline; white-space: normal;
  border-left: 2px solid var(--border2); padding-left: 9px; }
.rs-diffrow.added { border-left-color: var(--good-border); }
.rs-diffrow.removed { border-left-color: var(--bad-border); }
.rs-diffmark { flex: 0 0 auto; width: 9px; font: 700 12px var(--mono); color: var(--muted); }
/* The restatement rows reuse the diff row but put a cosine in the gutter, which
   does not fit the 9px a "+"/"−" needs. Sized in ch so the number never clips. */
.rs-cos { flex: 0 0 auto; width: 5ch; font: 700 11px var(--mono); color: var(--muted);
  font-variant-numeric: tabular-nums; }
.rs-diffrow.removed .rs-cos { color: var(--bad); }
.rs-diffrow.added .rs-diffmark { color: var(--good); }
.rs-diffrow.removed .rs-diffmark { color: var(--bad); }
.rs-diffbody { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column;
  gap: 1px; }

.rs-plan { margin: 8px 0 0; padding-left: 22px; display: flex; flex-direction: column;
  gap: 8px; }
.rs-plan li { white-space: normal; color: var(--muted); }
.rs-plan li.in-r1 { color: var(--text2); }
.rs-facet { display: block; font: 600 12px var(--mono); color: var(--text); }
.rs-question { display: block; font-size: 12px; line-height: 1.5; color: inherit;
  overflow-wrap: anywhere; }
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
