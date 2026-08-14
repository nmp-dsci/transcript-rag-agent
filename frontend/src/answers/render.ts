/**
 * Answer rendering: agent output → HTML.
 *
 * A TypeScript port of ANSWER_RENDER_JS in src/chat/frontend.py, which still
 * serves the static dashboard/chat.html viewer. Both must stay behaviourally
 * identical, which is what render.test.ts pins down.
 */

import type { Reference } from '../api/types';

export function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * The agent sometimes wraps its answer in a JSON payload (trailing, or inside a
 * ```json fence). When present that object's "answer" field is the canonical
 * markdown, so prefer it and drop the surrounding prose.
 */
function extractJsonAnswer(text: string): string | null {
  const match = text.match(/\{\s*"(?:question|answer|references)"\s*:/);
  if (!match || match.index === undefined) return null;
  const start = match.index;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const char = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === '{') depth++;
    else if (char === '}') {
      depth--;
      if (depth === 0) {
        try {
          const parsed = JSON.parse(text.slice(start, i + 1));
          return parsed && typeof parsed.answer === 'string' ? parsed.answer : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

const PREAMBLE_NOISE =
  /\b(evidence|comprehensive|here(?:'s| is)|based on|let me|i'?ll|i will|sufficient|i now have|compile|gathered)\b/i;

/**
 * Strip agent noise: prefer the embedded JSON "answer", else drop a trailing
 * payload, code fences, and a short meta preamble before the first heading.
 */
export function cleanAnswer(text: string | null | undefined): string {
  let out = String(text ?? '');
  const fromJson = extractJsonAnswer(out);
  if (fromJson != null && fromJson.trim()) return fromJson.trim();
  const jsonIndex = out.search(/\n\s*(?:```json\s*)?\{\s*"(question|answer|references)"\s*:/);
  if (jsonIndex !== -1) out = out.slice(0, jsonIndex);
  out = out.replace(/```[a-z]*\s*$/i, '').trim();
  out = out.replace(/^\s*[^#\n][^\n]*\n+(?=#|```)/, (match) =>
    PREAMBLE_NOISE.test(match) ? '' : match,
  );
  return out.replace(/```[a-z]*\s*$/i, '').trim();
}

export function fmtSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return '';
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** Params that already point YouTube at a moment, in every spelling it accepts. */
const SEEK_PARAMS = new Set(['t', 'start', 'time_continue']);

/**
 * Deep-link into a video at the second the UI is displaying.
 *
 * It lives beside `fmtSeconds` on purpose: every caller labels the link with
 * `fmtSeconds(start)` and hrefs it with this, so the clock and the offset are
 * built from one number in one module and cannot drift apart.
 *
 * Existing seek params are **replaced**, not appended to. Nine of this corpus's
 * source URLs were ingested with a `t=` already on them (a YouTube share link
 * carries one), and YouTube honours the *first* `t` it sees — so the older
 * idiom of `url + '&t=' + seconds` produced a link whose label read 1:13 and
 * whose player opened at 0:51. `start` and `time_continue` are stripped for the
 * same reason: either would outrank the `t` appended after it.
 *
 * The query is edited as a string rather than through `URL`, because a stored
 * `source_url` may not parse (a fixture, a relative path, a truncated value)
 * and because round-tripping through `URLSearchParams` would re-encode params
 * this function has no business touching — the `pp=…%3D` tracking blobs on real
 * share links, for instance. Nothing here throws: a missing URL yields no link,
 * and an unparseable one still gets its offset.
 */
export function videoTimestampUrl(
  sourceUrl: string | null | undefined,
  startSeconds: number | null | undefined,
): string | null {
  const base = typeof sourceUrl === 'string' ? sourceUrl.trim() : '';
  if (!base) return null;
  const raw = Number(startSeconds);
  const seconds = Number.isFinite(raw) ? Math.max(0, Math.floor(raw)) : 0;

  // Keep any fragment behind the query, where it belongs.
  const hashAt = base.indexOf('#');
  const fragment = hashAt === -1 ? '' : base.slice(hashAt);
  const addressed = hashAt === -1 ? base : base.slice(0, hashAt);

  const queryAt = addressed.indexOf('?');
  const path = queryAt === -1 ? addressed : addressed.slice(0, queryAt);
  const existing = queryAt === -1 ? [] : addressed.slice(queryAt + 1).split('&');
  const kept = existing.filter(
    (param) => param !== '' && !SEEK_PARAMS.has((param.split('=')[0] ?? '').toLowerCase()),
  );
  kept.push(`t=${seconds}s`);
  return `${path}?${kept.join('&')}${fragment}`;
}

export function fmtTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

/** Map citation number -> reference, keyed by the digits in each label. */
export function buildRefMap(references: Reference[] | undefined): Record<string, Reference> {
  const map: Record<string, Reference> = {};
  for (const reference of references ?? []) {
    const digits = String(reference.label ?? '').match(/\d+/);
    if (digits) map[digits[0]] = reference;
  }
  return map;
}

/** Inline formatting: escape, bold, and turn [n] citations into linked chips. */
function inline(text: string, refMap: Record<string, Reference>): string {
  let out = escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Document-section markers, before the corpus ones: [§3] points at the card
  // above the answer, not at a transcript, and a reader has to be able to tell
  // the two apart at a glance. It carries no link because the section it names
  // is already on screen.
  out = out.replace(/\[§\s*(\d+)\]/g, (_match, number: string) => {
    return `<span class="cite-doc" title="Section ${escapeHtml(number)} of the reviewed document">§${escapeHtml(number)}</span>`;
  });
  out = out.replace(/\[(\d+)\]/g, (_match, number: string) => {
    const reference = refMap[number];
    if (!reference) return `<span class="cite-missing">${number}</span>`;
    const url = escapeHtml(reference.timestamp_url || reference.source_url || '#');
    return `<a class="cite" href="${url}" target="_blank" rel="noreferrer" title="Open source at timestamp">${number}</a>`;
  });
  return out;
}

/** Paragraphs and ordered/unordered lists; headings are handled by sections. */
function renderBlocks(text: string, refMap: Record<string, Reference>): string {
  const lines = String(text ?? '').split('\n');
  let html = '';
  let list: 'ol' | 'ul' | null = null;
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${inline(paragraph.join(' '), refMap)}</p>`;
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      html += `</${list}>`;
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (!line.trim()) {
      flushParagraph();
      continue;
    }
    const ordered = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ordered) {
      flushParagraph();
      if (list !== 'ol') {
        closeList();
        html += '<ol>';
        list = 'ol';
      }
      html += `<li>${inline(ordered[1] ?? '', refMap)}</li>`;
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      if (list !== 'ul') {
        closeList();
        html += '<ul>';
        list = 'ul';
      }
      html += `<li>${inline(bullet[1] ?? '', refMap)}</li>`;
      continue;
    }
    closeList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  closeList();
  return html;
}

interface Section {
  level: number;
  title: string;
  body: string[];
}

export function parseSections(text: string): { intro: string; sections: Section[] } {
  const intro: string[] = [];
  const sections: Section[] = [];
  let current: Section | null = null;
  for (const line of String(text ?? '').split('\n')) {
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      current = { level: heading[1]!.length, title: heading[2] ?? '', body: [] };
      sections.push(current);
    } else if (current) {
      current.body.push(line);
    } else {
      intro.push(line);
    }
  }
  return { intro: intro.join('\n'), sections };
}

const SUMMARY_HEADING = /key findings|summary|tl;?dr|bottom line|takeaways?/i;

export function renderAnswer(text: string, references?: Reference[]): string {
  const refMap = buildRefMap(references);
  const { intro, sections } = parseSections(text);
  let html = renderBlocks(intro, refMap);
  for (const section of sections) {
    const highlight = SUMMARY_HEADING.test(section.title);
    const tag = `h${Math.min(section.level + 1, 5)}`;
    html +=
      `<section class="ans-section${highlight ? ' summary' : ''}">` +
      `<${tag} class="ans-h">${inline(section.title, refMap)}</${tag}>` +
      renderBlocks(section.body.join('\n'), refMap) +
      '</section>';
  }
  return html;
}
