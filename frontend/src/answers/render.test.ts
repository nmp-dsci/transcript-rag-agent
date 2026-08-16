import { describe, expect, it } from 'vitest';

import {
  buildRefMap,
  cleanAnswer,
  escapeHtml,
  fmtSeconds,
  parseSections,
  renderAnswer,
  videoTimestampUrl,
} from './render';

const REFS = [
  { label: '[1]', video_id: 'abc', timestamp_url: 'https://youtu.be/abc?t=5' },
  { label: '[2]', video_id: 'def', source_url: 'https://youtu.be/def' },
];

describe('escapeHtml', () => {
  it('escapes markup and nullish values', () => {
    expect(escapeHtml('<script>&')).toBe('&lt;script&gt;&amp;');
    expect(escapeHtml(null)).toBe('');
    expect(escapeHtml(undefined)).toBe('');
  });
});

describe('cleanAnswer', () => {
  it('prefers the embedded JSON answer field', () => {
    const raw = 'Here is my answer:\n{"question": "q", "answer": "The real answer."}';
    expect(cleanAnswer(raw)).toBe('The real answer.');
  });

  it('handles braces and escapes inside JSON strings', () => {
    const raw = '{"answer": "Uses {braces} and a \\"quote\\"."}';
    expect(cleanAnswer(raw)).toBe('Uses {braces} and a "quote".');
  });

  it('drops a trailing JSON payload when it has no answer field', () => {
    const raw = '# Findings\n\nSome prose.\n{"references": [1]}';
    expect(cleanAnswer(raw)).toBe('# Findings\n\nSome prose.');
  });

  it('strips a noisy meta preamble before the first heading', () => {
    const raw = 'I now have sufficient evidence to answer.\n\n# Key Findings\n\nBody.';
    expect(cleanAnswer(raw)).toBe('# Key Findings\n\nBody.');
  });

  it('keeps a meaningful first line that is not agent chatter', () => {
    const raw = 'The market is cooling.\n\n# Detail\n\nBody.';
    expect(cleanAnswer(raw)).toContain('The market is cooling.');
  });

  it('strips trailing code fences and handles empty input', () => {
    expect(cleanAnswer('Body text\n```')).toBe('Body text');
    expect(cleanAnswer('')).toBe('');
    expect(cleanAnswer(null)).toBe('');
  });
});

describe('buildRefMap', () => {
  it('keys references by the digits in their label', () => {
    const map = buildRefMap(REFS);
    expect(map['1']?.video_id).toBe('abc');
    expect(map['2']?.video_id).toBe('def');
  });

  it('tolerates missing references', () => {
    expect(buildRefMap(undefined)).toEqual({});
    expect(buildRefMap([{ video_id: 'x' }])).toEqual({});
  });
});

describe('renderAnswer', () => {
  it('links citations that resolve to a reference', () => {
    const html = renderAnswer('Prices fell [1].', REFS);
    expect(html).toContain('<a class="cite" href="https://youtu.be/abc?t=5"');
    expect(html).toContain('>1</a>');
  });

  it('falls back to source_url when there is no timestamp link', () => {
    expect(renderAnswer('See [2].', REFS)).toContain('href="https://youtu.be/def"');
  });

  it('marks citations with no matching reference', () => {
    expect(renderAnswer('Unknown [9].', REFS)).toContain('<span class="cite-missing">9</span>');
  });

  it('marks a document section apart from a corpus citation', () => {
    const html = renderAnswer('[§3] opens with "responsible for" [1].', REFS);
    expect(html).toContain('<span class="cite-doc"');
    expect(html).toContain('>§3</span>');
    expect(html).toContain('<a class="cite"');
  });

  it('does not read a section marker as an unresolved corpus citation', () => {
    expect(renderAnswer('Tighten [§2].', REFS)).not.toContain('cite-missing');
  });

  it('renders ordered and unordered lists', () => {
    const html = renderAnswer('- first\n- second\n\n1. one\n2. two', []);
    expect(html).toContain('<ul><li>first</li><li>second</li></ul>');
    expect(html).toContain('<ol><li>one</li><li>two</li></ol>');
  });

  it('renders bold text', () => {
    expect(renderAnswer('This is **important**.', [])).toContain(
      '<strong>important</strong>',
    );
  });

  it('escapes HTML in the answer body', () => {
    expect(renderAnswer('A <script>alert(1)</script> tag', [])).not.toContain('<script>');
  });

  it('highlights summary sections and demotes heading levels', () => {
    const html = renderAnswer('## Key Findings\n\nBody.', []);
    expect(html).toContain('class="ans-section summary"');
    expect(html).toContain('<h3 class="ans-h">');
  });

  it('does not highlight ordinary sections', () => {
    const html = renderAnswer('## Detail\n\nBody.', []);
    expect(html).toContain('class="ans-section"');
  });

  it('joins wrapped paragraph lines into one paragraph', () => {
    expect(renderAnswer('one line\nsecond line', [])).toBe('<p>one line second line</p>');
  });
});

describe('parseSections', () => {
  it('splits intro text from headed sections', () => {
    const { intro, sections } = parseSections('Intro.\n\n# One\nA\n\n## Two\nB');
    expect(intro.trim()).toBe('Intro.');
    expect(sections.map((s) => s.title)).toEqual(['One', 'Two']);
    expect(sections[0]?.level).toBe(1);
    expect(sections[1]?.level).toBe(2);
  });
});

describe('fmtSeconds', () => {
  it('formats as m:ss and handles null', () => {
    expect(fmtSeconds(0)).toBe('0:00');
    expect(fmtSeconds(75)).toBe('1:15');
    expect(fmtSeconds(3599)).toBe('59:59');
    expect(fmtSeconds(null)).toBe('');
  });
});

describe('videoTimestampUrl', () => {
  /** How a browser reads the link — YouTube honours the first `t` it is given. */
  const seekParam = (url: string) => new URLSearchParams(url.split('?')[1]).get('t');
  const countT = (url: string) =>
    [...new URLSearchParams(url.split('?')[1]).keys()].filter((key) => key === 't').length;

  it('appends the start second to a watch URL that has no t', () => {
    expect(videoTimestampUrl('https://www.youtube.com/watch?v=x', 91.7)).toBe(
      'https://www.youtube.com/watch?v=x&t=91s',
    );
  });

  it('starts the query string when the URL has none', () => {
    expect(videoTimestampUrl('https://youtu.be/x', 5)).toBe('https://youtu.be/x?t=5s');
  });

  // The V4 defect: this corpus stores nine share links that already carry a
  // t=, and appending a second one opened the player 22s before the clock the
  // row displayed.
  it('replaces a pre-existing t rather than appending a second one', () => {
    const stored = 'https://www.youtube.com/watch?v=by8wrrXW3So&t=51s&pp=ygUScmVzdW1lIGFpIGVuZ2luZWVy';
    const url = videoTimestampUrl(stored, 73)!;
    expect(seekParam(url)).toBe('73s');
    expect(countT(url)).toBe(1);
    expect(url).not.toContain('t=51s');
  });

  it('agrees with the clock the row displays', () => {
    const stored = 'https://www.youtube.com/watch?v=by8wrrXW3So&t=51s';
    // 1:13 on screen must be 73s in the href, not the stored 51s.
    expect(fmtSeconds(73)).toBe('1:13');
    expect(seekParam(videoTimestampUrl(stored, 73)!)).toBe('73s');
  });

  it('strips start and time_continue, which would outrank the appended t', () => {
    const url = videoTimestampUrl('https://www.youtube.com/watch?v=x&start=51&time_continue=51', 73)!;
    expect(seekParam(url)).toBe('73s');
    expect(url).not.toContain('start=');
    expect(url).not.toContain('time_continue=');
  });

  it('preserves every other query param, encoding untouched', () => {
    const stored = 'https://www.youtube.com/watch?v=X4dEHRzBLmc&t=951s&pp=ygUObGxtIGFzIGEganVkZ2U%3D';
    const url = videoTimestampUrl(stored, 1200)!;
    expect(url).toContain('v=X4dEHRzBLmc');
    expect(url).toContain('pp=ygUObGxtIGFzIGEganVkZ2U%3D');
    expect(seekParam(url)).toBe('1200s');
    expect(countT(url)).toBe(1);
  });

  it('links to the very start for offset 0', () => {
    expect(videoTimestampUrl('https://www.youtube.com/watch?v=x', 0)).toBe(
      'https://www.youtube.com/watch?v=x&t=0s',
    );
    expect(videoTimestampUrl('https://www.youtube.com/watch?v=x&t=51s', 0)).toBe(
      'https://www.youtube.com/watch?v=x&t=0s',
    );
  });

  it('treats a missing or unusable timestamp as the start of the video', () => {
    expect(videoTimestampUrl('https://youtu.be/x', null)).toBe('https://youtu.be/x?t=0s');
    expect(videoTimestampUrl('https://youtu.be/x', undefined)).toBe('https://youtu.be/x?t=0s');
    expect(videoTimestampUrl('https://youtu.be/x', Number.NaN)).toBe('https://youtu.be/x?t=0s');
    expect(videoTimestampUrl('https://youtu.be/x', -30)).toBe('https://youtu.be/x?t=0s');
  });

  it('has no link without a source URL', () => {
    expect(videoTimestampUrl(null, 10)).toBeNull();
    expect(videoTimestampUrl(undefined, 10)).toBeNull();
    expect(videoTimestampUrl('', 10)).toBeNull();
    expect(videoTimestampUrl('   ', 10)).toBeNull();
  });

  it('degrades rather than throwing on a malformed or non-YouTube URL', () => {
    expect(() => videoTimestampUrl('not a url', 10)).not.toThrow();
    expect(videoTimestampUrl('not a url', 10)).toBe('not a url?t=10s');
    expect(videoTimestampUrl('https://vimeo.com/123?t=9s', 10)).toBe('https://vimeo.com/123?t=10s');
    expect(videoTimestampUrl('/local/clip', 10)).toBe('/local/clip?t=10s');
    expect(videoTimestampUrl('https://www.youtube.com/watch?v=x&', 10)).toBe(
      'https://www.youtube.com/watch?v=x&t=10s',
    );
  });

  it('keeps a fragment behind the query', () => {
    expect(videoTimestampUrl('https://www.youtube.com/watch?v=x&t=51s#note', 73)).toBe(
      'https://www.youtube.com/watch?v=x&t=73s#note',
    );
  });
});
