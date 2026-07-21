import { describe, expect, it } from 'vitest';

import {
  buildRefMap,
  cleanAnswer,
  escapeHtml,
  fmtSeconds,
  parseSections,
  renderAnswer,
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
