import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { TraceStep } from '../api/types';
import { AnswerTrace } from './AnswerTrace';

function step(overrides: Partial<TraceStep> = {}): TraceStep {
  return {
    phase: 'retrieve',
    label: 'Retrieve candidates',
    detail: 'semantic search over the whole corpus — 10 candidates',
    chunk_ids: ['chunk:vid1:0', 'chunk:vid1:1'],
    model: null,
    elapsed_ms: 42,
    iteration: null,
    ...overrides,
  };
}

describe('AnswerTrace', () => {
  it('renders nothing for an empty trace', () => {
    const { container } = render(<AnswerTrace steps={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders each step with its phase, label, detail, and counts, collapsed', () => {
    render(
      <AnswerTrace
        steps={[
          step(),
          step({
            phase: 'llm',
            label: 'Answer',
            detail: 'one answer call over the retrieved chunks',
            chunk_ids: [],
            model: 'deepseek-v4-flash',
            elapsed_ms: null,
          }),
        ]}
      />,
    );
    expect(screen.getByText('trace — 2 steps')).toBeInTheDocument();
    expect(screen.getByText('retrieve')).toBeInTheDocument();
    expect(screen.getByText('llm')).toBeInTheDocument();
    expect(screen.getByText(/semantic search over the whole corpus/)).toBeInTheDocument();
    expect(screen.getByText(/2 chunks · 42ms/)).toBeInTheDocument();
    // Collapsed by default: the details element is not open.
    expect(document.querySelector('details[open]')).toBeNull();
  });

  it('labels a graph route decision distinctly', () => {
    render(
      <AnswerTrace
        steps={[
          step({
            phase: 'route',
            label: 'Route → temporal',
            detail: 'router named entities: negative gearing',
            chunk_ids: [],
          }),
        ]}
      />,
    );
    expect(screen.getByText('route')).toBeInTheDocument();
    expect(screen.getByText('Route → temporal')).toBeInTheDocument();
  });
  it('shows the query a retrieval actually searched for, in full', () => {
    // A document review searches for the criteria the document should be judged
    // against, not the words the user typed. Truncating that hides the one
    // thing that makes the retrieval checkable.
    const query =
      'how to present engineering projects on a personal portfolio site so recruiters and hiring managers take them seriously';
    render(<AnswerTrace steps={[step({ query })]} />);

    expect(screen.getByText('query')).toBeInTheDocument();
    expect(screen.getByText(query)).toBeInTheDocument();
  });

  it('shows no query row for steps that did not search', () => {
    render(<AnswerTrace steps={[step({ phase: 'llm', label: 'Answer', query: null })]} />);

    expect(screen.queryByText('query')).not.toBeInTheDocument();
  });

  it('shows the videos a summary filter routed to, in full', () => {
    // The count is in the detail line; which videos is the check on it, and a
    // list of five in a nowrap field shows two of them.
    const note =
      'Job Interview Simulation (5kxPMauR4fs) 0.33 · What AI Engineer JOBS Need on a Resume (by8wrrXW3So) 0.28 · My Entire Job Search Process (uC-FmLvw1u0) 0.27';
    render(
      <AnswerTrace
        steps={[step({ phase: 'filter', label: 'Summary filter', chunk_ids: [], note })]}
      />,
    );

    expect(screen.getByText('videos')).toBeInTheDocument();
    expect(screen.getByText(note)).toBeInTheDocument();
  });

  it('shows a corpus-coverage warning in full rather than clipped to its first clause', () => {
    const note =
      'This document does not match a kind the corpus has criteria for (resume, portfolio, professional profile, cover letter). The retrieved chunks are the closest the corpus holds, which may be advice about a different kind of document entirely — say so rather than applying it.';
    render(
      <AnswerTrace
        steps={[
          step({
            phase: 'route',
            label: 'Corpus coverage',
            detail: 'the corpus may hold no criteria for this document',
            chunk_ids: [],
            note,
          }),
        ]}
      />,
    );

    expect(screen.getByText('warning')).toBeInTheDocument();
    expect(screen.getByText(note)).toBeInTheDocument();
  });

  it('shows no note row for steps that carry none', () => {
    render(<AnswerTrace steps={[step()]} />);

    expect(screen.queryByText('videos')).not.toBeInTheDocument();
    expect(screen.queryByText('warning')).not.toBeInTheDocument();
  });
});
