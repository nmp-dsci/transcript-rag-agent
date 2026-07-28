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
});
