import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ReviewedDocument } from '../api/types';
import { DocumentCard } from './DocumentCard';

function doc(overrides: Partial<ReviewedDocument> = {}): ReviewedDocument {
  return {
    id: 'doc:abc',
    url: 'https://example.com/cv',
    requested_url: 'https://example.com/cv',
    title: 'A resume',
    truncated: false,
    fetched_at: '2026-07-29T00:00:00+00:00',
    sections: [
      { index: 0, heading: null, text: 'Nathan — Sydney' },
      { index: 1, heading: 'Experience', text: 'Led a team of six.' },
      { index: 2, heading: 'Education', text: 'BSc computer science.' },
    ],
    ...overrides,
  };
}

describe('DocumentCard', () => {
  it('names the document and links to the page it came from', () => {
    render(<DocumentCard document={doc()} />);

    expect(screen.getByText('A resume')).toBeInTheDocument();
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com/cv');
  });

  it('numbers sections to match the [§N] citations in the answer', () => {
    render(<DocumentCard document={doc()} />);

    expect(screen.getByText('§1')).toBeInTheDocument();
    expect(screen.getByText('§2')).toBeInTheDocument();
    expect(screen.getByText('§3')).toBeInTheDocument();
  });

  it('keeps section text collapsed until it is asked for', () => {
    render(<DocumentCard document={doc()} />);

    expect(screen.queryByText('Led a team of six.')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Experience'));

    expect(screen.getByText('Led a team of six.')).toBeInTheDocument();
  });

  it('opens the section a citation points at', () => {
    render(<DocumentCard document={doc()} openSection={2} />);

    expect(screen.getByText('BSc computer science.')).toBeInTheDocument();
  });

  it('says when the page was cut short rather than showing a partial page as whole', () => {
    render(<DocumentCard document={doc({ truncated: true })} />);

    expect(screen.getByText(/cut short/)).toBeInTheDocument();
  });

  it('marks the sections that were not sent to the model', () => {
    render(<DocumentCard document={doc({ narrowed: true, sections_selected: [1] })} />);

    expect(screen.getByText(/partly shown/)).toBeInTheDocument();
    expect(screen.getAllByText('not reviewed')).toHaveLength(2);
  });

  it('marks nothing as unreviewed when the whole document was read', () => {
    render(<DocumentCard document={doc()} />);

    expect(screen.queryByText('not reviewed')).not.toBeInTheDocument();
    expect(screen.queryByText(/partly shown/)).not.toBeInTheDocument();
  });

  it('says when the document was reused rather than fetched again', () => {
    render(<DocumentCard document={doc({ reused: true })} />);

    expect(screen.getByText('reused')).toBeInTheDocument();
  });

  it('falls back to the host when the page has no title', () => {
    render(<DocumentCard document={doc({ title: null })} />);

    expect(screen.getAllByText(/example\.com/).length).toBeGreaterThan(0);
  });

  it('states how much of the document the answer above it read', () => {
    render(
      <DocumentCard
        document={doc({ detail: 'fetched — whole document — all 3 sections in context' })}
      />,
    );

    expect(
      screen.getByText('fetched — whole document — all 3 sections in context'),
    ).toBeInTheDocument();
  });

  it('states the selection even when the server did not send one', () => {
    render(<DocumentCard document={doc()} />);

    expect(screen.getByText(/whole document — all 3 sections in context/)).toBeInTheDocument();
  });

  it('never claims a whole-document review of a narrowed selection', () => {
    render(<DocumentCard document={doc({ sections_selected: [1], narrowed: true })} />);

    expect(screen.queryByText(/whole document/)).not.toBeInTheDocument();
    expect(screen.getByText('1 of 3 sections selected for this question')).toBeInTheDocument();
  });
});
