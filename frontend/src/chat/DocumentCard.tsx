import { useState } from 'react';

import type { ReviewedDocument } from '../api/types';

/**
 * The page a message's URL pointed at, rendered above the answer about it.
 *
 * Extracted text, not the live page: an iframe is refused by most real sites
 * via X-Frame-Options/CSP, and a screenshot would need a headless browser. What
 * is shown here is exactly what the model read, which is the point — a reviewer
 * should be able to see whether the thing being critiqued is the thing they
 * shared.
 *
 * Sections are collapsed by default and numbered [§N] to match the citations in
 * the answer, so clicking a citation's section opens the text it refers to.
 */
export function DocumentCard({
  document,
  openSection,
}: {
  document: ReviewedDocument;
  /** Section index to expand on mount — the one a citation points at. */
  openSection?: number | null;
}) {
  const [open, setOpen] = useState<number | null>(openSection ?? null);
  const selected = document.sections_selected;
  const host = hostOf(document.url);

  return (
    <div className="doccard">
      <div className="doccard-head">
        <span className="doccard-kind">document</span>
        <span className="doccard-title" title={document.title ?? document.url}>
          {document.title || host}
        </span>
        <a className="doccard-link" href={document.url} target="_blank" rel="noreferrer noopener">
          {host} ↗
        </a>
      </div>

      {/*
        What the answer above it actually read. Stated on every review, not only
        the narrowed ones: "all 9 sections in context" is the claim that makes
        the feedback a review of the document rather than of part of it, and a
        claim nobody states is a claim nobody can check.
      */}
      <p className="doccard-detail">{document.detail ?? fallbackDetail(document)}</p>

      <div className="doccard-meta">
        <span>
          {document.sections.length} section{document.sections.length === 1 ? '' : 's'}
        </span>
        {document.reused ? <span className="doccard-tag">reused</span> : null}
        {document.truncated ? (
          <span className="doccard-warn" title="The page was longer than the fetch limit.">
            cut short — the end of this page is missing
          </span>
        ) : null}
        {document.narrowed ? (
          <span
            className="doccard-warn"
            title="Only the sections chosen for this question were sent to the model."
          >
            partly shown — {selected?.length ?? 0} of {document.sections.length} sections
            reviewed
          </span>
        ) : null}
      </div>

      <div className="doccard-sections">
        {document.sections.map((section) => {
          const isOpen = open === section.index;
          // A narrowed document sent only some sections to the model; the rest
          // are still readable here, but must not look like they were reviewed.
          const reviewed = !selected || selected.includes(section.index);
          return (
            <div
              className={`docsec${isOpen ? ' on' : ''}${reviewed ? '' : ' skipped'}`}
              key={section.index}
            >
              <button
                type="button"
                className="docsec-head"
                aria-expanded={isOpen}
                onClick={() => setOpen(isOpen ? null : section.index)}
              >
                <span className="docsec-num">§{section.index + 1}</span>
                <span className="docsec-name">
                  {section.heading || <em>opening</em>}
                </span>
                {reviewed ? null : <span className="docsec-skip">not reviewed</span>}
              </button>
              {isOpen ? <div className="docsec-body">{section.text}</div> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * The selection line for a document that predates the field, or that was
 * fetched by `GET /api/documents/:id` with no entry to pair it with.
 *
 * Derived from the section list rather than left blank, so the card never
 * silently implies a whole-document review it cannot vouch for.
 */
function fallbackDetail(document: ReviewedDocument): string {
  const total = document.sections.length;
  const selected = document.sections_selected;
  if (selected && selected.length < total) {
    return `${selected.length} of ${total} sections selected for this question`;
  }
  return `whole document — all ${total} section${total === 1 ? '' : 's'} in context`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
