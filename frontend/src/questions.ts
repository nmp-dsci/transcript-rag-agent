/**
 * The canonical example questions, in one place.
 *
 * The landing page and the chat empty state both offer starter questions, and
 * both used to carry their own hand-written copy — which had already drifted:
 * "Where do these videos disagree with each other?" was maintained twice, and
 * the landing's four were not the chat's four. The landing shows CURATED
 * verbatim; chat prefers questions naming videos actually in the corpus and
 * falls back to CORPUS_WIDE.
 */

/** Stable, corpus-independent questions — safe to print before any fetch. */
export const CURATED_QUESTIONS = [
  'How do I make my resume ATS-friendly?',
  'Where do these videos disagree with each other?',
  'Is a modular monolith a better default than microservices?',
  'What do recruiters actually look for on an AI engineer resume?',
] as const;

/** Questions that read well against any corpus, used to pad the chat list. */
export const CORPUS_WIDE_QUESTIONS = [
  'What are the main themes across the indexed transcripts?',
  'Where do these videos disagree with each other?',
] as const;
