/** The front door — what this is, what it does, what it measures.
 *
 * Mirrors data-qa-agent's landing flow: full-screen card, benefit grid,
 * example questions, outcome metrics, one Enter button. No auth here (the
 * app has none): the landing is pure narrative plus a door, and the App
 * decides when it shows (first visit per session; deep links bypass it).
 *
 * Copy principles carried over from that page: lead with the question, never
 * the feature; the outcome metrics are properties of the product ("every
 * claim cited"), not deployment counters; the fine print is honest about
 * what is replayed versus live.
 */

import { useEffect } from 'react';

import { captureEvent } from '../analytics';
import type { Corpus } from '../api/types';
import { Logo } from '../Logo';

/** Real recorded questions, in the shape people actually type them. */
const EXAMPLE_QUESTIONS = [
  'How do I make my resume ATS-friendly?',
  'Where do these videos disagree with each other?',
  'Is a modular monolith a better default than microservices?',
  'What do recruiters actually look for on an AI engineer resume?',
];

const BENEFITS = [
  {
    step: '01 · ASK',
    title: 'Q&A with receipts',
    body:
      'Every answer cites the exact video second it came from. Browse recorded ' +
      'conversations across expert channels — resumes, system design, AI engineering.',
    visual: (
      <>
        “How do I make my resume ATS-friendly?”
        <br />→ 10 chunks · <b>7 sources</b> · every claim <b>¹ ²</b> links to its timestamp
      </>
    ),
  },
  {
    step: '02 · SEE',
    title: 'The pipeline, visible',
    body:
      'Chunk similarity graph, a 2,000-entity knowledge graph, cross-video themes, and ' +
      'where experts genuinely disagree — the retrieval path is never a black box.',
    visual: (
      <>
        <b>chunk graph</b> · every indexed chunk
        <br />
        <b>knowledge graph</b> · 2,000 entities
        <br />
        <b>30 themes</b> · <b>7 disagreements</b>
      </>
    ),
  },
  {
    step: '03 · MEASURE',
    title: 'Scored, not vibes',
    body:
      'Four retrieval setups race on 20 golden questions. A depth-aware judge scores ' +
      'faithfulness, relevancy, precision, insight, specificity and coverage — with the ' +
      'weights on the table.',
    visual: (
      <>
        faithful <span className="l-bar" style={{ width: 52 }} />
        0.95
        <br />
        relevant <span className="l-bar" style={{ width: 37 }} />
        0.66
        <br />
        precision <span className="l-bar" style={{ width: 43 }} />
        0.77 · <span className="g">win rate 12/20</span>
      </>
    ),
  },
  {
    step: '04 · DISTIL',
    title: 'Experts → rubric packs',
    body:
      'An automated research loop distils creators’ advice into reviewable rubric packs — ' +
      'then proves itself against a held-out expert the system never saw.',
    visual: (
      <>
        4 packs · resume / job-search / system-design / app-arch
        <br />
        criteria recall <b>0.42</b> vs hand-built <b>0.26</b> ·{' '}
        <span className="w">held-out · 0 leaks</span>
      </>
    ),
  },
];

/** Product properties, not deployment telemetry — true on any corpus. */
const IMPACT = [
  { value: 'every claim', label: 'cited to a video second' },
  { value: '6 metrics', label: 'no hidden composite' },
  { value: '4 setups', label: 'judged head-to-head' },
  { value: '0 leaks', label: 'held-out eval, verified' },
];

interface Props {
  corpus: Corpus | null;
  /** null while /api/health is in flight — the fine print waits for truth. */
  demo: boolean | null;
  onEnter: () => void;
}

export function Landing({ corpus, demo, onEnter }: Props) {
  useEffect(() => {
    captureEvent('demo_landing_view');
  }, []);

  const counts = corpus?.totals.videos
    ? `${corpus.totals.videos} videos · ${corpus.totals.chunks.toLocaleString()} chunks indexed`
    : 'corpus loading…';

  return (
    <div className="landing" role="main">
      <div className="l-mark">
        <Logo />
      </div>
      <h1 className="l-title">
        transcript<em>·lab</em>
      </h1>
      <p className="l-tag">RAG you can audit.</p>

      <div className="l-chips" aria-hidden="true">
        <span className="l-chip on">{counts}</span>
        <span className="l-chip demo">replays real judged runs</span>
        <span className="l-chip on">no account needed</span>
      </div>

      <button
        type="button"
        className="l-enter"
        onClick={() => {
          captureEvent('demo_enter_click');
          onEnter();
        }}
      >
        Enter {demo === false ? 'workbench' : 'demo'}&ensp;→
      </button>
      <p className="l-fine">
        {demo === null
          ? '\u00a0'
          : demo
            ? 'Read-only walkthrough — answers were produced against the live corpus and ' +
              'judged with RAGAS; asking new questions is disabled here. Anonymous usage ' +
              'analytics only.'
            : 'Dev build — everything live: ask, judge, index, eval runs.'}
      </p>

      <div className="l-grid">
        {BENEFITS.map((benefit) => (
          <div className="l-bene" key={benefit.step}>
            <span className="l-step">{benefit.step}</span>
            <h4>{benefit.title}</h4>
            <p>{benefit.body}</p>
            <div className="l-visual">{benefit.visual}</div>
          </div>
        ))}
      </div>

      <div className="l-qs">
        {EXAMPLE_QUESTIONS.map((question) => (
          <span className="l-q" key={question}>
            {question}
          </span>
        ))}
      </div>

      <div className="l-metrics">
        {IMPACT.map((item) => (
          <div className="l-metric" key={item.label}>
            <b>{item.value}</b>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
