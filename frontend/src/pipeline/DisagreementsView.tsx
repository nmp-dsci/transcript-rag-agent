import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type {
  Conflict,
  ConflictList,
  ConflictProbe,
  ConflictSide,
  ConflictStats,
} from "../api/types";
import { fmtSeconds } from "../answers/render";

/**
 * Where the corpus contradicts itself — named, not averaged.
 *
 * Every other view in this app moves the corpus towards one answer: retrieval
 * picks the closest chunks, the theme layer writes one summary per cluster, the
 * chat tab produces one paragraph. When two creators genuinely disagree, all
 * three blend them, and the blend reads fluently while hiding that a choice was
 * made on the reader's behalf.
 *
 * So the layout here is a constraint, not decoration. The two sides are one
 * grid with two equal columns and no order that means anything — same width,
 * same styling, same everything — because any asymmetry at all (a wider column,
 * a first position, a green tick) would be read as the app picking a side, and
 * this data has no side to pick. There is no winner field in the payload to
 * render even if the layout wanted one.
 *
 * The things a reader needs in order to disbelieve any of this are on the
 * screen with it: the calibration probes (an adjudicator that says "conflict"
 * to everything would score full marks on the planted pairs and fail the
 * complementary ones), the rejection tally beside the count, so a count can be
 * read as "N out of 478 pairs looked at" rather than as a number with no
 * denominator, and — the one that was missing when this view first shipped —
 * the **vote on every card**. The first build of the layer adjudicated each
 * pair once; re-running its four cards three times each drew 1/3, 2/3, 3/3 and
 * 1/3, so three of the four were coin flips and the view had no way to say so.
 * Now a card carries how many independent looks agreed with it, and the header
 * carries how often the judge agrees with itself at all.
 *
 * The subtler version of the same bug is what :func:`Spread` exists for. An
 * artifact built before ``stability_statistics`` has no ``count_sd_estimate``,
 * and every uncertainty chip here is gated on the field being a number — so on
 * that artifact the ± is *omitted*, which is honest, and nothing else on the
 * page argues against a confident reading, which is not. Self-agreement at 90%
 * and a 3/3 on every card both read as "this count is firm" while measuring
 * something else entirely, so an absent spread has to be said out loud rather
 * than left as a gap. What is printed in that case is deliberately not a
 * number this run produced: the build recorded 3 looks per pair, three looks is
 * the regime where this layer has already been observed to return 4 and then 2
 * over an identical candidate pool, and that observation — not a modelled SD
 * dressed up as a measurement — is what the reader is given. A build that does
 * record its own spread prints the measured numbers and drops the paragraph.
 */

/** 8:34 — the moment in the video, not a duration. */
function stamp(side: ConflictSide): string {
  return fmtSeconds(side.start_seconds);
}

/**
 * One side of an axis.
 *
 * `quote` was cut out of the stored transcript server-side, so it is verbatim
 * including any ASR damage (this corpus renders "write skew" as "right skew"
 * throughout). It is shown as-is rather than tidied: the point of the quote is
 * that a reader can open the video at the timestamp beside it and hear these
 * words, and a corrected quote would not survive that check.
 */
function Side({ side, label }: { side: ConflictSide; label: string }) {
  return (
    <div className="dis-side">
      <div className="dis-sidehead">
        <span className="dis-sidelabel">{label}</span>
        <span className="dis-creator">{side.channel_name}</span>
      </div>
      <p className="dis-position">{side.position}</p>
      <blockquote className="dis-quote">“{side.quote}”</blockquote>
      <div className="dis-prov">
        <a
          href={side.watch_url}
          target="_blank"
          rel="noreferrer"
          className="dis-ts"
        >
          ▸ {side.title || side.video_id} at {stamp(side)}
        </a>
        <span
          className="dis-ratio"
          title="Share of the adjudicator's quote found verbatim in the stored chunk. The words above are the store's, not the model's."
        >
          quote {Math.round(side.quote_ratio * 100)}% in transcript
        </span>
      </div>
    </div>
  );
}

function ConflictCard({ conflict }: { conflict: Conflict }) {
  const factual = conflict.kind === "factual";
  const unanimous = conflict.votes >= conflict.repeats;
  // Three looks can only score 0, 1/3, 2/3 or 1, so a clean sweep is the top of
  // a three-rung ladder rather than evidence of certainty — a pair the judge is
  // evenly split on still returns one about one run in eight. Drawing that the
  // same green as 9/9 is the confident reading this view has to stop making, so
  // at this resolution the chip says what it is and loses the tick colour.
  const coarse = conflict.repeats > 0 && conflict.repeats <= 3;
  return (
    <article className={`dis-card${factual ? " factual" : ""}`}>
      <header className="dis-cardhead">
        <span className="microlabel" style={{ color: "var(--accent2)" }}>
          {factual ? "factual contradiction" : "axis"}
        </span>
        <h3 className="dis-axis">{conflict.axis}</h3>
        {/* A factual contradiction is not a matter of perspective, and framing
            it evenly would be a worse lie than picking a side. This layer still
            names no winner — it can check a claim against the corpus, not
            against the world — but it says which kind of thing this is. */}
        {factual ? (
          <p className="dis-factual">
            This question has one true answer, so one of these two is simply
            wrong. Which one is not something this corpus can settle — both are
            below, with timestamps, so you can check them yourself.
          </p>
        ) : null}
        <div className="dis-taglist">
          <span
            className={conflict.cross_channel ? "th-tag cross" : "th-tag warn"}
          >
            {conflict.cross_channel
              ? "two channels"
              : "same channel — not a corpus disagreement"}
          </span>
          {/* The adjudicator does not agree with itself, so a card that
              persuaded 2 of 3 looks is a weaker claim than one that persuaded
              3 of 3, and a reader cannot tell without the tally. */}
          <span
            className={
              !unanimous
                ? "th-tag warn"
                : coarse
                  ? "th-tag plain"
                  : "th-tag cross"
            }
            title={
              unanimous && coarse
                ? `How many independent adjudications called this a conflict — and ${conflict.repeats} looks is the coarsest grid this layer runs, so this is the top of a very short scale. A pair the judge is evenly split on still comes back ${conflict.votes}/${conflict.repeats} about one run in ${2 ** conflict.repeats}. The same chip at more looks per pair is a much stronger claim.`
                : "How many independent adjudications called this a conflict. A split vote means the judgement is near its own threshold."
            }
          >
            {!unanimous
              ? `split — only ${conflict.votes} of ${conflict.repeats} looks`
              : coarse
                ? `agreed ${conflict.votes}/${conflict.repeats} looks — the most ${conflict.repeats} can show`
                : `agreed ${conflict.votes}/${conflict.repeats} looks`}
          </span>
          <span className="th-tag plain">
            claim similarity {conflict.similarity.toFixed(3)}
          </span>
        </div>
      </header>

      {/* Two columns, equal, unordered. See the file docstring. */}
      <div className="dis-sides">
        <Side side={conflict.left} label="one view" />
        <Side side={conflict.right} label="the other" />
      </div>

      <footer className="dis-why">
        <span className="microlabel">why one person could not hold both</span>
        <p>{conflict.why_incompatible}</p>
      </footer>
    </article>
  );
}

/**
 * The calibration strip.
 *
 * Two planted contradictions that must surface and three complementary pairs
 * that must not, all put to the same adjudicator that produced the cards below,
 * on the same run. Without the second half a perfect score means nothing — a
 * model that answers "conflict" every time passes the planted pairs.
 */
function Probes({ probes }: { probes: ConflictProbe[] }) {
  const [open, setOpen] = useState(false);
  if (probes.length === 0) return null;
  const passed = probes.filter((probe) => probe.passed).length;
  const planted = probes.filter((probe) => probe.expect === "conflict");
  const plantedPassed = planted.filter((probe) => probe.passed).length;
  // The half that matters more: an adjudicator that answers "conflict" every
  // time passes every planted pair and fails all of these.
  const complementary = probes.filter((probe) => probe.expect !== "conflict");
  const complementaryPassed = complementary.filter(
    (probe) => probe.passed,
  ).length;
  return (
    <div className="dis-probes">
      <button
        type="button"
        className="dis-probehead"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <span
          className={passed === probes.length ? "th-tag cross" : "th-tag warn"}
        >
          calibration {passed}/{probes.length}
        </span>
        <span className="dis-probesum">
          {plantedPassed}/{planted.length} planted contradictions surfaced as
          conflicts; {complementaryPassed}/{complementary.length} complementary
          pairs correctly rejected
        </span>
        <span className="dis-chev">{open ? "▾" : "▸"}</span>
      </button>
      {open ? (
        <ul className="dis-probelist">
          {probes.map((probe) => (
            <li key={probe.probe_id} className="dis-probe">
              <div className="dis-probeline">
                <span className={probe.passed ? "th-tag cross" : "th-tag warn"}>
                  {probe.passed ? "pass" : "FAIL"}
                </span>
                <b>{probe.probe_id}</b>
                <span className="dis-probeverdicts">
                  expected {probe.expect} · got {probe.verdicts.join(", ")}
                </span>
              </div>
              <p className="dis-probewhy">{probe.why}</p>
              {probe.axis ? (
                <p className="dis-probeaxis">
                  named the axis: <i>{probe.axis}</i>
                  {probe.position_a && probe.position_b ? (
                    <>
                      {" "}
                      — “{probe.position_a}” against “{probe.position_b}”, kept
                      apart rather than fused into one statement.
                    </>
                  ) : null}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** Was the run's own run-to-run spread on the count recorded at all? */
function spreadRecorded(stats: ConflictStats): boolean {
  // `typeof === "number"` and not `> 0`: a run whose every pair is unanimous
  // genuinely measures a spread of zero, and a measured 0.0 is a result that
  // must print rather than fall through to the caveat below.
  return typeof stats.count_sd_estimate === "number";
}

/**
 * The count's error bar when the build never recorded one.
 *
 * The artifact this app ships with was built before :func:`stability_statistics`
 * existed. It has `verdict_agreement` and nothing else, so the ±, the firm
 * subset, the vote histogram and the three-look sub-runs are all absent — and
 * because each of those is gated on its field being a number, an absent
 * statistic is silently omitted. Omission is the honest half: nothing renders a
 * `0` or a `± 0.0` that was never measured. The dishonest half is what is left
 * standing once they are gone. Self-agreement at 90% and a clean sweep on every
 * card are both true, both prominent, and both point at confidence — while
 * measuring the judge's consistency on a *pair*, which is not a bound on how
 * many pairs clear the vote. A reader of that page leaves believing the layer
 * found exactly two disagreements.
 *
 * So the missing spread is stated rather than left as a gap. What is stated is
 * deliberately not a modelled `± 1.06` formatted to look like this run's own
 * measurement — the run did not measure it, and a number in the same slot and
 * the same shape as a measured one would be a worse lie than the omission it
 * replaced. What the run *does* record is its repeat count, and this project
 * has an observation at that repeat count: two builds at three looks over a
 * provably identical candidate pool returned 4 and then 2. That is history, it
 * is labelled as history, and it puts the reader within about one of the truth
 * without inventing a decimal.
 *
 * Renders nothing once a build records its own spread — at which point the ±,
 * the firm chip and the sub-run sentence carry the same information, measured.
 */
function Spread({ stats, count }: { stats: ConflictStats; count: number }) {
  if (spreadRecorded(stats)) return null;
  const repeats =
    typeof stats.adjudicate_repeats === "number" && stats.adjudicate_repeats > 0
      ? stats.adjudicate_repeats
      : null;
  const minority = stats.rejected?.minority_verdict ?? 0;
  return (
    <p className="dis-caveat">
      <b>This build recorded a count, not its spread.</b> It was made before
      this layer measured how far the count moves between runs, so there is no ±
      to print beside the {count} above
      {repeats !== null && repeats <= 3 ? (
        <>
          {" "}
          — and {repeats} looks per pair is the regime where it moves most. Two
          builds of this layer at three looks, over a provably identical
          candidate pool, returned 4 disagreements and then 2, two cards going
          3/3 to 0/3. That is an observation this project has already made
          rather than a re-measurement of this run, and it is the reason to read
          the {count} as <i>about {count}, give or take roughly one</i> instead
          of as exactly {count}.
        </>
      ) : (
        <>
          . Read it as approximate: the pairs this judge is undecided about
          carry a majority about half the time however often they are asked, so
          a re-run over the same candidates would not reliably return the same
          number.
        </>
      )}
      {minority > 0 ? (
        <>
          {" "}
          {minority} of the pairs rejected here won a look without winning a
          majority; any one of them carrying on a re-run moves the count.
        </>
      ) : null}{" "}
      Neither figure on this page that looks like an error bar is one: the
      judge&rsquo;s agreement with itself is measured on repeated looks at the
      same pair and says nothing about how many pairs survive the vote, and
      {repeats !== null ? ` ${repeats}/${repeats}` : " a clean sweep"} on a card
      is the highest score {repeats ?? "these"} looks can produce, which an
      evenly split pair returns about one run in{" "}
      {repeats !== null ? 2 ** repeats : "a handful"}. A build that records its
      own spread prints the measured ±, the firm subset and three independent
      sub-runs of its own draws in place of this paragraph.
    </p>
  );
}

export function DisagreementsView() {
  const [list, setList] = useState<ConflictList | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void api
      .conflicts()
      .then((payload) => {
        if (live) setList(payload);
      })
      .catch((err) => {
        if (live) setError((err as Error).message);
      });
    return () => {
      live = false;
    };
  }, []);

  const stats = list?.stats ?? {};
  const conflicts = list?.conflicts ?? [];
  const count = stats.conflicts ?? conflicts.length;
  const rejected = stats.rejected ?? {};
  const rejectedRows = useMemo(
    () => Object.entries(rejected).filter(([, count]) => count > 0),
    [rejected],
  );

  if (error) return <p className="th-toplevel">{error}</p>;
  if (!list) return <p className="th-toplevel">Loading disagreements…</p>;
  if (conflicts.length === 0) {
    return (
      <p className="th-toplevel">
        No disagreement layer built yet. Run <code>{list.build_command}</code>{" "}
        to pair claims from different creators about the same thing and ask of
        each pair whether one person could hold both views.
      </p>
    );
  }

  return (
    <div className="dis-wrap">
      <div className="dis-head">
        <span className="microlabel" style={{ color: "var(--accent2)" }}>
          disagreement layer
        </span>
        <p className="dis-blurb">
          Claims from different creators about the same subject, each pair put
          to one test: <b>could one person hold both of these views?</b> If yes
          it is complementary detail and is not here. Each pair is put to that
          test several times and carries only on a majority, because the judge
          does not agree with itself; the tally is on every card. What survives
          is an axis and two sides — never a verdict, because the corpus does
          not have one.
        </p>
        <div className="dis-statline">
          <span className="th-tag cross">
            {count} disagreements
            {typeof stats.count_sd_estimate === "number"
              ? ` ± ${stats.count_sd_estimate.toFixed(1)}`
              : null}
          </span>
          {/* The count never appears alone. When the build measured its spread
              the ± above is the qualifier; when it did not, this is — because
              an omitted ± leaves a bare integer, and a bare integer beside 90%
              self-agreement and two unanimous cards reads as a firm 2. Worded
              as a bracket rather than a decimal on purpose: this run produced
              no decimal, and see `Spread` for why one is not invented. */}
          {spreadRecorded(stats) ? null : (
            <span
              className="th-tag warn"
              title="This build predates the spread measurement, so the count carries no ± of its own. It is not a spread of zero — see the note below the tally."
            >
              spread not recorded — nearer {count} ± 1 than exactly {count}
            </span>
          )}
          {/* The count is a measurement, and a measurement without its spread
              invites being read as exact. Two builds of this layer over an
              identical candidate set returned 4 and then 2 — so how many of
              these the judge is actually sure about is the number that carries
              the claim, and it is printed beside the headline rather than
              inferred from the per-card tallies. */}
          {typeof stats.firm_conflicts === "number" ? (
            <span
              className={
                stats.firm_conflicts === count ? "th-tag cross" : "th-tag warn"
              }
              title="Carried by at least two thirds of looks, not merely a majority. A pair the judge is evenly split on carries a majority half the time however often it is asked."
            >
              {stats.firm_conflicts} firm
              {typeof stats.undecided_pairs === "number"
                ? ` · ${stats.undecided_pairs} pairs undecided`
                : null}
            </span>
          ) : null}
          {/* The denominator stays worded as the count's own qualifier: "4
              disagreements" alone is the number this layer must never be
              reducible to, since a run that looked at six pairs and one that
              looked at 478 print the same 4. The repeat count rides alongside
              it because a pair looked at once and a pair looked at three times
              are not the same measurement either. */}
          <span className="th-tag plain">
            from {stats.candidates_adjudicated ?? "—"} candidate pairs
            adjudicated × {stats.adjudicate_repeats ?? 1} looks each ={" "}
            {stats.adjudications ?? "—"} calls
          </span>
          <span
            className="th-tag plain"
            title="Conflicts over pairs adjudicated. Proposing more candidates can only lower it, so there is no number here that rises by being noisier."
          >
            precision{" "}
            {typeof stats.conflict_precision === "number"
              ? stats.conflict_precision.toFixed(3)
              : "—"}
          </span>
          <span className="th-tag plain">
            {stats.channels_involved ?? 0} channels ·{" "}
            {stats.videos_involved ?? 0} videos
          </span>
          {/* Published because the first build of this layer drew once per pair
              and could not report it at all — three of the four conflicts it
              shipped did not reproduce. But it is *not* the error bar on the
              count, and shown alone at 90% it reads as one: it measures the
              judge repeating itself on a pair, and the count is a sum over
              pairs. So the complement travels with it. The pairs the judge
              answers differently between looks are exactly the pairs sitting on
              the threshold, and one of them moving moves the headline by one —
              which is the direction this number actually points. */}
          {typeof stats.verdict_agreement === "number" ? (
            <span
              className="th-tag plain"
              title="Share of candidate pairs that drew the same verdict on every repeat. It is the adjudicator's consistency on a pair, not an error bar on the count: the pairs in the other share are the ones near the threshold, and any one of them flipping changes the count by one."
            >
              judge agrees with itself on{" "}
              {Math.round(stats.verdict_agreement * 100)}% of pairs
              {typeof stats.pairs_with_split_verdicts === "number"
                ? ` · answered ${stats.pairs_with_split_verdicts} differently between looks`
                : null}
            </span>
          ) : null}
          {typeof stats.min_quote_ratio === "number" ? (
            <span className="th-tag plain">
              every quote ≥ {Math.round(stats.min_quote_ratio * 100)}% verbatim
            </span>
          ) : null}
          {/* A conflict count is a measurement of a population, and this one
              moved from 1372 to 1792 chunks while the layer was being built.
              Without the corpus on screen, two counts taken a day apart cannot
              be told apart from two counts over two different corpora. */}
          {list.corpus?.chunks ? (
            <span
              className="th-tag plain"
              title="The corpus this count was taken over, digested over chunk id and text. Two runs are only comparable if this matches."
            >
              over {list.corpus.videos} videos / {list.corpus.chunks} chunks
              {list.corpus.digest
                ? ` · ${list.corpus.digest.slice(0, 8)}`
                : null}
            </span>
          ) : null}
        </div>
        {rejectedRows.length > 0 ? (
          <p className="dis-rejected">
            Rejected on the way here:{" "}
            {rejectedRows
              .map(([reason, count]) => `${count} ${reason.replace(/_/g, " ")}`)
              .join(" · ")}
            . A low count is a result, not a gap — the failure this layer has to
            avoid is inventing a disagreement, not missing one.
          </p>
        ) : null}
        {/* Either the spread was measured, in which case it is already on the
            chips above and this renders nothing, or it was not and the reader
            is told so in words. */}
        <Spread stats={stats} count={count} />
        {/* The spread, measured rather than modelled: this run's draws split
            into disjoint groups of three and the whole resolution re-run on
            each, so these are what independent three-look builds of this same
            data would have printed. */}
        {stats.subsample_counts_at_3 &&
        stats.subsample_counts_at_3.length > 1 ? (
          <p className="dis-rejected">
            Split this run's looks into independent groups of three and the same
            data yields {stats.subsample_counts_at_3.join(", ")} disagreements.
            That spread is the judge, not the corpus — which is why the headline
            carries a ± and why every card shows how many looks backed it.
          </p>
        ) : null}
        <Probes probes={list.probes} />
      </div>

      <div className="dis-list">
        {conflicts.map((conflict) => (
          <ConflictCard key={conflict.conflict_id} conflict={conflict} />
        ))}
      </div>

      <p className="dis-foot">
        Adjudicated by {list.adjudicator_model || "an LLM"}; candidates found
        with {list.embedding_model || "the shipped bi-encoder"}. Quotes are cut
        from the stored transcript server-side and are verbatim including
        transcription errors — the timestamp beside each one opens the video at
        the second it was said, so every side here can be checked against the
        source.
      </p>
    </div>
  );
}
