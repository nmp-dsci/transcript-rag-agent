import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { Conflict, ConflictList, ConflictProbe } from "../api/types";
import { DisagreementsView } from "./DisagreementsView";

const conflicts = vi.fn();
vi.mock("../api/client", () => ({
  api: { conflicts: () => conflicts() },
}));

function conflict(): Conflict {
  return {
    conflict_id: "conflict:2",
    axis: "What font size should the body text of a resume be?",
    why_incompatible:
      "One person cannot both recommend 11 to 12 point text and 10 point text for the same resume body.",
    left: {
      video_id: "_MT4SgfQ8QY",
      chunk_id: "chunk:_MT4SgfQ8QY:5",
      channel_name: "Jean Lee",
      title: "Engineering Resume Hack (from Big Tech Hiring Manager)",
      start_seconds: 310.6,
      end_seconds: 382.72,
      position: "11 to 12 points",
      quote: "also keeping it around 11 to 12 points makes it easier to read",
      quote_ratio: 1,
      watch_url: "https://www.youtube.com/watch?v=_MT4SgfQ8QY&t=310",
    },
    right: {
      video_id: "owIP4Qr39BY",
      chunk_id: "chunk:owIP4Qr39BY:10",
      channel_name: "Greg Langstaff",
      title: "Write a Better Resume - Step-by-Step Resume Upgrade",
      start_seconds: 664.24,
      end_seconds: 748.079,
      position: "10 point font",
      quote: "I'm just going down to 10point font.",
      quote_ratio: 1,
      watch_url: "https://www.youtube.com/watch?v=owIP4Qr39BY&t=664",
    },
    similarity: 0.7004,
    cross_channel: true,
    kind: "axis",
    votes: 3,
    repeats: 3,
  };
}

/** A conflict with one true answer, and one the judge only half believed. */
function factualConflict(): Conflict {
  return {
    ...conflict(),
    conflict_id: "conflict:1",
    axis: "What are the current vacancy rates in Brisbane, Adelaide and Perth?",
    kind: "factual",
    votes: 2,
    repeats: 3,
  };
}

function probes(): ConflictProbe[] {
  return [
    {
      probe_id: "planted-flat",
      expect: "conflict",
      why: "One says never, the other says always, about the same decision.",
      verdicts: ["conflict", "conflict"],
      passed: true,
      unanimous: true,
      axis: "Should you put a cache in front of your database?",
      position_a: "Never cache.",
      position_b: "Always cache.",
    },
    {
      probe_id: "complementary-same-subject",
      expect: "complementary",
      why: "Same subject, different aspects of it.",
      verdicts: ["complementary", "complementary"],
      passed: true,
      unanimous: true,
      axis: "",
      position_a: "",
      position_b: "",
    },
  ];
}

function list(overrides: Partial<ConflictList> = {}): ConflictList {
  return {
    conflicts: [conflict()],
    stats: {
      conflicts: 1,
      cross_channel_conflicts: 1,
      unanimous_conflicts: 1,
      split_conflicts: 0,
      adjudicate_repeats: 3,
      adjudications: 1434,
      pairs_with_split_verdicts: 40,
      verdict_agreement: 0.916,
      candidates_adjudicated: 478,
      conflict_precision: 0.0084,
      channels_involved: 2,
      videos_involved: 2,
      min_quote_ratio: 1,
      rejected: {
        not_a_conflict: 477,
        minority_verdict: 11,
        unstated_axis: 0,
        quote_not_in_transcript: 0,
        duplicate_evidence: 0,
      },
    },
    probes: probes(),
    generated_at: "2026-08-10T04:50:58+00:00",
    adjudicator_model: "deepseek-v4-flash",
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    config: {},
    build_command: "uv run python -m src.cli index-conflicts",
    ...overrides,
  };
}

describe("DisagreementsView", () => {
  it("shows the axis as a question and both sides, and names no winner", async () => {
    conflicts.mockResolvedValue(list());
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(
        screen.getByText("What font size should the body text of a resume be?"),
      ).toBeTruthy(),
    );
    // Both creators, both quotes. The failure this guards is a view that shows
    // one side plus a summary, which is the blend the layer exists to prevent.
    expect(screen.getByText("Jean Lee")).toBeTruthy();
    expect(screen.getByText("Greg Langstaff")).toBeTruthy();
    expect(
      screen.getByText(/11 to 12 points makes it easier to read/),
    ).toBeTruthy();
    expect(screen.getByText(/going down to 10point font/)).toBeTruthy();
    // Nothing anywhere may declare an outcome.
    expect(document.body.textContent).not.toMatch(
      /\bwinner\b|\bcorrect answer\b/i,
    );
  });

  it("links each side to the second it was said, with one t parameter", async () => {
    conflicts.mockResolvedValue(list());
    render(<DisagreementsView />);

    const links = await screen.findAllByRole("link");
    const hrefs = links.map((link) => link.getAttribute("href") ?? "");
    expect(hrefs).toContain(
      "https://www.youtube.com/watch?v=_MT4SgfQ8QY&t=310",
    );
    // Eight videos in this corpus already carry t= in their stored source_url
    // and YouTube honours the first one it sees, so a second is a silent
    // ~22s-early link. Every href here is built from the video id instead.
    for (const href of hrefs) {
      expect(href.split("t=").length).toBeLessThanOrEqual(2);
    }
  });

  it("reports the count with the denominator it came out of", async () => {
    conflicts.mockResolvedValue(list());
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText(/1 disagreements/)).toBeTruthy(),
    );
    // A count with no denominator cannot be told apart from a lazy sweep.
    expect(
      screen.getByText(
        /478 candidate pairs adjudicated × 3 looks each = 1434 calls/,
      ),
    ).toBeTruthy();
    expect(screen.getByText(/477 not a conflict/)).toBeTruthy();
  });

  it("prints the count with its spread, not as an exact number", async () => {
    // Two builds over an identical candidate set returned 4 and then 2. A bare
    // headline cannot express that, so the spread, the firm subset and the
    // directly measured three-look sub-runs all have to be on screen.
    conflicts.mockResolvedValue(
      list({
        stats: {
          ...list().stats,
          conflicts: 2,
          firm_conflicts: 1,
          undecided_pairs: 5,
          count_sd_estimate: 0.9,
          subsample_counts_at_3: [3, 2, 2],
        },
      }),
    );
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText(/2 disagreements ± 0\.9/)).toBeTruthy(),
    );
    expect(screen.getByText(/1 firm · 5 pairs undecided/)).toBeTruthy();
    expect(screen.getByText(/yields 3, 2, 2 disagreements/)).toBeTruthy();
    // ...and the stand-in for a missing spread is gone, because the real one is
    // here. A caveat that outlives the gap it covered is its own lie.
    expect(screen.queryByText(/recorded a count, not its spread/)).toBeNull();
    expect(screen.queryByText(/spread not recorded/)).toBeNull();
  });

  it("says the spread is missing when the build predates measuring it", async () => {
    // The shipped artifact: built at three looks, before stability_statistics,
    // so it has verdict_agreement and none of the count's own error bar. Every
    // chip that carries uncertainty is gated on a number and therefore absent —
    // which leaves 90% self-agreement and two clean sweeps as the only signals
    // on the page, both of them arguing the wrong way.
    conflicts.mockResolvedValue(
      list({ stats: { ...list().stats, conflicts: 2 } }),
    );
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText(/nearer 2 ± 1 than exactly 2/)).toBeTruthy(),
    );
    // Not a zero and not a confident value: no ± is printed, and the reader is
    // put within about one of the truth by an observation, not by a decimal
    // this run never produced.
    expect(screen.queryByText(/2 disagreements ±/)).toBeNull();
    const caveat =
      screen.getByText(/recorded a count, not its spread/).closest("p") ??
      document.body;
    expect(caveat.textContent).toMatch(
      /returned 4 disagreements and then 2, two cards going 3\/3 to 0\/3/,
    );
    expect(caveat.textContent).toMatch(/about 2, give or take roughly one/);
    // The two numbers that read as confidence, disarmed in the same breath.
    expect(caveat.textContent).toMatch(
      /says nothing about how many pairs survive the vote/,
    );
    expect(caveat.textContent).toMatch(
      /3\/3 on a card is the highest score 3 looks can produce, which an evenly split pair returns about one run in 8/,
    );
    // The rejected minority verdicts are the pairs that would move it.
    expect(caveat.textContent).toMatch(
      /11 of the pairs rejected here won a look without winning a majority/,
    );
  });

  it("prints a measured spread of zero rather than treating it as missing", async () => {
    // A run whose every pair is unanimous genuinely measures ± 0.0. That is a
    // result and must print as one — the caveat is for builds that never looked.
    conflicts.mockResolvedValue(
      list({ stats: { ...list().stats, count_sd_estimate: 0 } }),
    );
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText(/1 disagreements ± 0\.0/)).toBeTruthy(),
    );
    expect(screen.queryByText(/recorded a count, not its spread/)).toBeNull();
  });

  it("does not let the judge's self-agreement stand in for the count's error bar", async () => {
    conflicts.mockResolvedValue(list());
    render(<DisagreementsView />);

    const agreement = await screen.findByText(/judge agrees with itself on/);
    // The complement travels with the headline figure: consistency on a pair is
    // not stability of a sum over pairs, and the pairs it answered differently
    // are exactly the ones on the threshold.
    expect(agreement.textContent).toMatch(
      /judge agrees with itself on 92% of pairs · answered 40 differently between looks/,
    );
    expect(agreement.getAttribute("title")).toMatch(
      /not an error bar on the count/,
    );
    expect(document.body.textContent).not.toMatch(/honest error bar/);
  });

  it("shows the calibration in both directions", async () => {
    conflicts.mockResolvedValue(list());
    render(<DisagreementsView />);

    const head = await screen.findByRole("button", { name: /calibration/ });
    // An adjudicator that says "conflict" to everything passes the planted half
    // alone, so the complementary half is what makes the count believable.
    expect(head.textContent).toMatch(/1\/1 planted contradictions surfaced/);
    expect(head.textContent).toMatch(
      /1\/1 complementary pairs correctly rejected/,
    );

    await userEvent.click(head);
    const list_ = await screen.findByRole("list");
    expect(within(list_).getByText("planted-flat")).toBeTruthy();
    expect(
      within(list_).getByText(
        /kept apart rather than fused into one statement/,
      ),
    ).toBeTruthy();
  });

  it("says how to build the layer when there is none", async () => {
    conflicts.mockResolvedValue(list({ conflicts: [] }));
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText(/No disagreement layer built yet/)).toBeTruthy(),
    );
    expect(
      screen.getByText("uv run python -m src.cli index-conflicts"),
    ).toBeTruthy();
  });

  it("puts the vote on every card, so 2/3 cannot read as 3/3", async () => {
    conflicts.mockResolvedValue(
      list({ conflicts: [conflict(), factualConflict()] }),
    );
    render(<DisagreementsView />);

    // The failure this guards is the one the layer shipped with: a card built
    // from a single draw, presented with the same confidence as one three
    // independent looks agreed on.
    // At three looks a clean sweep is the top of a three-rung ladder, so it is
    // said as one and does not get the tick colour that 9/9 earns. An evenly
    // split pair returns 3/3 one run in eight.
    const sweep = await screen.findByText(
      "agreed 3/3 looks — the most 3 can show",
    );
    expect(sweep.className).not.toMatch(/cross/);
    expect(sweep.getAttribute("title")).toMatch(/about one run in 8/);
    expect(screen.getByText("split — only 2 of 3 looks")).toBeTruthy();
    expect(
      screen.getByText(/judge agrees with itself on 92% of pairs/),
    ).toBeTruthy();
  });

  it("keeps the plain unanimous chip once the looks are fine-grained", async () => {
    // The other direction: at nine looks 9/9 is a strong claim and reads as one.
    conflicts.mockResolvedValue(
      list({ conflicts: [{ ...conflict(), votes: 9, repeats: 9 }] }),
    );
    render(<DisagreementsView />);

    const sweep = await screen.findByText("agreed 9/9 looks");
    expect(sweep.className).toMatch(/cross/);
  });

  it("marks a factual contradiction as one instead of framing it evenly", async () => {
    conflicts.mockResolvedValue(list({ conflicts: [factualConflict()] }));
    render(<DisagreementsView />);

    await waitFor(() =>
      expect(screen.getByText("factual contradiction")).toBeTruthy(),
    );
    expect(screen.getByText(/This question has one true answer/)).toBeTruthy();
    // Still no winner: the layer can check a claim against the corpus, not
    // against the world.
    expect(
      screen.getByText(/not something this corpus can settle/),
    ).toBeTruthy();
    expect(document.body.textContent).not.toMatch(
      /\bwinner\b|\bcorrect answer\b/i,
    );
  });
});
