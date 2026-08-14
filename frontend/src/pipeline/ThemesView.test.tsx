import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { Theme, ThemeDetail, ThemeList } from "../api/types";
import { ThemesView } from "./ThemesView";

const themes = vi.fn();
const theme = vi.fn();
vi.mock("../api/client", () => ({
  api: {
    themes: () => themes(),
    theme: (id: string) => theme(id),
  },
}));

function crossVideoTheme(): Theme {
  return {
    theme_id: "theme:0",
    title:
      "Tailor the résumé to the posting and let the ATS read it — a claim four creators make",
    summary:
      "Every creator here starts from the job posting rather than the candidate's history.",
    member_count: 79,
    video_count: 12,
    channel_count: 11,
    cross_video: true,
    domain: "job_search",
    domain_mix: { job_search: 1 },
    property_share: 0,
    videos: [
      {
        video_id: "v1",
        title: "How to Write a Winning Tech Resume",
        channel_name: "Anthony D. Mays",
        member_count: 2,
        domain: "job_search",
      },
      {
        video_id: "v2",
        title: "How to make a Dev resume that actually gets you hired",
        channel_name: "Anthony Sistilli",
        member_count: 1,
        domain: "job_search",
      },
    ],
  };
}

function singleVideoTheme(): Theme {
  return {
    ...crossVideoTheme(),
    theme_id: "theme:1",
    title: "One video's outline",
    member_count: 33,
    video_count: 1,
    channel_count: 1,
    cross_video: false,
    videos: [
      {
        video_id: "v9",
        title: "Balancing Coupling in Software Design",
        channel_name: "DDD Europe",
        member_count: 33,
        domain: "system_design",
      },
    ],
  };
}

function list(overrides: Partial<ThemeList> = {}): ThemeList {
  return {
    themes: [crossVideoTheme(), singleVideoTheme()],
    stats: {
      themes: 2,
      cross_video_themes: 1,
      single_video_themes: 1,
      max_videos_in_a_theme: 12,
      chunks_clustered: 1329,
      videos_covered: 53,
    },
    generated_at: "2026-08-10T00:00:00+00:00",
    summary_model: "deepseek-v4-flash",
    embedding_model: "all-MiniLM-L6-v2",
    build_command: "uv run python -m src.cli index-themes",
    ...overrides,
  };
}

function detail(): ThemeDetail {
  const base = crossVideoTheme();
  return {
    theme: base,
    videos: [
      {
        ...base.videos[0]!,
        chunks: [
          {
            chunk_id: "chunk:v1:3",
            chunk_index: 3,
            probability: 0.98,
            text: "Read the posting, pull the exact words out of it, and put them in your bullets.",
            start_seconds: 120,
            end_seconds: 180,
            source_url: "https://www.youtube.com/watch?v=v1",
          },
          {
            chunk_id: "chunk:v1:4",
            chunk_index: 4,
            probability: 0.91,
            text: "The screener is looking for the title they advertised.",
            start_seconds: 181,
            end_seconds: 240,
            source_url: "https://www.youtube.com/watch?v=v1",
          },
        ],
      },
      {
        ...base.videos[1]!,
        chunks: [
          {
            chunk_id: "chunk:v2:8",
            chunk_index: 8,
            probability: 0.77,
            text: "Different creator, same advice: mirror the job description.",
            start_seconds: 400,
            end_seconds: 460,
            source_url: "https://www.youtube.com/watch?v=v2",
          },
        ],
      },
    ],
  };
}

describe("ThemesView", () => {
  it("names the build command when no theme layer exists yet", async () => {
    themes.mockResolvedValue(list({ themes: [], stats: {} }));
    render(<ThemesView />);
    expect(await screen.findByText(/No theme layer built yet/)).toBeInTheDocument();
    expect(
      screen.getByText("uv run python -m src.cli index-themes"),
    ).toBeInTheDocument();
  });

  it("surfaces a failed request", async () => {
    themes.mockRejectedValue(new Error("themes unavailable"));
    render(<ThemesView />);
    expect(await screen.findByText("themes unavailable")).toBeInTheDocument();
  });

  it("labels each theme with the videos and creators it spans", async () => {
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(detail());
    render(<ThemesView />);

    expect(await screen.findByText("1 of 2 span 2+ videos")).toBeInTheDocument();
    const listPane = document.querySelector(".th-list") as HTMLElement;
    expect(
      within(listPane).getByText("12 videos · 11 creators"),
    ).toBeInTheDocument();
    // The case the hypothesis fails on is labelled, not hidden.
    expect(within(listPane).getByText("1 video only")).toBeInTheDocument();
  });

  it("opens the first theme and groups its members by video", async () => {
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(detail());
    render(<ThemesView />);

    await screen.findByText("1 of 2 span 2+ videos");
    expect(theme).toHaveBeenCalledWith("theme:0");

    const panel = document.querySelector(".th-detail") as HTMLElement;
    const groups = panel.querySelectorAll(".th-group");
    expect(groups.length).toBe(2);
    expect(
      within(panel).getByText("How to Write a Winning Tech Resume"),
    ).toBeInTheDocument();
    expect(within(panel).getByText(/Anthony Sistilli/)).toBeInTheDocument();
  });

  it("clicking a member chunk reveals its transcript text and timestamp link", async () => {
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(detail());
    render(<ThemesView />);
    await screen.findByText("1 of 2 span 2+ videos");

    const heads = document.querySelectorAll(".th-chunkhead");
    expect(heads.length).toBe(3);
    expect(document.querySelectorAll(".th-chunkbody").length).toBe(0);
    await userEvent.click(heads[0]!);

    const body = document.querySelector(".th-chunkbody") as HTMLElement;
    expect(body).not.toBeNull();
    expect(
      within(body).getByText(
        "Read the posting, pull the exact words out of it, and put them in your bullets.",
      ),
    ).toBeInTheDocument();
    const link = within(body).getByRole("link", { name: /open at 2:00/ });
    expect(link).toHaveAttribute(
      "href",
      "https://www.youtube.com/watch?v=v1&t=120s",
    );
  });

  // Nine of this corpus's stored source_urls are share links that already
  // carry a t=. Appending a second one gave a row labelled 1:13 an href the
  // player honoured as 0:51, because YouTube reads the first t it is given.
  it("links to the clock it displays even when source_url already has a t", async () => {
    const withSeek = detail();
    withSeek.videos[0]!.chunks[0] = {
      ...withSeek.videos[0]!.chunks[0]!,
      start_seconds: 73,
      end_seconds: 147,
      source_url:
        "https://www.youtube.com/watch?v=by8wrrXW3So&t=51s&pp=ygUScmVzdW1lIGFpIGVuZ2luZWVy",
    };
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(withSeek);
    render(<ThemesView />);
    await screen.findByText("1 of 2 span 2+ videos");

    await userEvent.click(document.querySelectorAll(".th-chunkhead")[0]!);
    const body = document.querySelector(".th-chunkbody") as HTMLElement;
    const link = within(body).getByRole("link", { name: /open at 1:13/ });
    const href = link.getAttribute("href")!;
    const params = new URLSearchParams(href.split("?")[1]);
    expect(params.get("t")).toBe("73s");
    expect([...params.keys()].filter((key) => key === "t").length).toBe(1);
    // The rest of the share link survives untouched.
    expect(params.get("v")).toBe("by8wrrXW3So");
    expect(href).toContain("pp=ygUScmVzdW1lIGFpIGVuZ2luZWVy");
  });

  it("selecting another theme fetches it", async () => {
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(detail());
    render(<ThemesView />);
    await screen.findByText("1 of 2 span 2+ videos");

    await userEvent.click(screen.getByText("One video's outline"));
    expect(theme).toHaveBeenLastCalledWith("theme:1");
  });

  it("says plainly when a theme adds nothing over its video's own summary", async () => {
    themes.mockResolvedValue(list());
    const single = singleVideoTheme();
    theme.mockResolvedValue({
      theme: single,
      videos: [{ ...single.videos[0]!, chunks: [] }],
    });
    render(<ThemesView />);
    await screen.findByText("1 of 2 span 2+ videos");

    expect(
      await screen.findByText(/Every member of this theme came from one video/),
    ).toBeInTheDocument();
  });

  it("never truncates a theme title or summary with CSS", async () => {
    // jsdom has no layout, so a clipped title still reads back in full from the
    // DOM. What is assertable here is the rule itself: nothing on the text
    // elements may set nowrap, which is how this app has clipped titles before.
    themes.mockResolvedValue(list());
    theme.mockResolvedValue(detail());
    render(<ThemesView />);
    await screen.findByText("1 of 2 span 2+ videos");

    const clipped = [
      ...document.querySelectorAll(
        ".th-row-title, .th-title, .th-summary, .th-groupname, .th-tag",
      ),
    ].filter((node) => getComputedStyle(node).whiteSpace === "nowrap");
    expect(clipped).toEqual([]);
  });
});
