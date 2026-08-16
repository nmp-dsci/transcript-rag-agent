import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import type { Theme, ThemeChunk, ThemeDetail, ThemeList } from "../api/types";
import { fmtSeconds, videoTimestampUrl } from "../answers/render";

/**
 * RAPTOR level 2 — the layer the per-video summaries cannot produce.
 *
 * The whole question this view has to answer at a glance is whether a theme is
 * a claim several creators make or one video's outline under a new name, so the
 * video and channel counts are on every row, the cross-video ones sort first,
 * and a single-video theme is labelled as such rather than quietly mixed in.
 */

function timestampUrl(chunk: ThemeChunk): string | null {
  return videoTimestampUrl(chunk.source_url, chunk.start_seconds);
}

function ThemeRow({
  theme,
  selected,
  onSelect,
}: {
  theme: Theme;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`th-row${selected ? " on" : ""}`}
      aria-current={selected ? "true" : undefined}
      onClick={onSelect}
    >
      <span className="th-row-title">{theme.title}</span>
      <span className="th-row-meta">
        <span className={theme.cross_video ? "th-tag cross" : "th-tag single"}>
          {theme.cross_video
            ? `${theme.video_count} videos · ${theme.channel_count} creators`
            : "1 video only"}
        </span>
        <span className="th-tag plain">{theme.member_count} chunks</span>
        <span className="th-tag plain">{theme.domain.replace(/_/g, " ")}</span>
      </span>
    </button>
  );
}

export function ThemesView() {
  const [list, setList] = useState<ThemeList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<ThemeDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [openChunk, setOpenChunk] = useState<string | null>(null);

  // Guards against an out-of-order response overwriting a newer selection.
  const requestRef = useRef(0);

  const select = (themeId: string) => {
    const request = ++requestRef.current;
    setSelected(themeId);
    setDetail(null);
    setDetailError(null);
    setOpenChunk(null);
    void api
      .theme(themeId)
      .then((payload) => {
        if (requestRef.current === request) setDetail(payload);
      })
      .catch((err) => {
        if (requestRef.current === request)
          setDetailError((err as Error).message);
      });
  };

  useEffect(() => {
    let live = true;
    void api
      .themes()
      .then((payload) => {
        if (!live) return;
        setList(payload);
        // Open the first theme straight away: an empty right pane next to a
        // list of thirty titles reads as a broken view, not as a prompt.
        const first = payload.themes[0];
        if (first) select(first.theme_id);
      })
      .catch((err) => {
        if (live) setError((err as Error).message);
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stats = list?.stats ?? {};
  const themes = list?.themes ?? [];
  const crossVideo = useMemo(
    () => themes.filter((theme) => theme.cross_video).length,
    [themes],
  );

  if (error) {
    return <p className="th-toplevel">{error}</p>;
  }
  if (!list) {
    return <p className="th-toplevel">Loading themes…</p>;
  }
  if (themes.length === 0) {
    return (
      <p className="th-toplevel">
        No theme layer built yet. Run <code>{list.build_command}</code> to
        cluster the stored chunk embeddings across every video and summarize
        each cluster.
      </p>
    );
  }

  return (
    <div className="th-layout">
      <div className="th-list">
        <div className="th-listhead">
          <span className="microlabel" style={{ color: "var(--accent2)" }}>
            raptor level 2
          </span>
          <p className="th-blurb">
            Clusters over chunk embeddings from the whole corpus at once, so a
            theme can hold the same argument from creators who never met. The
            per-video summaries one level down cannot do this — their unit is
            the video.
          </p>
          <div className="th-statline">
            <span className="th-tag cross">
              {crossVideo} of {themes.length} span 2+ videos
            </span>
            {stats.max_videos_in_a_theme ? (
              <span className="th-tag plain">
                widest {stats.max_videos_in_a_theme} videos
              </span>
            ) : null}
            {stats.chunks_clustered ? (
              <span className="th-tag plain">
                {stats.chunks_clustered} chunks · {stats.videos_covered} videos
              </span>
            ) : null}
          </div>
        </div>
        {themes.map((theme) => (
          <ThemeRow
            key={theme.theme_id}
            theme={theme}
            selected={theme.theme_id === selected}
            onSelect={() => select(theme.theme_id)}
          />
        ))}
      </div>

      <div className="th-detail">
        {detailError ? <p className="th-toplevel">{detailError}</p> : null}
        {!detail && !detailError ? (
          <p className="th-empty">Loading theme…</p>
        ) : null}
        {detail ? (
          <>
            <h3 className="th-title">{detail.theme.title}</h3>
            <p className="th-summary">{detail.theme.summary}</p>
            <div className="th-statline">
              <span
                className={
                  detail.theme.cross_video ? "th-tag cross" : "th-tag single"
                }
              >
                {detail.theme.video_count} video
                {detail.theme.video_count === 1 ? "" : "s"} ·{" "}
                {detail.theme.channel_count} creator
                {detail.theme.channel_count === 1 ? "" : "s"}
              </span>
              <span className="th-tag plain">
                {detail.theme.member_count} member chunks
              </span>
              {/* Only a warning when property material has leaked into a
                  theme that is about something else — a property theme being
                  100% property is the clustering working. */}
              {detail.theme.domain !== "property" &&
              detail.theme.property_share > 0.2 ? (
                <span className="th-tag warn">
                  {Math.round(detail.theme.property_share * 100)}% property —
                  mixed with {detail.theme.domain.replace(/_/g, " ")}
                </span>
              ) : null}
            </div>
            {!detail.theme.cross_video ? (
              <p className="th-note">
                Every member of this theme came from one video, so it repeats
                what that video's own summary already says — kept visible rather
                than hidden, because it is the case where this layer adds
                nothing.
              </p>
            ) : null}

            {detail.videos.map((group) => (
              <div className="th-group" key={group.video_id}>
                <div className="th-grouphead">
                  <span className="th-groupname">
                    {group.title || group.video_id}
                  </span>
                  <span className="th-groupmeta">
                    {group.channel_name || "Unknown creator"} ·{" "}
                    {group.member_count} chunk
                    {group.member_count === 1 ? "" : "s"}
                  </span>
                </div>
                {group.chunks.map((chunk) => {
                  const open = chunk.chunk_id === openChunk;
                  const link = timestampUrl(chunk);
                  return (
                    <div
                      className={`th-chunk${open ? " on" : ""}`}
                      key={chunk.chunk_id}
                    >
                      <button
                        type="button"
                        className="th-chunkhead"
                        aria-expanded={open}
                        onClick={() =>
                          setOpenChunk(open ? null : chunk.chunk_id)
                        }
                      >
                        <span className="th-chunkid">#c{chunk.chunk_index}</span>
                        <span className="th-chunktime">
                          {fmtSeconds(chunk.start_seconds)}–
                          {fmtSeconds(chunk.end_seconds)}
                        </span>
                        <span className="th-chunkpreview">
                          {chunk.text.slice(0, 120) || "(no text stored)"}
                        </span>
                      </button>
                      {open ? (
                        <div className="th-chunkbody">
                          <p>{chunk.text || "No text stored for this chunk."}</p>
                          {link ? (
                            <a href={link} target="_blank" rel="noreferrer">
                              ▸ open at {fmtSeconds(chunk.start_seconds)}
                            </a>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ))}
          </>
        ) : null}
      </div>
    </div>
  );
}
