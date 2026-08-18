import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { Chunk, Corpus } from "../api/types";
import { ChunkGraphView } from "./ChunkGraphView";
import { DisagreementsView } from "./DisagreementsView";
import { CorpusSummary } from "./CorpusSummary";
import { CorpusTree, type SortKey } from "./CorpusTree";
import { IndexPanel } from "./IndexPanel";
import { KnowledgeGraphView } from "./KnowledgeGraphView";
import { RetrievalLab } from "./RetrievalLab";
import { useDemo } from "../demo";
import { ThemesView } from "./ThemesView";
import { VideoDetail } from "./VideoDetail";
import { type TreeFilter, applyFilter } from "./insights";
import { PIPELINE_STYLES } from "./styles";

type SubTab = "corpus" | "graph" | "knowledge" | "themes" | "conflicts";

const SUBTABS: { id: SubTab; label: string }[] = [
  { id: "corpus", label: "Corpus & retrieval" },
  { id: "graph", label: "Chunk graph" },
  { id: "knowledge", label: "Knowledge graph" },
  { id: "themes", label: "Themes" },
  { id: "conflicts", label: "Disagreements" },
];

interface Props {
  corpus: Corpus | null;
  onCorpusChange: () => void;
  onAskAbout: (url: string) => void;
  /**
   * Reported by /api/health. Optional so App can pass the health it already
   * holds; when omitted this view fetches it once rather than going without.
   */
  embeddingModel?: string | null;
}

export function PipelineView({
  corpus,
  onCorpusChange,
  onAskAbout,
  embeddingModel,
}: Props) {
  // Ingestion and the retrieval lab drive refused routes in demo mode, so
  // they are not rendered there; browsing (tree, graphs, themes) is untouched.
  const demo = useDemo();
  const [sub, setSub] = useState<SubTab>("corpus");
  // The graph fetches on mount, so once opened it stays mounted and is merely
  // hidden — switching sub-tabs must not rebuild a 281-node projection.
  const [graphMounted, setGraphMounted] = useState(false);
  const [knowledgeMounted, setKnowledgeMounted] = useState(false);
  const [themesMounted, setThemesMounted] = useState(false);
  const [conflictsMounted, setConflictsMounted] = useState(false);
  const [sort, setSort] = useState<SortKey>("views");
  const [filter, setFilter] = useState<TreeFilter | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<number | null>(null);
  // video_id -> chunks; undefined means "not fetched yet", so the tree can
  // show a loading row without a second piece of state.
  const [chunks, setChunks] = useState<Record<string, Chunk[]>>({});
  const [fetchedModel, setFetchedModel] = useState<string | null>(null);

  useEffect(() => {
    if (embeddingModel !== undefined) return;
    let live = true;
    void api
      .health()
      .then((health) => {
        if (live) setFetchedModel(health.embedding_model);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [embeddingModel]);

  const loadChunks = useCallback(
    async (videoId: string) => {
      if (chunks[videoId] !== undefined) return;
      try {
        const payload = await api.chunks(videoId);
        setChunks((current) => ({ ...current, [videoId]: payload.chunks }));
      } catch {
        setChunks((current) => ({ ...current, [videoId]: [] }));
      }
    },
    [chunks],
  );

  const selectVideo = (videoId: string) => {
    setSelectedVideo(videoId);
    setSelectedChunk(null);
    void loadChunks(videoId);
  };

  const selectChunk = (videoId: string, chunkIndex: number) => {
    setSelectedVideo(videoId);
    setSelectedChunk(chunkIndex);
    void loadChunks(videoId);
  };

  const allVideos = corpus?.videos ?? [];

  /** An insight chip narrows the tree, and jumps to the first video it names. */
  const changeFilter = (next: TreeFilter | null) => {
    setFilter(next);
    setSub("corpus");
    const first = applyFilter(allVideos, next)[0];
    if (next && first) selectVideo(first.video_id);
  };

  const viewIndexedVideo = (videoId: string) => {
    setFilter(null);
    setSub("corpus");
    selectVideo(videoId);
  };

  const showSub = (next: SubTab) => {
    if (next === "graph") setGraphMounted(true);
    if (next === "knowledge") setKnowledgeMounted(true);
    if (next === "themes") setThemesMounted(true);
    if (next === "conflicts") setConflictsMounted(true);
    setSub(next);
  };

  const videos = applyFilter(allVideos, filter);
  const video =
    allVideos.find((item) => item.video_id === selectedVideo) ?? null;

  return (
    <section className="view" style={{ flexDirection: "column" }}>
      <style>{PIPELINE_STYLES}</style>

      <CorpusSummary
        corpus={corpus}
        embeddingModel={embeddingModel ?? fetchedModel}
        filter={filter}
        onFilterChange={changeFilter}
      >
        <div className="modes" role="group" aria-label="Pipeline view">
          {SUBTABS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={sub === option.id ? "on" : ""}
              aria-current={sub === option.id ? "page" : undefined}
              onClick={() => showSub(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </CorpusSummary>

      {!demo && <IndexPanel onIndexed={onCorpusChange} onViewVideo={viewIndexedVideo} />}

      <div className="pipe-pane" hidden={sub !== "corpus"}>
        {!demo && <RetrievalLab
          scopeVideoId={selectedVideo}
          scopeLabel={
            video
              ? (video.title || video.video_id).slice(0, 34)
              : "Whole corpus"
          }
          selectedChunk={
            selectedVideo != null && selectedChunk != null
              ? `${selectedVideo}:${selectedChunk}`
              : null
          }
          onSelectChunk={selectChunk}
        />}

        <div className="libbody">
          {allVideos.length === 0 ? (
            <div className="detail">
              <div className="empty">
                <h2>The library is empty</h2>
                <p>
                  Index a video or a channel above, then explore its chunks
                  here.
                </p>
              </div>
            </div>
          ) : videos.length === 0 ? (
            <div className="detail">
              <div className="empty">
                <h2>No videos match this filter</h2>
                <p>
                  The insight you selected names videos that are no longer in
                  the corpus.
                </p>
                <button
                  type="button"
                  className="btn"
                  onClick={() => changeFilter(null)}
                >
                  Clear filter
                </button>
              </div>
            </div>
          ) : (
            <>
              <CorpusTree
                videos={videos}
                sort={sort}
                onSortChange={setSort}
                expandChannels={filter !== null}
                selectedVideo={selectedVideo}
                selectedChunk={selectedChunk}
                chunks={chunks}
                onSelectVideo={selectVideo}
                onSelectChunk={selectChunk}
              />
              <VideoDetail
                video={video}
                chunks={selectedVideo ? chunks[selectedVideo] : undefined}
                selectedChunk={selectedChunk}
                onAskAbout={demo ? undefined : onAskAbout}
              />
            </>
          )}
        </div>
      </div>

      {graphMounted ? (
        <div className="pipe-pane" hidden={sub !== "graph"}>
          <ChunkGraphView />
        </div>
      ) : null}

      {knowledgeMounted ? (
        <div className="pipe-pane" hidden={sub !== "knowledge"}>
          <KnowledgeGraphView />
        </div>
      ) : null}

      {themesMounted ? (
        <div className="pipe-pane" hidden={sub !== "themes"}>
          <ThemesView />
        </div>
      ) : null}

      {conflictsMounted ? (
        <div className="pipe-pane" hidden={sub !== "conflicts"}>
          <DisagreementsView />
        </div>
      ) : null}
    </section>
  );
}
