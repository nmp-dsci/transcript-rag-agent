import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { Chunk, Corpus, WebChunk, WebSourceList } from "../api/types";
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
import { WebSourceDetail } from "./WebSourceDetail";
import { ChannelsPanel } from "./ChannelsPanel";
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
  // The corpus's web half. Fetched separately from /api/corpus rather than
  // folded into it, because the two halves live in different collections and
  // a web poll must not invalidate the video corpus (or the reverse).
  const [web, setWeb] = useState<WebSourceList | null>(null);
  const [webChunks, setWebChunks] = useState<Record<string, WebChunk[]>>({});
  const [selectedWebSource, setSelectedWebSource] = useState<string | null>(null);
  const [selectedWebChunk, setSelectedWebChunk] = useState<number | null>(null);

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

  const loadWeb = useCallback(async () => {
    try {
      setWeb(await api.webSources());
    } catch {
      setWeb({ channels: [], totals: { channels: 0, sources: 0, chunks: 0 } });
    }
  }, []);

  useEffect(() => {
    if (demo) return;
    void loadWeb();
  }, [demo, loadWeb]);

  /** A finished job can have changed either half, so refresh both.
   *
   * The cached chunk lists go with it: a re-poll that re-chunked a document
   * leaves this instance holding the previous revision's text, and nothing
   * else would ever invalidate it. */
  const refreshCorpus = useCallback(() => {
    onCorpusChange();
    setWebChunks({});
    void loadWeb();
  }, [onCorpusChange, loadWeb]);

  const loadWebChunks = useCallback(
    async (key: string) => {
      if (webChunks[key] !== undefined) return;
      try {
        const payload = await api.webChunks(key);
        setWebChunks((current) => ({ ...current, [key]: payload.chunks }));
      } catch {
        setWebChunks((current) => ({ ...current, [key]: [] }));
      }
    },
    [webChunks],
  );

  const selectWebSource = (key: string) => {
    setSelectedWebSource(key);
    setSelectedWebChunk(null);
    setSelectedVideo(null);
    void loadWebChunks(key);
  };

  const selectWebChunk = (key: string, chunkIndex: number) => {
    setSelectedWebSource(key);
    setSelectedWebChunk(chunkIndex);
    setSelectedVideo(null);
    void loadWebChunks(key);
  };

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
    setSelectedWebSource(null);
    void loadChunks(videoId);
  };

  const selectChunk = (videoId: string, chunkIndex: number) => {
    setSelectedVideo(videoId);
    setSelectedChunk(chunkIndex);
    setSelectedWebSource(null);
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
  const webChannels = web?.channels ?? [];
  const webSource =
    webChannels
      .flatMap((group) => group.sources)
      .find((item) => item.key === selectedWebSource) ?? null;
  // The corpus is only empty when *both* halves are. A poll that brought in
  // web documents before any video was indexed still has something to show.
  const corpusEmpty = allVideos.length === 0 && webChannels.length === 0;

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

      {!demo && <IndexPanel onIndexed={refreshCorpus} onViewVideo={viewIndexedVideo} />}
      {!demo && <ChannelsPanel onPolled={refreshCorpus} />}

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
          {corpusEmpty ? (
            <div className="detail">
              <div className="empty">
                <h2>The library is empty</h2>
                <p>
                  Index a video or a channel above, then explore its chunks
                  here.
                </p>
              </div>
            </div>
          ) : videos.length === 0 && filter !== null ? (
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
                webChannels={webChannels}
                selectedWebSource={selectedWebSource}
                selectedWebChunk={selectedWebChunk}
                webChunks={webChunks}
                onSelectWebSource={selectWebSource}
                onSelectWebChunk={selectWebChunk}
              />
              {webSource ? (
                <WebSourceDetail
                  source={webSource}
                  chunks={selectedWebSource ? webChunks[selectedWebSource] : undefined}
                  selectedChunk={selectedWebChunk}
                  onAskAbout={demo ? undefined : onAskAbout}
                />
              ) : (
                <VideoDetail
                  video={video}
                  chunks={selectedVideo ? chunks[selectedVideo] : undefined}
                  selectedChunk={selectedChunk}
                  onAskAbout={demo ? undefined : onAskAbout}
                />
              )}
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
