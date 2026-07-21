import { useCallback, useState } from 'react';

import { api } from '../api/client';
import type { Chunk, Corpus } from '../api/types';
import { CorpusTree, type SortKey } from './CorpusTree';
import { IndexPanel } from './IndexPanel';
import { RetrievalLab } from './RetrievalLab';
import { VideoDetail } from './VideoDetail';

interface Props {
  corpus: Corpus | null;
  onCorpusChange: () => void;
  onAskAbout: (url: string) => void;
}

export function LibraryView({ corpus, onCorpusChange, onAskAbout }: Props) {
  const [sort, setSort] = useState<SortKey>('views');
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<number | null>(null);
  // video_id -> chunks; undefined means "not fetched yet", so the tree can
  // show a loading row without a second piece of state.
  const [chunks, setChunks] = useState<Record<string, Chunk[]>>({});

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

  const videos = corpus?.videos ?? [];
  const video = videos.find((item) => item.video_id === selectedVideo) ?? null;

  return (
    <section className="view" style={{ flexDirection: 'column' }}>
      <IndexPanel onIndexed={onCorpusChange} />
      <RetrievalLab
        scopeVideoId={selectedVideo}
        scopeLabel={video ? (video.title || video.video_id).slice(0, 34) : 'Whole corpus'}
        selectedChunk={
          selectedVideo != null && selectedChunk != null
            ? `${selectedVideo}:${selectedChunk}`
            : null
        }
        onSelectChunk={selectChunk}
      />

      <div className="libbody">
        {videos.length === 0 ? (
          <div className="detail">
            <div className="empty">
              <h2>The library is empty</h2>
              <p>Index a video or a channel above, then explore its chunks here.</p>
            </div>
          </div>
        ) : (
          <>
            <CorpusTree
              videos={videos}
              sort={sort}
              onSortChange={setSort}
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
              onAskAbout={onAskAbout}
            />
          </>
        )}
      </div>
    </section>
  );
}
