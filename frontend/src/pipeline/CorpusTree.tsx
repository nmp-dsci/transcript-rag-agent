import type { Chunk, Video } from '../api/types';

export type SortKey = 'views' | 'recent' | 'chunks' | 'title';

export const SORTS: { key: SortKey; label: string }[] = [
  { key: 'views', label: 'top views' },
  { key: 'recent', label: 'most recent' },
  { key: 'chunks', label: 'most chunks' },
  { key: 'title', label: 'title' },
];

export interface Channel {
  name: string;
  videos: Video[];
  chunkTotal: number;
  viewTotal: number;
  newest: string;
}

/** Group videos by channel, then order both levels by the chosen sort. */
export function groupByChannel(videos: Video[], sort: SortKey): Channel[] {
  const byChannel = new Map<string, Video[]>();
  for (const video of videos) {
    const name = video.channel_name || 'Unknown channel';
    byChannel.set(name, [...(byChannel.get(name) ?? []), video]);
  }

  const channels: Channel[] = [...byChannel.entries()].map(([name, items]) => ({
    name,
    videos: sortVideos(items, sort),
    chunkTotal: items.reduce((total, video) => total + (video.chunk_count ?? 0), 0),
    viewTotal: items.reduce((total, video) => total + (video.view_count ?? 0), 0),
    newest: items.reduce(
      (latest, video) => (String(video.upload_date ?? '') > latest ? String(video.upload_date ?? '') : latest),
      '',
    ),
  }));

  channels.sort((a, b) => {
    if (sort === 'views') return b.viewTotal - a.viewTotal;
    if (sort === 'chunks') return b.chunkTotal - a.chunkTotal;
    if (sort === 'recent') return b.newest === a.newest ? 0 : b.newest < a.newest ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  return channels;
}

function sortVideos(videos: Video[], sort: SortKey): Video[] {
  const copy = [...videos];
  copy.sort((a, b) => {
    if (sort === 'views') return (b.view_count ?? 0) - (a.view_count ?? 0);
    if (sort === 'chunks') return (b.chunk_count ?? 0) - (a.chunk_count ?? 0);
    if (sort === 'recent') {
      const bDate = String(b.upload_date ?? '');
      const aDate = String(a.upload_date ?? '');
      return bDate === aDate ? 0 : bDate < aDate ? -1 : 1;
    }
    return (a.title ?? a.video_id).localeCompare(b.title ?? b.video_id);
  });
  return copy;
}

interface Props {
  videos: Video[];
  sort: SortKey;
  onSortChange: (sort: SortKey) => void;
  /** Open every channel — used when an insight has narrowed the tree. */
  expandChannels?: boolean;
  selectedVideo: string | null;
  selectedChunk: number | null;
  chunks: Record<string, Chunk[]>;
  onSelectVideo: (videoId: string) => void;
  onSelectChunk: (videoId: string, chunkIndex: number) => void;
}

const CHUNK_PREVIEW_LIMIT = 40;

export function CorpusTree({
  videos,
  sort,
  onSortChange,
  expandChannels = false,
  selectedVideo,
  selectedChunk,
  chunks,
  onSelectVideo,
  onSelectChunk,
}: Props) {
  const channels = groupByChannel(videos, sort);
  const totalChunks = videos.reduce((total, video) => total + (video.chunk_count ?? 0), 0);

  return (
    <nav className="tree" aria-label="Corpus">
      <div className="treehead">
        <span className="microlabel">sort</span>
        <select
          value={sort}
          onChange={(event) => onSortChange(event.target.value as SortKey)}
          aria-label="Sort corpus"
        >
          {SORTS.map((option) => (
            <option key={option.key} value={option.key}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <details open>
        <summary>
          <span className="label">
            <b>All videos</b>
          </span>
          <span className="cnt">
            {videos.length} · {totalChunks}
          </span>
        </summary>
        <div className="lvl">
          {channels.map((channel) => (
            <details key={channel.name} open={expandChannels}>
              <summary>
                <span className="label">{channel.name}</span>
                <span className="cnt">
                  {channel.videos.length} · {channel.chunkTotal}
                </span>
              </summary>
              <div className="lvl">
                {channel.videos.map((video) => {
                  const loaded = chunks[video.video_id];
                  return (
                    <details
                      key={video.video_id}
                      onToggle={(event) => {
                        if ((event.currentTarget as HTMLDetailsElement).open) {
                          onSelectVideo(video.video_id);
                        }
                      }}
                    >
                      <summary className={video.video_id === selectedVideo ? 'on' : ''}>
                        {video.thumbnail_url ? (
                          <img className="thumb" src={video.thumbnail_url} alt="" loading="lazy" />
                        ) : null}
                        <span className="label">{video.title || video.video_id}</span>
                        <span className="cnt">{video.chunk_count}</span>
                      </summary>
                      <div className="lvl">
                        {loaded === undefined ? (
                          <div className="chunkrow">
                            <span className="ct">loading chunks…</span>
                          </div>
                        ) : loaded.length === 0 ? (
                          <div className="chunkrow">
                            <span className="ct">no chunks stored</span>
                          </div>
                        ) : (
                          <>
                            {loaded.slice(0, CHUNK_PREVIEW_LIMIT).map((chunk) => (
                              <button
                                type="button"
                                key={chunk.chunk_index}
                                className={`chunkrow${
                                  video.video_id === selectedVideo &&
                                  chunk.chunk_index === selectedChunk
                                    ? ' on'
                                    : ''
                                }`}
                                onClick={() => onSelectChunk(video.video_id, chunk.chunk_index)}
                              >
                                <span className="ci">#c{chunk.chunk_index}</span>
                                <span className="ct">{chunk.text.slice(0, 42)}</span>
                              </button>
                            ))}
                            {loaded.length > CHUNK_PREVIEW_LIMIT ? (
                              <div className="chunkrow">
                                <span className="ct">
                                  … {loaded.length - CHUNK_PREVIEW_LIMIT} more in the detail pane
                                </span>
                              </div>
                            ) : null}
                          </>
                        )}
                      </div>
                    </details>
                  );
                })}
              </div>
            </details>
          ))}
        </div>
      </details>
    </nav>
  );
}
