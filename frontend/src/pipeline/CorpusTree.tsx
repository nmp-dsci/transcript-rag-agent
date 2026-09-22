import type { Chunk, Video, WebChannelGroup, WebChunk, WebSourceSummary } from '../api/types';

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

/** The same four sorts, read against what a document actually has.
 *
 * A web document has no view count, so `views` falls back to size — the only
 * honest analogue of "biggest first". `recent` reads last_changed_at rather
 * than published_at: the tree is a view of the store, and what matters there
 * is when this copy last moved, not when the author first posted it. */
export function sortWebSources(sources: WebSourceSummary[], sort: SortKey): WebSourceSummary[] {
  const label = (item: WebSourceSummary) => (item.title || item.url).toLowerCase();
  const copy = [...sources];
  copy.sort((a, b) => {
    if (sort === 'views' || sort === 'chunks') return b.chunk_count - a.chunk_count;
    if (sort === 'recent') return b.last_changed_at.localeCompare(a.last_changed_at);
    return label(a).localeCompare(label(b));
  });
  return copy;
}

/** Channel groups in the order the chosen sort implies. */
export function sortWebChannels(channels: WebChannelGroup[], sort: SortKey): WebChannelGroup[] {
  const copy = channels.map((channel) => ({
    ...channel,
    sources: sortWebSources(channel.sources, sort),
  }));
  copy.sort((a, b) => {
    if (sort === 'views' || sort === 'chunks') return b.chunk_count - a.chunk_count;
    if (sort === 'recent') {
      const newest = (group: WebChannelGroup) =>
        group.sources.reduce((latest, item) => (item.last_changed_at > latest ? item.last_changed_at : latest), '');
      return newest(b).localeCompare(newest(a));
    }
    return a.label.localeCompare(b.label);
  });
  return copy;
}

/** How a document's recorded state reads in the tree. `live` is the ordinary
 * case and says nothing; the rest are the ones worth a badge. */
const STATE_BADGE: Record<string, string> = {
  changed: 'acc',
  moved: 'acc',
  truncated: 'warn',
  gone: 'bad',
  blocked: 'bad',
};

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
  /** The corpus's web half. Empty until a channel has been polled. */
  webChannels: WebChannelGroup[];
  selectedWebSource: string | null;
  selectedWebChunk: number | null;
  webChunks: Record<string, WebChunk[]>;
  onSelectWebSource: (key: string) => void;
  onSelectWebChunk: (key: string, chunkIndex: number) => void;
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
  webChannels,
  selectedWebSource,
  selectedWebChunk,
  webChunks,
  onSelectWebSource,
  onSelectWebChunk,
}: Props) {
  const channels = groupByChannel(videos, sort);
  const totalChunks = videos.reduce((total, video) => total + (video.chunk_count ?? 0), 0);
  const webGroups = sortWebChannels(webChannels, sort);
  const webSourceTotal = webGroups.reduce((total, group) => total + group.sources.length, 0);
  const webChunkTotal = webGroups.reduce((total, group) => total + group.chunk_count, 0);

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

      {/* Both roots start closed once there are two of them. Left open, the
          video half runs to some forty channel rows and the web root opens
          below the fold — which is how it came to read as absent. */}
      <details open={expandChannels || webSourceTotal === 0}>
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

      <details>
        <summary>
          <span className="label">
            <b>Websites &amp; docs</b>
          </span>
          <span className="cnt">
            {webSourceTotal} · {webChunkTotal}
          </span>
        </summary>
        <div className="lvl">
          {webGroups.length === 0 ? (
            <div className="chunkrow">
              <span className="ct">no watched sources yet — poll a channel above</span>
            </div>
          ) : (
            webGroups.map((group) => (
              <details key={group.channel_id} open={expandChannels}>
                <summary>
                  <span className="label">{group.label}</span>
                  <span className="cnt">
                    {group.sources.length} · {group.chunk_count}
                  </span>
                </summary>
                <div className="lvl">
                  {group.sources.map((source) => {
                    const loaded = webChunks[source.key];
                    const badge = STATE_BADGE[source.state];
                    return (
                      <details
                        key={source.key}
                        onToggle={(event) => {
                          if ((event.currentTarget as HTMLDetailsElement).open) {
                            onSelectWebSource(source.key);
                          }
                        }}
                      >
                        <summary className={source.key === selectedWebSource ? 'on' : ''}>
                          <span className="label" title={source.url}>
                            {source.title || source.url}
                          </span>
                          {badge ? (
                            <span
                              className={`badge ${badge} docstate`}
                              title={source.state_reason ?? source.state}
                            >
                              {source.state}
                            </span>
                          ) : null}
                          <span className="cnt">{source.chunk_count}</span>
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
                                    source.key === selectedWebSource &&
                                    chunk.chunk_index === selectedWebChunk
                                      ? ' on'
                                      : ''
                                  }`}
                                  onClick={() => onSelectWebChunk(source.key, chunk.chunk_index)}
                                >
                                  <span className="ci">#c{chunk.chunk_index}</span>
                                  <span className="ct">
                                    {chunk.heading ? `§ ${chunk.heading}` : chunk.text.slice(0, 42)}
                                  </span>
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
            ))
          )}
        </div>
      </details>
    </nav>
  );
}
