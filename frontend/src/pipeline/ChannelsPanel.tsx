import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { ChannelList, WatchedChannel } from "../api/types";

interface Props {
  /** Called when a poll finishes, so the corpus counts above can refresh. */
  onPolled: () => void;
}

/** What each channel kind is, in one word the reader can act on. */
const KIND_LABEL: Record<WatchedChannel["kind"], string> = {
  rss: "RSS",
  atom: "Atom",
  sitemap: "sitemap",
  github_docs: "repo docs",
  url_list: "pinned",
};

/** A channel's state, distinguishing the three that need different actions:
 * switch it on in the file, reset it, or simply wait. */
function statusOf(channel: WatchedChannel): { badge: string; text: string } {
  if (channel.disabled_reason) {
    return { badge: "bad", text: channel.disabled_reason };
  }
  if (!channel.enabled) {
    return { badge: "warn", text: "off — set enabled: true in channels.yaml" };
  }
  if (channel.last_error) {
    return { badge: "warn", text: channel.last_error };
  }
  if (!channel.last_polled_at) {
    return { badge: "warn", text: "never polled" };
  }
  return { badge: "good", text: `checked every ${Math.round(channel.interval_hours)}h` };
}

function relative(iso: string | null): string {
  if (!iso) return "never";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "never";
  const hours = (Date.now() - then) / 3_600_000;
  if (hours < 1) return "just now";
  if (hours < 24) return `${Math.round(hours)}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** The watched text sources, and a way to poll them.
 *
 * Deliberately separate from the video ingest form above it: a channel is a
 * source of URLs that this pipeline checks on a schedule, while a video is a
 * link somebody pastes once. Conflating them would hide which half of the
 * corpus a given run touched. */
export function ChannelsPanel({ onPolled }: Props) {
  const [list, setList] = useState<ChannelList | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setList(await api.channels());
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const poll = async (channelIds: string[], label: string) => {
    setBusy(label);
    setNote(null);
    setFailure(null);
    try {
      await api.pollChannels(channelIds);
      // The job runs on the shared ingestion queue, so its progress appears
      // in the queue above rather than being re-reported here.
      setNote(`Queued a poll of ${label}. Watch it in the queue above.`);
      onPolled();
      await load();
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  };

  if (failure && !list) {
    return (
      <section className="panel chan">
        <h2>Watched sources</h2>
        <p className="chan-empty">Could not read the channel register: {failure}</p>
      </section>
    );
  }

  const channels = list?.channels ?? [];
  const totals = list?.totals ?? { channels: 0, enabled: 0, sources: 0 };

  return (
    <section className="panel chan">
      <div className="chan-head">
        <h2>Watched sources</h2>
        <span className="chan-totals">
          {totals.enabled} of {totals.channels} on · {totals.sources} documents
        </span>
        <button
          type="button"
          className="chan-poll"
          disabled={busy !== null || totals.enabled === 0}
          onClick={() => void poll([], "every enabled channel")}
        >
          {busy ? "Queuing…" : "Poll all"}
        </button>
      </div>

      {list?.error && <p className="chan-empty">channels.yaml could not be read: {list.error}</p>}
      {note && <p className="chan-note">{note}</p>}
      {failure && <p className="chan-note bad">{failure}</p>}

      {channels.length === 0 && !list?.error ? (
        <p className="chan-empty">
          No channels configured. Add one with <code>channels add</code>, then poll it here.
        </p>
      ) : (
        <ul className="chan-list">
          {channels.map((channel) => {
            const status = statusOf(channel);
            return (
              <li key={channel.id} className={channel.enabled ? "chan-row" : "chan-row off"}>
                <span className="chan-name" title={channel.url ?? channel.id}>
                  {channel.label}
                </span>
                <span className="chan-kind">{KIND_LABEL[channel.kind]}</span>
                {channel.body_in_feed && (
                  <span className="chan-kind feed" title="The feed carries the article, so the page is never fetched">
                    full text
                  </span>
                )}
                <span className="chan-count">{channel.sources}</span>
                <span className={`chan-badge ${status.badge}`} title={status.text}>
                  {relative(channel.last_polled_at)}
                </span>
                <button
                  type="button"
                  className="chan-one"
                  disabled={busy !== null}
                  onClick={() => void poll([channel.id], channel.label)}
                >
                  Poll
                </button>
              </li>
            );
          })}
        </ul>
      )}
      <p className="chan-foot">
        Text sources only — videos stay a manual paste in the form above, so nothing here spends a
        transcript credit. Web documents live in their own collections and are queried alongside
        transcripts, never mixed into them.
      </p>
    </section>
  );
}
