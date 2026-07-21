"""Corpus channel grouping and insight derivation."""

from src.api.corpus import build_channels, build_insights


def video(video_id, channel_id, channel_name, chunks, summary="s"):
    return {
        "video_id": video_id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "chunk_count": chunks,
        "summary": summary,
    }


def test_groups_videos_by_channel_and_sums_chunks():
    channels = build_channels(
        [
            video("v1", "UC1", "Alpha", 10),
            video("v2", "UC1", "Alpha", 5),
            video("v3", "UC2", "Beta", 40),
        ]
    )
    assert [c["channel_name"] for c in channels] == ["Beta", "Alpha"]
    alpha = next(c for c in channels if c["channel_name"] == "Alpha")
    assert alpha["video_count"] == 2
    assert alpha["chunk_count"] == 15
    assert sorted(alpha["video_ids"]) == ["v1", "v2"]


def test_channel_without_id_still_gets_a_stable_key():
    channels = build_channels([video("v1", None, "Alpha", 3)])
    assert channels[0]["channel_id"] == "name:Alpha"


def test_flags_channel_that_dominates_the_index():
    videos = [video("v1", "UC1", "Alpha", 80), video("v2", "UC2", "Beta", 20)]
    insights = build_insights(videos, build_channels(videos), 100)
    skew = next(i for i in insights if i["kind"] == "channel_skew")
    assert skew["level"] == "warn"
    assert "80%" in skew["message"]
    assert skew["channel_id"] == "UC1"


def test_balanced_corpus_reports_no_skew():
    videos = [video(f"v{i}", f"UC{i}", f"Ch{i}", 10) for i in range(5)]
    insights = build_insights(videos, build_channels(videos), 50)
    assert not [i for i in insights if i["kind"] == "channel_skew"]


def test_flags_videos_the_summary_filter_can_never_select():
    videos = [video("v1", "UC1", "Alpha", 10, summary=None), video("v2", "UC1", "Alpha", 10)]
    insights = build_insights(videos, build_channels(videos), 20)
    missing = next(i for i in insights if i["kind"] == "missing_summaries")
    assert missing["video_ids"] == ["v1"]


def test_flags_transcripts_with_no_chunks_as_invisible():
    videos = [video("v1", "UC1", "Alpha", 0)]
    insights = build_insights(videos, build_channels(videos), 0)
    unindexed = next(i for i in insights if i["kind"] == "unindexed")
    assert unindexed["level"] == "bad"
    assert unindexed["video_ids"] == ["v1"]


def test_empty_corpus_produces_no_insights():
    assert build_insights([], [], 0) == []


def test_channel_keys_match_the_ids_stamped_on_chunks():
    """The UI filters chunks by this id, so it must be the real channel id."""
    channels = build_channels(
        [video("v1", "UCwC81boH8aT3ognPmLiE6kw", "Smart Property Investment", 10)]
    )
    assert channels[0]["channel_id"] == "UCwC81boH8aT3ognPmLiE6kw"
