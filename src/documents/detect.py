"""Finding the URL in a chat message, if there is one.

The chat input stays exactly what it was: a text box. A message with no URL
behaves as it always has. A message that *contains* one gains a behaviour —
the page is fetched and reviewed alongside the corpus — with no mode switch,
no second tab, and nothing for the user to turn on.

Two rules make that safe to do implicitly:

* **YouTube links keep their existing meaning.** They already scope retrieval
  to one video, and quietly turning that into "fetch and review the YouTube
  page" would break a behaviour people rely on to answer a completely
  different question.
* **Only the first non-YouTube URL is taken.** A message that pastes three
  links is asking about something the review flow cannot represent — one
  document card, one set of anchors — so it reviews the first and the rest stay
  ordinary words in the question.
"""

from __future__ import annotations

import re

#: Deliberately conservative: a scheme is required. Bare ``example.com`` is
#: far more often prose ("compare it to example.com's approach") than a request
#: to fetch something, and a fetch is not a side effect to trigger on a guess.
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\])}]+", re.IGNORECASE)

_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "music.youtube.com",
    }
)

#: Trailing characters that are almost always sentence punctuation rather than
#: part of the URL — "see https://example.com/cv." should not fetch ``/cv.``.
_TRAILING_PUNCTUATION = ".,;:!?"


def _trim(url: str) -> str:
    return url.rstrip(_TRAILING_PUNCTUATION)


def is_youtube_url(url: str) -> bool:
    """Whether a URL is a YouTube link, which the chat already handles."""
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in _YOUTUBE_HOSTS


def find_urls(text: str) -> list[str]:
    """Every http(s) URL in the message, in order, punctuation trimmed."""
    return [trimmed for raw in _URL_PATTERN.findall(text) if (trimmed := _trim(raw))]


def find_document_url(text: str) -> str | None:
    """The URL this message wants reviewed, or ``None`` for an ordinary question.

    ``None`` is the common case and the default: the chat is a chat, and this
    only ever adds a behaviour to a message that already contains a link.
    """
    for url in find_urls(text):
        if not is_youtube_url(url):
            return url
    return None
