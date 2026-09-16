"""Field guides: long-form, cited write-ups the corpus writes and the reader edits.

A guide is a directory under ``guides/<slug>/`` holding the human page
(``guide.html``), the agent-facing copy (``guide.md``), the verified citations
(``claims.json``), the extraction evidence, reader commentary and an immutable
version history. Everything the app serves is a committed file; the modules
here read, verify and (via the Agent SDK) write those files.
"""
