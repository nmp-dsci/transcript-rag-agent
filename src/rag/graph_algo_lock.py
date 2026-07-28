"""Serializes access to igraph's process-global RNG.

``igraph.set_random_number_generator`` mutates a single C-extension-level
generator shared by the whole process, not a per-call or per-thread one. Both
:func:`~src.rag.graph_view._fr_layout_coordinates` (FastAPI request thread)
and :func:`~src.rag.communities.detect_communities` seed it before an igraph
call that reads from it; without a shared lock, two overlapping calls on
different threads can interleave a seed from one with the algorithm run of
the other, corrupting both results.
"""

from __future__ import annotations

import threading

igraph_rng_lock = threading.Lock()
