# Demo image: the public read-only walkthrough deployment.
#
# One container, no database, no secrets. Everything a visitor browses is
# baked in at build time — the Chroma index, the knowledge-graph snapshot,
# the committed eval runs and packs, the recorded chat history — so what a
# given image serves is exactly what its commit says it serves. The server
# runs with YT_AGENT_DEMO_MODE=1 (set here, not in infra) and refuses every
# mutating route; no provider keys exist in the environment.
#
#   docker build -t yt-agent-demo .
#   docker run --rm -p 8080:8080 yt-agent-demo    # no env file, no keys
#
# Build context requirements (all in-repo, but two are generated):
#   frontend/dist            — built in the node stage below
#   .yt-agent/chroma         — the indexed corpus (dev state, ~150 MB)
#   .yt-agent/graph_snapshot — scripts/export_graph_snapshot.py output

# ── Stage 1: frontend bundle ────────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: runtime ────────────────────────────────────────────────────
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# CPU-only torch: the lockfile pins the CUDA-bundled manylinux wheel plus
# the whole nvidia-*/triton companion set (~8 GB installed) that the demo —
# which never trains and only ever *imports* torch via sentence-transformers
# — has no use for. The +cpu build goes in first and satisfies the later
# `torch==` pin (PEP 440 local versions match); the nvidia/triton pins are
# stripped from the export so pip never pulls them.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
    && grep -vE '^(nvidia-|triton==)' requirements.txt > requirements-cpu.txt \
    && TORCH_PIN=$(grep -E '^torch==' requirements-cpu.txt) \
    && uv pip install --system --index-url https://download.pytorch.org/whl/cpu "$TORCH_PIN" \
    && uv pip install --system -r requirements-cpu.txt \
    && rm requirements.txt requirements-cpu.txt

# Code, UI, and the frozen data the read routes serve.
COPY src/ src/
COPY --from=frontend /build/dist frontend/dist
COPY dashboard/chat_history.json dashboard/chat_history.json
COPY evals/runs evals/runs
COPY experts experts
COPY .yt-agent/chroma .yt-agent/chroma
COPY .yt-agent/graph_snapshot .yt-agent/graph_snapshot
COPY .yt-agent/themes.json .yt-agent/conflicts.json .yt-agent/
COPY .yt-agent/documents .yt-agent/documents

ENV YT_AGENT_DEMO_MODE=1 \
    YT_AGENT_EMBEDDING_DEVICE=cpu \
    MLFLOW_TRACKING_URI=file:.yt-agent/mlruns \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
