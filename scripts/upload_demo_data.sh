#!/usr/bin/env bash
# Sync the demo image's non-git data (Chroma index, graph snapshot, reviewed
# documents, theme/conflict layers) to the demo-data bucket so CI can build
# the image from a fresh checkout. Run after re-indexing or re-exporting:
#
#   ./scripts/upload_demo_data.sh            # uses AWS_PROFILE or data-qa
#
# Everything uploaded is derived state, reproducible from the dev corpus.
set -euo pipefail

PROFILE="${AWS_PROFILE:-data-qa}"
BUCKET="${DEMO_DATA_BUCKET:-yt-agent-demo-data-089783391188}"

sync() {
  aws s3 sync "$1" "s3://${BUCKET}/$1" --delete --profile "$PROFILE"
}

sync .yt-agent/chroma
sync .yt-agent/graph_snapshot
sync .yt-agent/documents
aws s3 cp .yt-agent/themes.json "s3://${BUCKET}/.yt-agent/themes.json" --profile "$PROFILE"
aws s3 cp .yt-agent/conflicts.json "s3://${BUCKET}/.yt-agent/conflicts.json" --profile "$PROFILE"

echo "demo data synced to s3://${BUCKET}"
