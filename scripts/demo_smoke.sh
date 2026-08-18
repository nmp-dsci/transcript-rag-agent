#!/usr/bin/env bash
# Smoke test the live demo URL. Asserts the three properties that make the
# deployment what it claims to be: it is up, it serves the corpus, and it
# refuses writes. A deploy that fails any of these fails the workflow.
#
#   ./scripts/demo_smoke.sh https://xyz.ap-southeast-2.awsapprunner.com
set -euo pipefail

BASE="${1:?usage: demo_smoke.sh <base-url>}"
BASE="${BASE%/}"

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

# 1. Health says demo.
health="$(curl -fsS --max-time 15 "$BASE/api/health")" || fail "health unreachable"
echo "$health" | grep -q '"mode":"demo"' || fail "health mode is not demo: $health"

# 2. The corpus is served.
videos="$(curl -fsS --max-time 15 "$BASE/api/corpus" | python3 -c 'import json,sys; print(json.load(sys.stdin)["totals"]["videos"])')"
[[ "$videos" -gt 0 ]] || fail "corpus is empty"

# 3. The gate holds: a POST is refused with the demo detail.
code="$(curl -s -o /tmp/smoke_ask.json -w '%{http_code}' --max-time 15 \
  -X POST "$BASE/api/ask" -H 'Content-Type: application/json' -d '{}')"
[[ "$code" == "403" ]] || fail "POST /api/ask returned $code, expected 403"
grep -q '"demo"' /tmp/smoke_ask.json || fail "403 body is not the demo refusal"

# 4. The React shell serves.
curl -fsS --max-time 15 "$BASE/" | grep -qi '<!doctype html' || fail "index page did not render"

echo "SMOKE PASS: $BASE — mode=demo, $videos videos, writes refused"
