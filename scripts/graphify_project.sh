#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEMANTIC=0
if [[ "${1:-}" == "--semantic" ]]; then
  SEMANTIC=1
fi

if ! command -v graphify >/dev/null 2>&1; then
  echo "graphify command not found. Install with: uv tool install graphifyy" >&2
  exit 127
fi

EXCLUDES=(
  --exclude ".claude/"
  --exclude ".opencode/node_modules/"
  --exclude "TrainV3/target/"
  --exclude "DesignAssets/"
  --exclude "assets/"
  --exclude "outputs/"
  --exclude "output/"
)

BACKEND_ARGS=()
if [[ "$SEMANTIC" != "1" ]]; then
  EXCLUDES+=(
    --exclude "*.md"
    --exclude "*.html"
    --exclude "*.csv"
    --exclude "requirements.txt"
  )
else
  BACKEND_ARGS=(
    --backend "${GRAPHIFY_BACKEND:-deepseek}"
    --mode "${GRAPHIFY_MODE:-deep}"
    --max-concurrency "${GRAPHIFY_MAX_CONCURRENCY:-1}"
    --token-budget "${GRAPHIFY_TOKEN_BUDGET:-80000}"
  )
fi

mkdir -p graphify-out
rm -f graphify-out/graph.json graphify-out/manifest.json

GRAPHIFY_MAX_WORKERS="${GRAPHIFY_MAX_WORKERS:-1}" graphify extract . \
  --out . \
  --no-cluster \
  --max-workers "${GRAPHIFY_MAX_WORKERS:-1}" \
  ${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"} \
  "${EXCLUDES[@]}"
