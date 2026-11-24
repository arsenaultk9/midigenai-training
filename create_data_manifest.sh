#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

folder="${1:-./data/orchestral}"
out="${2:-./data/orchestral/manifest.jsonl}"

uv run python ./create_data_manifest.py "$folder" --out "$out"
