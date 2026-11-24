#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="./data/orchestral"
OUT_DIR=""
TAG=""

usage() {
  cat <<'EOF'
Usage: ./prepare_data.sh [DATA_DIR] [--out PREPARED_DIR] [--tag TAG]

Prepare all .mid/.midi files below DATA_DIR. By default:
  manifest: DATA_DIR/manifest.jsonl
  output:   DATA_DIR/prepared
  tag:      DATA_DIR basename

Examples:
  ./prepare_data.sh
  ./prepare_data.sh ./data/jazz
  ./prepare_data.sh /mnt/midi-library --out ./data/library-prepared --tag library
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      [[ $# -ge 2 ]] || { echo "--out requires a directory." >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "--tag requires a value." >&2; exit 2; }
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "$DATA_DIR" != "./data/orchestral" ]]; then
        echo "Only one DATA_DIR may be specified." >&2
        usage >&2
        exit 2
      fi
      DATA_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data folder not found: $DATA_DIR" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$DATA_DIR/prepared}"
if [[ -z "$TAG" ]]; then
  TAG="$(basename "$DATA_DIR")"
  TAG="${TAG//[^[:alnum:]_-]/_}"
fi
MANIFEST="$DATA_DIR/manifest.jsonl"

uv run python ./create_data_manifest.py "$DATA_DIR" --out "$MANIFEST"

uv run python -m midigenai.data.build_dataset \
  --manifest "$MANIFEST" \
  --out "$OUT_DIR" \
  --tag "$TAG" \
  --scheme v4
