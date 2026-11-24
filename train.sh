#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Run 'uv sync' first." >&2
  exit 1
fi

RUN_NAME="local-v5"
CORPUS="orchestral"
DATA_DIR=""
OUT_DIR=""
SIZE="medium"
HUB_VERSION="v5-rl"
RESUME=false
RESUME_FROM=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./train.sh [options] [-- midigenai.train options]

Options:
  --run-name NAME          Output run name (default: local-v5)
  --corpus NAME            Prepared corpus under data/NAME/prepared
                            (default: orchestral)
  --data DIR               Explicit prepared-data directory
  --out DIR                Explicit checkpoint/output directory
  --size SIZE              pilot, medium, large, or production
                            (default: medium)
  --resume                 Resume the newest local checkpoint; if none
                            exists, download the v5-rl Hub checkpoint
  --resume-from FILE       Resume from this local checkpoint
  --hub-version VERSION    Hub version used by --resume (default: v5-rl)
  -h, --help               Show this help

Arguments after -- are passed to:
  python -m midigenai.train

Examples:
  ./train.sh --run-name orchestral-v5 --corpus orchestral --size medium --resume
  ./train.sh --run-name quick-test --size pilot -- --max-steps 100 --batch-size 4
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Option $1 requires a value." >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      need_value "$@"
      RUN_NAME="$2"
      shift 2
      ;;
    --corpus)
      need_value "$@"
      CORPUS="$2"
      shift 2
      ;;
    --data)
      need_value "$@"
      DATA_DIR="$2"
      shift 2
      ;;
    --out)
      need_value "$@"
      OUT_DIR="$2"
      shift 2
      ;;
    --size)
      need_value "$@"
      SIZE="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --resume-from)
      need_value "$@"
      RESUME=true
      RESUME_FROM="$2"
      shift 2
      ;;
    --hub-version)
      need_value "$@"
      HUB_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Put midigenai.train options after --." >&2
      usage >&2
      exit 2
      ;;
  esac
done

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/$CORPUS/prepared}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}"

if [[ ! -f "$DATA_DIR/manifest.json" || ! -d "$DATA_DIR/shards" ]]; then
  echo "Prepared corpus not found at $DATA_DIR." >&2
  echo "Run ./prepare_data.sh first, or pass --data." >&2
  exit 1
fi

HAS_BLOCK_SIZE=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--block-size" || "$arg" == --block-size=* ]]; then
    HAS_BLOCK_SIZE=true
    break
  fi
done

if ! "$HAS_BLOCK_SIZE"; then
  BLOCK_SIZE="$(
    "$PYTHON" - "$DATA_DIR/shards" <<'PY'
import sys
from pathlib import Path

import numpy as np

shards = sorted(Path(sys.argv[1]).glob("*.npy"))
if not shards:
    raise SystemExit("No .npy shards found")

shortest = min(np.load(path, mmap_mode="r").shape[0] for path in shards)
largest_safe = shortest - 1
block_size = min(1024, largest_safe)

if block_size < 32:
    raise SystemExit(f"Shortest shard has only {shortest} tokens; at least 33 are required")

if block_size < 1024:
    block_size = 1 << (block_size.bit_length() - 1)

print(block_size)
PY
  )"
  EXTRA_ARGS+=(--block-size "$BLOCK_SIZE")
else
  BLOCK_SIZE="explicit"
fi

if "$RESUME"; then
  if [[ -n "$RESUME_FROM" ]]; then
    CHECKPOINT="$RESUME_FROM"
  else
    CHECKPOINT="$(
      find "$OUT_DIR" -maxdepth 1 -type f -name 'ckpt_*.pt' \
        -printf '%T@ %p\n' 2>/dev/null |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
    )"
    if [[ -z "$CHECKPOINT" ]]; then
      echo "No local checkpoint found; downloading Hub checkpoint '$HUB_VERSION'..."
      CHECKPOINT="$(
        "$PYTHON" -c \
          'import sys; from midigenai.hub import download_files; print(download_files(version=sys.argv[1])[0])' \
          "$HUB_VERSION"
      )"
    fi
  fi

  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Resume checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--resume "$CHECKPOINT")
fi

echo "Starting local training:"
echo "  data:   $DATA_DIR"
echo "  output: $OUT_DIR"
echo "  size:   $SIZE"
echo "  block:  $BLOCK_SIZE"
[[ -z "${CHECKPOINT:-}" ]] || echo "  resume: $CHECKPOINT"

exec "$PYTHON" -m midigenai.train \
  --data "$DATA_DIR" \
  --out "$OUT_DIR" \
  --size "$SIZE" \
  "${EXTRA_ARGS[@]}"
