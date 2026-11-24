#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./export_onnx.sh [run-directory] [-- tokenizer|extra exporter options]

Exports runs/<name>/ckpt_final.pt to models/<name>/ using midigenai.export_onnx.
The tokenizer is inferred from runs/<name>/train_config.json.

Example:
  ./export_onnx.sh runs/orchestral-scratch
  ./export_onnx.sh runs/orchestral-scratch --check-midi data/orchestral/example.mid
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_DIR="$SCRIPT_DIR/runs/orchestral-scratch"
if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_DIR="$1"
  shift
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi
RUN_DIR="$(cd -- "$RUN_DIR" && pwd)"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Run 'uv sync' first." >&2
  exit 1
fi

if [[ ! -f "$RUN_DIR/ckpt_final.pt" ]]; then
  echo "Final checkpoint not found: $RUN_DIR/ckpt_final.pt" >&2
  exit 1
fi

CONFIG="$RUN_DIR/train_config.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing run configuration: $CONFIG" >&2
  exit 1
fi

DATA_DIR="$(
  "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
print(config["data_dir"])
PY
)"
TOKENIZER="$DATA_DIR/tokenizer.json"

if [[ ! -f "$TOKENIZER" ]]; then
  echo "Tokenizer not found: $TOKENIZER" >&2
  exit 1
fi

RUN_NAME="$(basename "$RUN_DIR")"
OUT_DIR="$SCRIPT_DIR/models/$RUN_NAME"

echo "Exporting:"
echo "  checkpoint: $RUN_DIR/ckpt_final.pt"
echo "  tokenizer:  $TOKENIZER"
echo "  output:     $OUT_DIR"

exec "$PYTHON" -m midigenai.export_onnx \
  --checkpoint "$RUN_DIR/ckpt_final.pt" \
  --tokenizer "$TOKENIZER" \
  --out "$OUT_DIR" \
  "$@"
