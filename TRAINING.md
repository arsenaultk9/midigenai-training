# Local training

Training runs entirely on the local machine. GPU selection is automatic:
CUDA, Apple MPS, then CPU.

The wrapper defaults to a 1,024-token context, but automatically reduces it
to a power of two that fits the shortest train or validation shard. Pass
`-- --block-size N` to choose one explicitly.

## Prepare data

```bash
./prepare_data.sh
```

This creates `data/orchestral/prepared/` with token shards and tokenizer data.
To prepare another MIDI folder:

```bash
./prepare_data.sh ./data/jazz
```

That creates `data/jazz/manifest.jsonl` and `data/jazz/prepared/`. The output
directory and shard tag can be overridden with `--out` and `--tag`:

```bash
./prepare_data.sh /mnt/midi-library \
  --out ./data/library-prepared \
  --tag library
```

## Train locally

Start from the current `v5-rl` checkpoint on Hugging Face:

```bash
./train.sh --run-name orchestral-v5 \
  --corpus orchestral \
  --size medium \
  --resume
```

`--resume` uses the newest checkpoint in `runs/<run-name>/` when one exists.
For a new run it downloads `v5-rl` from `nicholasbien/midigenai` into the
standard Hugging Face cache and resumes from that checkpoint.

To train without a checkpoint:

```bash
./train.sh --run-name orchestral-scratch --corpus orchestral --size medium
```

Pass additional local trainer options after `--`:

```bash
./train.sh --run-name orchestral-v5 --resume -- \
  --max-steps 20000 \
  --batch-size 8 \
  --grad-accum 4 \
  --block-size 128 \
  --save-interval 500
```

Use another local checkpoint with:

```bash
./train.sh --run-name continued \
  --resume-from runs/orchestral-v5/ckpt_005000.pt
```

## Output

Run configuration, metrics, and checkpoints are written under:

```text
runs/<run-name>/
```

Run `./train.sh --help` for wrapper options, or
`.venv/bin/python -m midigenai.train --help` for all trainer options.
