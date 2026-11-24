# Local fine-tuning

```bash
uv sync
./prepare_data.sh
./train.sh --run-name orchestral-v5 --corpus orchestral --size medium --resume
```

Pass another MIDI folder to the preparation script:

```bash
./prepare_data.sh ./data/jazz
```

Training runs on the local CUDA, MPS, or CPU device. `--resume` uses the
newest checkpoint for the run, or downloads the `v5-rl` checkpoint from
Hugging Face for a new run. See [TRAINING.md](./TRAINING.md) for more options.