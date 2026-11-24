# Quick Start: Music Generation with ONNX Models

## 30-Second Setup

You're ready to go! Just run:

```bash
python generate_with_onnx.py models/orchestral-scratch
```

This will generate a MIDI file with:
- Random single-note primer (C2-C6 pitch range)
- 512 generated tokens
- Top-p sampling (p=0.9)
- Saved to `generated.mid`

## Common Commands

### Try It Out
```bash
python generate_with_onnx.py models/orchestral-scratch -t 50
```

### Save with Custom Name
```bash
python generate_with_onnx.py models/orchestral-scratch -o my_song.mid
```

### Reproducible Results
```bash
python generate_with_onnx.py models/orchestral-scratch --seed 42
```

### Quick 10-token Test
```bash
python generate_with_onnx.py models/orchestral-scratch -t 10 --seed 1
```

### Creative (High Diversity)
```bash
python generate_with_onnx.py models/orchestral-scratch -p 0.95 --temperature 1.3
```

### Conservative (Predictable)
```bash
python generate_with_onnx.py models/orchestral-scratch -p 0.7 --temperature 0.5
```

## Full Help

```bash
python generate_with_onnx.py --help
```

## What's Happening

1. **Load Model**: Reads ONNX decoder from models folder
2. **Random Primer**: Creates a random single note to seed generation
3. **Encode**: Converts MIDI primer to tokens using midigenai tokenizer
4. **Generate**: Produces new tokens using top-p sampling
5. **Decode**: Converts tokens back to MIDI
6. **Save**: Writes generated.mid

## For More Details

See [GENERATION.md](./GENERATION.md) for:
- Detailed parameter explanations
- Advanced examples
- Troubleshooting
- Technical details
