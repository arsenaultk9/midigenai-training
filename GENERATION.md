# ONNX Model Music Generation

This guide explains how to use the `generate_with_onnx.py` script to generate music from trained ONNX models using the midigenai tokenizer and top-p sampling.

## Overview

The script performs the following steps:

1. **Loads an ONNX model** from the models folder (e.g., `models/orchestral-scratch/`)
2. **Creates a random single-note primer** using symusic with:
   - Random pitch (configurable range, default: C2-C6)
   - Random duration (configurable range, default: 0.25-2.0 beats)
3. **Encodes the primer** using the midigenai tokenizer
4. **Generates a sequence** using top-p (nucleus) sampling
5. **Decodes to MIDI** using the midigenai tokenizer
6. **Outputs the result** as a MIDI file

## Prerequisites

- Python 3.14+
- Project dependencies installed: `uv sync`
- A trained ONNX model exported to the `models/` folder (see [TRAINING.md](./TRAINING.md) for exporting)
- A tokenizer file (auto-detected from common locations)

## Basic Usage

### Generate with Default Settings

```bash
python generate_with_onnx.py models/orchestral-scratch
```

This will:
- Use the CPU model (int8 quantized)
- Generate 512 tokens with top-p=0.9 and temperature=1.0
- Save output to `generated.mid`
- Use a random seed each time

### Generate with Specific Output File

```bash
python generate_with_onnx.py models/orchestral-scratch -o my_generation.mid
```

### Generate with Reproducible Results

Use `--seed` to get consistent generations:

```bash
python generate_with_onnx.py models/orchestral-scratch --seed 42 -o seeded.mid
```

## Advanced Options

### Token Generation

Control the number of tokens generated:

```bash
python generate_with_onnx.py models/orchestral-scratch -t 256
```

Shorter sequences for quick testing:

```bash
python generate_with_onnx.py models/orchestral-scratch -t 50
```

### Sampling Parameters

**Top-p Sampling (Nucleus Sampling)**

Top-p controls diversity. Lower values = more conservative, higher values = more diverse:

```bash
# Conservative (more predictable)
python generate_with_onnx.py models/orchestral-scratch -p 0.7

# Balanced (default)
python generate_with_onnx.py models/orchestral-scratch -p 0.9

# Diverse (more creative)
python generate_with_onnx.py models/orchestral-scratch -p 0.95
```

**Temperature**

Temperature controls randomness. Lower = more deterministic, higher = more random:

```bash
# Deterministic
python generate_with_onnx.py models/orchestral-scratch --temperature 0.5

# Balanced (default)
python generate_with_onnx.py models/orchestral-scratch --temperature 1.0

# Very creative
python generate_with_onnx.py models/orchestral-scratch --temperature 1.5
```

### Primer Note

Control the random primer note characteristics:

**Custom pitch range** (MIDI pitch values: 0-127, e.g., 36=C2, 60=C4, 84=C6):

```bash
# Bass range
python generate_with_onnx.py models/orchestral-scratch --min-pitch 36 --max-pitch 60

# High range
python generate_with_onnx.py models/orchestral-scratch --min-pitch 72 --max-pitch 96
```

### Device Selection

By default, the script tries to use GPU if available, falling back to CPU:

```bash
# Force CPU
python generate_with_onnx.py models/orchestral-scratch --cpu

# Use GPU (default if available)
python generate_with_onnx.py models/orchestral-scratch
```

### Custom Tokenizer Path

If the tokenizer is not auto-detected:

```bash
python generate_with_onnx.py models/orchestral-scratch \
  --tokenizer-path /path/to/tokenizer.json
```

## Complete Examples

### Quick Test Generation

```bash
python generate_with_onnx.py models/orchestral-scratch \
  -t 50 \
  --seed 42 \
  -o test.mid
```

### High-Quality Generation

```bash
python generate_with_onnx.py models/orchestral-scratch \
  -t 512 \
  -p 0.85 \
  --temperature 1.0 \
  --seed 123 \
  -o high_quality.mid
```

### Batch Generation with Different Seeds

```bash
for seed in 1 2 3 4 5; do
  python generate_with_onnx.py models/orchestral-scratch \
    -t 256 \
    --seed $seed \
    -o generation_seed_$seed.mid
done
```

### Creative Generation (High Diversity)

```bash
python generate_with_onnx.py models/orchestral-scratch \
  -t 256 \
  -p 0.95 \
  --temperature 1.3 \
  -o creative.mid
```

## Command-Line Arguments

```
usage: generate_with_onnx.py [-h] [--tokenizer-path TOKENIZER_PATH]
                             [-o OUTPUT] [-t TOKENS] [-p TOP_P]
                             [--temperature TEMPERATURE] [-s SEED]
                             [--min-pitch MIN_PITCH] [--max-pitch MAX_PITCH]
                             [--cpu]
                             model_path

positional arguments:
  model_path            Path to the model folder (e.g., models/orchestral-scratch)

options:
  --tokenizer-path TOKENIZER_PATH
                        Path to tokenizer.json (auto-detected if not provided)
  -o, --output OUTPUT   Output MIDI file path (default: generated.mid)
  -t, --tokens TOKENS   Maximum tokens to generate (default: 512)
  -p, --top-p TOP_P     Top-p sampling parameter (default: 0.9)
  --temperature TEMPERATURE
                        Sampling temperature (default: 1.0)
  -s, --seed SEED       Random seed for reproducibility (optional)
  --min-pitch MIN_PITCH Minimum MIDI pitch for primer note (default: 36 = C2)
  --max-pitch MAX_PITCH Maximum MIDI pitch for primer note (default: 84 = C6)
  --cpu                 Prefer CPU model over GPU
```

## Technical Details

### ONNX Model Support

The script supports ONNX models exported from midigenai training runs:

- **GPU model**: `decoder-webgpu-q4.onnx` (4-bit quantized)
- **CPU model**: `decoder-wasm-int8.onnx` (8-bit quantized)

The script automatically selects the appropriate model based on available hardware.

### Top-p Sampling

Top-p (nucleus) sampling works by:
1. Sorting tokens by probability
2. Selecting the top tokens whose cumulative probability exceeds `top_p`
3. Sampling uniformly from this subset

This avoids generating low-probability tokens while maintaining diversity.

### Token Encoding/Decoding

Tokens are encoded/decoded using the midigenai tokenizer, which represents music as a sequence of discrete tokens. The primer is converted to MIDI, encoded to tokens, used as context, and the generated tokens are decoded back to MIDI.

## Troubleshooting

### "Tokenizer not found"

The script looks for `tokenizer.json` in:
1. Model folder
2. Parent of model folder
3. `data/orchestral/prepared/tokenizer.json`

If still not found, provide it explicitly:

```bash
python generate_with_onnx.py models/orchestral-scratch \
  --tokenizer-path /path/to/tokenizer.json
```

### Inference Performance

Generation speed depends on:
- Hardware (GPU >> CPU)
- Model size (reflected in layer/head counts from `manifest.json`)
- Number of tokens to generate
- Quantization (int8 is faster than fp32, but slightly lower quality)

For CPU generation, int8 models are significantly faster than full precision.

### Generated Music Quality

Generation quality depends on:
1. Model training quality
2. Primer note (some pitches may work better than others)
3. Sampling parameters
4. Number of tokens generated

Experiment with different seeds and parameters to find what works best for your use case.

## See Also

- [Training](./TRAINING.md) - How to train models
- [Main Documentation](./README.md) - Project overview
- [midigenai Repository](https://github.com/nicholasbien/midigenai) - Original library
