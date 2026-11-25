#!/usr/bin/env python3
"""
Generate music using an ONNX model from the models folder.

This script:
1. Loads an ONNX model and tokenizer from a specified model folder
2. Creates a random single-note primer (pitch and duration)
3. Generates a sequence using top-p sampling
4. Outputs the result as MIDI
"""

import argparse
import random
import tempfile
from pathlib import Path
import json
import numpy as np

import onnxruntime as rt
from midigenai import tokenizer
import symusic


def load_model_config(model_path: Path) -> dict:
    """Load model configuration from manifest.json"""
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    with open(manifest_path) as f:
        return json.load(f)


def get_onnx_model_path(model_path: Path, prefer_gpu: bool = True) -> Path:
    """Get the appropriate ONNX model file based on preference"""
    config = load_model_config(model_path)
    files = config["files"]
    
    if prefer_gpu and "gpu" in files:
        model_file = files["gpu"]["file"]
    elif "cpu" in files:
        model_file = files["cpu"]["file"]
    else:
        # Fallback to first available
        model_file = list(files.values())[0]["file"]
    
    model_full_path = model_path / model_file
    if not model_full_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_full_path}")
    
    return model_full_path


def create_random_note_primer(
    min_pitch: int = 36,
    max_pitch: int = 84,
    min_duration: float = 0.25,
    max_duration: float = 2.0,
    velocity: int = 80,
) -> symusic.Score:
    """
    Create a random single-note primer using symusic.
    
    Args:
        min_pitch: Minimum MIDI pitch (default: 36 = C2)
        max_pitch: Maximum MIDI pitch (default: 84 = C6)
        min_duration: Minimum note duration in beats (default: 0.25)
        max_duration: Maximum note duration in beats (default: 2.0)
        velocity: MIDI velocity (default: 80)
    
    Returns:
        symusic.Score containing a single random note
    """
    pitch = random.randint(min_pitch, max_pitch)
    
    # Random duration in beats
    duration_beats = random.uniform(min_duration, max_duration)
    
    # Create a score with standard 4/4 time
    score = symusic.Score()
    track = symusic.Track()
    
    # Convert beats to ticks (assuming 480 ticks per beat)
    ticks_per_beat = 480
    start_tick = 0
    end_tick = int(duration_beats * ticks_per_beat)
    
    # Add the random note
    note = symusic.Note(
        time=start_tick,
        pitch=pitch,
        duration=end_tick - start_tick,
        velocity=velocity,
    )
    track.notes.append(note)
    
    score.tracks.append(track)
    
    # Add tempo (120 BPM = 500000 microseconds per beat)
    score.tempos.append(symusic.Tempo(time=0, mspq=500000))
    
    # Add time signature (4/4)
    score.time_signatures.append(symusic.TimeSignature(time=0, numerator=4, denominator=4))
    
    return score


def top_p_sampling(
    logits: np.ndarray,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> int:
    """
    Sample from logits using top-p (nucleus) sampling.
    
    Args:
        logits: Model output logits for the vocabulary
        top_p: Nucleus probability threshold
        temperature: Sampling temperature
    
    Returns:
        Selected token ID
    """
    # Apply temperature
    logits = logits / temperature
    
    # Convert logits to probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Sort probabilities and select top-p tokens
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Find the cutoff
    cutoff_idx = np.searchsorted(cum_probs, top_p)
    cutoff_idx = min(cutoff_idx + 1, len(sorted_probs))  # At least one token
    
    # Renormalize probabilities
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = probs[top_p_indices]
    top_p_probs = top_p_probs / np.sum(top_p_probs)
    
    # Sample from top-p distribution
    sampled_idx = np.random.choice(len(top_p_indices), p=top_p_probs)
    return int(top_p_indices[sampled_idx])


def continuation_prompt(tok, prompt_ids: list[int]) -> list[int]:
    """Prefix the MIDI body with BOS, as in the model's training documents."""
    bos_id = tokenizer.special_id(tok, "BOS")
    if bos_id is not None and (not prompt_ids or prompt_ids[0] != bos_id):
        return [bos_id, *prompt_ids]
    return list(prompt_ids)


def repeated_ngram_next_tokens(tokens: list[int], size: int) -> set[int]:
    """Find tokens that would complete a previously seen n-gram."""
    if size < 2 or len(tokens) < size - 1:
        return set()
    prefix = tokens[-(size - 1):]
    return {
        tokens[i + size - 1]
        for i in range(len(tokens) - size + 1)
        if tokens[i:i + size - 1] == prefix
    }


def generate_with_onnx(
    session: rt.InferenceSession,
    prompt_ids: list[int],
    config: dict,
    max_tokens: int,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int | None = None,
    eos_id: int | None = None,
    ban_ids: list[int] | None = None,
    no_repeat_ngram_size: int = 0,
) -> list[int]:
    """
    Generate tokens using an ONNX model session with KV cache support.
    
    Args:
        session: ONNX Runtime inference session
        prompt_ids: Initial token sequence
        config: Model configuration from manifest.json
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        temperature: Sampling temperature
        seed: Random seed
        eos_id: Stop when this token is sampled, without including it
        ban_ids: Token IDs that are invalid in a continuation
        no_repeat_ngram_size: Forbid repeated token sequences of this size (0 disables)
    
    Returns:
        List of generated token IDs
    """
    if no_repeat_ngram_size < 0 or no_repeat_ngram_size == 1:
        raise ValueError("no_repeat_ngram_size must be 0 or at least 2")
    generated = []
    
    # Get model info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"Model inputs: {input_names[:5]}... ({len(input_names)} total)")
    print(f"Model outputs: {output_names[:5]}... ({len(output_names)} total)")
    
    n_layers = config.get("n_layers", 12)
    if "new_k0" in output_names:
        cache_output_prefix = "new"
        cache_axis = 1
        empty_cache_shape = (1, 0, config["n_heads"], config["head_dim"])
    elif "present_k0" in output_names:
        cache_output_prefix = "present"
        cache_axis = 2
        empty_cache_shape = (1, config["n_heads"], 0, config["head_dim"])
    else:
        raise ValueError("Unsupported ONNX model: unrecognized KV cache outputs")

    expected_outputs = [
        f"{cache_output_prefix}_{kind}{i}"
        for i in range(n_layers)
        for kind in ("k", "v")
    ]
    missing_outputs = set(expected_outputs) - set(output_names)
    if missing_outputs:
        raise ValueError(f"Missing KV cache outputs: {sorted(missing_outputs)}")

    tokens = np.array(prompt_ids, dtype=np.int64).reshape(1, -1)
    kv_cache = {
        f"past_{kind}{i}": np.zeros(empty_cache_shape, dtype=np.float32)
        for i in range(n_layers)
        for kind in ("k", "v")
    }
    
    for step in range(max_tokens):
        if step == 0:
            # First step: process entire prompt
            input_ids = tokens
        else:
            # Subsequent steps: only process last token
            input_ids = tokens[:, -1:]
        
        inputs = {"ids": input_ids, **kv_cache}
        outputs = session.run(output_names, inputs)
        outputs_by_name = dict(zip(output_names, outputs))
        
        logits = outputs_by_name["logits"]
        
        # Get logits for the last token position
        if len(logits.shape) == 2:
            # Shape: (batch, vocab_size) - only last token
            last_logits = logits[0, :]
        else:
            # Shape: (batch, seq_len, vocab_size) - full sequence
            last_logits = logits[0, -1, :]
        
        banned = set(ban_ids or [])
        if no_repeat_ngram_size:
            banned.update(
                repeated_ngram_next_tokens(prompt_ids + generated, no_repeat_ngram_size)
            )
        if banned:
            last_logits = last_logits.copy()
            last_logits[list(banned)] = -np.inf
            if not np.isfinite(last_logits).any():
                raise ValueError("No valid tokens remain after applying sampling restrictions")

        # Sample next token using top-p
        next_token = top_p_sampling(last_logits, top_p=top_p, temperature=temperature)
        if next_token == eos_id:
            print(f"  End-of-sequence token sampled after {len(generated)} tokens")
            break
        generated.append(next_token)
        
        # Append to token sequence
        tokens = np.concatenate([tokens, [[next_token]]], axis=1)
        
        # Update KV cache for next iteration
        for i in range(n_layers):
            for kind in ("k", "v"):
                input_name = f"past_{kind}{i}"
                cache_output = outputs_by_name[f"{cache_output_prefix}_{kind}{i}"]
                if cache_output_prefix == "new":
                    kv_cache[input_name] = np.concatenate(
                        (kv_cache[input_name], cache_output), axis=cache_axis
                    )
                else:
                    kv_cache[input_name] = cache_output
        
        if (step + 1) % 50 == 0 or step < 5:
            print(f"  Generated {step + 1}/{max_tokens} tokens (token_id={next_token})")
    
    return generated


def generate_music(
    model_path: str,
    tokenizer_path: str | None = None,
    output_path: str = "generated.mid",
    max_tokens: int = 512,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int | None = None,
    min_pitch: int = 36,
    max_pitch: int = 84,
    prefer_gpu: bool = True,
    no_repeat_ngram_size: int = 0,
) -> None:
    """
    Generate music using an ONNX model and top-p sampling.
    
    Args:
        model_path: Path to the model folder containing ONNX files and manifest
        tokenizer_path: Path to tokenizer.json (auto-detected if not provided)
        output_path: Output MIDI file path
        max_tokens: Maximum number of tokens to generate
        top_p: Top-p (nucleus) sampling parameter (0.0 to 1.0)
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        min_pitch: Minimum MIDI pitch for primer note
        max_pitch: Maximum MIDI pitch for primer note
        prefer_gpu: Prefer GPU model if available
        no_repeat_ngram_size: Forbid repeated token sequences of this size (0 disables)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Load model configuration
    config = load_model_config(model_path)
    print(f"Model info: {config['version']}, vocab_size={config['vocab_size']}")
    
    # Select the model for the execution provider actually available.
    use_gpu = prefer_gpu and "CUDAExecutionProvider" in rt.get_available_providers()
    if prefer_gpu and not use_gpu:
        print("CUDA execution provider unavailable; using CPU.")
    onnx_model_path = get_onnx_model_path(model_path, prefer_gpu=use_gpu)
    print(f"Using model: {onnx_model_path.name}")
    
    # Auto-detect tokenizer if not provided
    if tokenizer_path is None:
        # Try common locations
        common_paths = [
            model_path / "tokenizer.json",
            model_path.parent / "tokenizer.json",
            Path("data/orchestral/prepared/tokenizer.json"),
        ]
        for tp in common_paths:
            if tp.exists():
                tokenizer_path = str(tp)
                break
        
        if tokenizer_path is None:
            raise FileNotFoundError(
                f"Tokenizer not found. Searched: {common_paths}. "
                "Provide with --tokenizer-path"
            )
    
    print(f"Using tokenizer: {tokenizer_path}")
    tok = tokenizer.load_tokenizer(tokenizer_path)
    if len(tok.vocab) != config["vocab_size"]:
        raise ValueError(
            f"Tokenizer has {len(tok.vocab)} tokens, but the model expects "
            f"{config['vocab_size']}; use the tokenizer matching the model."
        )
    
    # Initialize ONNX Runtime session
    print("Initializing ONNX Runtime session...")
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Choose execution provider
    providers = []
    if use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    session = rt.InferenceSession(str(onnx_model_path), sess_options, providers=providers)
    print(f"Using execution provider: {session.get_providers()}")
    
    # Create random single-note primer
    print(f"Creating random note primer (pitch range: {min_pitch}-{max_pitch})...")
    primer_score = create_random_note_primer(
        min_pitch=min_pitch,
        max_pitch=max_pitch,
    )
    
    # Save primer temporarily and encode it
    with tempfile.TemporaryDirectory() as temp_dir:
        primer_temp_path = Path(temp_dir) / "primer.mid"
        primer_score.dump_midi(str(primer_temp_path))
        print("Encoding primer...")
        prompt_ids = tokenizer.encode_midi(tok, primer_temp_path)
    print(f"Primer encoded: {len(prompt_ids)} tokens")
    print(f"First 10 prompt tokens: {prompt_ids[:10]}")
    model_prompt = continuation_prompt(tok, prompt_ids)
    eos_id = tokenizer.special_id(tok, "EOS")
    ban_ids = []
    if tokenizer.is_v4(tok):
        ban_ids = [
            token_id
            for name in ("BOS", "SEP", "MASK")
            if (token_id := tokenizer.special_id(tok, name)) is not None
        ]
    
    # Generate using top-p sampling
    print(
        f"Generating {max_tokens} tokens with top_p={top_p}, "
        f"temperature={temperature}..."
    )
    
    generated_ids = generate_with_onnx(
        session=session,
        prompt_ids=model_prompt,
        config=config,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        seed=seed,
        eos_id=eos_id,
        ban_ids=ban_ids,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    
    print(f"Generated {len(generated_ids)} tokens")
    
    # Combine prompt and generated tokens
    all_ids = prompt_ids + generated_ids
    
    # Decode to MIDI
    output_path = Path(output_path)
    print(f"Decoding to MIDI: {output_path}")
    tokenizer.decode_to_midi(tok, all_ids, output_path)
    
    print(f"✓ Generation complete! Output: {output_path}")
    
    # Also try to load and display info about the generated file
    try:
        generated_score = symusic.Score.from_file(str(output_path))
        print(f"Generated score info:")
        print(f"  - Tracks: {len(generated_score.tracks)}")
        for i, track in enumerate(generated_score.tracks):
            print(f"    Track {i}: {len(track.notes)} notes")
    except Exception as e:
        print(f"Note: Could not load generated MIDI for info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate music using an ONNX model from the models folder"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model folder (e.g., models/orchestral-scratch)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-detected if not provided)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="generated.mid",
        help="Output MIDI file path (default: generated.mid)",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "-p",
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--min-pitch",
        type=int,
        default=36,
        help="Minimum MIDI pitch for primer note (default: 36 = C2)",
    )
    parser.add_argument(
        "--max-pitch",
        type=int,
        default=84,
        help="Maximum MIDI pitch for primer note (default: 84 = C6)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Prefer CPU model over GPU",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Forbid repeated token sequences of this size (0 disables; minimum 2)",
    )
    
    args = parser.parse_args()
    
    generate_music(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        max_tokens=args.tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
        prefer_gpu=not args.cpu,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )


if __name__ == "__main__":
    main()
