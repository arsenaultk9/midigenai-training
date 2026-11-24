import argparse
import json
from pathlib import Path

from symusic import Score


def create_manifest(folder: Path, output: Path) -> tuple[int, int]:
    folder = folder.expanduser().resolve()
    output = output.expanduser().resolve()
    cwd = Path.cwd().resolve()

    if not folder.is_dir():
        raise NotADirectoryError(f"MIDI input folder does not exist: {folder}")

    midi_paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    if not midi_paths:
        raise ValueError(f"No .mid or .midi files found in: {folder}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0

    with output.open("w", encoding="utf-8") as manifest:
        for midi_path in midi_paths:
            try:
                score = Score(str(midi_path)).to("second")
                duration = float(score.end())
                if duration < 0:
                    raise ValueError("negative MIDI duration")

                entry = {
                    "path": (
                        midi_path.relative_to(cwd).as_posix()
                        if midi_path.is_relative_to(cwd)
                        else str(midi_path)
                    ),
                    "duration_seconds": duration,
                }
                manifest.write(json.dumps(entry) + "\n")
                written += 1
            except Exception as exc:
                failed += 1
                print(f"[manifest] failed: {midi_path}: {exc}")

    return written, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a midigenai JSONL manifest for a folder of MIDI files."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="folder containing .mid/.midi files (searched recursively)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/orchestral/manifest.jsonl"),
        help="output JSONL manifest",
    )
    args = parser.parse_args()

    written, failed = create_manifest(args.folder, args.out)
    print(f"Wrote {written} manifest entries to {args.out.resolve()}")
    if failed:
        print(f"Skipped {failed} unreadable MIDI files")


if __name__ == "__main__":
    main()
