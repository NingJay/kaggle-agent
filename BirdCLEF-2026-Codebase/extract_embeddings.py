from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an audio manifest as a placeholder embedding cache step.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_root = Path(__file__).resolve().parent
    data_root = runtime_root / "birdclef-2026" / "train_audio"
    output_dir = runtime_root / "outputs" / "embeddings_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_path"])
        for item in sorted(data_root.rglob("*.ogg")):
            writer.writerow([str(item)])
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

