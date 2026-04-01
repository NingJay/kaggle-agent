from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="candidate_manifest.json")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_submission_csv"]).resolve()
    output_path = Path(args.output).resolve()
    rows = list(csv.reader(sample_path.open("r", encoding="utf-8", newline="")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row_index, row in enumerate(rows):
            if row_index == 0:
                writer.writerow(row)
                continue
            writer.writerow([row[0], *(["0"] * (len(row) - 1))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
