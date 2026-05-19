#!/usr/bin/env python3
"""Extract a small RepCount squat subset for local baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import io
import random
import tarfile
from pathlib import Path


ANNOTATION_FILES = (
    ("test", "RepCount_pose/annotation/test.csv"),
    ("valid", "RepCount_pose/annotation/valid.csv"),
)
SQUAT_TYPES = {"squat", "squant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic squat subset from RepCount_pose.tar.gz."
    )
    parser.add_argument("--archive", required=True, help="Path to RepCount_pose.tar.gz")
    parser.add_argument("--count", type=int, default=20, help="Number of clips to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffle")
    parser.add_argument(
        "--videos-dir",
        default="data/videos/repcount_squat",
        help="Output directory for extracted videos",
    )
    parser.add_argument(
        "--manifest",
        default="data/videos.csv",
        help="Output CSV manifest consumed by eval.py",
    )
    return parser.parse_args()


def collect_rows(tar: tarfile.TarFile) -> list[dict[str, str]]:
    archive_names = set(tar.getnames())
    rows: list[dict[str, str]] = []

    for split, annotation_path in ANNOTATION_FILES:
        annotation = tar.extractfile(annotation_path)
        if annotation is None:
            raise FileNotFoundError(annotation_path)

        text = annotation.read().decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(text)):
            exercise_type = (row.get("type") or "").strip().lower()
            if exercise_type not in SQUAT_TYPES:
                continue

            gt_count = int(float(row["count"]))
            if gt_count <= 0:
                continue

            video_name = row["name"].strip()
            member_name = f"RepCount_pose/video/{split}/{video_name}"
            if member_name not in archive_names:
                raise FileNotFoundError(member_name)

            rows.append(
                {
                    "member_name": member_name,
                    "video_name": video_name,
                    "gt_count": str(gt_count),
                    "category": f"repcount_{split}_{exercise_type}",
                }
            )

    return rows


def extract_subset(
    tar: tarfile.TarFile, rows: list[dict[str, str]], videos_dir: Path
) -> list[dict[str, str]]:
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        output_name = f"{index:03d}_{Path(row['video_name']).name}"
        output_path = videos_dir / output_name

        with tar.extractfile(row["member_name"]) as src:
            if src is None:
                raise FileNotFoundError(row["member_name"])
            output_path.write_bytes(src.read())

        manifest_rows.append(
            {
                "video_path": str(Path("repcount_squat") / output_name).replace("\\", "/"),
                "gt_count": row["gt_count"],
                "category": row["category"],
            }
        )

    return manifest_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "gt_count", "category"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    videos_dir = Path(args.videos_dir)
    manifest = Path(args.manifest)

    with tarfile.open(archive, "r:gz") as tar:
        rows = collect_rows(tar)
        if len(rows) < args.count:
            print(
                f"WARNING: Only {len(rows)} squat clips found in archive, "
                f"fewer than requested --count={args.count}. "
                f"Using all {len(rows)} available clips."
            )
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        subset = rows[: args.count]
        manifest_rows = extract_subset(tar, subset, videos_dir)

    write_manifest(manifest, manifest_rows)
    print(f"Extracted {len(manifest_rows)} clips to {videos_dir}")
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
