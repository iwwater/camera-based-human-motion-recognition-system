"""Generate a horizontal contact sheet of N frames from a video.

Usage:
    python scripts/make_contact_sheet.py \
        --video data/videos/repcount_squat/015_stu6_65.mp4 \
        --out demo_assets/failure_cases/015_stu6_65_contact.jpg \
        --n-frames 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_contact_sheet(
    video_path: str,
    out_path: str,
    n_frames: int = 5,
    target_height: int = 270,
    gap: int = 4,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = [int(total * (i + 0.5) / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        new_w = int(w * target_height / h)
        frame = cv2.resize(frame, (new_w, target_height))
        label = f"f={idx}"
        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    if gap > 0:
        spacer = 255 * np.ones((target_height, gap, 3), dtype=np.uint8)
        spaced = []
        for i, frm in enumerate(frames):
            if i > 0:
                spaced.append(spacer)
            spaced.append(frm)
        sheet = cv2.hconcat(spaced) if len(spaced) > 1 else frames[0]
    else:
        sheet = cv2.hconcat(frames)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"Contact sheet saved to {out}  ({len(frames)} frames, {sheet.shape[1]}x{sheet.shape[0]} px)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--out", required=True, help="Path to output image (.jpg)")
    ap.add_argument("--n-frames", type=int, default=5, help="Number of equally-spaced frames")
    ap.add_argument("--target-height", type=int, default=270, help="Height of each frame row in px")
    args = ap.parse_args()
    make_contact_sheet(args.video, args.out, args.n_frames, args.target_height)
