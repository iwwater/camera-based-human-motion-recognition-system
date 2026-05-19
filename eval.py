"""
Offline evaluation of the rule-based squat rep counter.

This script is a faithful Python port of the fixed-threshold recognition
pipeline used by index.html. It uses MediaPipe Pose to extract BlazePose
landmarks, then applies the same finite state machine so recorded videos can be
evaluated with OBO accuracy and MAE.

Inputs:
    CSV with columns: video_path, gt_count
    Optional column: category

Outputs:
    Per-video CSV with predicted count, absolute error, and OBO flag.
    Summary printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

REQUIRED_LOWER = [
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

VISIBILITY_THRESHOLD = 0.55
DESCEND = 128.0
BOTTOM = 112.0
ASCEND = 148.0
TOP = 160.0
STABLE_FRAMES = 3
REP_COOLDOWN_MS = 750
EMA_ALPHA = 0.35
HIP_BELOW_SHOULDER_OFFSET = 0.14


def angle_deg(a, b, c) -> float:
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    dot = abx * cbx + aby * cby
    mag_ab = math.hypot(abx, aby)
    mag_cb = math.hypot(cbx, cby)
    cos = max(-1.0, min(1.0, dot / (mag_ab * mag_cb + 1e-8)))
    return math.degrees(math.acos(cos))


def min_required_visibility(lms) -> float:
    return min(lms[i].visibility for i in REQUIRED_LOWER)


def count_reps(video_path: Path, verbose: bool = False) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cooldown_frames = int(REP_COOLDOWN_MS * fps / 1000.0)

    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoothed_knee = None
    stage = "Ready"
    bottom_frames = 0
    top_frames = 0
    last_rep_frame = -10**9
    rep_count = 0
    frame_idx = 0
    weak_pose_streak = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                weak_pose_streak += 1
                if weak_pose_streak >= 8 and stage in (
                    "Descending",
                    "Bottom",
                    "Ascending",
                ):
                    stage = "Ready"
                    bottom_frames = 0
                    top_frames = 0
                continue

            lms = res.pose_landmarks.landmark
            if min_required_visibility(lms) < VISIBILITY_THRESHOLD:
                weak_pose_streak += 1
                if weak_pose_streak >= 8 and stage != "Ready":
                    stage = "Ready"
                    bottom_frames = 0
                    top_frames = 0
                continue
            weak_pose_streak = 0

            left_knee_ang = angle_deg(lms[LEFT_HIP], lms[LEFT_KNEE], lms[LEFT_ANKLE])
            right_knee_ang = angle_deg(lms[RIGHT_HIP], lms[RIGHT_KNEE], lms[RIGHT_ANKLE])
            avg_knee = (left_knee_ang + right_knee_ang) / 2.0

            if smoothed_knee is None:
                smoothed_knee = avg_knee
            else:
                smoothed_knee = smoothed_knee * (1.0 - EMA_ALPHA) + avg_knee * EMA_ALPHA

            hip_y = (lms[LEFT_HIP].y + lms[RIGHT_HIP].y) / 2.0
            shoulder_y = (lms[LEFT_SHOULDER].y + lms[RIGHT_SHOULDER].y) / 2.0

            if stage == "Ready":
                if (
                    smoothed_knee < DESCEND
                    and hip_y > shoulder_y + HIP_BELOW_SHOULDER_OFFSET
                ):
                    stage = "Descending"
                    bottom_frames = 0

            elif stage == "Descending":
                if smoothed_knee > TOP:
                    stage = "Ready"
                    bottom_frames = 0
                else:
                    bottom_frames = bottom_frames + 1 if smoothed_knee < BOTTOM else 0
                    if bottom_frames >= STABLE_FRAMES:
                        stage = "Bottom"

            elif stage == "Bottom":
                if smoothed_knee > ASCEND:
                    stage = "Ascending"
                    top_frames = 0

            elif stage == "Ascending":
                top_frames = top_frames + 1 if smoothed_knee > TOP else 0
                if (
                    top_frames >= STABLE_FRAMES
                    and (frame_idx - last_rep_frame) > cooldown_frames
                ):
                    rep_count += 1
                    last_rep_frame = frame_idx
                    stage = "Ready"
                    bottom_frames = 0
                    top_frames = 0

            if verbose and frame_idx % 30 == 0:
                print(
                    f"    frame {frame_idx}  stage={stage}  "
                    f"knee={smoothed_knee:.1f}  reps={rep_count}"
                )
    finally:
        cap.release()
        pose.close()

    return rep_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-csv", required=True)
    parser.add_argument("--videos-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.videos_root)
    rows_out = []

    with open(args.videos_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = root / row["video_path"]
            gt = int(row["gt_count"])
            category = row.get("category", "") or ""
            print(f"[{len(rows_out) + 1}] {video_path.name}  (gt={gt}, cat={category!r})")

            try:
                pred = count_reps(video_path, verbose=args.verbose)
                err = abs(pred - gt)
                obo = 1 if err <= 1 else 0
                print(f"     pred={pred}  |err|={err}  obo={obo}")
            except Exception as exc:
                print(f"     ERROR: {exc}")
                pred, err, obo = -1, "", ""

            rows_out.append(
                {
                    "video_path": row["video_path"],
                    "category": category,
                    "gt_count": gt,
                    "pred_count": pred,
                    "abs_error": err,
                    "obo": obo,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "category",
                "gt_count",
                "pred_count",
                "abs_error",
                "obo",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    valid = [r for r in rows_out if r["pred_count"] != -1]
    n = len(valid)
    print("\n=== Summary ===")
    print(f"Wrote per-video results to: {out_path}")
    print(f"n = {n}  (skipped {len(rows_out) - n} due to errors)")

    if n == 0:
        return

    mae = sum(r["abs_error"] for r in valid) / n
    obo_sum = sum(r["obo"] for r in valid)
    obo_acc = obo_sum / n
    print(f"MAE          = {mae:.3f}")
    print(f"OBO accuracy = {obo_sum}/{n} = {obo_acc:.3f}")

    cats = sorted({r["category"] for r in valid if r["category"]})
    if cats:
        print("\nPer-category breakdown:")
        for cat in cats:
            sub = [r for r in valid if r["category"] == cat]
            sub_mae = sum(r["abs_error"] for r in sub) / len(sub)
            sub_obo = sum(r["obo"] for r in sub) / len(sub)
            print(
                f"  [{cat:<12s}] n={len(sub):3d}  "
                f"MAE={sub_mae:.3f}  OBO={sub_obo:.3f}"
            )


if __name__ == "__main__":
    main()
