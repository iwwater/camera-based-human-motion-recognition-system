#!/usr/bin/env python3
"""Weakly supervised pose-feature baseline for squat rep counting.

The RepCount annotations provide repetition intervals, not explicit per-frame
up/down labels. This script derives weak labels from those intervals, trains a
small logistic classifier on pose features, and counts down-to-up transitions.
It is intended as a comparison point for the fixed-threshold rule baseline.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_ANNOTATION = "RepCount_pose/annotation/video_train.csv"
EVAL_ANNOTATIONS = (
    "RepCount_pose/annotation/test.csv",
    "RepCount_pose/annotation/valid.csv",
)
POSE_DIR = "RepCount_pose/test_poses"
SQUAT_TYPES = {"squat", "squant"}

FEATURE_NAMES = [
    "smoothed_knee_angle",
    "hip_y_minus_shoulder_y",
    "ankle_width",
    "knee_gap",
    "min_lower_visibility",
]

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class Sample:
    video_name: str
    gt_count: int
    intervals: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to RepCount_pose.tar.gz")
    parser.add_argument("--videos-csv", default="data/videos.csv")
    parser.add_argument("--out", default="results/ml_baseline_eval.csv")
    parser.add_argument("--train-cache", default="_downloads/ml_train_cache")
    parser.add_argument("--max-train-clips", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    return parser.parse_args()


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a[:2] - b[:2]
    cb = c[:2] - b[:2]
    denom = np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8
    cos = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def pose_features(pose: np.ndarray) -> np.ndarray:
    lms = pose.reshape(33, 3)
    left_knee = angle_deg(lms[LEFT_HIP], lms[LEFT_KNEE], lms[LEFT_ANKLE])
    right_knee = angle_deg(lms[RIGHT_HIP], lms[RIGHT_KNEE], lms[RIGHT_ANKLE])
    hip_y = float((lms[LEFT_HIP, 1] + lms[RIGHT_HIP, 1]) / 2.0)
    shoulder_y = float((lms[LEFT_SHOULDER, 1] + lms[RIGHT_SHOULDER, 1]) / 2.0)
    ankle_width = abs(float(lms[LEFT_ANKLE, 0] - lms[RIGHT_ANKLE, 0]))
    shoulder_width = abs(float(lms[LEFT_SHOULDER, 0] - lms[RIGHT_SHOULDER, 0]))
    knee_gap = abs(float(lms[LEFT_KNEE, 0] - lms[RIGHT_KNEE, 0]))
    return np.array(
        [
            left_knee,
            right_knee,
            (left_knee + right_knee) / 2.0,
            hip_y - shoulder_y,
            ankle_width,
            shoulder_width,
            knee_gap,
        ],
        dtype=np.float32,
    )


def landmarks_to_pose(landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32,
    ).reshape(99)


def parse_intervals(row: dict[str, str]) -> list[tuple[int, int]]:
    values = []
    for key in sorted((k for k in row if k.startswith("L")), key=lambda x: int(x[1:])):
        value = (row.get(key) or "").strip()
        if value:
            values.append(int(float(value)))

    intervals = []
    for start, end in zip(values[0::2], values[1::2]):
        if end > start:
            intervals.append((start, end))
    return intervals


def read_samples(tar: tarfile.TarFile, annotation_path: str) -> list[Sample]:
    annotation = tar.extractfile(annotation_path)
    if annotation is None:
        raise FileNotFoundError(annotation_path)

    samples: list[Sample] = []
    text = annotation.read().decode("utf-8-sig")
    for row in csv.DictReader(io.StringIO(text)):
        exercise_type = (row.get("type") or "").strip().lower()
        if exercise_type not in SQUAT_TYPES:
            continue
        gt_count = int(float(row["count"]))
        intervals = parse_intervals(row)
        if gt_count <= 0 or not intervals:
            continue
        samples.append(Sample(row["name"].strip(), gt_count, intervals))
    return samples


def load_pose(tar: tarfile.TarFile, video_name: str) -> np.ndarray:
    pose_name = f"{POSE_DIR}/{Path(video_name).stem}.npy"
    pose_file = tar.extractfile(pose_name)
    if pose_file is None:
        raise FileNotFoundError(pose_name)
    return np.load(io.BytesIO(pose_file.read()))


def extract_video(tar: tarfile.TarFile, member_name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / Path(member_name).name
    if output_path.exists():
        return output_path

    video_file = tar.extractfile(member_name)
    if video_file is None:
        raise FileNotFoundError(member_name)
    output_path.write_bytes(video_file.read())
    return output_path


EMA_ALPHA = 0.35

REQUIRED_LOWER = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]


def extract_features(lms) -> np.ndarray:
    left_knee = angle_deg(
        np.array([lms[LEFT_HIP].x, lms[LEFT_HIP].y]),
        np.array([lms[LEFT_KNEE].x, lms[LEFT_KNEE].y]),
        np.array([lms[LEFT_ANKLE].x, lms[LEFT_ANKLE].y]),
    )
    right_knee = angle_deg(
        np.array([lms[RIGHT_HIP].x, lms[RIGHT_HIP].y]),
        np.array([lms[RIGHT_KNEE].x, lms[RIGHT_KNEE].y]),
        np.array([lms[RIGHT_ANKLE].x, lms[RIGHT_ANKLE].y]),
    )
    raw_knee = (left_knee + right_knee) / 2.0
    hip_y = (lms[LEFT_HIP].y + lms[RIGHT_HIP].y) / 2.0
    shoulder_y = (lms[LEFT_SHOULDER].y + lms[RIGHT_SHOULDER].y) / 2.0
    ankle_width = abs(lms[LEFT_ANKLE].x - lms[RIGHT_ANKLE].x)
    knee_gap = abs(lms[LEFT_KNEE].x - lms[RIGHT_KNEE].x)
    min_vis = min(lms[i].visibility for i in REQUIRED_LOWER)
    return np.array(
        [raw_knee, hip_y - shoulder_y, ankle_width, knee_gap, min_vis],
        dtype=np.float32,
    )


def video_features(
    video_path: Path, pose_model, labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    xs = []
    ys = []
    frame_idx = 0
    smoothed_knee = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            feats = extract_features(result.pose_landmarks.landmark)
            raw_knee = feats[0]
            if smoothed_knee is None:
                smoothed_knee = raw_knee
            else:
                smoothed_knee = smoothed_knee * (1.0 - EMA_ALPHA) + raw_knee * EMA_ALPHA
            feats[0] = smoothed_knee
            xs.append(feats)
            if labels is not None and frame_idx < len(labels):
                ys.append(labels[frame_idx])
        frame_idx += 1

    cap.release()
    if len(xs) == 0:
        return np.zeros((0, 5), dtype=np.float32), None
    if labels is None:
        return np.vstack(xs), None
    return np.vstack(xs), np.array(ys, dtype=np.float32)


def weak_labels(frame_count: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    labels = np.full(frame_count, -1, dtype=np.int8)
    labels[:] = 0

    for start, end in intervals:
        start = max(0, min(frame_count - 1, start))
        end = max(0, min(frame_count - 1, end))
        if end <= start:
            continue
        span = end - start
        down_start = start + int(span * 0.35)
        down_end = start + int(span * 0.65)
        labels[down_start : down_end + 1] = 1

    return labels


def build_training_set(
    tar: tarfile.TarFile, cache_dir: Path, max_train_clips: int
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    skipped = 0
    used = 0

    pose_model = mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for sample in read_samples(tar, TRAIN_ANNOTATION):
        if used >= max_train_clips:
            break
        member_name = f"RepCount_pose/video/train/{sample.video_name}"
        try:
            video_path = extract_video(tar, member_name, cache_dir)
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            labels = weak_labels(frame_count, sample.intervals)
            features, feature_labels = video_features(video_path, pose_model, labels)
        except (FileNotFoundError, RuntimeError, ValueError):
            skipped += 1
            continue

        if len(features) == 0 or feature_labels is None or len(feature_labels) == 0:
            skipped += 1
            continue

        stride = max(1, len(features) // 300)
        xs.extend(features[::stride])
        ys.extend(feature_labels[::stride])
        used += 1

    if skipped:
        print(f"Skipped {skipped} training clips without usable video/pose features")
    print(f"Used {used} training clips")
    return np.vstack(xs), np.array(ys, dtype=np.float32)


def train_sklearn_models(
    x: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, MLPClassifier, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(x_scaled, y)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    mlp.fit(x_scaled, y)
    return lr, mlp, scaler


def predict_probs(features: np.ndarray, model, scaler: StandardScaler) -> np.ndarray:
    x_scaled = scaler.transform(features)
    return model.predict_proba(x_scaled)[:, 1]


def count_transitions(probs: np.ndarray) -> int:
    state = "up"
    stable_down = 0
    stable_up = 0
    reps = 0
    cooldown = 10
    cooldown_left = 0

    for prob in probs:
        if cooldown_left:
            cooldown_left -= 1

        if state == "up":
            if prob >= 0.6:
                stable_down += 1
                if stable_down >= 3:
                    state = "down"
                    stable_up = 0
            else:
                stable_down = 0
        else:
            if prob <= 0.4:
                stable_up += 1
                if stable_up >= 3 and not cooldown_left:
                    reps += 1
                    state = "up"
                    stable_down = 0
                    cooldown_left = cooldown
            else:
                stable_up = 0

    return reps


def original_name(manifest_path: str) -> str:
    name = Path(manifest_path).name
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else name


def read_eval_names(path: Path) -> list[tuple[str, str, int, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    row["video_path"],
                    original_name(row["video_path"]),
                    int(row["gt_count"]),
                    row.get("category", ""),
                )
            )
    return rows


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "category",
                "gt_count",
                "pred_count_lr",
                "abs_error_lr",
                "obo_lr",
                "pred_count_mlp",
                "abs_error_mlp",
                "obo_mlp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_feature_summary(
    path: Path, x_all: np.ndarray, video_names: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic"] + FEATURE_NAMES)
        writer.writerow(["mean"] + [f"{v:.4f}" for v in x_all.mean(axis=0)])
        writer.writerow(["std"] + [f"{v:.4f}" for v in x_all.std(axis=0)])
        writer.writerow(["min"] + [f"{v:.4f}" for v in x_all.min(axis=0)])
        writer.writerow(["max"] + [f"{v:.4f}" for v in x_all.max(axis=0)])


def main() -> None:
    args = parse_args()
    with tarfile.open(args.archive, "r:gz") as tar:
        x_train, y_train = build_training_set(
            tar, Path(args.train_cache), args.max_train_clips
        )
        print(f"Training set: {x_train.shape[0]} frames, {x_train.shape[1]} features")
        lr_model, mlp_model, scaler = train_sklearn_models(x_train, y_train)

        pose_model = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        all_features = []
        rows = []
        for manifest_path, video_name, gt_count, category in read_eval_names(
            Path(args.videos_csv)
        ):
            video_path = Path("data/videos") / manifest_path
            features, _ = video_features(video_path, pose_model)
            if features.shape[0] == 0:
                print(f"{video_name}: SKIP (no features)")
                continue
            all_features.append(features)
            probs_lr = predict_probs(features, lr_model, scaler)
            probs_mlp = predict_probs(features, mlp_model, scaler)
            pred_lr = count_transitions(probs_lr)
            pred_mlp = count_transitions(probs_mlp)
            err_lr = abs(pred_lr - gt_count)
            err_mlp = abs(pred_mlp - gt_count)
            rows.append(
                {
                    "video_name": video_name,
                    "category": category,
                    "gt_count": gt_count,
                    "pred_count_lr": pred_lr,
                    "abs_error_lr": err_lr,
                    "obo_lr": 1 if err_lr <= 1 else 0,
                    "pred_count_mlp": pred_mlp,
                    "abs_error_mlp": err_mlp,
                    "obo_mlp": 1 if err_mlp <= 1 else 0,
                }
            )
            print(
                f"{video_name}: gt={gt_count}  lr={pred_lr} (err={err_lr})  "
                f"mlp={pred_mlp} (err={err_mlp})"
            )

    if all_features:
        x_all = np.vstack(all_features)
        write_feature_summary(
            Path("results/feature_summary.csv"),
            x_all,
            [r["video_name"] for r in rows],
        )
        print(f"Feature summary written to results/feature_summary.csv")

    write_results(Path(args.out), rows)
    n = len(rows)
    mae_lr = sum(int(row["abs_error_lr"]) for row in rows) / n
    obo_lr = sum(int(row["obo_lr"]) for row in rows)
    mae_mlp = sum(int(row["abs_error_mlp"]) for row in rows) / n
    obo_mlp = sum(int(row["obo_mlp"]) for row in rows)
    print("\n=== Weak ML Baseline Summary ===")
    print(f"Wrote per-video results to: {args.out}")
    print(f"n = {n}")
    print(f"Logistic  MAE = {mae_lr:.3f}  OBO = {obo_lr}/{n} = {obo_lr / n:.3f}")
    print(f"MLP      MAE = {mae_mlp:.3f}  OBO = {obo_mlp}/{n} = {obo_mlp / n:.3f}")


if __name__ == "__main__":
    main()
