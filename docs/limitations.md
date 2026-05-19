# Limitations

## Baseline Limitations

- **Fixed thresholds**: The FSM uses hand-tuned knee-angle thresholds that do
  not adapt to individual anatomy, camera distance, or exercise style.
- **Single exercise type**: Only squat is evaluated. Generalization to other
  exercises (push-up, lunge, etc.) requires new state-machine designs.
- **Weak supervision noise**: Interval-derived labels approximate the down phase
  as the middle 30% of each repetition interval. The actual bottom position can
  drift within the interval, introducing label noise.
- **Pose estimator dependence**: All features depend on BlazePose landmark
  quality. Heavy occlusion, low light, or unusual camera angles degrade the
  input signal before any recognition logic runs.
- **Small evaluation set**: 33 clips, not a full benchmark. Per-category
  breakdowns (see `results/eval.csv`) show performance variation across subjects
  and splits.

## Failure Modes

The FSM recognizer exhibits four distinct failure patterns on the RepCount squat
subset. Representative examples are documented in the README Failure Cases
section.

1. **Missed threshold crossing**: The subject performs visible squats, but the
   knee angle never crosses the fixed BOTTOM=112 threshold. Caused by shallow
   motion, camera framing that compresses apparent knee flexion, or individual
   range-of-motion differences.
2. **Side-view occlusion**: Lateral camera angles cause one side of the body to
   be partially occluded, reducing lower-body visibility below the 0.55
   threshold and resetting the state machine.
3. **Partial body framing**: The lower body is cropped out of frame, so hip,
   knee, or ankle landmarks are unavailable or have low visibility.
4. **High inter-subject variance**: Subjects with different body proportions or
   squat styles produce knee-angle trajectories outside the fixed threshold
   ranges, even when the movement is correctly performed.

## Next Steps

- **Per-frame phase annotation**: Manually label a small subset of frames with
  ground-truth movement phases (up/down/bottom) to replace weak interval-derived
  labels and train stronger models.
- **Cross-subject evaluation**: Train on one subject group and evaluate on
  another to assess generalization.
- **Multi-exercise extension**: Add state machines and features for push-ups,
  lunges, and other bodyweight exercises.
- **Learned feature extraction**: Replace hand-engineered knee-angle features
  with learned representations from the full 33-landmark pose.
