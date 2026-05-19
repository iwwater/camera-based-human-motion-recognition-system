# Method

## Finite State Machine (FSM)

The rule-based recognizer uses a four-state machine to gate repetition counting:

```
Ready -> Descending -> Bottom -> Ascending -> Ready
```

**Transition conditions** (knee angle = hip-knee-ankle, averaged left/right, EMA-smoothed alpha=0.35):

| Transition | Condition |
|---|---|
| Ready -> Descending | knee angle < 128 AND hip_y > shoulder_y + 0.14 |
| Descending -> Ready | knee angle > 160 (abort before bottom) |
| Descending -> Bottom | knee angle < 112 for 3 consecutive frames |
| Bottom -> Ascending | knee angle > 148 |
| Ascending -> Ready | knee angle > 160 for 3 consecutive frames AND > 750 ms since last rep |

**Guard mechanisms**:
- **Hysteresis**: Separate descend (128), bottom (112), ascend (148), and top (160) thresholds prevent oscillation.
- **Stable frames** (3): Threshold crossing must persist for 3 consecutive frames.
- **Cooldown** (750 ms): Prevents double-counting a single movement.
- **Visibility gate**: All 6 lower-body landmarks (hips, knees, ankles) must have visibility >= 0.55. An 8-frame streak of weak pose resets the state to Ready.

### Threshold Parameters

All thresholds are fixed and shared between `index.html` (browser demo) and `eval.py` (offline evaluation).

```
LEFT_SHOULDER = 11    RIGHT_SHOULDER = 12
LEFT_HIP = 23         RIGHT_HIP = 24
LEFT_KNEE = 25        RIGHT_KNEE = 26
LEFT_ANKLE = 27       RIGHT_ANKLE = 28

VISIBILITY_THRESHOLD       = 0.55
DESCEND  (knee angle, deg) = 128.0
BOTTOM   (knee angle, deg) = 112.0
ASCEND   (knee angle, deg) = 148.0
TOP      (knee angle, deg) = 160.0
STABLE_FRAMES              = 3
REP_COOLDOWN_MS            = 750
EMA_ALPHA                  = 0.35
HIP_BELOW_SHOULDER_OFFSET  = 0.14
```

## Weakly Supervised ML Baseline

### Weak Label Derivation

The RepCount dataset provides repetition interval annotations (start/end frame of each repetition), not per-frame movement-phase labels. Weak labels are derived as follows:

1. Each repetition interval `[start, end]` defines a full up-down-up cycle.
2. The middle 30% of each interval (from `start + 0.35*span` to `start + 0.65*span`) is labeled as "down" (1).
3. All other frames are labeled as "up" (0).

This is a noisy approximation: the actual bottom of a squat rarely aligns precisely with the interval midpoint, and the transition duration varies across subjects.

### Feature Engineering

Five features are computed per frame from BlazePose landmarks:

| Feature | Definition | Rationale |
|---|---|---|
| `smoothed_knee_angle` | Knee flexion angle (hip-knee-ankle), averaged left/right, EMA alpha=0.35 | Primary signal for squat depth |
| `hip_y_minus_shoulder_y` | Vertical offset between hip and shoulder midpoints | Detects hip drop below shoulders |
| `ankle_width` | Absolute horizontal distance between left and right ankles | Proxy for stance width and body orientation |
| `knee_gap` | Absolute horizontal distance between left and right knees | Captures knee valgus/varus and side-view collapse |
| `min_lower_visibility` | Minimum visibility among the 6 lower-body landmarks | Signals occlusion or partial body framing |

The EMA smoothing uses the same alpha (0.35) as the rule-based FSM for comparability.

### Training

Both classifiers (logistic regression and MLP with 32-16 hidden units) are trained on the same features and weak labels using scikit-learn, with `random_state=42` and `max_iter=500`. Features are standardized with `StandardScaler` before training.
