# Evaluation

## Data

The evaluation uses a deterministic subset of squat clips from the
[RepCount](https://github.com/MiracleDance/PoseRAC) dataset (Hu et al. 2022),
accessed via the PoseRAC project's public `RepCount_pose.tar.gz` package.

- **Source**: PoseRAC / RepCount benchmark
- **Exercise type**: squat (including `squant` — source annotation spelling preserved)
- **Subset size**: 33 clips (all available squat clips with valid annotations)
- **Sampling**: deterministic shuffle with `--seed 42` via `prepare_repcount_sample.py`
- **Splits represented**: `repcount_test_squat`, `repcount_test_squant`, `repcount_valid_squat`, `repcount_valid_squant`

The 33-clip count is the archive ceiling — there are 33 squat clips with valid
repetition annotations in `RepCount_pose.tar.gz`. The sampling script prints a
warning when `--count` exceeds the available pool.

## Metrics

Following Hu et al. 2022 (TransRAC):

- **MAE (Mean Absolute Error)**: `(1/n) * sum(|pred_i - gt_i|)`. Lower is better.
- **OBO (Off-By-One) accuracy**: fraction of clips where `|pred_i - gt_i| <= 1`.

## Reproduction

Create environment:

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate
# Windows
python -m venv .venv && .venv\Scripts\activate

pip install -r requirements.txt
```

Download `RepCount_pose.tar.gz` from [PoseRAC](https://github.com/MiracleDance/PoseRAC) to `_downloads/`, then:

```bash
# Extract deterministic 33-clip squat subset
python prepare_repcount_sample.py \
    --archive _downloads/RepCount_pose.tar.gz \
    --count 33 --seed 42

# Rule-based FSM evaluation
python eval.py \
    --videos-csv data/videos.csv \
    --videos-root data/videos \
    --out results/eval.csv

# Weak ML baseline (logistic + MLP)
python ml_baseline.py \
    --archive _downloads/RepCount_pose.tar.gz \
    --videos-csv data/videos.csv \
    --out results/ml_baseline_eval.csv \
    --max-train-clips 33
```

The `--seed 42` flag on `prepare_repcount_sample.py` ensures the same 33 clips
are selected every run. The per-clip outputs in `results/eval.csv` and
`results/ml_baseline_eval.csv` are the source of truth for all reported metrics.

## Results (33-clip RepCount squat subset, seed 42)

| Method | Clips | MAE | OBO accuracy |
|---|---|---|---|
| Fixed-threshold FSM | 33 | 7.455 | 6/33 = 0.182 |
| Weak logistic phase baseline | 33 | 4.424 | 19/33 = 0.576 |
| Weak MLP phase baseline (32-16 hidden) | 33 | 4.788 | 13/33 = 0.394 |

The logistic baseline outperforms the MLP on this subset. The FSM predicts 0
repetitions on 18 of the 33 clips, confirming that fixed thresholds do not
generalize across camera viewpoints and exercise styles.
