# PathSnap model training and evaluation

## Reproduce in VS Code

Open this folder in VS Code and select **Terminal → New Terminal** (PowerShell).
Use Python 3.11 or newer with the package versions in requirements.txt.

```powershell
python -m pip install -r requirements.txt
```

Extract the raw student ZIP and Person 1's pipeline. In Person 1's folder,
run the candidate and alignment stages if the four compressed CSVs have not
already been generated:

```powershell
python s2_candidates.py --data-root "C:\path\to\Student folder" --out out
python s3_align.py --data-root "C:\path\to\Student folder" --out out
```

Return to this folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_person2.ps1 -DataDir "C:\path\to\pathsnap_p1\out" -StudentGraphRoot "C:\path\to\Student folder\student_test"
```

The graph folder supplies directed arcs for the continuity diagnostic. Use
`-BaselineOnly -OutputDir baseline_results` to produce an early Logistic
Regression handoff.

## Experimental design

- Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest,
  and Histogram Gradient Boosting score binary road candidates. Median
  imputation and scaling are fitted inside each training fold.
- Three GroupKFold folds keep observations from the same trip together.
  Person 1's suspect labels (`label_ok=0`, aligned distance over 100 m)
  are excluded from fitting. Validation and student-test evaluations include
  all supplied reference observations.
- Baseline selection uses mean training-fold candidate Macro F1. The selected
  algorithm is tuned using a Gaussian-process surrogate and an upper-confidence
  bound acquisition over a discrete grid. The student-test partition does not
  enter fitting, optimisation, or the export decision.
- Candidate Macro F1, precision, recall, and log loss are binary row metrics.
  Top-1 selects the highest scoring arc per observation (lower arc ID breaks
  ties). Observation Macro F1 averages over trip-local reference arc classes.
  Observations with no candidate count as misses. Candidate miss rate includes
  observations where the reference arc was absent from the candidate set.
- Direct route continuity is the fraction of consecutive GPS observation
  pairs whose predicted arcs are identical or immediately connected in the
  directed graph. Missing predictions count as discontinuities. This
  diagnostic does not test whether two non-adjacent arcs are connected by a
  longer path. The Viterbi comparison belongs to Person 3.

## Outputs and integration

- `comparison.csv`, `cv_folds.csv`, `bayesian_trials.csv`, and
  `run_config.json` contain the results and configuration. PNGs visualise the
  comparison. Timings are machine-specific.
- `final_model.joblib` contains the selected fitted pipeline and the exact
  feature order. `run_config.json` identifies its run. Use a trusted artifact.
- Send the selected run's score CSV to the decoder owner. Read `trip_id` as a
  string to preserve leading zeros. Scores are positive-class probabilities
  per candidate, keyed by `trip_id, obs_idx, arc_id`.

```python
import joblib
from pathsnap_models import score_candidates

artifact = joblib.load("results/final_model.joblib")
scores = score_candidates(artifact["pipeline"], candidate_frame, artifact["features"])
```

`candidate_frame` is the inference-safe Person 1 candidate table. Review the
team's data-sharing rules before publishing row-level outputs derived from
the dataset.
