# PathSnap model results

Person 1's candidate generation and ordered reference alignment were rerun
on the supplied 80 training and 10 student-testing trips. The resulting
tables contained 1,843,151 training candidates and 208,057 student-test
candidates. Fitting excluded candidates attached to suspect training labels;
all 19,160 student-test reference observations were scored.

| Run | CV candidate Macro F1 | Student candidate Macro F1 | Student Top-1 | Student observation Macro F1 | Candidate log loss | Direct continuity | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 88.97% | 89.48% | 84.66% | 85.71% | 0.0911 | 91.43% | 0.073 |
| naive_bayes | 81.85% | 81.00% | 82.91% | 85.67% | 0.4226 | 91.53% | 0.108 |
| decision_tree | 91.64% | 91.96% | 85.22% | 87.25% | 0.0670 | 91.87% | 0.077 |
| random_forest | 91.72% | 92.09% | 85.71% | 87.29% | 0.0638 | 92.22% | 0.308 |
| hist_boosting | 91.81% | 92.43% | 85.99% | 87.54% | 0.0685 | 91.91% | 0.614 |
| hist_boosting_tuned | 91.90% | 92.42% | 86.00% | 87.50% | 0.0630 | 92.30% | 0.581 |

**Exported model:** `hist_boosting_tuned`.
The Bayesian search tested six values of `learning_rate`
and selected `0.07` on mean training-fold candidate Macro F1.
CV Macro F1 changed from 91.81% to 91.90%.
Student-test Top-1 changed from 85.99% to 86.00%.
The largest measured student-test Top-1 was 86.00% for `hist_boosting_tuned`.

The reference arc was absent for 27 of 19,160 student-test observations
(0.14%). These misses remain in the Top-1 denominator. Results measure
agreement with Person 1's aligned development reference. The direct
continuity figures are for classifier-only predictions, before Viterbi.

The model workflow was implemented and run with Codex assistance.
