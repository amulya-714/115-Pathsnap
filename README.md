# PathSnap: Distributed Map-Matching and Route Reconstruction

[![Course](https://img.shields.io/badge/PES_University-UE24CS352A_Machine_Learning-1E3A8A.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.11_|_3.12_|_3.13-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.163.2-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic_Evaluation-lightgrey.svg)]()

---

## 1. Project Overview
**PathSnap** is an end-to-end, distributed machine learning and sequential map-matching system that reconstructs real vehicle trajectories from noisy, low-frequency GPS tracking observations. Developed for **UE24CS352A (Machine Learning) at PES University**, the system bridges discrete spatial indexing, supervised classification, Hidden Markov Model (HMM) Viterbi dynamic programming, and a distributed two-laptop client-server architecture.

---

## 2. Problem Statement
Urban GPS observations suffer from severe signal degradation caused by tall buildings, satellite multipath reflections, atmospheric attenuation, GPS dropouts, and irregular sampling intervals (~1 Hz with occasional multi-second gaps). As a result:
- Raw GPS points appear to jump across buildings, drift into parallel corridors, or make physically impossible turns.
- Independent, pointwise classification of road segments produces disconnected trajectories that violate road network topology and traffic rules.

The core challenge of **PathSnap** is to take raw, noisy GPS fixes $(\text{lon}, \text{lat}, t)$ and accurately identify the exact sequence of directed road segments traversed by the vehicle while maintaining $100\%$ topological continuity and sub-second inference latency.

---

## 3. Dataset Description
The dataset contains raw vehicle trajectories and corresponding OpenStreetMap-derived road networks partitioned into three strict subsets:
- **Training Set (80 trips):** Used exclusively for exploratory data analysis (EDA), candidate generation, feature extraction, 3-fold `GroupKFold` cross-validation, and Bayesian hyperparameter tuning.
- **Student Testing Set (10 trips):** Permitted development-stage evaluation, model comparison, HMM parameter calibration, and integration testing.
- **Evaluator Testing Set (10 trips):** Reserved exclusively for live judging. Evaluator data remains isolated and untouched during development.

### Raw File Formats (Per Trip)
| File Extension | Content & Format | Meaning |
|---|---|---|
| `<trip>.track` | Whitespace-separated: `lon`, `lat`, `t_s` (~1 Hz, 96.5% at 1 s) | Raw, noisy GPS observation fixes |
| `<trip>.nodes` | Whitespace-separated: `lon`, `lat` (0-based row index = `node_id`) | Physical junction/road coordinates |
| `<trip>.arcs` | Whitespace-separated: `u`, `v` (0-based row index = `arc_id`) | Directed road segments connecting `u` $\rightarrow$ `v` |
| `<trip>.route` | Integer list: ground truth sequence of `arc_id` values | Historical reference route (Server-Side Only) |

---

## 4. System Architecture
PathSnap follows a decoupled four-stage machine learning pipeline and a distributed two-laptop deployment:

```
+---------------------------------------------------------------------------------------------------------+
|                                    LAPTOP 1: ML SERVICE & API HOST                                     |
|                                                                                                         |
|   [Raw GPS Track] ----> [1. Candidate Generator] ----> [2. Feature Extraction]                          |
|         (lon, lat, t)          KDTree Radius=50m              13 Kinematic Features                     |
|                                                                     │                                   |
|                                                                     ▼                                   |
|                                                        [3. Tuned HistGradientBoosting]                  |
|                                                               Candidate Scoring                         |
|                                                                     │                                   |
|                                                                     ▼                                   |
|   [Reconstructed Route] <---- [5. Route Bridging] <---- [4. Log-Space Viterbi HMM]                      |
|      Contiguous Arcs              Dijkstra Gaps               Network Transitions                       |
|             │                                                                                           |
|             ▼                                                                                           |
|   [FastAPI REST Service] (0.0.0.0:8000) <---------------- Server-Side Evaluation (Confidential)        |
+---------------------------------------------------------------------------------------------------------+
                                          ▲                  │
                               GET /trips │                  │ POST /predict Response
                                          │                  ▼
+---------------------------------------------------------------------------------------------------------+
|                                  LAPTOP 2: EVALUATOR USER INTERFACE                                     |
|                                                                                                         |
|   [Streamlit Web App] ----> [Trip Selector] ----> [Folium Visualizer] ----> [Performance KPI Cards]     |
+---------------------------------------------------------------------------------------------------------+
```

---

## 5. Team Responsibilities
| Member | Role | Core Deliverables |
|---|---|---|
| **Person 1** | Data & EDA | Ingestion loader, KDTree candidate generation, 13 features contract, monotone reference alignment, EDA reports, quality flags |
| **Person 2** | ML Models & Tuning | 5-model comparison, GroupKFold CV, Bayesian GP optimisation, model selection (`HistGradientBoosting`), joblib artifacts |
| **Person 3** | Sequential HMM / Viterbi | Hidden Markov Model formulation, log emissions & transitions, local Dijkstra routing, Viterbi recurrence, route reconstruction |
| **Person 4** | Backend, UI & Deployment | FastAPI service (`0.0.0.0:8000`), Streamlit evaluator UI, two-laptop networking, interactive Folium mapping, end-to-end testing |

---

## 6. Data Preprocessing
- **Local Metric Equirectangular Projection:** Converts spherical coordinates $(\text{lon}, \text{lat})$ into local Euclidean coordinates $(x, y)$ in metres centered at the bounding box centroid of each trip's road network, guaranteeing distance distortion $< 0.4\%$.
- **Directed Graph Representation:** Road arcs are strictly directed ($u \rightarrow v$). Reverse twin arcs ($v \rightarrow u$) represent two-way roads.
- **Handling Data Anomalies:** Identifies sampling rate variations, GPS gaps ($\ge 5\text{ s}$), duplicate arcs (128 in test), and zero-length segments.

---

## 7. Feature Engineering (13-Feature Contract)
For each observation $t$ and candidate road segment $k$ within 50 m, 13 inference-safe features are computed without accessing reference route files:
1. `dist_m`: Orthogonal Euclidean distance from GPS fix to road segment (metres).
2. `heading_diff_deg`: Smallest angular difference between vehicle movement heading and road segment bearing ($0^\circ \text{ to } 180^\circ$).
3. `heading_cos`: Cosine of heading difference ($\cos \Delta\theta$).
4. `heading_valid`: Binary flag ($1$ if displacement $\ge 6\text{ m}$ over $\pm 2$ fixes, else $0$).
5. `along_frac`: Position of perpendicular projection foot-point along the segment ($0.0 = \text{tail node}$, $1.0 = \text{head node}$).
6. `arc_len_m`: Total length of the road segment in metres.
7. `speed_mps`: Estimated vehicle velocity ($\text{m/s}$).
8. `dist_rank`: Integer rank of candidate ordered by distance ($0 = \text{closest}$).
9. `dist_minus_min_m`: Excess distance above the nearest candidate ($d - d_{\min}$).
10. `n_cand`: Total number of candidates retained for the observation ($k \le 50$).
11. `dt_prev_s`: Elapsed time since previous GPS fix ($0\text{ s}$ for first fix).
12. `dt_next_s`: Time to next GPS fix ($0\text{ s}$ for last fix).
13. `has_reverse_arc`: Binary indicator ($1$ if reverse-direction twin arc exists in graph).

---

## 8. Five-Model Experimentation
Five candidate models across diverse algorithm families were trained using the 80 training trips (1,843,151 candidate pairs) and evaluated on all 10 student-test trips (208,057 candidate pairs):
1. **Logistic Regression** (Linear / Probabilistic baseline)
2. **Gaussian Naive Bayes** (Generative probabilistic)
3. **Decision Tree** (Non-linear rule-based, `max_depth=8`)
4. **Random Forest** (Bagged ensemble, 40 estimators, `max_depth=12`, `max_samples=0.25`)
5. **Histogram Gradient Boosting** (Sequential gradient boosted decision trees)

---

## 9. Model Comparison & Results
All models used a unified preprocessing pipeline: `SimpleImputer(strategy='median')` followed by `StandardScaler()`. 3-fold `GroupKFold` cross-validation grouped by `trip_id` ensured zero data leakage across folds.

| Model | Algorithm Family | CV Macro F1 | Student Top-1 | Student Macro F1 | Candidate Log Loss | Direct Continuity | Inference Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | Linear / Probabilistic | 88.97% | 84.66% | 85.71% | 0.0911 | 91.43% | 0.073 s |
| **Gaussian Naive Bayes** | Probabilistic | 81.85% | 82.91% | 85.67% | 0.4226 | 91.53% | 0.108 s |
| **Decision Tree** | Tree-based | 91.64% | 85.22% | 87.25% | 0.0670 | 91.87% | 0.077 s |
| **Random Forest** | Ensemble | 91.72% | 85.71% | 87.29% | 0.0638 | 92.22% | 0.308 s |
| **HistGradientBoosting** | Boosting (Untuned) | 91.81% | 85.99% | 87.54% | 0.0685 | 91.91% | 0.614 s |
| **HistGradientBoosting (Tuned)** | Boosting (Selected) | **91.90%** | **86.00%** | **87.50%** | **0.0630** | **92.30%** | 0.581 s |

---

## 10. Bayesian Hyperparameter Optimisation
Histogram Gradient Boosting was chosen as the top performer. Bayesian optimisation was conducted using a Gaussian Process Regressor with a Matérn kernel and Upper Confidence Bound (UCB) acquisition over learning rate configurations on the 80 training trips:

| Trial | Learning Rate ($\eta$) | CV Candidate Macro F1 | Trial Duration |
|---|---:|---:|---:|
| 1 | 0.01 | 88.48% | 46.88 s |
| 2 | 0.30 | 91.56% | 32.46 s |
| 3 | 0.15 | 91.72% | 41.24 s |
| 4 | 0.20 | 91.43% | 35.77 s |
| 5 | 0.10 | 91.81% | 35.60 s |
| **6 (Best)** | **0.07** | **91.90%** | 75.14 s |

**Selected Configuration:** `learning_rate = 0.07`, yielding optimal log-loss (0.0630) and generalisation stability.

---

## 11. HMM and Viterbi Sequential Decoding
To solve the spatial disconnects of classifier-only predictions, Person 3 implemented a sequential Hidden Markov Model:
- **Hidden States:** The road segment candidates $C_t$ generated for each observation fix.
- **Emission Log-Probability:** $\log P(O_t \mid c_{t, k}) = \log(\max(P_{\text{model}}(c_{t, k}), 10^{-6}))$.
- **Transition Log-Probability:** Evaluated using the physical difference between network routing distance $d_{\text{net}}$ and Euclidean GPS displacement $d_{\text{gps}}$:
  $$\log P(v \mid u) = -\frac{\left| d_{\text{net}}(u, v) - d_{\text{gps}} \right|}{\beta} - \text{penalties}$$
  where $\beta = 10.0\text{ m}$. Bounded Dijkstra calculates shortest paths up to $2.5 \cdot d_{\text{gps}} + 50\text{ m}$.
- **Zero-Candidate & Disconnect Handling:** Observations with zero candidates carry forward trellis state distributions without crashing. Disconnected transitions receive a soft penalty ($\log P_{\text{disc}} = -12.0$), allowing Viterbi to bridge across unmapped alleys or GPS jumps.

---

## 12. Route Reconstruction & Comparison
Comparing Classifier-Only vs HMM/Viterbi across all 19,160 observations in the student-test partition:

| Trip ID | GPS Fixes | Classifier Top-1 | HMM Decoded Top-1 | Top-1 Gain | Classifier Continuity | HMM Continuity | Continuity Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 94.37% | **96.60%** | +2.23% | 96.52% | **99.48%** | +2.96% |
| `00000007` | 1,354 | 94.17% | **99.63%** | +5.47% | 99.70% | **99.93%** | +0.22% |
| `00000009` | 1,049 | 85.13% | **89.32%** | +4.19% | 88.55% | **94.94%** | +6.39% |
| `00000040` | 2,995 | 93.09% | **96.19%** | +3.11% | 95.99% | **99.30%** | +3.31% |
| `00000052` | 262 | 95.42% | **98.09%** | +2.67% | 94.25% | **96.17%** | +1.92% |
| `00000056` | 3,958 | 88.66% | **94.47%** | +5.81% | 89.03% | **96.46%** | +7.43% |
| `00000065` | 768 | 83.07% | **87.50%** | +4.43% | 92.31% | **98.17%** | +5.87% |
| `00000076` | 1,486 | 89.23% | **93.27%** | +4.04% | 89.16% | **96.84%** | +7.68% |
| `00000077` | 3,911 | 68.65% | **76.27%** | +7.62% | 90.61% | **97.83%** | +7.21% |
| `00000091` | 874 | 85.81% | **91.42%** | +5.61% | 87.86% | **97.02%** | +9.16% |
| **OVERALL** | **19,160** | **85.99%** | **90.92%** | **+4.93%** | **92.40%** | **97.61%** | **+5.21%** |

---

## 13. FastAPI Backend
The backend service resides in `backend/` and binds to `0.0.0.0:8000`:
- `GET /health`: Returns service status, loaded model pipeline, timestamp, and version.
- `GET /trips`: Dynamically lists all available evaluation trips (prioritising evaluator test sets).
- `POST /predict`: Accepts `{"trip_id": "00000000"}`, runs full map-matching inference, and returns predicted route geometry, timing breakdowns, and server-side metrics.
- **Privacy Enforcement:** Ground-truth `.route` data remains strictly internal; no ground-truth geometry or labels are ever returned over the API.

---

## 14. Streamlit Frontend
The evaluator interface resides in `frontend/app.py`:
- Configurable backend URL pointing to Laptop 1 LAN IP (`http://<ip>:8000`).
- Live health badge and dynamic trip selector.
- KPI cards showing Top-1 Accuracy, Route Continuity, Latency, and Arc Counts.
- Interactive Folium map with toggleable layers.
- Traversed arc segment inspector and raw API payload inspection.

---

## 15. Folium Visualisation
- **Noisy GPS observations:** Rendered as semi-transparent circular markers (`#EF4444`).
- **Predicted reconstructed path:** Rendered as high-visibility polylines (`#10B981`) following the actual geometry of the road segments.
- **Origin & Destination:** Highlighted with distinct start (green) and flag (red) markers.
- **Layer Controls & Measure Tools:** Enabled for interactive inspection of road geometry.

---

## 16. Two-Laptop Deployment Guide

```
Laptop 1 (MacBook Air - Server Host)        Laptop 2 (Evaluator UI / Client)
IP: e.g. 192.168.1.15                        IP: e.g. 192.168.1.20
Running: FastAPI on 0.0.0.0:8000             Running: Streamlit on localhost:8501
```

### Laptop 1 Commands (MacBook Air):
```bash
# 1. Navigate to project repository
cd ~/Downloads/115-Pathsnap

# 2. Activate Python virtual environment
source .venv/bin/activate

# 3. Find your local Wi-Fi IP address
ipconfig getifaddr en0
# (Example: 192.168.1.15)

# 4. Start FastAPI server binding to 0.0.0.0:8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Laptop 2 Commands (Teammate / Evaluator):
```bash
# Windows PowerShell / Mac Terminal
cd 115-Pathsnap
source .venv/bin/activate  # (or .venv\Scripts\Activate.ps1 on Windows)

# Start Streamlit Frontend
streamlit run frontend/app.py --server.port 8501
```
In the Streamlit sidebar, enter `http://192.168.1.15:8000` as the FastAPI URL and click **Match & Reconstruct Route**.

### Testing via Command Line from Laptop 2:
```bash
# Health check:
curl -X GET http://192.168.1.15:8000/health

# Query available trips:
curl -X GET http://192.168.1.15:8000/trips

# Request prediction for trip 00000000:
curl -X POST http://192.168.1.15:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "00000000"}'
```

---

## 17. Installation
```bash
git clone https://github.com/amulya-714/115-Pathsnap.git
cd 115-Pathsnap

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 18. Execution Instructions

### Run Automated Test Suite
```bash
pytest tests/ -v
```

### Run HMM Systematic Evaluation
```bash
python hmm/evaluate_hmm.py
```

### Generate Standalone Folium HTML Maps
```bash
python scripts/generate_maps.py
```

---

## 19. API Documentation
Interactive OpenAPI Swagger documentation is automatically generated by FastAPI:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 20. Automated Testing & Verification
The test suite in `tests/` covers 17 comprehensive unit and integration tests:
- `test_person1.py`: Ingestion, graph topology, KDTree candidate contract, feature AST route independence.
- `test_person2.py`: Model artifact loading, feature signature matching, top-1 parity with saved scores.
- `test_person3.py`: Viterbi decoder execution, zero-candidate observation handling, continuity guarantees.
- `test_backend.py`: REST endpoints (`/`, `/health`, `/trips`, `/predict`), input validation (422), missing trips (404), and privacy guarantees.

---

## 21. Summary of Experimental Results
- **Overall Top-1 Match Rate:** **90.92%** across all student testing observations.
- **Overall Route Continuity:** **97.61%** topologically continuous transitions.
- **Sub-Second Latency:** Average inference latency of **0.863 seconds per trip** (< 1 s requirement).
- **Candidate Miss Rate:** Low 0.14% candidate miss rate in student-test partition.

---

## 22. Limitations & Edge Cases
1. **Unmapped Private Alleys / Road Construction:** If a vehicle travels through unindexed private roads or temporary detours, candidate generation yields zero candidates within 50 m. The system relies on soft disconnected transitions.
2. **Dense Multi-Level Overpasses:** GPS vertical altitude is noisy; stacked bridges (elevated highways above surface streets) can produce ambiguous candidate headings.
3. **Severe GPS Jumps (> 100 m/s):** Sudden multipath reflection spikes are flagged by quality controls but can temporarily degrade route continuity.

---

## 23. References
1. Newson, P., & Krumm, J. (2009). *Hidden Markov map matching through noise and sparseness.* Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 336-343.
2. Lou, Y., et al. (2009). *Map-matching for low-sampling-rate GPS trajectories.* ACM SIGSPATIAL.
3. OpenStreetMap contributors. (2024). *Planet dump.* Retrieved from https://www.openstreetmap.org.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
5. Tiangolo, S. (2020). *FastAPI framework.* https://fastapi.tiangolo.com.
