# PathSnap: Live Evaluation Demonstration Script & Viva Defense Guide

**Team:** 115  
**Project:** PathSnap: Distributed Map-Matching and Route Reconstruction  
**Course:** UE24CS352A (Machine Learning) | PES University  
**System Setup:** Laptop 1 (MacBook Air M1 Host) & Laptop 2 (Client / Evaluator UI)

---

## 1. Quick Live Demo Checklist (Before Evaluator Arrives)

- [ ] Connect both Laptop 1 and Laptop 2 to the same local Wi-Fi network (or mobile hotspot).
- [ ] On Laptop 1, get your local IP address: `ipconfig getifaddr en0` (e.g. `192.168.1.15`).
- [ ] On Laptop 1, launch the FastAPI server: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
- [ ] On Laptop 2, open the Streamlit web app or browser. In the sidebar, enter `http://192.168.1.15:8000` and confirm the green **Connected (ok)** badge.
- [ ] Ensure terminal window on Laptop 1 is visible side-by-side to display real-time incoming server logs.

---

## 2. Step-by-Step Live Demonstration Protocol

### Step 1: Show Distributed Connectivity
1. Point out the live connection status badge in the Streamlit UI on Laptop 2.
2. In Laptop 2 terminal (PowerShell / Command Prompt), run:
   ```bash
   curl -X GET http://192.168.1.15:8000/health
   ```
3. Direct the evaluator's attention to Laptop 1 terminal: show the immediate incoming GET request logged with client IP `192.168.1.20`.

### Step 2: Query Permitted / Evaluator Trips
1. Show the trip selector dropdown in the Streamlit sidebar.
2. Explain:
   > *"Our backend dynamically queries the available trips using `GET /trips`. During development, it serves the 10 permitted student-testing trips (`00000000` to `00000091`). When the official evaluation folder is mounted, it seamlessly switches to the 10 reserved evaluator trips."*

### Step 3: Trigger Real End-to-End Prediction
1. Ask the evaluator to pick any trip (e.g., `00000000` or `00000007`).
2. Click **🚀 Match & Reconstruct Route**.
3. Point to Laptop 1 terminal: show the 3 pipeline stages executing in real-time:
   ```
   [INFO] Stage 1 (Candidates & Features) completed in 158.4 ms.
   [INFO] Stage 2 (Model Scoring) completed in 42.1 ms.
   [INFO] Stage 3 (HMM/Viterbi Decoding) completed in 186.2 ms.
   [INFO] Total pipeline execution: 0.412 s (200 OK)
   ```
4. Point to Laptop 2 UI: observe the results rendering in $< 0.8\text{ s}$ total round-trip time.

### Step 4: Explain Returned Evaluation Metrics
1. Show the KPI metric cards:
   - **Observation Top-1 Accuracy:** ~96.60% (matches ground truth road segment).
   - **Route Continuity:** ~99.48% (continuous road segment sequence without jumping blocks).
   - **Total Latency:** ~0.41 s (< 1.0 s requirement).
2. Show the Subsystem Timing Breakdown expander to demonstrate pipeline efficiency.

### Step 5: Explore the Interactive Folium Map
1. Zoom in on the Folium map.
2. Toggle layers using the top-right layer control:
   - **Raw GPS Fixes (Blue/Red circles):** Show how raw GPS points drift into parallel alleys and across buildings.
   - **Reconstructed Route (Green polyline):** Show how PathSnap cleanly snaps the trajectory to the true road centerlines and respects one-way traffic directionality.

---

## 3. Comprehensive Viva Defense Q&A

### Q1: How is the raw dataset structured? What does each file represent?
> **Answer:** *"Each trip folder contains four headerless whitespace-separated files:  
> 1. `.track`: 3 columns (`lon`, `lat`, `timestamp_seconds`), sampled at ~1 Hz ($96.5\%$ of fixes are exactly 1 s apart).  
> 2. `.nodes`: 2 columns (`lon`, `lat`), where 0-based row index defines the node ID.  
> 3. `.arcs`: 2 integer columns (`from_node_id`, `to_node_id`), where 0-based row index defines the local arc ID. Arcs are strictly directed ($u \rightarrow v$). Bidirectional streets have two distinct reverse twin arcs.  
> 4. `.route`: Single column of chronological ground-truth arc IDs traversed by the vehicle. Crucially, `.route` is confidential and kept server-side; it is never accessed during inference."*

### Q2: How did you preprocess geographical coordinates? Why not use Euclidean distance in degrees?
> **Answer:** *"Longitude and latitude are angular coordinates on an oblate spheroid. Ordinary Euclidean distance in degrees $\sqrt{\Delta\text{lon}^2 + \Delta\text{lat}^2}$ causes severe distortion because a degree of longitude shrinks by $\cos(\text{latitude})$.  
> We implemented a **local metric equirectangular projection** centered at each trip's bounding box centroid:  
> $x = (\text{lon} - \text{lon}_0) \cdot R \cos(\text{lat}_0) \cdot \frac{\pi}{180}$, $y = (\text{lat} - \text{lat}_0) \cdot R \cdot \frac{\pi}{180}$.  
> In our Stage 1 verification, this achieved $< 0.4\%$ maximum relative error compared to geodesic Haversine while computing in nanoseconds."*

### Q3: What features did you engineer, and why were they chosen?
> **Answer:** *"We engineered 13 inference-safe features strictly without lookahead or reference leakage:  
> - **Spatial closeness:** `dist_m`, `along_frac`, `dist_rank`, `dist_minus_min_m`.  
> - **Kinematic & directional alignment:** `heading_diff_deg` and `heading_cos`. Crucially, `heading_valid` acts as a guard flag: if displacement is $< 6\text{ m}$ over a $\pm 2$ fix window, heading difference defaults to $90^\circ$ and cosine to $0$ to prevent GPS jitter from misclassifying stationary vehicles.  
> - **Network context:** `arc_len_m`, `has_reverse_arc`, `speed_mps`, candidate density `n_cand`, and time deltas `dt_prev_s` and `dt_next_s`."*

### Q4: Which 5 models did you train, and why did you select Histogram Gradient Boosting?
> **Answer:** *"We trained 5 models across diverse algorithm families: Logistic Regression (linear baseline), Gaussian Naive Bayes (probabilistic generative), Decision Tree (rule-based non-linear), Random Forest (bagged ensemble), and Histogram Gradient Boosting (sequential boosting).  
> While Random Forest achieved strong accuracy (85.71%), Histogram Gradient Boosting delivered the best candidate log loss (0.0630) and highest validation CV Macro F1 (91.90%). Crucially for HMM decoding, Gradient Boosting produces well-calibrated class probability estimates rather than discrete step functions, which directly improves Viterbi emission calculations."*

### Q5: How was Bayesian hyperparameter optimization conducted?
> **Answer:** *"We optimized the learning rate $\eta$ of Histogram Gradient Boosting using a Gaussian Process Regressor with a Matérn kernel ($\nu=2.5$) and Upper Confidence Bound (UCB) acquisition on the 80 training trips using 3-fold GroupKFold. Across 6 trials, $\eta=0.07$ outperformed $\eta=0.10$, reducing log loss from 0.0685 to 0.0630 and boosting observation Top-1 accuracy to 86.00%."*

### Q6: How does your HMM/Viterbi sequential decoder work?
> **Answer:** *"Pointwise classifiers suffer from temporal myopia: they pick parallel roads or illegal turn jumps because each fix is evaluated independently.  
> Our HMM models the trajectory as a hidden sequence of road segments:  
> 1. **Emission Likelihood:** $\log P(O_t \mid c_{t, j}) = \log(\max(P_{\text{model}}(c_{t, j}), 10^{-6}))$, derived from our tuned ML pipeline.  
> 2. **Transition Likelihood:** Following Newson & Krumm, $\log P(v \mid u) = -\frac{|d_{\text{net}} - d_{\text{gps}}|}{\beta}$, where $d_{\text{net}}$ is the road network distance between segments computed using a bounded local Dijkstra search ($r = 2.5 \cdot d_{\text{gps}} + 50\text{ m}$).  
> 3. **Viterbi Recurrence:** In log-space, dynamic programming tracks the highest probability path and backpointers.  
> 4. **Edge Cases:** Zero-candidate fixes carry forward previous state distributions. Disconnected hops receive a soft log penalty of $-12.0$, ensuring the trellis never collapses.  
> Across all 10 student test trips, HMM decoding boosted Top-1 accuracy from **85.99% $\rightarrow$ 90.92% (+4.93%)** and route continuity from **92.40% $\rightarrow$ 97.61% (+5.21%)**."*

### Q7: How do you guarantee evaluation privacy and prevent data leakage?
> **Answer:** *"1. **Partition Isolation:** Training and cross-validation were fitted exclusively on the 80 training trips.  
> 2. **No Route Access at Inference:** `features.py` never imports, opens, or reads `.route` files (enforced by AST static inspection in our tests).  
> 3. **Server-Side Security:** The inference API only accepts `trip_id` and outputs predicted geometry. Ground-truth route labels and coordinates are strictly evaluated on Laptop 1 and never sent across the network."*
