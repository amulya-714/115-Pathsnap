# PathSnap Person 3: Hidden Markov Model & Viterbi Sequential Route Reconstruction

## Overview
Independent candidate predictions from supervised machine learning classifiers can produce spatially disjoint, unphysical routes due to GPS multi-path errors, parallel urban corridors, and measurement noise. 
Person 3's HMM / Viterbi sequential decoder models vehicle trajectories as transitions across hidden road segment states constrained by road network topology and physical kinematics.

## Mathematical Formulation

### 1. Hidden State Space
For observation $t \in \{0, \dots, T-1\}$, hidden states $S_t \in C_t = \{c_{t, 1}, \dots, c_{t, K_t}\}$ represent the discrete set of candidate road arcs within a 50 m search radius of GPS fix $O_t = (x_t, y_t)$.

### 2. Emission Log-Likelihood
Emission probabilities reflect the likelihood of observing fix $O_t$ given the vehicle is traversing road arc $c_{t, j}$:
$$\log P(O_t \mid S_t = c_{t, j}) = \log\left(\max(P_{\text{model}}(c_{t, j}), \epsilon)\right)$$
where $P_{\text{model}}(c_{t, j}) \in [0, 1]$ is the calibrated candidate probability from Person 2's tuned Histogram Gradient Boosting pipeline (`final_model.joblib`), and $\epsilon = 10^{-6}$ ensures numerical stability.

### 3. Transition Log-Likelihood (Newson & Krumm 2009)
Between consecutive observations $t-1$ and $t$, the vehicle moves from arc $u = c_{t-1, i}$ to arc $v = c_{t, j}$.
Let $d_{\text{gps}} = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$ be the Euclidean GPS displacement.
The network routing distance $d_{\text{net}}(u, v)$ is calculated based on road graph connectivity:
- **Same Segment ($u == v$):** 
  $$d_{\text{net}} = \Delta\text{along} \times \text{length}(u)$$
- **Successor / Network Path ($v$ reachable from $u$):**
  $$d_{\text{net}} = (1 - f_u)\cdot \text{length}(u) + d_{\text{shortest\_path}}(\text{head}(u), \text{tail}(v)) + f_v \cdot \text{length}(v)$$
  where $d_{\text{shortest\_path}}$ is computed using a bounded local Dijkstra search within radius $r = 2.5 \cdot d_{\text{gps}} + 50\text{ m}$.
- **Transition Probability:**
  $$\log P(v \mid u) = -\frac{\left| d_{\text{net}}(u, v) - d_{\text{gps}} \right|}{\beta} - \mathbb{I}[v = \text{reverse}(u)] \cdot \text{penalty}_{\text{U-turn}}$$
  with $\beta = 10.0\text{ m}$ and $\text{penalty}_{\text{U-turn}} = 4.0$.
- **Disconnected Fallback:**
  If $v$ is unreachable from $u$ in the directed graph, a soft penalty is applied:
  $$\log P_{\text{disconnect}}(v \mid u) = -\frac{d_{\text{gps}}}{\beta} - \text{penalty}_{\text{disconnect}}$$
  with $\text{penalty}_{\text{disconnect}} = 12.0$, preventing trellis collapse during GPS jumps or tunnel dropouts.

### 4. Dynamic Programming Recurrence
The Viterbi trellis $V(t, j)$ and backpointers $BP(t, j)$ are computed in log-space:
$$V(t, j) = \max_{i \in C_{t-1}} \left[ V(t-1, i) + \log P(c_{t, j} \mid c_{t-1, i}) \right] + \log P(O_t \mid c_{t, j})$$
$$BP(t, j) = \arg\max_{i \in C_{t-1}} \left[ V(t-1, i) + \log P(c_{t, j} \mid c_{t-1, i}) \right]$$
The optimal sequence $S_0^*, \dots, S_{T-1}^*$ is retrieved by backtracking from $\arg\max_k V(T-1, k)$.

### 5. Contiguous Route Reconstruction
`reconstruct_route()` collapses repeated arc IDs and bridges any remaining topological hops along the shortest directed graph path, returning a continuous sequence of road segments suitable for navigation and map rendering.

## Verification & Key Results
Across all 10 student-test trips (19,160 GPS observations):
- **Classifier-Only Observation Top-1:** 85.99%
- **HMM / Viterbi Observation Top-1:** **90.92%** (+4.93% absolute gain)
- **Classifier-Only Route Continuity:** 92.40%
- **HMM / Viterbi Route Continuity:** **97.61%** (+5.21% absolute gain)
- **Mean Route Jaccard Match:** **92.71%**
- **Mean Inference Time:** **0.863 seconds per trip**
