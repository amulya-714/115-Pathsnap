"""viterbi.py - Hidden Markov Model and Viterbi decoding for map matching.

Implements sequential route reconstruction by finding the most likely sequence
of road segments given noisy GPS observations and road network topology.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


@dataclass
class HMMConfig:
    beta: float = 10.0                 # Newson & Krumm exponential scale for |d_net - d_gps|
    disconnect_penalty: float = 12.0    # Soft log penalty for unreachable transitions
    u_turn_penalty: float = 4.0        # Log penalty for immediate 180-degree U-turns
    emission_weight: float = 1.0       # Weight for classifier log emission probabilities
    min_prob_clip: float = 1e-6        # Epsilon to prevent log(0)
    dijkstra_cutoff_mult: float = 2.5  # Multiplier on d_gps for local Dijkstra radius
    dijkstra_cutoff_base: float = 50.0 # Base minimum search radius in metres


class ViterbiDecoder:
    """Sequential decoder using log-space Viterbi algorithm over road candidate states."""
    
    def __init__(self, config: HMMConfig = HMMConfig()):
        self.cfg = config

    def _get_local_shortest_paths(self, graph, source_node: int, max_dist: float) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Run bounded single-source Dijkstra from source_node up to max_dist metres.
        Returns:
            (distances: dict[node_id -> float], parent_arcs: dict[node_id -> arc_id])
        """
        dist = {source_node: 0.0}
        parent_arc = {}
        pq = [(0.0, source_node)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            start, end = graph._start[u], graph._start[u + 1]
            for arc_id in graph._order[start:end]:
                v = graph.v[arc_id]
                l = graph.length_m[arc_id]
                new_d = d + l
                if new_d <= max_dist and new_d < dist.get(v, float('inf')):
                    dist[v] = new_d
                    parent_arc[v] = arc_id
                    heapq.heappush(pq, (new_d, v))
        return dist, parent_arc

    def decode(
        self,
        obs: pd.DataFrame,
        candidates: pd.DataFrame,
        graph,
        scores: Optional[np.ndarray] = None
    ) -> Dict:
        """Decode optimal road segment sequence using log-space Viterbi.
        
        Args:
            obs: DataFrame of projected GPS fixes with columns ['obs_idx', 'x_m', 'y_m']
            candidates: DataFrame of candidate road segments with ['obs_idx', 'arc_id', 'along_frac']
            graph: RoadGraph object for the trip
            scores: Model predicted positive class probabilities (optional if present in candidates['score'])
            
        Returns:
            dict containing:
                'decoded_arcs': list of arc_id for each observation
                'reconstructed_route': contiguous list of road arc_ids traversing the graph
                'log_likelihood': final optimal log path score
                'top1_classifier_arcs': classifier-only greedy argmax arcs
                'n_obs': number of observations
                'continuity': direct graph continuity ratio
        """
        n_obs = len(obs)
        if n_obs == 0:
            return {
                "decoded_arcs": [],
                "reconstructed_route": [],
                "log_likelihood": 0.0,
                "top1_classifier_arcs": [],
                "n_obs": 0,
                "continuity": 1.0,
            }

        df = candidates.copy()
        if scores is not None:
            df["score"] = scores
        elif "score" not in df.columns:
            # Fallback to inverse distance if no model score provided
            df["score"] = 1.0 / (df["dist_m"] + 1.0)

        # Pre-group candidates by observation index
        cands_by_obs = {}
        for obs_idx, grp in df.groupby("obs_idx"):
            cands_by_obs[obs_idx] = {
                "arc_ids": grp["arc_id"].to_numpy(dtype=int),
                "scores": grp["score"].to_numpy(dtype=float),
                "along_fracs": grp["along_frac"].to_numpy(dtype=float),
                "dists": grp["dist_m"].to_numpy(dtype=float) if "dist_m" in grp else np.zeros(len(grp))
            }

        # Identify greedy classifier-only choice per observation
        top1_clf_arcs = []
        for t in range(n_obs):
            c = cands_by_obs.get(t)
            if c is not None and len(c["arc_ids"]) > 0:
                best_idx = np.lexsort((c["arc_ids"], -c["scores"]))[0]
                top1_clf_arcs.append(int(c["arc_ids"][best_idx]))
            else:
                top1_clf_arcs.append(-1)

        # Trellis arrays
        V: List[Dict[int, float]] = []
        BP: List[Dict[int, int]] = []
        
        # Track valid observation index for transitions (handling missing candidates)
        last_valid_t = -1
        
        for t in range(n_obs):
            c_curr = cands_by_obs.get(t)
            
            # Edge case: zero candidates for this observation
            if c_curr is None or len(c_curr["arc_ids"]) == 0:
                if len(V) == 0:
                    # Starting fix has no candidates
                    V.append({0: 0.0})
                    BP.append({0: 0})
                else:
                    # Carry forward previous probabilities with pass-through backpointer
                    V.append(V[-1].copy())
                    BP.append({k: k for k in V[-1].keys()})
                continue

            scores_curr = c_curr["scores"]
            log_em_curr = np.log(np.maximum(scores_curr, self.cfg.min_prob_clip))

            if last_valid_t == -1:
                # First observation with valid candidates: initialize prior
                V0 = {j: float(log_em_curr[j]) for j in range(len(scores_curr))}
                V.append(V0)
                BP.append({})
                last_valid_t = t
                continue

            c_prev = cands_by_obs[last_valid_t]
            dx = obs.x_m.iloc[t] - obs.x_m.iloc[last_valid_t]
            dy = obs.y_m.iloc[t] - obs.y_m.iloc[last_valid_t]
            d_gps = float(np.hypot(dx, dy))
            cutoff = max(d_gps * self.cfg.dijkstra_cutoff_mult + self.cfg.dijkstra_cutoff_base, 100.0)

            # Precompute Dijkstra from head node of each previous candidate
            dijkstra_cache = {}
            for i, u_arc in enumerate(c_prev["arc_ids"]):
                h_node = graph.v[u_arc]
                if h_node not in dijkstra_cache:
                    dijkstra_cache[h_node], _ = self._get_local_shortest_paths(graph, h_node, cutoff)

            Vt: Dict[int, float] = {}
            BPt: Dict[int, int] = {}
            prev_V = V[-1]

            for j, v_arc in enumerate(c_curr["arc_ids"]):
                best_val = -float("inf")
                best_prev = 0
                v_tail = graph.u[v_arc]
                v_along = c_curr["along_fracs"][j] * graph.length_m[v_arc]

                for i, u_arc in enumerate(c_prev["arc_ids"]):
                    if i not in prev_V:
                        continue
                    u_head = graph.v[u_arc]
                    u_rem = (1.0 - c_prev["along_fracs"][i]) * graph.length_m[u_arc]

                    if u_arc == v_arc:
                        # Same arc transition
                        delta_along = (c_curr["along_fracs"][j] - c_prev["along_fracs"][i]) * graph.length_m[u_arc]
                        if delta_along >= -5.0:
                            d_net = max(delta_along, 0.0)
                            log_tr = -abs(d_net - d_gps) / self.cfg.beta
                        else:
                            # Significant backward movement on directed arc (jitter penalty)
                            log_tr = -abs(abs(delta_along) - d_gps) / self.cfg.beta - 3.0
                    elif v_tail in dijkstra_cache[u_head]:
                        # Reachable through road network
                        path_len = dijkstra_cache[u_head][v_tail]
                        d_net = u_rem + path_len + v_along
                        log_tr = -abs(d_net - d_gps) / self.cfg.beta
                        if graph.reverse_id[u_arc] == v_arc:
                            log_tr -= self.cfg.u_turn_penalty
                    else:
                        # Disconnected / unreachable transition (soft fallback penalty)
                        log_tr = -abs(d_gps) / self.cfg.beta - self.cfg.disconnect_penalty

                    candidate_val = prev_V[i] + log_tr
                    if candidate_val > best_val:
                        best_val = candidate_val
                        best_prev = i

                Vt[j] = best_val + self.cfg.emission_weight * float(log_em_curr[j])
                BPt[j] = best_prev

            V.append(Vt)
            BP.append(BPt)
            last_valid_t = t

        # Backtrack optimal sequence
        best_final_k = max(V[-1].keys(), key=lambda k: V[-1][k])
        decoded_indices = [best_final_k]
        for t in range(n_obs - 1, 0, -1):
            prev_k = BP[t].get(decoded_indices[-1], 0)
            decoded_indices.append(prev_k)
        decoded_indices.reverse()

        # Map state indices back to arc_ids
        decoded_arcs = []
        for t, k in enumerate(decoded_indices):
            c = cands_by_obs.get(t)
            if c is not None and len(c["arc_ids"]) > 0 and k < len(c["arc_ids"]):
                decoded_arcs.append(int(c["arc_ids"][k]))
            else:
                # For zero candidate fix, borrow from previous fix if available
                prev_arc = decoded_arcs[-1] if decoded_arcs else -1
                decoded_arcs.append(prev_arc)

        # Reconstruct traversed route by bridging gaps between non-adjacent arcs
        reconstructed_route = self.reconstruct_route(decoded_arcs, graph)

        # Calculate continuity on decoded arcs
        arr = np.array(decoded_arcs, dtype=int)
        if len(arr) > 1:
            valid_pairs = (arr[:-1] >= 0) & (arr[1:] >= 0)
            b, a = arr[:-1][valid_pairs], arr[1:][valid_pairs]
            connected = (b == a) | (graph.v[b] == graph.u[a])
            continuity = float(connected.mean()) if len(connected) > 0 else 1.0
        else:
            continuity = 1.0

        return {
            "decoded_arcs": decoded_arcs,
            "reconstructed_route": reconstructed_route,
            "log_likelihood": float(V[-1][best_final_k]),
            "top1_classifier_arcs": top1_clf_arcs,
            "n_obs": n_obs,
            "continuity": continuity,
        }

    def reconstruct_route(self, decoded_arcs: List[int], graph) -> List[int]:
        """Reconstruct full traversed route by collapsing duplicates and bridging gaps.
        
        Args:
            decoded_arcs: Decoded road arc_id for each GPS observation
            graph: RoadGraph object
            
        Returns:
            Contiguous sequence of traversed arc_ids
        """
        if not decoded_arcs:
            return []

        # Step 1: Collapse consecutive identical arcs
        collapsed = [decoded_arcs[0]]
        for arc in decoded_arcs[1:]:
            if arc != collapsed[-1]:
                collapsed.append(arc)

        # Step 2: Bridge any non-adjacent hops with shortest directed path
        reconstructed = []
        for idx in range(len(collapsed)):
            curr_arc = collapsed[idx]
            if curr_arc < 0:
                continue

            if not reconstructed:
                reconstructed.append(curr_arc)
                continue

            prev_arc = reconstructed[-1]
            if prev_arc == curr_arc:
                continue

            # Check if immediately connected
            if graph.v[prev_arc] == graph.u[curr_arc]:
                reconstructed.append(curr_arc)
            else:
                # Find shortest path from prev_arc head node to curr_arc tail node
                h_node = graph.v[prev_arc]
                t_node = graph.u[curr_arc]
                
                dist, parent_arc = self._get_local_shortest_paths(graph, h_node, max_dist=1000.0)
                if t_node in parent_arc:
                    # Reconstruct intermediate path
                    bridge = []
                    curr_node = t_node
                    while curr_node in parent_arc and curr_node != h_node:
                        p_arc = parent_arc[curr_node]
                        bridge.append(p_arc)
                        curr_node = graph.u[p_arc]
                    bridge.reverse()
                    reconstructed.extend(bridge)
                reconstructed.append(curr_arc)

        return reconstructed
