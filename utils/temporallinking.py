#!/usr/bin/env python3
"""
temporallinking.py

Temporal linking utilities for Ichthys:
- Cost-based Hungarian matching between consecutive frames
- Distance + appearance gating
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_fallback_matches(
    *,
    matches,
    groups_tp1,
    track_memory,
    frame_num_tp1,
    dist_thresh,
    sim_thresh,
    gap_dist_scale=0.15,
    gap_sim_drop=0.04,
):
    """Build multi-frame fallback matches by searching older frames' groups.

    This mirrors the inlined logic previously in `train.py`.

    Args:
        matches: list of (i_g, j_g, w) from immediate-frame `temporal_linking`.
        groups_tp1: list of current (t+1) groups. Each group should have:
            - 'emb_tensor' (torch 1D tensor, normalized)
            - 'X' (3D point list)
            - 'conf' (float)
        track_memory: deque/list of dicts: {'frame': int, 'groups': [group_like]}.
            Each stored group_like should have 'emb_tensor', 'X', 'conf'.
        frame_num_tp1: frame index for t+1.
        dist_thresh: base distance threshold.
        sim_thresh: base similarity threshold.
        gap_dist_scale: multiplicative expansion factor per gap frame.
        gap_sim_drop: similarity threshold drop per gap frame.

    Returns:
        matched_new: set of j indices in groups_tp1 that are matched (immediate or fallback).
        fallback_matches: list of (emb_old_tensor, emb_new_tensor, weight).
    """
    # local import so this module stays light for non-training use
    import torch

    matched_new = set([m[1] for m in matches])
    fallback_matches = []  # list of (emb_old_tensor, emb_new_tensor, weight)

    for mem_entry in track_memory:
        past_frame_num = mem_entry["frame"]
        gap = max(1, int(frame_num_tp1) - int(past_frame_num))
        dist_gate = float(dist_thresh) * (1.0 + float(gap_dist_scale) * gap)
        sim_gate = float(sim_thresh) - float(gap_sim_drop) * gap

        for new_idx, g_new in enumerate(groups_tp1):
            if new_idx in matched_new:
                continue  # already matched via immediate previous frame

            emb_new = g_new.get("emb_tensor", None)
            X_new = np.array(g_new.get("X", [0, 0, 0]), dtype=np.float32)
            conf_new = float(g_new.get("conf", 0.0))
            if emb_new is None:
                continue

            best_candidate = None
            best_score = -1.0
            for g_old in mem_entry.get("groups", []):
                emb_old = g_old.get("emb_tensor", None)
                X_old = np.array(g_old.get("X", [0, 0, 0]), dtype=np.float32)
                conf_old = float(g_old.get("conf", 0.0))
                if emb_old is None:
                    continue

                d3 = float(np.linalg.norm(X_new - X_old))
                sim = float(torch.dot(emb_old, emb_new).item())
                if d3 <= dist_gate and sim >= sim_gate:
                    # score prefer high similarity and lower distance (weighted sum)
                    score = sim - 0.25 * (d3 / (dist_gate + 1e-8))
                    if score > best_score:
                        best_score = score
                        w_fb = min(conf_old, conf_new)
                        best_candidate = (emb_old, emb_new, w_fb)

            if best_candidate is not None:
                fallback_matches.append(best_candidate)
                matched_new.add(new_idx)  # mark as bridged

    return matched_new, fallback_matches


def temporal_linking(groups_t, groups_tp1,
                     alpha=0.6, beta=0.4,
                     dist_thresh=1.0, sim_thresh=0.6):
    """
    Link groups between consecutive frames using Hungarian algorithm.
    
    This version does NOT use active track velocity predictions. Matching cost is 
    a combination of normalized 3D distance and appearance dissimilarity (1 - cosine).
    
    Args:
        groups_t: list of group dicts from frame t with 'X', 'emb_np', 'conf'
        groups_tp1: list of group dicts from frame t+1
        alpha: weight for distance term
        beta: weight for appearance term
        dist_thresh: distance gating threshold
        sim_thresh: similarity gating threshold
        
    Returns:
        matches: list of (idx_i, idx_j, weight) tuples for matched groups
    """
    M = len(groups_t)
    N = len(groups_tp1)
    if M == 0 or N == 0:
        return []

    # Build cost matrix (lower cost = better match)
    cost = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        Xi = np.array(groups_t[i]['X'])
        emb_i = groups_t[i].get('emb_np', None)
        conf_i = groups_t[i].get('conf', 0.0)
        for j in range(N):
            Xj = np.array(groups_tp1[j]['X'])
            emb_j = groups_tp1[j].get('emb_np', None)
            conf_j = groups_tp1[j].get('conf', 0.0)
            d3 = np.linalg.norm(Xj - Xi)
            # appearance dissimilarity (1 - cosine)
            if emb_i is None or emb_j is None:
                app = 1.0
            else:
                app = 1.0 - float(np.dot(emb_i, emb_j) / ((np.linalg.norm(emb_i) * np.linalg.norm(emb_j)) + 1e-8))
            # cost: normalized distance + appearance dissimilarity
            cost_ij = alpha * (d3 / max(1.0, dist_thresh)) + beta * app
            cost[i, j] = cost_ij

    # Solve assignment (Hungarian) to minimize total cost
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        # gating
        Xi = np.array(groups_t[r]['X'])
        Xj = np.array(groups_tp1[c]['X'])
        d3 = np.linalg.norm(Xj - Xi)
        emb_i = groups_t[r].get('emb_np', None)
        emb_j = groups_tp1[c].get('emb_np', None)
        if emb_i is not None and emb_j is not None:
            sim = float(np.dot(emb_i, emb_j) / ((np.linalg.norm(emb_i) * np.linalg.norm(emb_j)) + 1e-8))
        else:
            sim = 0.0
        conf_pair = min(groups_t[r].get('conf', 0.0), groups_tp1[c].get('conf', 0.0))
        if d3 <= dist_thresh and sim >= sim_thresh:
            matches.append((r, c, float(conf_pair)))
    return matches
