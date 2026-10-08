#!/usr/bin/env python3
"""
grouping.py

3D grouping and triangulation utilities for Ichthys:
- Multi-view triangulation
- Hybrid grouping using pairwise probabilities and embeddings
- Per-camera Hungarian matching with transitive merge
"""
import numpy as np
from collections import defaultdict


def triangulate_multiview(pts, Ks, RTs):
    """
    Triangulate a 3D point from multiple views.
    
    Args:
        pts: list of (x,y) image coordinates (len m)
        Ks: list of camera intrinsic matrices
        RTs: list of camera extrinsic matrices [R|T]
        
    Returns:
        X: 3D point [x, y, z]
    """
    A_rows = []
    for (x, y), K, RT in zip(pts, Ks, RTs):
        P = K @ RT
        A_rows.append(x * P[2, :] - P[0, :])
        A_rows.append(y * P[2, :] - P[1, :])
    A = np.stack(A_rows, axis=0)
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X = X / X[-1]
    return X[:3]


def group_and_triangulate(P_np, Fp_np, detections, cams,
                          thr_assoc=0.6,
                          match_mode: str = "hungarian",
                          debug=False, debug_file=None):
    """
    Hybrid grouping: use pairwise probability matrix P and embeddings Fp to group detections.
    
    Args:
        P_np: numpy array N x N (pairwise association probabilities)
        Fp_np: numpy array N x d (detection embeddings)
        detections: list of detection dicts
        cams: camera dict with 'camX-K' and 'camX-R|T'
        thr_assoc: association threshold for matching
        debug: if True, write debug logs
        debug_file: path to debug log file
        
    Returns:
        groups: list of group dicts with 'members', 'X', 'conf'
    """

    # NOTE: reprojection error gating is intentionally NOT used here.
    # For small fish, the centroid can be noisy enough that reprojection px error
    # becomes too strict and hurts recall.
    N = P_np.shape[0]
    groups = []
    if N == 0:
        return groups

    # compute cosine similarity matrix from embeddings
    eps = 1e-8
    F_norm = Fp_np / (np.linalg.norm(Fp_np, axis=1, keepdims=True) + eps)
    cos_sim = F_norm @ F_norm.T

    # combined affinity: weight pairwise probability and embedding similarity
    wp = 0.6
    wf = 0.4
    A = wp * P_np + wf * ((cos_sim + 1.0) / 2.0)  # normalize cos_sim to [0,1]

    # build index lists per camera
    cam_ids = [d['cam_id'] for d in detections]
    cam_to_idxs = defaultdict(list)
    for idx, cid in enumerate(cam_ids):
        cam_to_idxs[cid].append(idx)

    edges = set()
    try:
        def _greedy_match(A_sub: np.ndarray):
            """Return greedy max-matching indices for a rectangular affinity matrix.

            Returns:
                row_ind, col_ind: lists of matched row/col indices.
            """
            if A_sub.size == 0:
                return [], []
            # Sort all pairs by descending affinity.
            flat = A_sub.ravel()
            order = np.argsort(flat)[::-1]
            rows, cols = np.unravel_index(order, A_sub.shape)
            taken_r = set()
            taken_c = set()
            row_ind = []
            col_ind = []
            for r, c in zip(rows.tolist(), cols.tolist()):
                if r in taken_r or c in taken_c:
                    continue
                row_ind.append(r)
                col_ind.append(c)
                taken_r.add(r)
                taken_c.add(c)
                if len(taken_r) == A_sub.shape[0] or len(taken_c) == A_sub.shape[1]:
                    break
            return row_ind, col_ind

        match_mode_l = str(match_mode or "hungarian").lower()
        use_greedy = match_mode_l in {"greedy", "greedygroup", "approx", "fast"}
        if not use_greedy:
            # Import SciPy lazily so greedy mode can run without SciPy.
            from scipy.optimize import linear_sum_assignment
        # For each unordered camera pair, match detections
        cam_list = list(cam_to_idxs.keys())
        for i in range(len(cam_list)):
            for j in range(i + 1, len(cam_list)):
                ca = cam_list[i]
                cb = cam_list[j]
                idxs_a = cam_to_idxs[ca]
                idxs_b = cam_to_idxs[cb]
                if len(idxs_a) == 0 or len(idxs_b) == 0:
                    continue
                A_sub = A[np.ix_(idxs_a, idxs_b)]
                # Hungarian solves min-cost; we want max-affinity -> cost = -A_sub
                if use_greedy:
                    row_ind, col_ind = _greedy_match(A_sub)
                else:
                    try:
                        row_ind, col_ind = linear_sum_assignment(-A_sub)
                    except Exception:
                        # if SciPy errors, fall back to greedy matching
                        row_ind, col_ind = _greedy_match(A_sub)
                # keep matches above thr_assoc
                for ra, cb_idx in zip(row_ind, col_ind):
                    affinity = float(A_sub[ra, cb_idx])
                    if affinity >= thr_assoc:
                        edges.add(tuple(sorted((idxs_a[ra], idxs_b[cb_idx]))))
        
        # build graph and compute connected components (transitive merge)
        # simple union-find
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra
        for (a, b) in edges:
            union(a, b)
        comps = defaultdict(list)
        for idx in range(N):
            comps[find(idx)].append(idx)

    # For each component, prune conflicting same-camera detections by keeping highest det_score
        for root, members in comps.items():
            # Extract only detections that actually participated in edges (ignore isolated indices)
            comp_edges_present = any((tuple(sorted((a, b))) in edges) for a in members for b in members if a != b)
            if not comp_edges_present:
                continue
            # group members by camera and pick best per camera
            keep = []
            by_cam = defaultdict(list)
            for m in members:
                by_cam[cam_ids[m]].append(m)
            for cam, lst in by_cam.items():
                if len(lst) == 1:
                    keep.append(lst[0])
                else:
                    # choose highest detection score
                    best = max(lst, key=lambda x: float(detections[x].get('det_score', 0.0)))
                    keep.append(best)
            # require at least 2 unique cameras (3-view datasets often have missing detections)
            cams_in_group = set([cam_ids[k] for k in keep])
            n_cams = len(cams_in_group)
            if n_cams < 2:
                if debug and debug_file is not None:
                    with open(debug_file, 'a') as df:
                        df.write(f"Component {root} rejected: only {n_cams} cameras\n")
                continue

            # Stricter association requirement for 2-view groups to reduce false triangulations.
            if n_cams == 2:
                best_aff = -1.0
                for ia in keep:
                    for ib in keep:
                        if ia == ib:
                            continue
                        if cam_ids[ia] == cam_ids[ib]:
                            continue
                        best_aff = max(best_aff, float(A[ia, ib]))
                # heuristic: require higher score than 3-view case
                thr_assoc_2cam = max(float(thr_assoc), 0.80)
                if best_aff < thr_assoc_2cam:
                    if debug and debug_file is not None:
                        with open(debug_file, 'a') as df:
                            df.write(
                                f"Component {root} rejected: 2-cam best_aff={best_aff:.3f} < thr_assoc_2cam={thr_assoc_2cam:.3f}\n"
                            )
                    continue

            # triangulate using kept members' centroids
            pts = []
            Ks = []
            RTs = []
            for idx in keep:
                pts.append(detections[idx].get('segmentation-centroid'))
                cam_id = detections[idx]['cam_id']
                Ks.append(np.array(cams[f"cam{cam_id}-K"]))
                RTs.append(np.array(cams[f"cam{cam_id}-R|T"]))
            try:
                X = triangulate_multiview(pts, Ks, RTs)
            except Exception as e:
                if debug and debug_file is not None:
                    with open(debug_file, 'a') as df:
                        df.write(f"Component {root} triangulation failed: {e}\n")
                continue

            # bbox containment check for both 2cams and 3cams
            contained_all = True
            for (x, y), K, RT, idx in zip(pts, Ks, RTs, keep):
                Pm = K @ RT
                proj = (Pm @ np.hstack([X, 1.0]))
                if abs(proj[2]) < 1e-9:
                    contained_all = False
                    continue
                proj = proj / proj[2]
                preds = proj[:2]
                # bbox containment check
                bbox = detections[idx]['bbox']
                x1, y1, x2, y2 = bbox
                px, py = float(preds[0]), float(preds[1])
                if not (px >= x1 and px <= x2 and py >= y1 and py <= y2):
                    contained_all = False
                    
            
            if not contained_all:
                if debug and debug_file is not None:
                    with open(debug_file, 'a') as df:
                        df.write(f"Component {root} rejected: bbox containment failed\n")
                continue

            
            # Accept group if the triangulated point reprojection lies inside each detection bbox.
            # confidence based on number of cameras
            # confidence: 3-view groups are more reliable than 2-view
            conf = min(1.0, 0.2 + 0.2 * n_cams)  # 2 cams -> 0.6, 3 cams -> 0.8
            if n_cams == 2:
                conf *= 0.75
            if debug and debug_file is not None:
                with open(debug_file, 'a') as df:
                    df.write(f"Component {root} accepted by bbox containment. conf={conf:.3f}, members={keep}\n")
            groups.append({'members': keep, 'X': X.tolist(), 'conf': conf, 'num_cams': int(n_cams)})
    except Exception as e:
        # fallback to empty groups but log debug
        if debug and debug_file is not None:
            with open(debug_file, 'a') as df:
                df.write(f"group_and_triangulate crashed: {e}\n")
    return groups
