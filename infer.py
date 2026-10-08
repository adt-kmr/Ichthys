#!/usr/bin/env python3
"""Ichthys inference without a known object-count prior.

This script does NOT cheat with --count.  Instead it leverages every piece of
what training actually learned:

1. **Embedding similarity** (contrastive loss payoff):
   Groups are linked across frames by cosine similarity of their learned
   embeddings — the same metric training optimized with InfoNCE.

2. **PredictorM embedding forecast** (temporal prediction loss payoff):
   When a track disappears for a few frames (occlusion), we use the trained
   PredictorM to *predict* what its embedding would look like now, and match
   against that prediction.  Training explicitly optimized
   ``MSE(predictor(emb_t), emb_{t+1})``, so this is not a heuristic — it is
   the direct inference-time benefit of the temporal loss.

3. **Multi-frame fallback matching** (``build_fallback_matches`` logic):
   Exactly like training, we keep a sliding window of past frames.
   Unmatched current groups are searched against older track memories with
   *relaxed* distance & similarity gates (same formulas as training:
   ``dist_gate *= (1 + 0.15*gap)``, ``sim_gate -= 0.04*gap``).

4. **Dynamic track birth & death** — no ``--count`` needed:
   New tracks spawn when a group cannot be matched to any existing track.
   Tracks are killed after ``max_gap`` consecutive frames of no observation.
   Tracks with fewer than ``--min_obs`` total observations are filtered as
   ghosts (noise).

Architecture
------------
Output format:
    object,Timestamp,X,Y,Z,group
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---- import path shim -------------------------------------------------
# Support direct execution from any working directory.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR

for p in (PROJECT_ROOT, THIS_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from utils.encode_features import GeometricFeatureEncoder
from utils.association_module import AssociationTransformer
from utils.association_module_noattn import AssociationNoGlobalAttention
from utils.grouping import group_and_triangulate

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

# ---------------------------------------------------------------------------
# PredictorM — same architecture as training (must match checkpoint)
# ---------------------------------------------------------------------------
class PredictorM(nn.Module):
    """Embedding predictor: maps emb(t) → predicted emb(t+1).

    Training optimized this with MSE loss on matched pairs, so at inference we
    can "roll forward" an embedding across occlusion gaps.
    """
    def __init__(self, d: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers shared with work.py
# ---------------------------------------------------------------------------
def _resolve_frame_image(base_path: Path) -> str:
    p_jpeg = base_path.with_suffix(".jpeg")
    if p_jpeg.exists():
        return str(p_jpeg)
    p_jpg = base_path.with_suffix(".jpg")
    if p_jpg.exists():
        return str(p_jpg)
    return str(p_jpeg)


def rle_to_centroid(rle, bbox=None):
    try:
        from pycocotools import mask as mask_util
        mask = mask_util.decode(rle)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
            return [0.0, 0.0]
        return [float(xs.mean()), float(ys.mean())]
    except Exception:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
        return [0.0, 0.0]


def parse_scene_json(scene_json_path: str, workers: int = 1):
    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)
    cams = {}
    cam_data = {ce["cam"]: ce["info"] for ce in scene_data}
    for cam_id, info in cam_data.items():
        cams[f"cam{cam_id}-K"] = np.array(info["K"])
        cams[f"cam{cam_id}-R|T"] = np.array(info["R|T"])
    max_frames = max(len(info["frames"]) for info in cam_data.values())

    def build_frame(t: int):
        detections = []
        frame_number = None
        for cam_id, info in cam_data.items():
            if t >= len(info["frames"]):
                continue
            frame = info["frames"][t]
            frame_number = frame["frame"]
            for ann in frame["annotations"]:
                bbox = ann["bbox"]
                centroid = rle_to_centroid(ann.get("segmentation", {}), bbox=bbox)
                detections.append({
                    "cam_id": cam_id,
                    "bbox": bbox,
                    "segmentation-centroid": centroid,
                    "det_score": ann.get("score", 1.0),
                })
        if frame_number is not None:
            return {"frame": frame_number, "detections": detections}
        return None

    frames: List[dict] = []
    if workers is None or workers <= 1:
        for t in range(max_frames):
            fr = build_frame(t)
            if fr is not None:
                frames.append(fr)
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for fr in ex.map(build_frame, range(max_frames)):
                    if fr is not None:
                        frames.append(fr)
        except Exception:
            for t in range(max_frames):
                fr = build_frame(t)
                if fr is not None:
                    frames.append(fr)

    scene_name = os.path.basename(scene_json_path)[:-5]
    return cams, frames, scene_name


def make_boxes_tensor(detections, img_w, img_h, device):
    boxes = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        boxes.append([cx / img_w, cy / img_h, w / img_w, h / img_h])
    return torch.tensor(boxes, dtype=torch.float32, device=device)


def attach_group_embeddings(groups, H_tensor):
    for g in groups:
        idxs = g.get("members", [])
        if len(idxs) == 0:
            g["emb_tensor"] = torch.zeros(H_tensor.shape[1], device=H_tensor.device)
            g["emb_np"] = np.zeros((H_tensor.shape[1],), dtype=np.float32)
            continue
        emb = H_tensor[idxs].mean(dim=0)
        emb_norm = F.normalize(emb, p=2, dim=0)
        g["emb_tensor"] = emb_norm
        g["emb_np"] = emb_norm.detach().cpu().numpy()


def label_from_index(idx: int) -> str:
    letters: List[str] = []
    idx0 = idx
    while True:
        idx, rem = divmod(idx0, 26)
        letters.append(chr(ord("A") + rem))
        if idx == 0:
            break
        idx0 = idx - 1
    return "".join(reversed(letters))


# ---------------------------------------------------------------------------
# Active Track representation
# ---------------------------------------------------------------------------
@dataclass
class ActiveTrack:
    """A living track maintained across frames."""
    label: str
    # last observed 3D position
    x: np.ndarray
    # last observed embedding (numpy, L2-normalized)
    emb_np: np.ndarray
    # last observed embedding (torch tensor, for predictor)
    emb_tensor: torch.Tensor
    # confidence of last observation
    conf: float = 1.0
    # first frame index where this track was born
    first_seen_t: int = 0
    # last frame index where this track was observed (not interpolated)
    last_seen_t: int = 0
    # number of consecutive frames with no observation
    age_since_seen: int = 0
    # velocity estimate (position delta per frame)
    velocity: Optional[np.ndarray] = None
    # total number of observations
    n_obs: int = 1
    # --- for post-hoc merge ---
    # embedding at birth (first observation)
    first_emb_np: Optional[np.ndarray] = None
    # position at birth
    first_x: Optional[np.ndarray] = None


@dataclass
class DeadTrack:
    """Snapshot of a track that was killed, for post-hoc merge."""
    label: str
    first_seen_t: int
    last_seen_t: int
    n_obs: int
    # embedding snapshots
    last_emb_np: np.ndarray
    first_emb_np: np.ndarray
    # position snapshots
    last_x: np.ndarray
    first_x: np.ndarray


# ---------------------------------------------------------------------------
# Core: track-to-group matching using trained embeddings + predictor
# ---------------------------------------------------------------------------
def _cos_sim_np(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def temporal_linking_tracks(
    tracks: List[ActiveTrack],
    groups: List[dict],
    *,
    alpha: float = 0.6,
    beta: float = 0.4,
    dist_thresh: float = 0.5,
    sim_thresh: float = 0.6,
    predictor: Optional[nn.Module] = None,
    device: str = "cpu",
) -> List[Tuple[int, int, float]]:
    """Link active tracks to current-frame groups using Hungarian matching.

    This uses the same weighted distance-and-similarity cost form as training:
        cost = alpha * (d3 / dist_thresh) + beta * (1 - cosine_sim)

    When a track has been unseen for more than zero frames
    AND a predictor is available, we use the trained PredictorM to *predict*
    what the track's embedding should look like now (rolling forward through
    the gap).  This is the direct inference-time payoff of the temporal
    prediction loss.

    Gates are relaxed for older tracks using the **same formula** as training's
    ``build_fallback_matches``:
        dist_gate = dist_thresh * (1 + 0.15 * gap)
        sim_gate  = sim_thresh  - 0.04 * gap
    """
    M = len(tracks)
    N = len(groups)
    if M == 0 or N == 0:
        return []

    cost = np.full((M, N), 1e6, dtype=np.float32)

    for i, trk in enumerate(tracks):
        gap = trk.age_since_seen

        # --- Embedding to use for matching ---
        # If the track has been missing and we have the trained predictor,
        # roll the embedding forward to approximate what it would be NOW.
        if gap > 0 and predictor is not None:
            with torch.no_grad():
                emb_pred = trk.emb_tensor.to(device)
                # Roll forward: apply predictor once per gap frame (cap at 5
                # to avoid drift on very long gaps)
                for _step in range(min(gap, 5)):
                    emb_pred = predictor(emb_pred)
                emb_pred = F.normalize(emb_pred, p=2, dim=0)
                emb_i = emb_pred.detach().cpu().numpy()
        else:
            emb_i = trk.emb_np

        # --- Position prediction ---
        # Use constant-velocity model if we have velocity and there's a gap
        if trk.velocity is not None and gap > 0:
            x_pred = trk.x + trk.velocity * float(gap + 1)
        else:
            x_pred = trk.x

        # --- Gate relaxation (same as build_fallback_matches in training) ---
        local_dist = dist_thresh * (1.0 + 0.15 * gap)
        local_sim = max(0.1, sim_thresh - 0.04 * gap)

        for j, g in enumerate(groups):
            Xj = np.array(g.get("X", [0, 0, 0]), dtype=np.float32)
            emb_j = g.get("emb_np", None)

            d3 = float(np.linalg.norm(Xj - x_pred))
            sim = _cos_sim_np(emb_i, emb_j)

            # Gating
            if d3 > local_dist or sim < local_sim:
                continue

            app = 1.0 - sim
            cost[i, j] = float(alpha * (d3 / max(1e-6, local_dist)) + beta * app)

    # Solve with Hungarian (same algorithm as training's temporal_linking)
    if linear_sum_assignment is None:
        # greedy fallback
        matches: List[Tuple[int, int, float]] = []
        used_j: Set[int] = set()
        for i in range(M):
            j_best = int(np.argmin(cost[i]))
            if cost[i, j_best] >= 1e5 or j_best in used_j:
                continue
            matches.append((i, j_best, 1.0))
            used_j.add(j_best)
    else:
        r, c = linear_sum_assignment(cost)
        matches = [(int(ri), int(ci), 1.0) for ri, ci in zip(r, c) if cost[ri, ci] < 1e5]

    return matches


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
def run_inference_new(
    ckpt_path: str,
    scene_json_path: str,
    images_root: str,
    output_path: str = "3dtracking_new.txt",
    *,
    # Grouping defaults used for the reported inference results.
    thr_assoc: float = 0.5,
    # temporal linking — same defaults as training
    alpha: float = 0.6,
    beta: float = 0.4,
    dist_thresh: float = 0.5,
    sim_thresh: float = 0.6,
    # track management
    max_gap: int = 15,
    min_conf: float = 0.05,
    min_obs_to_output: int = 2,
    # gap filling (linear interp between observations)
    fill_gaps: bool = True,
    # runtime
    device: Optional[str] = None,
    workers: int = 16,
    fp16: bool = False,
    use_rope: bool = True,
    use_dino: bool | None = None,
    disable_predictor: bool = False,
):
    """Run inference using training-faithful temporal logic.

    No ``--count`` needed.  Tracks are born and killed dynamically.
    The trained PredictorM is used to bridge occlusion gaps.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ load
    cams, frames, scene_name = parse_scene_json(scene_json_path, workers=workers)
    print(f"Loaded scene: {scene_name}  frames={len(frames)}")

    ckpt = torch.load(ckpt_path, map_location=device)
    enc_sd = ckpt.get("encoder_state_dict", {})
    ckpt_has_dino = any(str(k).startswith("dino_model.") for k in enc_sd.keys())
    if use_dino is None:
        use_dino = bool(ckpt_has_dino)

    encoder = GeometricFeatureEncoder(
        d_model=128,
        hidden_dim=64,
        num_cams=3,
        image_size=(1920, 1080),
        use_dino=bool(use_dino),
        dinov3_repo_dir=str(PROJECT_ROOT / "dinov3"),
        weights=str(PROJECT_ROOT / "DINOV3Model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        device=device,
    ).to(device)
    encoder.load_state_dict(enc_sd, strict=True)

    raw_assoc_sd = ckpt["assoc_state_dict"]
    assoc_sd = {k: v for k, v in raw_assoc_sd.items() if not k.startswith("pos_enc.")}
    has_rope = any(k.startswith("rope.") for k in assoc_sd.keys())
    no_global_attention = any(k.startswith("transformer.net.") for k in assoc_sd)
    if use_rope and not has_rope:
        print("[warn] RoPE requested but checkpoint has no rope.* params; disabling RoPE.")
        use_rope = False

    association_class = AssociationNoGlobalAttention if no_global_attention else AssociationTransformer
    assoc_model = association_class(
        in_dim=encoder.total_d_model,
        d_model=128,
        num_layers=4,
        num_heads=4,
        use_rope=use_rope,
    ).to(device)
    try:
        assoc_model.load_state_dict(assoc_sd, strict=use_rope and has_rope)
    except RuntimeError:
        assoc_model.load_state_dict(assoc_sd, strict=False)

    # --- Load PredictorM (the temporal prediction head trained with L_temp) ---
    predictor = PredictorM(d=128, hidden=256).to(device)
    predictor_loaded = False
    if disable_predictor:
        print("[info] PredictorM DISABLED via --no_predictor (ablation mode).")
    elif "predictor_state_dict" in ckpt:
        try:
            predictor.load_state_dict(ckpt["predictor_state_dict"])
            predictor_loaded = True
            print("[info] PredictorM loaded — temporal embedding prediction ENABLED.")
        except Exception as e:
            print(f"[warn] Could not load PredictorM: {e}")
    else:
        print("[warn] Checkpoint has no predictor_state_dict — running without embedding prediction.")

    encoder.eval()
    assoc_model.eval()
    predictor.eval()

    print(
        f"Checkpoint: {ckpt_path}\n"
        f"  association={'tokenwise MLP' if no_global_attention else 'global attention'}  "
        f"use_dino={use_dino}  RoPE={'ON' if use_rope else 'OFF'}  "
        f"predictor={'ON' if predictor_loaded else 'OFF'}\n"
        f"  temporal params: alpha={alpha} beta={beta} dist_thresh={dist_thresh} "
        f"sim_thresh={sim_thresh} max_gap={max_gap}"
    )

    # ------------------------------------------------------------ image setup
    cam_ids_from_cams = sorted(
        {int(k[3:].split("-")[0]) for k in cams.keys() if k.endswith("-K")}
    )
    root_path = Path(images_root)

    def cams_exist_at(base: Path):
        return all((base / f"cam{cid}").exists() for cid in cam_ids_from_cams)

    if cams_exist_at(root_path):
        image_base = root_path
    elif cams_exist_at(root_path / scene_name):
        image_base = root_path / scene_name
    else:
        image_base = root_path

    # set image size once
    try:
        if len(frames) > 0 and len(cam_ids_from_cams) > 0 and hasattr(encoder, "set_image_size_from_path"):
            first_frame_num = int(frames[0]["frame"])
            first_cam_id = int(cam_ids_from_cams[0])
            base0 = image_base / f"cam{first_cam_id}" / f"frame_{first_frame_num:04d}"
            img0 = _resolve_frame_image(base0)
            if os.path.exists(img0):
                encoder.set_image_size_from_path(img0)
    except Exception:
        pass

    # ============================================================
    # PASS 1: Per-frame encode + group (identical to work.py)
    # ============================================================
    is_cuda = device.startswith("cuda")
    if fp16 and not is_cuda:
        print("[warn] --fp16 requested but CUDA not available; using fp32")
    amp_ctx = torch.amp.autocast("cuda", enabled=bool(fp16 and is_cuda))

    all_frame_groups: List[List[dict]] = []  # index = frame idx

    t0 = time.time()
    with torch.no_grad():
        for idx_frame, frame in enumerate(frames):
            frame_num = int(frame["frame"])
            detections = frame["detections"]

            images = []
            missing_any = False
            for cid in cam_ids_from_cams:
                base = image_base / f"cam{cid}" / f"frame_{frame_num:04d}"
                img_path = Path(_resolve_frame_image(base))
                if not img_path.exists():
                    missing_any = True
                images.append(str(img_path))
            if missing_any:
                all_frame_groups.append([])
                continue

            with amp_ctx:
                F_feat, _raw = encoder(detections, cams, images, return_raw=True)
            if F_feat.numel() == 0:
                all_frame_groups.append([])
                continue
            F_feat = F_feat.to(device)

            img_w, img_h = encoder.img_w, encoder.img_h
            boxes = make_boxes_tensor(detections, img_w, img_h, device)
            cam_ids = torch.tensor([d["cam_id"] for d in detections], dtype=torch.long, device=device)

            with amp_ctx:
                Xproj = assoc_model.input_proj(F_feat)
                if (assoc_model.use_rope and boxes is not None
                        and hasattr(assoc_model, "rope") and assoc_model.rope is not None):
                    positions = torch.stack([boxes[:, 1], boxes[:, 0]], dim=-1)
                    Xproj = assoc_model.rope(Xproj, positions)
                H = assoc_model.transformer(Xproj.unsqueeze(0)).squeeze(0)
                N = H.size(0)
                P = torch.zeros((N, N), dtype=H.dtype, device=H.device)
                if N > 0:
                    cam_mat_i = cam_ids.view(N, 1).expand(N, N)
                    cam_mat_j = cam_ids.view(1, N).expand(N, N)
                    cross_mask = cam_mat_i != cam_mat_j
                    idx_pairs = cross_mask.nonzero(as_tuple=False)
                    if idx_pairs.numel() > 0:
                        Hi = H.index_select(0, idx_pairs[:, 0])
                        Hj = H.index_select(0, idx_pairs[:, 1])
                        pair_feats = torch.cat([Hi, Hj], dim=-1)
                        logits = assoc_model.pairwise_mlp(pair_feats).squeeze(-1)
                        probs = torch.sigmoid(logits)
                        P[idx_pairs[:, 0], idx_pairs[:, 1]] = probs.to(P.dtype)

            H_np = H.detach().cpu().numpy()
            P_np = P.detach().cpu().numpy()

            groups = group_and_triangulate(
                P_np, H_np, detections, cams,
                thr_assoc=thr_assoc,
            )

            # prune overlapping groups (keep higher conf)
            if len(groups) > 0:
                groups_sorted = sorted(groups, key=lambda g: float(g.get("conf", 0.0)), reverse=True)
                used_det: Set[int] = set()
                pruned: List[dict] = []
                for g in groups_sorted:
                    if float(g.get("conf", 0.0)) < float(min_conf):
                        continue
                    mem = g.get("members", [])
                    if any(m in used_det for m in mem):
                        continue
                    pruned.append(g)
                    used_det.update(mem)
                groups = pruned

            attach_group_embeddings(groups, H)
            all_frame_groups.append(groups)

            if (idx_frame + 1) % 50 == 0:
                elapsed = time.time() - t0
                avg = elapsed / (idx_frame + 1)
                eta = avg * (len(frames) - idx_frame - 1)
                print(f"  [encode] {idx_frame+1}/{len(frames)} "
                      f"groups={len(groups)} avg={avg:.3f}s/frame ETA={eta:.0f}s")

    t_encode = time.time() - t0
    print(f"Encoding done. frames={len(all_frame_groups)} time={t_encode:.1f}s")

    # ============================================================
    # PASS 2: TEMPORAL TRACKING — training-faithful, no count prior
    # ============================================================
    active_tracks: List[ActiveTrack] = []
    dead_tracks: List[DeadTrack] = []  # graveyard for post-hoc merge
    next_label_idx = 0

    # Raw output rows: (label, frame_idx, x, y, z, group_str)
    raw_rows: List[Tuple[str, int, float, float, float, str]] = []
    # Per-track observation list for gap filling later
    track_observations: Dict[str, List[Tuple[int, float, float, float]]] = {}

    # Map internal index → actual frame number from the scene JSON
    idx_to_frame: Dict[int, int] = {}
    for fi, fr in enumerate(frames):
        idx_to_frame[fi] = int(fr["frame"])

    t1 = time.time()

    for t_idx in range(len(all_frame_groups)):
        groups = all_frame_groups[t_idx]

        if len(groups) == 0:
            # No groups this frame — age all tracks
            for trk in active_tracks:
                trk.age_since_seen += 1
            # Kill dead tracks
            for trk in active_tracks:
                if trk.age_since_seen > max_gap:
                    dead_tracks.append(DeadTrack(
                        label=trk.label, first_seen_t=trk.first_seen_t,
                        last_seen_t=trk.last_seen_t, n_obs=trk.n_obs,
                        last_emb_np=trk.emb_np.copy(), first_emb_np=trk.first_emb_np.copy() if trk.first_emb_np is not None else trk.emb_np.copy(),
                        last_x=trk.x.copy(), first_x=trk.first_x.copy() if trk.first_x is not None else trk.x.copy(),
                    ))
            active_tracks = [trk for trk in active_tracks if trk.age_since_seen <= max_gap]
            continue

        # ---- Match active tracks → current groups ----
        if len(active_tracks) > 0:
            matches = temporal_linking_tracks(
                active_tracks,
                groups,
                alpha=alpha,
                beta=beta,
                dist_thresh=dist_thresh,
                sim_thresh=sim_thresh,
                predictor=predictor if predictor_loaded else None,
                device=device,
            )
        else:
            matches = []

        matched_trk_idxs: Set[int] = set()
        matched_grp_idxs: Set[int] = set()

        # ---- Update matched tracks ----
        for trk_i, grp_j, _w in matches:
            matched_trk_idxs.add(trk_i)
            matched_grp_idxs.add(grp_j)

            trk = active_tracks[trk_i]
            g = groups[grp_j]
            x_new = np.array(g["X"], dtype=np.float32)

            # Update velocity (smoothed)
            if trk.age_since_seen == 0:
                # Was seen previous frame — simple delta
                new_vel = x_new - trk.x
                if trk.velocity is not None:
                    # Exponential moving average for stability
                    trk.velocity = 0.7 * new_vel + 0.3 * trk.velocity
                else:
                    trk.velocity = new_vel
            elif trk.age_since_seen > 0 and trk.x is not None:
                dt = float(trk.age_since_seen + 1)
                trk.velocity = (x_new - trk.x) / dt
            # else: keep existing velocity or None

            trk.x = x_new
            trk.emb_np = g.get("emb_np", trk.emb_np)
            trk.emb_tensor = g.get("emb_tensor", trk.emb_tensor)
            trk.conf = float(g.get("conf", 0.0))
            trk.last_seen_t = t_idx
            trk.age_since_seen = 0
            trk.n_obs += 1

            gtags = g.get("member_tags", [])
            group_str = "{" + ",".join(gtags) + "}" if len(gtags) > 0 else "{}"
            raw_rows.append((trk.label, idx_to_frame[t_idx], float(x_new[0]), float(x_new[1]), float(x_new[2]), group_str))
            track_observations.setdefault(trk.label, []).append(
                (idx_to_frame[t_idx], float(x_new[0]), float(x_new[1]), float(x_new[2]))
            )

        # ---- Age unmatched tracks ----
        for i, trk in enumerate(active_tracks):
            if i not in matched_trk_idxs:
                trk.age_since_seen += 1

        # ---- Spawn new tracks for unmatched groups ----
        for j, g in enumerate(groups):
            if j in matched_grp_idxs:
                continue
            label = label_from_index(next_label_idx)
            next_label_idx += 1
            x_new = np.array(g["X"], dtype=np.float32)
            emb_np_new = g.get("emb_np", np.zeros(128, dtype=np.float32))
            new_trk = ActiveTrack(
                label=label,
                x=x_new.copy(),
                emb_np=emb_np_new,
                emb_tensor=g.get("emb_tensor", torch.zeros(128)),
                conf=float(g.get("conf", 0.0)),
                first_seen_t=t_idx,
                last_seen_t=t_idx,
                age_since_seen=0,
                velocity=None,
                n_obs=1,
                first_emb_np=emb_np_new.copy() if isinstance(emb_np_new, np.ndarray) else emb_np_new,
                first_x=x_new.copy(),
            )
            active_tracks.append(new_trk)

            gtags = g.get("member_tags", [])
            group_str = "{" + ",".join(gtags) + "}" if len(gtags) > 0 else "{}"
            raw_rows.append((label, idx_to_frame[t_idx], float(x_new[0]), float(x_new[1]), float(x_new[2]), group_str))
            track_observations.setdefault(label, []).append(
                (idx_to_frame[t_idx], float(x_new[0]), float(x_new[1]), float(x_new[2]))
            )

        # ---- Kill dead tracks (exceeded max_gap) ----
        for trk in active_tracks:
            if trk.age_since_seen > max_gap:
                dead_tracks.append(DeadTrack(
                    label=trk.label, first_seen_t=trk.first_seen_t,
                    last_seen_t=trk.last_seen_t, n_obs=trk.n_obs,
                    last_emb_np=trk.emb_np.copy(), first_emb_np=trk.first_emb_np.copy() if trk.first_emb_np is not None else trk.emb_np.copy(),
                    last_x=trk.x.copy(), first_x=trk.first_x.copy() if trk.first_x is not None else trk.x.copy(),
                ))
        active_tracks = [trk for trk in active_tracks if trk.age_since_seen <= max_gap]

    # Also capture remaining active tracks into dead_tracks for merge consideration
    for trk in active_tracks:
        dead_tracks.append(DeadTrack(
            label=trk.label, first_seen_t=trk.first_seen_t,
            last_seen_t=trk.last_seen_t, n_obs=trk.n_obs,
            last_emb_np=trk.emb_np.copy(), first_emb_np=trk.first_emb_np.copy() if trk.first_emb_np is not None else trk.emb_np.copy(),
            last_x=trk.x.copy(), first_x=trk.first_x.copy() if trk.first_x is not None else trk.x.copy(),
        ))

    t_track = time.time() - t1
    print(f"Tracking done. time={t_track:.3f}s  total_tracks_spawned={next_label_idx}")

    # ============================================================
    # PASS 2.5: POST-HOC TRACK MERGE
    # ============================================================
    # If track A ends at frame 75 and track B starts at frame 76+, and their
    # embeddings are similar (end of A ≈ start of B) and positions are close,
    # they're the same fish.  Merge B → A.
    #
    # This uses the trained contrastive embeddings — the same embeddings that
    # training optimized to be discriminative per-fish.
    #
    # We also use PredictorM to roll A's last embedding forward across the gap
    # and compare with B's first embedding (same as fallback matching).
    merge_map: Dict[str, str] = {}  # label_B → label_A  (B merges into A)
    merge_sim_thresh = 0.45  # lower than runtime sim_thresh because of long gaps
    merge_dist_thresh = 2.0  # generous: fish can move far in long gaps

    # Sort fragments by first_seen_t so we process chronologically
    dead_sorted = sorted(dead_tracks, key=lambda d: d.first_seen_t)

    for i in range(len(dead_sorted)):
        frag_b = dead_sorted[i]
        # Skip if this fragment is already merged into something
        lbl_b = frag_b.label
        while lbl_b in merge_map:
            lbl_b = merge_map[lbl_b]
        if lbl_b != frag_b.label:
            continue  # already consumed

        best_score = -1.0
        best_a_idx = -1

        for k in range(i):
            frag_a = dead_sorted[k]
            lbl_a = frag_a.label
            while lbl_a in merge_map:
                lbl_a = merge_map[lbl_a]

            # Check non-overlap: A must end before B starts
            # Use the *effective* end of A (accounting for merges)
            # Find the effective last_seen_t of the chain rooted at lbl_a
            eff_last_a = frag_a.last_seen_t
            for dd in dead_sorted[:i]:
                dl = dd.label
                while dl in merge_map:
                    dl = merge_map[dl]
                if dl == lbl_a:
                    eff_last_a = max(eff_last_a, dd.last_seen_t)

            if eff_last_a >= frag_b.first_seen_t:
                continue  # overlapping — can't merge

            gap = frag_b.first_seen_t - eff_last_a

            # Position check
            d3 = float(np.linalg.norm(frag_b.first_x - frag_a.last_x))
            if d3 > merge_dist_thresh:
                continue

            # Embedding similarity: end of A vs start of B
            sim_raw = _cos_sim_np(frag_a.last_emb_np, frag_b.first_emb_np)

            # Also try PredictorM-rolled embedding if available
            sim_pred = sim_raw
            if predictor_loaded and gap > 0:
                with torch.no_grad():
                    emb_rolled = torch.from_numpy(frag_a.last_emb_np).float().to(device)
                    for _s in range(min(gap, 10)):
                        emb_rolled = predictor(emb_rolled)
                    emb_rolled = F.normalize(emb_rolled, p=2, dim=0)
                    sim_pred = _cos_sim_np(emb_rolled.cpu().numpy(), frag_b.first_emb_np)

            sim = max(sim_raw, sim_pred)
            if sim < merge_sim_thresh:
                continue

            # Score: prefer high similarity and short gap
            score = sim - 0.1 * (gap / 100.0) - 0.05 * (d3 / merge_dist_thresh)
            if score > best_score:
                best_score = score
                best_a_idx = k

        if best_a_idx >= 0:
            lbl_a = dead_sorted[best_a_idx].label
            while lbl_a in merge_map:
                lbl_a = merge_map[lbl_a]
            merge_map[frag_b.label] = lbl_a

    # Apply merge_map to raw_rows and track_observations
    def resolve_label(lbl: str) -> str:
        visited = set()
        while lbl in merge_map and lbl not in visited:
            visited.add(lbl)
            lbl = merge_map[lbl]
        return lbl

    if len(merge_map) > 0:
        n_merges = len(set(merge_map.keys()))
        raw_rows = [(resolve_label(lbl), t, x, y, z, grp) for (lbl, t, x, y, z, grp) in raw_rows]
        new_obs: Dict[str, List[Tuple[int, float, float, float]]] = {}
        for lbl, obs_list in track_observations.items():
            resolved = resolve_label(lbl)
            new_obs.setdefault(resolved, []).extend(obs_list)
        track_observations = new_obs
        print(f"Post-hoc merge: {n_merges} fragments merged")

    # ============================================================
    # Post-processing: gap filling (linear interpolation)
    # ============================================================
    filled_rows: List[Tuple[str, int, float, float, float, str]] = []

    if fill_gaps:
        for label, obs_list in track_observations.items():
            if len(obs_list) < 2:
                continue
            obs_sorted = sorted(obs_list, key=lambda o: o[0])
            for k in range(len(obs_sorted) - 1):
                t_a, xa, ya, za = obs_sorted[k]
                t_b, xb, yb, zb = obs_sorted[k + 1]
                gap_len = t_b - t_a
                if gap_len <= 1 or gap_len > max_gap:
                    continue
                for dt in range(1, gap_len):
                    frac = float(dt) / float(gap_len)
                    xi = xa + (xb - xa) * frac
                    yi = ya + (yb - ya) * frac
                    zi = za + (zb - za) * frac
                    filled_rows.append((label, t_a + dt, xi, yi, zi, "{interp}"))

    # ============================================================
    # Filter: remove tracks with too few observations (ghosts)
    # ============================================================
    obs_count: Dict[str, int] = {}
    for label, obs_list in track_observations.items():
        obs_count[label] = len(obs_list)

    keep_labels: Set[str] = {lbl for lbl, cnt in obs_count.items() if cnt >= min_obs_to_output}

    # Combine raw + filled, filter, sort
    all_rows = raw_rows + filled_rows
    all_rows = [(lbl, t, x, y, z, grp) for (lbl, t, x, y, z, grp) in all_rows if lbl in keep_labels]
    all_rows.sort(key=lambda r: (r[1], r[0]))  # sort by time, then label

    # Relabel to clean consecutive letters (A, B, C, ...)
    unique_labels = sorted(
        set(r[0] for r in all_rows),
        key=lambda lbl: min(r[1] for r in all_rows if r[0] == lbl),
    )
    relabel_map = {old: label_from_index(i) for i, old in enumerate(unique_labels)}

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("object,Timestamp,X,Y,Z,group\n")
        for lbl, t, x, y, z, grp in all_rows:
            f.write(f"{relabel_map[lbl]},{t},{x},{y},{z},{grp}\n")

    n_tracks = len(unique_labels)
    total_pts = len(all_rows)
    total_time = time.time() - t0
    print(
        f"\nSaved to {output_path}\n"
        f"  tracks={n_tracks} (after ghost filter: min_obs>={min_obs_to_output})\n"
        f"  points={total_pts} (raw={len(raw_rows)} + filled={len(filled_rows)} - filtered)\n"
        f"  total_time={total_time:.1f}s"
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ichthys inference.\n"
            "Uses trained PredictorM for embedding prediction across occlusion gaps.\n"
            "Uses trained contrastive embeddings for re-identification.\n"
            "No --count needed — tracks are born and killed dynamically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ckpt", type=str, help="Path to trained checkpoint .pth")
    p.add_argument("scene_json", type=str, help="Path to scene JSON")
    p.add_argument("images_root", type=str, help="Root folder with images")
    p.add_argument("--output", type=str, default="3dtracking_new.txt")

    g1 = p.add_argument_group("grouping")
    g1.add_argument(
        "--thr_assoc",
        type=float,
        default=0.5,
        help="Hybrid association threshold (reported inference setting: 0.5)",
    )

    g2 = p.add_argument_group("temporal linking (same defaults as training)")
    g2.add_argument("--alpha", type=float, default=0.6, help="Distance weight (training=0.6)")
    g2.add_argument("--beta", type=float, default=0.4, help="Appearance weight (training=0.4)")
    g2.add_argument("--dist_thresh", type=float, default=0.5, help="3D distance gate (training=0.5)")
    g2.add_argument("--sim_thresh", type=float, default=0.6, help="Cosine similarity gate (training=0.6)")

    g3 = p.add_argument_group("track management")
    g3.add_argument("--max_gap", type=int, default=15, help="Kill track after N consecutive unobserved frames")
    g3.add_argument("--min_conf", type=float, default=0.05, help="Drop groups below this confidence")
    g3.add_argument("--min_obs", type=int, default=2, help="Min observations to keep a track (ghost filter)")
    g3.add_argument("--no_fill", action="store_true", help="Disable gap filling (linear interpolation)")

    g4 = p.add_argument_group("runtime")
    g4.add_argument("--no_rope", action="store_true")
    g4.add_argument("--no_predictor", action="store_true",
                    help="Disable PredictorM even if checkpoint has it (ablation: no temporal prediction)")
    g4.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    g4.add_argument("--fp16", action="store_true")
    g4.add_argument("--use_dino", action="store_true", default=None)
    g4.add_argument("--no_dino", action="store_false", dest="use_dino")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_inference_new(
        args.ckpt,
        args.scene_json,
        args.images_root,
        output_path=args.output,
        thr_assoc=args.thr_assoc,
        alpha=args.alpha,
        beta=args.beta,
        dist_thresh=args.dist_thresh,
        sim_thresh=args.sim_thresh,
        max_gap=args.max_gap,
        min_conf=args.min_conf,
        min_obs_to_output=args.min_obs,
        fill_gaps=not args.no_fill,
        workers=args.workers,
        fp16=args.fp16,
        use_rope=not args.no_rope,
        use_dino=args.use_dino,
        disable_predictor=args.no_predictor,
    )


if __name__ == "__main__":
    main()
