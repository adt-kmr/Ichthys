from __future__ import annotations

import numpy as np
import os
from pathlib import Path
from typing import Tuple


PSEUDOGT_SUBDIR = "pseudo_gt"


def _pseudo_gt_path(*, dataset_root: str | Path, scene_name: str, frame_num: int) -> Path:
    root = Path(dataset_root)
    return root / PSEUDOGT_SUBDIR / scene_name / f"frame_{int(frame_num):06d}.npz"


def save_pseudo_gt(path: str | Path, G: np.ndarray, C: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Use compressed npz: typically small and fast enough.
    np.savez_compressed(p, G=G.astype(np.int32, copy=False), C=C.astype(np.float32, copy=False))


def load_pseudo_gt(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = np.load(str(path))
    return obj["G"], obj["C"]


def get_or_make_pseudo_gt(
    *,
    detections,
    cams,
    dataset_root: str | Path,
    scene_name: str,
    frame_num: int,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pseudo-GT from disk if present; otherwise compute and save.

    This keeps training loops simple and makes pseudo-GT generation a one-time cost.
    """
    p = _pseudo_gt_path(dataset_root=dataset_root, scene_name=scene_name, frame_num=frame_num)
    if (not force_rebuild) and p.exists():
        try:
            return load_pseudo_gt(p)
        except Exception:
            # Corrupted file: fall back to rebuild.
            pass

    G, C = make_pseudo_gt(detections, cams)
    try:
        save_pseudo_gt(p, G, C)
    except Exception:
        # If we can't write for any reason, still return computed values.
        pass
    return G, C


def build_pseudo_gt_cache_for_dataset(
    *,
    dataset,
    dataset_root: str | Path,
    name: str = "train",
    force_rebuild: bool = False,
    verbose: bool = True,
) -> None:
    """Preflight build pseudo-GT cache for all frames in a dataset.

    Writes files under:
        <dataset_root>/pseudo_gt/<scene>/frame_XXXXXX.npz
    """
    dataset_root = Path(dataset_root)
    cache_root = dataset_root / PSEUDOGT_SUBDIR

    total = 0
    missing = 0

    # First pass: find missing files (cheap filesystem checks).
    to_build = []
    for scene in dataset:
        scenename = scene["names"]
        frames = scene["frames"]
        for fr in frames:
            frame_num = int(fr["frame"])
            total += 1
            p = _pseudo_gt_path(dataset_root=dataset_root, scene_name=scenename, frame_num=frame_num)
            if force_rebuild or (not p.exists()):
                missing += 1
                to_build.append((scene, fr))

    if verbose:
        if force_rebuild:
            print(f"[pseudo_gt] {name}: rebuilding cache under: {cache_root}")
        else:
            print(f"[pseudo_gt] {name}: checking cache under: {cache_root}")
        print(f"[pseudo_gt] {name}: present {total - missing}/{total} | missing {missing}/{total}")

    if missing == 0:
        return

    # Second pass: build missing.
    for scene, fr in to_build:
        scenename = scene["names"]
        frame_num = int(fr["frame"])
        cams = scene["cams"]
        detections = fr["detections"]
        _ = get_or_make_pseudo_gt(
            detections=detections,
            cams=cams,
            dataset_root=dataset_root,
            scene_name=scenename,
            frame_num=frame_num,
            force_rebuild=force_rebuild,
        )

    if verbose:
        print(f"[pseudo_gt] {name}: build done. cache now at {cache_root}")

def triangulate_batch(pts1, K1, RT1, pts2, K2, RT2):
    """
    Vectorized triangulation for multiple point pairs between two cameras.
    pts1, pts2: (M,2) arrays
    Returns (M,3) triangulated 3D points
    """
    P1 = K1 @ RT1
    P2 = K2 @ RT2

    M = pts1.shape[0]
    Xs = np.zeros((M, 3))

    for idx in range(M):  # still a loop but only 1 SVD per pair
        x1, y1 = pts1[idx]
        x2, y2 = pts2[idx]

        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X /= X[-1]
        Xs[idx] = X[:3]

    return Xs

def make_pseudo_gt(detections, cams):
    """
    Vectorized pseudo-GT association matrix (faster).
    """
    N = len(detections)
    G = -1 * np.ones((N, N), dtype=np.int32)
    C = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        det_i = detections[i]
        cam_i = det_i["cam_id"]
        if f"cam{cam_i}-K" not in cams:  # skip invalid
            continue
        Ki = np.array(cams[f"cam{cam_i}-K"])
        RTi = np.array(cams[f"cam{cam_i}-R|T"])
        pt_i = np.array(det_i["segmentation-centroid"], dtype=np.float32)

        for j in range(i + 1, N):
            det_j = detections[j]
            cam_j = det_j["cam_id"]
            if cam_i == cam_j:
                continue
            Kj = np.array(cams[f"cam{cam_j}-K"])
            RTj = np.array(cams[f"cam{cam_j}-R|T"])
            pt_j = np.array(det_j["segmentation-centroid"], dtype=np.float32)

            # batch of size 1 (can later vectorize larger groups)
            Xs = triangulate_batch(np.array([pt_i]), Ki, RTi,
                                   np.array([pt_j]), Kj, RTj)

            # reproject the triangulated point back into each camera and check
            # whether the projected point falls inside the original detection bbox.
            X = Xs[0]
            # reproj into cam i
            Pi = Ki @ RTi
            proj_i = (Pi @ np.hstack([X, 1.0]))
            proj_i = proj_i / proj_i[2]
            pred_i = proj_i[:2]
            # reproj into cam j
            Pj = Kj @ RTj
            proj_j = (Pj @ np.hstack([X, 1.0]))
            proj_j = proj_j / proj_j[2]
            pred_j = proj_j[:2]

            # bbox containment check (bbox format: [x1,y1,x2,y2])
            bbox_i = np.array(det_i.get('bbox', [0,0,0,0]), dtype=np.float32)
            bbox_j = np.array(det_j.get('bbox', [0,0,0,0]), dtype=np.float32)
            def in_bbox(pt, bbox):
                x, y = float(pt[0]), float(pt[1])
                x1, y1, x2, y2 = bbox
                return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

            inside_i = in_bbox(pred_i, bbox_i)
            inside_j = in_bbox(pred_j, bbox_j)

            # compute avg reprojection error (for confidence), keep for logging
            err_i = np.linalg.norm(pred_i - pt_i)
            err_j = np.linalg.norm(pred_j - pt_j)
            avg_err = 0.5 * (err_i + err_j)

            # Decide positive if projected point lies inside both original bboxes.
            if inside_i and inside_j:
                G[i, j] = 1
                G[j, i] = 1
                conf = 1.0 / (1.0 + avg_err)  # Ranges from 0 to 1
                C[i, j] = conf
                C[j, i] = conf
            else:
                G[i, j] = 0
                G[j, i] = 0

    return G, C
