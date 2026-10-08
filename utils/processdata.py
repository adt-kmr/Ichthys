#!/usr/bin/env python3
"""
processdata.py

Data preprocessing utilities for Ichthys training:
- RLE decoding to centroids
- Scene parsing from JSON
- Dataset loading with caching
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def rle_to_centroid(rle, bbox=None):
    """
    Decode RLE to mask centroid. If mask is empty and bbox is provided,
    fall back to bbox center. Returns [x, y].
    """
    try:
        from pycocotools import mask as mask_util
        mask = mask_util.decode(rle)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                return [float(cx), float(cy)]
            return [0.0, 0.0]
        return [float(xs.mean()), float(ys.mean())]
    except Exception:
        # If mask decoding fails for any reason, fall back to bbox center if available
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return [float(cx), float(cy)]
        return [0.0, 0.0]


def parse_scene(scene_json):
    """
    Parse a scene JSON into camera matrices and per-frame detections.
    
    Returns:
        cams: dict with 'camX-K' and 'camX-R|T' keys
        frames: list of frame dicts with 'frame' number and 'detections' list
    """
    cams = {}
    cam_data = {cam_entry["cam"]: cam_entry["info"] for cam_entry in scene_json}
    for cam_id, info in cam_data.items():
        cams[f"cam{cam_id}-K"] = np.array(info["K"])
        cams[f"cam{cam_id}-R|T"] = np.array(info["R|T"])
    max_frames = max(len(info["frames"]) for info in cam_data.values())
    frames = []
    for t in range(max_frames):
        detections = []
        frame_number = None
        for cam_id, info in cam_data.items():
            if t >= len(info["frames"]):
                continue
            frame = info["frames"][t]
            frame_number = frame["frame"]
            for ann in frame["annotations"]:
                bbox = ann["bbox"]
                # pass bbox as fallback to centroid decoder
                centroid = rle_to_centroid(ann.get("segmentation", {}), bbox=bbox)
                hw = ann.get("segmentation", {}).get("size", [0, 0])

                detections.append({
                    "cam_id": cam_id,
                    "bbox": bbox,
                    "segmentation-centroid": centroid,
                    "det_score": ann.get("score", 1.0)
                })
        if frame_number is not None:
            frames.append({"frame": frame_number, "detections": detections})
    return cams, frames


def parse_single_file(path):
    """Parse a single JSON file into scene dict."""
    with open(path, "r") as jf:
        raw_scene = json.load(jf)
    cams, frames = parse_scene(raw_scene)
    return {"cams": cams, "frames": frames, "names": os.path.basename(path)[:-5]}


def load_and_parse_json_folder(folder_path, num_workers=32, use_cache=True):
    """
    Load and parse all JSON files in a folder (with optional caching).
    
    Args:
        folder_path: path to folder containing scene JSON files
        num_workers: number of parallel workers for parsing
        use_cache: if True, use cached parsed data if available
        
    Returns:
        dataset: list of scene dicts with 'cams', 'frames', 'names'
    """
    cache_file = os.path.join(folder_path, "_parsed_cache.pkl")
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    json_files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json"))
    dataset = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for parsed in ex.map(parse_single_file, json_files):
            dataset.append(parsed)
    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
    return dataset
