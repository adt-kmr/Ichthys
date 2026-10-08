"""Loaders for Ichthys artefacts.

Everything the pipeline writes is a plain text file, so the dashboard only needs
parsers:

===========================  ==========================================
Artefact                      Produced by
===========================  ==========================================
``object,Timestamp,X,Y,Z``    ``infer.py`` / ``sort3d.py``
``Actor,Timestamp,X,Y,Z``     SynFish / 3D-ZeF ground truth
scene ``*.json``              SynFish detections + calibration
epoch loss lines              ``train.py`` stdout
per-frame debug blocks        ``utils/debug.py``
===========================  ==========================================

The prediction and ground-truth formats are parsed exactly the way
``evaluate.py`` parses them, so numbers shown in the UI match the CLI.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRED_COLUMNS = ["object", "Timestamp", "X", "Y", "Z", "group"]
GT_COLUMNS = ["Actor", "Timestamp", "X", "Y", "Z"]

# Frame number used for interpolated rows written by infer.py gap filling.
INTERP_GROUP = "{interp}"


class ArtifactError(ValueError):
	"""Raised when a file does not look like a Ichthys artefact."""


# ---------------------------------------------------------------------------
# Track tables
# ---------------------------------------------------------------------------
def load_predictions(path: Path | str) -> pd.DataFrame:
	"""Load an ``infer.py`` / ``sort3d.py`` track file into a DataFrame.

	The ``group`` column holds brace-wrapped, comma-separated camera member tags
	(``{cam0-0,cam1-2}``) or ``{interp}`` for gap-filled rows, so lines are split
	with a ``maxsplit`` that keeps the tag list intact.
	"""
	path = Path(path)
	if not path.is_file():
		raise ArtifactError(f"No such prediction file: {path}")

	rows: List[Tuple[str, int, float, float, float, str]] = []
	with path.open("r", encoding="utf-8-sig") as handle:
		header = handle.readline().strip().split(",")
		if len(header) < 5 or header[0] != "object" or header[1] != "Timestamp":
			raise ArtifactError(
				f"Unexpected header in {path.name}: {header}. "
				"Expected 'object,Timestamp,X,Y,Z[,group]'."
			)
		has_group = len(header) >= 6
		for line in handle:
			line = line.strip()
			if not line:
				continue
			parts = line.split(",", 5) if has_group else line.split(",")
			if len(parts) < 5:
				continue
			try:
				label = parts[0].strip()
				frame = int(float(parts[1]))
				xyz = (float(parts[2]), float(parts[3]), float(parts[4]))
			except ValueError:
				continue
			group = parts[5].strip() if has_group and len(parts) > 5 else ""
			rows.append((label, frame, xyz[0], xyz[1], xyz[2], group))

	if not rows:
		raise ArtifactError(f"{path.name} contains no parsable rows.")

	frame = pd.DataFrame(rows, columns=PRED_COLUMNS)
	frame["Timestamp"] = frame["Timestamp"].astype(int)
	frame["is_interpolated"] = frame["group"].astype(str).str.contains("interp", case=False)
	return frame.sort_values(["Timestamp", "object"], kind="stable").reset_index(drop=True)


def load_ground_truth(path: Path | str) -> pd.DataFrame:
	"""Load a GT CSV (``Actor,Timestamp,X,Y,Z``) into a DataFrame."""
	path = Path(path)
	if not path.is_file():
		raise ArtifactError(f"No such ground-truth file: {path}")

	frame = pd.read_csv(path)
	frame.columns = [str(c).strip() for c in frame.columns]
	missing = [c for c in GT_COLUMNS if c not in frame.columns]
	if missing:
		raise ArtifactError(f"{path.name} is missing columns {missing}. Found {list(frame.columns)}.")

	frame = frame[GT_COLUMNS].copy()
	frame["Actor"] = frame["Actor"].astype(str).str.strip()
	for column in ("X", "Y", "Z"):
		frame[column] = pd.to_numeric(frame[column], errors="coerce")
	frame["Timestamp"] = pd.to_numeric(frame["Timestamp"], errors="coerce")
	frame = frame.dropna(subset=["Timestamp", "X", "Y", "Z"])
	frame["Timestamp"] = frame["Timestamp"].astype(int)

	# evaluate.py keeps only the first row per (actor, frame).
	frame = frame.drop_duplicates(subset=["Actor", "Timestamp"], keep="first")
	return frame.sort_values(["Timestamp", "Actor"], kind="stable").reset_index(drop=True)


def tracks_from_frame(frame: pd.DataFrame, id_column: str) -> Dict[str, pd.DataFrame]:
	"""Split a long table into ``{track_id: sub_table}`` sorted by frame."""
	if id_column not in frame.columns:
		raise ArtifactError(f"Column {id_column!r} not in {list(frame.columns)}")
	tracks: Dict[str, pd.DataFrame] = {}
	for track_id, sub in frame.groupby(id_column, sort=False):
		tracks[str(track_id)] = sub.sort_values("Timestamp", kind="stable").reset_index(drop=True)
	return tracks


def xyz_matrix(track: pd.DataFrame) -> np.ndarray:
	"""Return the ``(n, 3)`` float array of XYZ positions for one track."""
	return track[["X", "Y", "Z"]].to_numpy(dtype=np.float64)


def track_statistics(tracks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	"""Per-track summary: lifetime, observation count, gaps, path length, speed."""
	records: List[Dict[str, object]] = []
	for label, track in tracks.items():
		times = track["Timestamp"].to_numpy(dtype=np.float64)
		xyz = xyz_matrix(track)
		n_obs = int(len(times))
		observed = ~track["is_interpolated"].to_numpy(dtype=bool) if "is_interpolated" in track else np.ones(n_obs, dtype=bool)
		span = int(times[-1] - times[0]) + 1 if n_obs else 0
		gaps = int(n_obs - 1 - np.count_nonzero(np.diff(times) == 1)) if n_obs > 1 else 0
		steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if n_obs > 1 else np.zeros(0)
		path_length = float(steps.sum())
		records.append(
			{
				"track": label,
				"first_frame": int(times[0]),
				"last_frame": int(times[-1]),
				"lifetime_frames": span,
				"observations": n_obs,
				"real_observations": int(observed.sum()),
				"interpolated": int(n_obs - observed.sum()),
				"gaps": gaps,
				"path_length": path_length,
				"mean_speed": path_length / span if span else 0.0,
				"max_step": float(steps.max()) if steps.size else 0.0,
				"mean_group_size": float(_mean_group_size(track)),
			}
		)
	result = pd.DataFrame.from_records(records)
	if result.empty:
		return result
	return result.sort_values("first_frame", kind="stable").reset_index(drop=True)


def _mean_group_size(track: pd.DataFrame) -> float:
	"""Average number of camera member tags in the ``group`` column."""
	if "group" not in track.columns:
		return 0.0
	sizes: List[int] = []
	for value in track["group"].astype(str):
		tags = value.strip().strip("{}").strip()
		if not tags:
			continue
		sizes.append(len([tag for tag in tags.split(",") if tag.strip()]))
	return float(np.mean(sizes)) if sizes else 0.0


def match_tracks_to_gt(
	pred_tracks: Dict[str, pd.DataFrame],
	gt_frame: pd.DataFrame,
	dist_thr: float,
) -> pd.DataFrame:
	"""Greedy nearest-neighbour matching of predicted tracks to GT actors.

	Mirrors the gating in ``evaluate.py`` (Euclidean, gated at ``dist_thr``) but
	works on whole tracks instead of single frames, which is what you want for
	eyeballing identity quality. Returns one row per predicted track.
	"""
	gt_tracks = tracks_from_frame(gt_frame, "Actor")
	records: List[Dict[str, object]] = []
	for label, track in pred_tracks.items():
		pred_xyz = xyz_matrix(track)
		pred_times = track["Timestamp"].to_numpy(dtype=np.int64)
		best_actor, best_dist, best_frame, n_in_gate = "", float("inf"), -1, 0
		for actor, gt_track in gt_tracks.items():
			gt_xyz = xyz_matrix(gt_track)
			gt_times = gt_track["Timestamp"].to_numpy(dtype=np.int64)
			common, pred_idx, gt_idx = np.intersect1d(pred_times, gt_times, return_indices=True)
			if common.size == 0:
				continue
			dist = np.linalg.norm(pred_xyz[pred_idx] - gt_xyz[gt_idx], axis=1)
			n_in_gate += int(np.count_nonzero(dist <= dist_thr))
			mean_dist = float(dist.mean())
			if mean_dist < best_dist:
				best_actor, best_dist, best_frame = actor, mean_dist, int(common[0])
		records.append(
			{
				"track": label,
				"best_gt_actor": best_actor,
				"mean_dist": best_dist if np.isfinite(best_dist) else float("nan"),
				"first_common_frame": best_frame,
				"frames_in_gate": n_in_gate,
				"observations": int(len(pred_times)),
				"inlier_ratio": n_in_gate / len(pred_times) if len(pred_times) else 0.0,
			}
		)
	return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Scene JSON
# ---------------------------------------------------------------------------
@dataclass
class SceneInfo:
	"""Static facts about a SynFish scene JSON."""

	name: str
	path: Path
	camera_ids: List[int]
	intrinsics: Dict[int, np.ndarray]
	extrinsics: Dict[int, np.ndarray]
	frame_numbers: List[int]
	detection_counts: pd.DataFrame  # columns: cam, frame, detections

	@property
	def num_cameras(self) -> int:
		return len(self.camera_ids)

	@property
	def num_frames(self) -> int:
		return len(self.frame_numbers)

	@property
	def total_detections(self) -> int:
		if self.detection_counts.empty:
			return 0
		return int(self.detection_counts["detections"].sum())

	def camera_centers(self) -> pd.DataFrame:
		"""World-space camera centres derived from the ``R|T`` extrinsics."""
		rows = []
		for cam_id in self.camera_ids:
			rt = self.extrinsics.get(cam_id)
			if rt is None or rt.shape[1] < 4:
				continue
			rotation, translation = rt[:, :3], rt[:, 3]
			center = -rotation.T @ translation
			rows.append({"cam": cam_id, "X": center[0], "Y": center[1], "Z": center[2]})
		return pd.DataFrame(rows)


def load_scene_info(scene_json_path: Path | str) -> SceneInfo:
	"""Parse a scene JSON into calibration + detection statistics.

	Mirrors the structure ``infer.py:parse_scene_json`` expects: a list of
	``{"cam": id, "info": {"K": ..., "R|T": ..., "frames": [...]}}``.
	"""
	path = Path(scene_json_path)
	if not path.is_file():
		raise ArtifactError(f"No such scene JSON: {path}")
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list) or not payload:
		raise ArtifactError(f"{path.name}: expected a non-empty list of camera entries.")

	intrinsics: Dict[int, np.ndarray] = {}
	extrinsics: Dict[int, np.ndarray] = {}
	frame_numbers: List[int] = []
	count_rows: List[Tuple[int, int, int]] = []

	for entry in payload:
		info = entry.get("info", {})
		cam_id = int(entry["cam"])
		if "K" in info:
			intrinsics[cam_id] = np.asarray(info["K"], dtype=np.float64)
		if "R|T" in info:
			extrinsics[cam_id] = np.asarray(info["R|T"], dtype=np.float64)
		for frame in info.get("frames", []):
			frame_no = int(frame["frame"])
			annotations = frame.get("annotations", []) or []
			count_rows.append((cam_id, frame_no, len(annotations)))
			if frame_no not in frame_numbers:
				frame_numbers.append(frame_no)

	frame_numbers.sort()
	detection_counts = pd.DataFrame(count_rows, columns=["cam", "frame", "detections"])

	return SceneInfo(
		name=path.stem,
		path=path,
		camera_ids=sorted(intrinsics),
		intrinsics=intrinsics,
		extrinsics=extrinsics,
		frame_numbers=frame_numbers,
		detection_counts=detection_counts,
	)


def scene_detections_frame(scene_json_path: Path | str, frame_no: int) -> pd.DataFrame:
	"""Per-camera detections for one frame, with bbox geometry and scores."""
	path = Path(scene_json_path)
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	rows: List[Dict[str, object]] = []
	for entry in payload:
		info = entry.get("info", {})
		cam_id = int(entry["cam"])
		for frame in info.get("frames", []):
			if int(frame["frame"]) != int(frame_no):
				continue
			for index, annotation in enumerate(frame.get("annotations", []) or []):
				bbox = annotation.get("bbox", [np.nan] * 4)
				rows.append(
					{
						"cam": cam_id,
						"detection": index,
						"x1": bbox[0],
						"y1": bbox[1],
						"x2": bbox[2],
						"y2": bbox[3],
						"cx": (bbox[0] + bbox[2]) / 2.0,
						"cy": (bbox[1] + bbox[3]) / 2.0,
						"width": bbox[2] - bbox[0],
						"height": bbox[3] - bbox[1],
						"area": max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
						"score": float(annotation.get("score", np.nan)),
						"has_mask": bool(annotation.get("segmentation")),
					}
				)
	return pd.DataFrame.from_records(rows)


def scene_bbox_overview(scene_json_path: Path | str) -> pd.DataFrame:
	"""Cheap bbox statistics for every detection in a scene."""
	path = Path(scene_json_path)
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	rows: List[Dict[str, object]] = []
	for entry in payload:
		info = entry.get("info", {})
		cam_id = int(entry["cam"])
		for frame in info.get("frames", []):
			frame_no = int(frame["frame"])
			for index, annotation in enumerate(frame.get("annotations", []) or []):
				bbox = annotation.get("bbox", [np.nan] * 4)
				rows.append(
					{
						"cam": cam_id,
						"frame": frame_no,
						"detection": index,
						"width": float(bbox[2]) - float(bbox[0]),
						"height": float(bbox[3]) - float(bbox[1]),
						"area": max(0.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))),
						"score": float(annotation.get("score", np.nan)),
					}
				)
	return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
_EPOCH_DONE = re.compile(
	r"Epoch\s+(?P<epoch>\d+)\s+done\.\s*"
	r"Avg loss:\s*(?P<avg_loss>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_asso_sum\):\s*(?P<asso>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_ctr_sum\):\s*(?P<ctr>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_temp\):\s*(?P<temp>[-+0-9.eE]+)"
)
_EPOCH_VAL = re.compile(
	r"Epoch\s+(?P<epoch>\d+)\s+val\.\s*"
	r"Avg loss:\s*(?P<avg_loss>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_asso_sum\):\s*(?P<asso>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_ctr_sum\):\s*(?P<ctr>[-+0-9.eE]+)\s*\|\s*"
	r"mean\(L_temp\):\s*(?P<temp>[-+0-9.eE]+)"
)
_DEBUG_HEADER = re.compile(r"^Epoch\s+(?P<epoch>\d+),\s*Scene\s+(?P<scene>[\w.-]+),\s*Frame\s+(?P<frame>\d+)->(?P<next_frame>\d+)")
_DEBUG_LOSSES = re.compile(
	r"L_asso_t=(?P<L_asso_t>[-+0-9.eE]+),\s*"
	r"L_asso_tp1=(?P<L_asso_tp1>[-+0-9.eE]+),\s*"
	r"L_ctr_t=(?P<L_ctr_t>[-+0-9.eE]+),\s*"
	r"L_ctr_tp1=(?P<L_ctr_tp1>[-+0-9.eE]+),\s*"
	r"L_temp=(?P<L_temp>[-+0-9.eE]+),\s*"
	r"total=(?P<total>[-+0-9.eE]+)"
)
_DEBUG_GRAD = re.compile(r"Gradient norm:\s*(?P<grad_norm>[-+0-9.eE]+)")
_DEBUG_GROUPS = re.compile(r"Groups_t:\s*(?P<groups_t>\d+),\s*Groups_tp1:\s*(?P<groups_tp1>\d+)")
_DEBUG_P = re.compile(r"^P_t:\s*mean=(?P<mean>[-+0-9.eE]+),\s*min=(?P<min>[-+0-9.eE]+),\s*max=(?P<max>[-+0-9.eE]+)")


def parse_training_log(text: str) -> pd.DataFrame:
	"""Extract per-epoch train/validation losses from ``train.py`` stdout."""
	records: List[Dict[str, object]] = []
	for line in text.splitlines():
		match = _EPOCH_VAL.search(line)
		if match:
			records.append({"split": "val", **{k: float(v) for k, v in match.groupdict().items()}})
			continue
		match = _EPOCH_DONE.search(line)
		if match:
			records.append({"split": "train", **{k: float(v) for k, v in match.groupdict().items()}})
	return pd.DataFrame.from_records(records)


def parse_debug_log(text: str) -> pd.DataFrame:
	"""Extract per-frame debug blocks written by ``utils/debug.py``."""
	records: List[Dict[str, object]] = []
	current: Optional[Dict[str, object]] = None

	def flush() -> None:
		if current is not None:
			records.append(current)

	for line in text.splitlines():
		header = _DEBUG_HEADER.match(line.strip())
		if header:
			flush()
			current = {
				"epoch": int(header.group("epoch")),
				"scene": header.group("scene"),
				"frame": int(header.group("frame")),
			}
			continue
		if current is None:
			continue
		losses = _DEBUG_LOSSES.search(line)
		if losses:
			for key, value in losses.groupdict().items():
				current[key] = float(value)
			continue
		groups = _DEBUG_GROUPS.search(line)
		if groups:
			current["groups_t"] = int(groups.group("groups_t"))
			current["groups_tp1"] = int(groups.group("groups_tp1"))
			continue
		grad = _DEBUG_GRAD.search(line)
		if grad:
			current["grad_norm"] = float(grad.group("grad_norm"))
			continue
		probs = _DEBUG_P.match(line.strip())
		if probs:
			current["P_t_mean"] = float(probs.group("mean"))
			current["P_t_max"] = float(probs.group("max"))
	flush()

	frame = pd.DataFrame.from_records(records)
	if frame.empty:
		return frame
	return frame.sort_values(["epoch", "scene", "frame"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------
@dataclass
class Workspace:
	"""Files the dashboard can work with, discovered under one root."""

	root: Path
	predictions: List[Path]
	ground_truth: List[Path]
	scenes: List[Path]
	checkpoints: List[Path]
	logs: List[Path]
	images: Dict[Path, int]

	@property
	def is_empty(self) -> bool:
		return not (self.predictions or self.ground_truth or self.scenes or self.checkpoints or self.logs)


def discover_workspace(root: Path | str, max_depth: int = 5) -> Workspace:
	"""Find prediction / GT / scene / checkpoint / log files under ``root``.

	Deliberately shallow and suffix-driven so pointing the dashboard at a large
	SynFish tree does not crawl the whole image corpus.
	"""
	root = Path(root).expanduser()
	if not root.exists():
		raise ArtifactError(f"Workspace root does not exist: {root}")

	predictions: List[Path] = []
	ground_truth: List[Path] = []
	scene_candidates: List[Path] = []
	checkpoints: List[Path] = []
	logs: List[Path] = []
	images: Dict[Path, int] = {}
	scene_dirs: set[str] = set()

	# Skip caches, vendored weights and the dashboard's own scratch directories
	# (.ichthys_jobs / .ichthys_uploads), whose logs are not training logs.
	skip_dirs = {".git", "__pycache__", "dinov3", "DINOV3Model"}

	root_depth = len(root.parts)
	for dirpath, dirnames, filenames in os.walk(root):
		current = Path(dirpath)
		if len(current.parts) - root_depth >= max_depth:
			dirnames[:] = []
		dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".ichthys")]

		# A directory holding cam0/ cam1/ ... subdirectories is one scene's images.
		if any(re.fullmatch(r"cam\d+", name) for name in dirnames):
			scene_dirs.add(current.name)

		if current.name == "gt-traj":
			ground_truth.extend(sorted(current.glob("*.csv")))
			continue

		image_files = [f for f in filenames if f.lower().endswith((".jpeg", ".jpg", ".png"))]
		if image_files:
			images[current] = len(image_files)

		for filename in filenames:
			path = current / filename
			lower = filename.lower()
			if lower.endswith((".pth", ".pt", ".ckpt")):
				checkpoints.append(path)
			elif lower.endswith(".log"):
				logs.append(path)
			elif lower.endswith(".json"):
				scene_candidates.append(path)
			elif lower.endswith((".txt", ".csv")):
				if _looks_like_predictions(path):
					predictions.append(path)
				elif _looks_like_ground_truth(path):
					ground_truth.append(path)

	# A scene JSON is either named after a sibling cam-directory (the SynFish
	# layout: test1.json next to test1/cam0/...) or sits directly among images.
	scenes = [
		path for path in scene_candidates
		if path.stem in scene_dirs or path.parent in images
	]

	return Workspace(
		root=root,
		predictions=sorted(set(predictions)),
		ground_truth=sorted(set(ground_truth)),
		scenes=sorted(set(scenes)),
		checkpoints=sorted(set(checkpoints)),
		logs=sorted(set(logs)),
		images=dict(sorted(images.items(), key=lambda item: str(item[0]))),
	)


def _looks_like_predictions(path: Path) -> bool:
	try:
		with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
			header = handle.readline().strip().split(",")
	except OSError:
		return False
	return len(header) >= 5 and header[0].strip() == "object" and header[1].strip() == "Timestamp"


def _looks_like_ground_truth(path: Path) -> bool:
	try:
		with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
			header = handle.readline().strip().split(",")
	except OSError:
		return False
	required = {"actor", "timestamp", "x", "y", "z"}
	return required.issubset({h.strip().lower() for h in header})


def preview_text_file(path: Path | str, max_bytes: int = 200_000) -> str:
	"""Read the head of a text file for the log viewer."""
	path = Path(path)
	with path.open("r", encoding="utf-8", errors="replace") as handle:
		return handle.read(max_bytes)
