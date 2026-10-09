"""Evaluate 3D multi-object tracking using MOTMetrics (py-motmetrics).

Ichthys outputs 3D tracks as a CSV-like text file from `infer.py`:

	object,Timestamp,X,Y,Z,group

and GT is provided as a CSV with:

	Actor,Timestamp,X,Y,Z

We treat this as standard MOT: per-frame assignment + identity consistency.

Key idea
--------
At each frame t, build a cost matrix = Euclidean distance in 3D between each
GT track position and each predicted track position. Distances above a gating
threshold are marked as "no match" (NaN), which forces MOTMetrics to count
FP/FN/ID switches appropriately.

Paper-friendly metrics
----------------------
This script reports MOTChallenge-style metrics: MOTA, MOTP, IDF1, IDP, IDR,
IDSW, FP, FN, etc. These are commonly used for tracking evaluation and are
appropriate for 3D tracking when the distance cost is defined in 3D.

Example
-------
	python evaluate.py \
		--pred outputs/test1.txt \
		--gt synfish/test/gt-traj/test1-gt.csv \
		--match_pred_range
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def _read_pred_txt(pred_path: Path) -> Dict[int, List[Tuple[str, np.ndarray]]]:
	"""Return {t: [(id, xyz), ...]} from prediction txt."""
	# The file is comma-separated with a header.
	# IDs are letters (A, B, ..., AA) but could be any string.
	by_t: Dict[int, List[Tuple[str, np.ndarray]]] = {}
	seen_per_t: Dict[int, set[str]] = {}
	with pred_path.open("r", encoding="utf-8") as f:
		header = f.readline().strip().split(",")
		if len(header) < 5 or header[0] != "object" or header[1] != "Timestamp":
			raise ValueError(
				f"Unexpected pred header in {pred_path}: {header}. "
				"Expected: object,Timestamp,X,Y,Z,..."
			)
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = [p.strip() for p in line.split(",")]
			if len(parts) < 5:
				continue
			obj = parts[0]
			try:
				t = int(float(parts[1]))
				x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
			except ValueError:
				continue
			# motmetrics expects unique IDs per frame; if duplicates happen, keep the first.
			s = seen_per_t.setdefault(t, set())
			if obj in s:
				continue
			s.add(obj)
			by_t.setdefault(t, []).append((obj, np.array([x, y, z], dtype=np.float32)))
	return by_t


def _read_gt_csv(gt_path: Path) -> Dict[int, List[Tuple[str, np.ndarray]]]:
	"""Return {t: [(id, xyz), ...]} from GT csv."""
	import csv

	by_t: Dict[int, List[Tuple[str, np.ndarray]]] = {}
	seen_per_t: Dict[int, set[str]] = {}
	with gt_path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		required = {"Actor", "Timestamp", "X", "Y", "Z"}
		if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
			raise ValueError(
				f"Unexpected GT columns in {gt_path}: {reader.fieldnames}. "
				f"Required: {sorted(required)}"
			)
		for row in reader:
			try:
				actor = str(row["Actor"])
				t = int(float(row["Timestamp"]))
				x, y, z = float(row["X"]), float(row["Y"]), float(row["Z"])
			except Exception:
				continue
			s = seen_per_t.setdefault(t, set())
			if actor in s:
				continue
			s.add(actor)
			by_t.setdefault(t, []).append((actor, np.array([x, y, z], dtype=np.float32)))
	return by_t


def _euclidean_cost_matrix(
	gt_items: List[Tuple[str, np.ndarray]],
	pr_items: List[Tuple[str, np.ndarray]],
	dist_thr: float,
) -> np.ndarray:
	"""Compute |GT|x|PR| matrix with NaNs for distances > dist_thr."""
	if len(gt_items) == 0:
		return np.zeros((0, len(pr_items)), dtype=np.float32)
	if len(pr_items) == 0:
		return np.zeros((len(gt_items), 0), dtype=np.float32)

	gt_xyz = np.stack([p for _, p in gt_items], axis=0)  # (G,3)
	pr_xyz = np.stack([p for _, p in pr_items], axis=0)  # (P,3)
	# (G,1,3) - (1,P,3) -> (G,P,3)
	d = gt_xyz[:, None, :] - pr_xyz[None, :, :]
	dist = np.linalg.norm(d, axis=-1).astype(np.float32)  # (G,P)
	dist[dist > float(dist_thr)] = np.nan
	return dist


@dataclass
class EvalResult:
	summary_text: str


def _benchmark_dist_thr(gt_path: Path) -> float:
	"""Select the released evaluation protocol from a known GT filename."""
	if gt_path.name in {f"test{i}-gt.csv" for i in range(1, 7)}:
		return 1.0
	if gt_path.name in {"zebra02-new.csv", "zebra04.csv"}:
		return 0.5
	raise ValueError(
		f"Unknown benchmark GT filename: {gt_path.name}. "
		"Pass --dist_thr explicitly for a custom dataset."
	)


def evaluate_3d_mot(
	pred_path: Path,
	gt_path: Path,
	dist_thr: float = 0.75,
	t_min: int | None = None,
	t_max: int | None = None,
	ignore_empty_frames: bool = False,
	gt_t_offset: int = 0,
	pred_t_offset: int = 0,
	match_pred_range: bool = False,
) -> EvalResult:
	"""Evaluate with motmetrics and return a formatted summary."""
	try:
		import motmetrics as mm
	except Exception as e:
		raise RuntimeError(
			"motmetrics is required. Install with: pip install motmetrics"
		) from e

	pred_raw = _read_pred_txt(pred_path)
	gt_raw = _read_gt_csv(gt_path)

	pred: Dict[int, List[Tuple[str, np.ndarray]]] = {t + int(pred_t_offset): v for t, v in pred_raw.items()}
	gt: Dict[int, List[Tuple[str, np.ndarray]]] = {t + int(gt_t_offset): v for t, v in gt_raw.items()}

all_ts = sorted(set(pred.keys()) | set(gt.keys()))

	if match_pred_range:
		# Only evaluate frames where predictions exist, so missing pred frames
		# at the start/end don't inflate FN.
		pred_ts = set(pred.keys())
		if len(pred_ts) > 0:
			pred_t_min = min(pred_ts)
			pred_t_max = max(pred_ts)
			all_ts = [t for t in all_ts if pred_t_min <= t <= pred_t_max]
	if t_min is not None:
		all_ts = [t for t in all_ts if t >= t_min]
	if t_max is not None:
		all_ts = [t for t in all_ts if t <= t_max]

	acc = mm.MOTAccumulator(auto_id=True)

	# Some motmetrics/pandas versions attempt to cast IDs to float internally.

	# Some motmetrics/pandas versions attempt to cast IDs to float internally.
	# To be robust, map arbitrary string IDs -> stable integer IDs.
	gt_id_map: Dict[str, int] = {}
	pr_id_map: Dict[str, int] = {}
	next_gt = 1
	next_pr = 1

	def _map_ids(ids: List[str], mapping: Dict[str, int], next_id: int) -> Tuple[List[int], int]:
		out: List[int] = []
		for s in ids:
			if s not in mapping:
				mapping[s] = next_id
				next_id += 1
			out.append(mapping[s])
		return out, next_id

	for t in tqdm(all_ts, desc="Evaluating frames", mininterval=5, miniters=1):
		gt_items = gt.get(t, [])
		pr_items = pred.get(t, [])
		if ignore_empty_frames and len(gt_items) == 0 and len(pr_items) == 0:
			continue
		gt_ids_str = [i for i, _ in gt_items]
		pr_ids_str = [i for i, _ in pr_items]
		gt_ids, next_gt = _map_ids(gt_ids_str, gt_id_map, next_gt)
		pr_ids, next_pr = _map_ids(pr_ids_str, pr_id_map, next_pr)
		C = _euclidean_cost_matrix(gt_items, pr_items, dist_thr=dist_thr)
		acc.update(gt_ids, pr_ids, C)

	# --- Compute MTBFm (Mean Time Between Failures) ---
	# A "failure" for a GT object is any frame where it is not matched (missed)
	# or where an ID switch occurs. MTBFm = average length of continuous
	# successfully-tracked segments across all GT objects.
	#
	# We reconstruct per-GT-object match/miss history from the accumulator events.
	events = acc.events
	# events has columns: ['Type', 'OId', 'HId', 'D']
	# Type: 'MATCH', 'SWITCH', 'FP', 'MISS', 'TRANSFER', 'ASCEND', 'MIGRATE'
	# OId = GT object id (for MATCH/SWITCH/MISS), HId = hypothesis id
	gt_frame_status: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
	# group events by GT object id
	for (frame_idx, event_idx), row in events.iterrows():
		etype = row["Type"]
		oid = row["OId"]
		if etype in ("MATCH",):
			gt_frame_status[oid].append((frame_idx, "ok"))
		elif etype in ("SWITCH", "TRANSFER", "ASCEND", "MIGRATE"):
			gt_frame_status[oid].append((frame_idx, "switch"))
		elif etype == "MISS":
			gt_frame_status[oid].append((frame_idx, "miss"))
		# FP events have no GT object, skip them

	all_segment_lengths = []
	for oid, frame_events in gt_frame_status.items():
		# Sort by frame index
		frame_events.sort(key=lambda x: x[0])
		current_segment = 0
		for _, status in frame_events:
			if status == "ok":
				current_segment += 1
			else:
				# failure: end current segment, start new one
				if current_segment > 0:
					all_segment_lengths.append(current_segment)
				current_segment = 0
				if status == "switch":
					# A switch is still a detection, start new segment
					current_segment = 1
		if current_segment > 0:
			all_segment_lengths.append(current_segment)

	if len(all_segment_lengths) > 0:
		mtbfm = float(np.mean(all_segment_lengths))
	else:
		mtbfm = 0.0

	mh = mm.metrics.create()
	metrics = [
		"num_frames",
		"num_objects",
		"num_predictions",
		"recall",
		"precision",
		"idf1",
		"idp",
		"idr",
		"mota",
		"motp",
		"num_switches",
		"num_false_positives",
		"num_misses",
		"num_matches",
		"num_fragmentations",
		"mostly_tracked",
		"partially_tracked",
		"mostly_lost",
		"num_unique_objects",
	]

	# motmetrics expects motp = (sum(costs)/num_matches) if "distance" is cost.
	# That's meaningful here: smaller is better (meters/units of your world).
	summary = mh.compute(acc, metrics=metrics, name="scene")

	# Add MT% and ML% (as percentage of unique objects) and MTBFm
	n_unique = int(summary.loc["scene", "num_unique_objects"])
	mt_count = int(summary.loc["scene", "mostly_tracked"])
	pt_count = int(summary.loc["scene", "partially_tracked"])
	ml_count = int(summary.loc["scene", "mostly_lost"])
	mt_pct = (mt_count / n_unique * 100.0) if n_unique > 0 else 0.0
	ml_pct = (ml_count / n_unique * 100.0) if n_unique > 0 else 0.0

	summary["MT"] = mt_count
	summary["PT"] = pt_count
	summary["ML"] = ml_count
	summary["MT%"] = f"{mt_pct:.1f}%"
	summary["ML%"] = f"{ml_pct:.1f}%"
	summary["MTBFm"] = f"{mtbfm:.1f}"

	text = mm.io.render_summary(
		summary,
		formatters=mh.formatters,
		namemap={
			"num_switches": "IDSW",
			"num_false_positives": "FP",
			"num_misses": "FN",
			"num_fragmentations": "Frag",
			"num_matches": "Matches",
		},
	)
	return EvalResult(summary_text=text)


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="3D tracking evaluation with MOTMetrics")
	p.add_argument("--pred", type=str, required=True, help="Prediction text file from infer.py")
	p.add_argument("--gt", type=str, required=True, help="GT csv (Actor,Timestamp,X,Y,Z)")
	p.add_argument(
		"--dist_thr",
		type=float,
		default=None,
		help="Override the benchmark-specific matching gate for custom datasets",
	)
	p.add_argument("--t_min", type=int, default=None)
	p.add_argument("--t_max", type=int, default=None)
	p.add_argument(
		"--gt_t_offset",
		type=int,
		default=0,
		help="Add this integer offset to GT timestamps (frame indices)",
	)
	p.add_argument(
		"--pred_t_offset",
		type=int,
		default=0,
		help="Add this integer offset to prediction timestamps (frame indices)",
	)
	p.add_argument(
		"--ignore_empty_frames",
		action="store_true",
		help="Skip frames where both GT and pred are empty",
	)
	p.add_argument(
		"--match_pred_range",
		action="store_true",
		help="Only evaluate frames within the prediction time range (min_pred_t..max_pred_t). "
		     "Prevents FN inflation when pred has fewer frames than GT.",
	)
	return p


def main() -> None:
    args = build_argparser().parse_args()

    # Validate input files exist
    pred_path = Path(args.pred)
    gt_path = Path(args.gt)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.pred}\n"
            "Please verify the path is correct and the file exists with the expected header: object,Timestamp,X,Y,Z,..."
        )
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {args.gt}\n"
            "Please verify the path is correct and the file exists with the expected columns: Actor,Timestamp,X,Y,Z"
        )

    gt_path = Path(args.gt)
    try:
        dist_thr = _benchmark_dist_thr(gt_path) if args.dist_thr is None else args.dist_thr
    except ValueError as exc:
        build_argparser().error(str(exc))
    res = evaluate_3d_mot(
        pred_path=Path(args.pred),
        gt_path=gt_path,
        dist_thr=float(dist_thr),
        t_min=args.t_min,
        t_max=args.t_max,
        ignore_empty_frames=bool(args.ignore_empty_frames),
        gt_t_offset=int(args.gt_t_offset),
        pred_t_offset=int(args.pred_t_offset),
        match_pred_range=bool(args.match_pred_range),
    )
    print(res.summary_text)


if __name__ == "__main__":
	main()
