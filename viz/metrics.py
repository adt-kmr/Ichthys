"""MOT metrics for the dashboard.

This deliberately reuses ``evaluate.py``'s readers and cost function so the
numbers in the UI are identical to ``python evaluate.py`` on the CLI. The only
addition is a per-frame series, which the CLI does not expose but which is what
you actually want when staring at a bad run.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPOSITORY_ROOT))

from evaluate import _benchmark_dist_thr, _euclidean_cost_matrix, _read_gt_csv, _read_pred_txt


@dataclass
class MotReport:
	"""Summary metrics plus the per-frame series behind them."""

	summary: pd.DataFrame        # one row, MOTMetrics columns + MT/PT/ML/MTBFm
	per_frame: pd.DataFrame      # frame, n_gt, n_pred, n_matched, mean_dist, precision, recall
	mtbfm: float
	dist_thr: float
	frame_range: Tuple[int, int]

	@property
	def headline(self) -> Dict[str, float]:
		"""The handful of numbers worth putting in big text."""
		if self.summary.empty:
			return {}
		row = self.summary.iloc[0]
		keys = ["mota", "idf1", "recall", "precision", "motp", "num_switches"]
		return {key: float(row[key]) for key in keys if key in self.summary.columns}


def resolve_dist_thr(gt_path: Path | str, override: Optional[float] = None) -> float:
	"""Use the benchmark gate from the GT filename unless overridden."""
	if override is not None:
		return float(override)
	return float(_benchmark_dist_thr(Path(gt_path)))


def compute_mot_report(
	pred_path: Path | str,
	gt_path: Path | str,
	dist_thr: Optional[float] = None,
	match_pred_range: bool = True,
	ignore_empty_frames: bool = False,
) -> MotReport:
	"""Run the MOT evaluation and return both summary and per-frame series.

	``match_pred_range`` defaults to True because SynFish GT CSVs carry a few
	trailing frames with no detections, which would otherwise show up as misses.
	"""
	try:
		import motmetrics as mm
	except ImportError as exc:  # pragma: no cover - dependency guard
		raise RuntimeError("motmetrics is required for the Metrics tab: pip install motmetrics") from exc

	gate = resolve_dist_thr(gt_path, dist_thr)

	pred_raw = _read_pred_txt(Path(pred_path))
	gt_raw = _read_gt_csv(Path(gt_path))

	all_ts = sorted(set(pred_raw) | set(gt_raw))
	if match_pred_range and pred_raw:
		pred_ts = sorted(pred_raw)
		low, high = pred_ts[0], pred_ts[-1]
		all_ts = [t for t in all_ts if low <= t <= high]

	acc = mm.MOTAccumulator(auto_id=True)

	gt_id_map: Dict[str, int] = {}
	pr_id_map: Dict[str, int] = {}
	next_gt, next_pr = 1, 1

	def _map_ids(ids: List[str], mapping: Dict[str, int], next_id: int) -> Tuple[List[int], int]:
		out: List[int] = []
		for value in ids:
			if value not in mapping:
				mapping[value] = next_id
				next_id += 1
			out.append(mapping[value])
		return out, next_id

	per_frame_rows: List[Dict[str, object]] = []

	for t in all_ts:
		gt_items = gt_raw.get(t, [])
		pr_items = pred_raw.get(t, [])
		if ignore_empty_frames and not gt_items and not pr_items:
			continue

		gt_ids, next_gt = _map_ids([i for i, _ in gt_items], gt_id_map, next_gt)
		pr_ids, next_pr = _map_ids([i for i, _ in pr_items], pr_id_map, next_pr)
		cost = _euclidean_cost_matrix(gt_items, pr_items, dist_thr=gate)
		acc.update(gt_ids, pr_ids, cost)

		# Per-frame view of the same cost matrix.
		masked = np.where(np.isnan(cost), np.inf, cost)
		n_gt, n_pred = len(gt_items), len(pr_items)
		if n_gt and n_pred:
			# Hungarian on the same gated matrix the accumulator uses.
			from scipy.optimize import linear_sum_assignment

			rows, cols = linear_sum_assignment(np.nan_to_num(masked, nan=1e6))
			in_gate = np.isfinite(masked[rows, cols])
			n_matched = int(in_gate.sum())
			mean_dist = float(masked[rows, cols][in_gate].mean()) if n_matched else float("nan")
		else:
			n_matched = 0
			mean_dist = float("nan")

		per_frame_rows.append(
			{
				"frame": t,
				"n_gt": n_gt,
				"n_pred": n_pred,
				"n_matched": n_matched,
				"mean_dist": mean_dist,
				"precision": n_matched / n_pred if n_pred else np.nan,
				"recall": n_matched / n_gt if n_gt else np.nan,
			}
		)

	per_frame = pd.DataFrame.from_records(per_frame_rows)

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
	mh = mm.metrics.create()
	summary = mh.compute(acc, metrics=metrics, name="scene")

	n_unique = int(summary.loc["scene", "num_unique_objects"])
	mt_count = int(summary.loc["scene", "mostly_tracked"])
	pt_count = int(summary.loc["scene", "partially_tracked"])
	ml_count = int(summary.loc["scene", "mostly_lost"])
	summary["MT"] = mt_count
	summary["PT"] = pt_count
	summary["ML"] = ml_count
	summary["MT%"] = (mt_count / n_unique * 100.0) if n_unique else 0.0
	summary["ML%"] = (ml_count / n_unique * 100.0) if n_unique else 0.0

	mtbfm = _mean_time_between_failures(acc)

	frame_range = (int(all_ts[0]), int(all_ts[-1])) if all_ts else (0, 0)
	return MotReport(
		summary=summary,
		per_frame=per_frame,
		mtbfm=mtbfm,
		dist_thr=gate,
		frame_range=frame_range,
	)


def _mean_time_between_failures(acc) -> float:
	"""MTBFm, matching the segment reconstruction in ``evaluate.py``."""
	gt_frame_status: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
	for (frame_idx, _), row in acc.events.iterrows():
		event_type, oid = row["Type"], row["OId"]
		if event_type == "MATCH":
			gt_frame_status[oid].append((frame_idx, "ok"))
		elif event_type in ("SWITCH", "TRANSFER", "ASCEND", "MIGRATE"):
			gt_frame_status[oid].append((frame_idx, "switch"))
		elif event_type == "MISS":
			gt_frame_status[oid].append((frame_idx, "miss"))

	segments: List[int] = []
	for frame_events in gt_frame_status.values():
		frame_events.sort(key=lambda item: item[0])
		current = 0
		for _, status in frame_events:
			if status == "ok":
				current += 1
				continue
			if current > 0:
				segments.append(current)
			current = 1 if status == "switch" else 0
		if current > 0:
			segments.append(current)
	return float(np.mean(segments)) if segments else 0.0


def format_summary(report: MotReport) -> pd.DataFrame:
	"""Reshape the summary into ``metric, value, display`` rows.

	MOTMetrics stores the ratio metrics as fractions and its own formatters
	multiply them by 100, so percentages are rendered here to match the CLI table.
	"""
	if report.summary.empty:
		return pd.DataFrame(columns=["metric", "value", "display"])
	labels: Dict[str, str] = {
		"num_frames": "Frames",
		"num_objects": "GT objects",
		"num_predictions": "Predictions",
		"recall": "Recall",
		"precision": "Precision",
		"idf1": "IDF1",
		"idp": "IDP",
		"idr": "IDR",
		"mota": "MOTA",
		"motp": "MOTP (mean dist)",
		"num_switches": "IDSW",
		"num_false_positives": "FP",
		"num_misses": "FN",
		"num_matches": "Matches",
		"num_fragmentations": "Frag",
		"mostly_tracked": "MT",
		"partially_tracked": "PT",
		"mostly_lost": "ML",
		"num_unique_objects": "Unique objects",
		"MT%": "MT%",
		"ML%": "ML%",
	}
	percent_metrics = {"recall", "precision", "idf1", "idp", "idr", "mota", "MT%", "ML%"}

	row = report.summary.iloc[0]
	rows: List[Dict[str, object]] = []
	for key, label in labels.items():
		if key not in report.summary.columns:
			continue
		value = float(row[key])
		if key in percent_metrics:
			# 1 decimal, matching motmetrics' own formatters so the dashboard and
			# `python evaluate.py` render identical strings.
			display = f"{value * 100:.1f}%" if key not in {"MT%", "ML%"} else f"{value:.1f}%"
		elif key == "motp":
			display = f"{value:.3f}"
		else:
			display = f"{value:.0f}"
		rows.append({"metric": label, "value": value, "display": display})

	rows.append({"metric": "MTBFm", "value": report.mtbfm, "display": f"{report.mtbfm:.1f}"})
	rows.append({"metric": "Matching gate", "value": report.dist_thr, "display": f"{report.dist_thr:.2f}"})
	return pd.DataFrame(rows)
