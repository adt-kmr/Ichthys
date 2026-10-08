"""Smoke test every dashboard loader, metric and figure against the demo workspace.

Run:  python scripts/check_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from viz import data_io, plots
from viz.metrics import compute_mot_report, format_summary

ROOT = REPO_ROOT / "demo_workspace"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
	status = "ok  " if condition else "FAIL"
	print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
	if not condition:
		failures.append(label)


def main() -> int:
	if not ROOT.exists():
		print(f"demo workspace missing at {ROOT}; run scripts/make_demo_workspace.py first")
		return 1

	# ---------------------------------------------------------------- discovery
	workspace = data_io.discover_workspace(ROOT)
	check("discovery finds prediction file", len(workspace.predictions) == 1,
		", ".join(p.name for p in workspace.predictions))
	check("discovery finds ground truth", len(workspace.ground_truth) == 1,
		", ".join(p.name for p in workspace.ground_truth))
	check("discovery finds scene json", len(workspace.scenes) == 1,
		", ".join(p.name for p in workspace.scenes))
	check("discovery finds checkpoints", len(workspace.checkpoints) == 2,
		", ".join(p.name for p in workspace.checkpoints))
	check("discovery finds logs", len(workspace.logs) == 2,
		", ".join(p.name for p in workspace.logs))
	check("discovery counts image dirs", len(workspace.images) == 3,
		f"{sum(workspace.images.values())} images")

	# -------------------------------------------------------------- predictions
	pred_path = workspace.predictions[0]
	pred = data_io.load_predictions(pred_path)
	check("prediction row count", len(pred) > 0, f"{len(pred)} rows")
	check("prediction columns", list(pred.columns)[:6] == data_io.PRED_COLUMNS, str(list(pred.columns)))
	check("group tags with commas survive split",
		pred[pred["object"] == "A"]["group"].iloc[0].count(",") == 2,
		pred[pred["object"] == "A"]["group"].iloc[0])
	check("interpolated rows flagged", bool(pred["is_interpolated"].any()),
		f"{int(pred['is_interpolated'].sum())} interp rows")
	tracks = data_io.tracks_from_frame(pred, "object")
	check("tracks split", len(tracks) >= 14, f"{len(tracks)} tracks")
	check("id switch split fish 5 into two ids",
		any(t["Timestamp"].max() == 59 for t in tracks.values())
		and any(t["Timestamp"].min() == 60 for t in tracks.values()))

	stats = data_io.track_statistics(tracks)
	check("track stats shape", list(stats.columns)[:4] == ["track", "first_frame", "last_frame", "lifetime_frames"])
	check("track stats non-empty", len(stats) == len(tracks), f"{len(stats)} rows")
	# The ZZ ghost has a single observation, so only real tracks have path length.
	moving = stats[stats["observations"] > 1]
	check("path length positive for multi-frame tracks", bool((moving["path_length"] > 0).all()),
		f"{len(moving)} tracks, min {float(moving['path_length'].min()):.3f}")
	check("interpolated counted", int(stats["interpolated"].sum()) >= 1,
		f"{int(stats['interpolated'].sum())} interp")
	check("mean group size ~3", 2.0 < float(stats["mean_group_size"].mean()) < 4.0,
		f"{float(stats['mean_group_size'].mean()):.2f}")

	# ------------------------------------------------------------ ground truth
	gt = data_io.load_ground_truth(workspace.ground_truth[0])
	check("gt row count", len(gt) == 14 * 120, f"{len(gt)} rows")
	check("gt columns", list(gt.columns) == data_io.GT_COLUMNS, str(list(gt.columns)))

	matches = data_io.match_tracks_to_gt(tracks, gt, 1.0)
	check("identity match rows", len(matches) == len(tracks), f"{len(matches)} rows")
	clean = matches[matches["inlier_ratio"] > 0.5]
	check("most tracks matched to a gt actor", len(clean) >= 12, f"{len(clean)}/{len(matches)}")
	check("matched distances are small", float(clean["mean_dist"].max()) < 0.5,
		f"max {float(clean['mean_dist'].max()):.4f}")

	# ------------------------------------------------------------------ metrics
	report = compute_mot_report(pred_path, workspace.ground_truth[0])
	check("metrics summary produced", not report.summary.empty)
	check("benchmark gate inferred for test1-gt.csv", report.dist_thr == 1.0, f"{report.dist_thr}")
	check("per-frame series length", len(report.per_frame) > 0, f"{len(report.per_frame)} frames")
	check("per-frame matched > 0", int(report.per_frame["n_matched"].sum()) > 0,
		f"{int(report.per_frame['n_matched'].sum())} matches")
	check("per-frame mean_dist populated", bool(report.per_frame["mean_dist"].notna().any()))
	headline = report.headline
	check("headline metrics present", {"mota", "idf1", "recall", "precision"} <= set(headline), str(sorted(headline)))
	check("MOTA in sane range", -1.5 < headline["mota"] <= 1.0, f"{headline['mota']:.4f}")
	check("recall is high", headline["recall"] > 0.85, f"{headline['recall']:.4f}")
	check("MTBFm computed", report.mtbfm > 0, f"{report.mtbfm:.2f}")
	summary_table = format_summary(report)
	check("summary display table", len(summary_table) > 15 and "display" in summary_table.columns,
		f"{len(summary_table)} rows")
	display = dict(zip(summary_table["metric"], summary_table["display"]))
	check("MOTA displayed as percent", display["MOTA"].endswith("%"), display["MOTA"])
	check("MOTP displayed as distance", "." in display["MOTP (mean dist)"], display["MOTP (mean dist)"])

	# --------------------------------------------------------------------- scene
	info = data_io.load_scene_info(workspace.scenes[0])
	check("scene cameras", info.num_cameras == 3, f"{info.camera_ids}")
	check("scene frames", info.num_frames == 120, f"{info.num_frames}")
	check("scene detections", info.total_detections > 0, f"{info.total_detections}")
	centers = info.camera_centers()
	check("camera centers computed", len(centers) == 3 and np.isfinite(centers[["X", "Y", "Z"]].to_numpy()).all(),
		f"radius ~{float(np.linalg.norm(centers[['X', 'Y']].to_numpy(), axis=1).mean()):.2f}")
	frame_detections = data_io.scene_detections_frame(workspace.scenes[0], 0)
	check("per-frame detections", len(frame_detections) > 0, f"{len(frame_detections)} detections at frame 0")
	check("bbox geometry columns", {"cx", "cy", "width", "height", "area", "score"} <= set(frame_detections.columns))
	boxes = data_io.scene_bbox_overview(workspace.scenes[0])
	check("bbox overview", len(boxes) == info.total_detections, f"{len(boxes)} rows")

	# ---------------------------------------------------------------------- logs
	train_text = data_io.preview_text_file(ROOT / "train.log")
	epochs = data_io.parse_training_log(train_text)
	check("epoch lines parsed", len(epochs) > 0, f"{len(epochs)} rows")
	check("train and val splits", set(epochs["split"]) == {"train", "val"}, str(sorted(set(epochs['split']))))
	check("epoch count", int(epochs["epoch"].max()) == 12, f"max epoch {int(epochs['epoch'].max())}")
	check("loss decreases", float(epochs[epochs.split == 'train'].sort_values('epoch')['avg_loss'].iloc[-1])
		< float(epochs[epochs.split == 'train'].sort_values('epoch')['avg_loss'].iloc[0]))

	debug_text = data_io.preview_text_file(workspace.logs[0] if workspace.logs[0].name == "debug.log" else ROOT / "checkpoints" / "demo-model" / "debug.log")
	debug = data_io.parse_debug_log(debug_text)
	check("debug blocks parsed", len(debug) > 0, f"{len(debug)} blocks")
	check("debug loss columns", {"L_asso_t", "L_ctr_t", "L_temp", "total", "grad_norm"} <= set(debug.columns))
	check("debug grad norm", bool((debug["grad_norm"] > 0).all()))
	check("debug scenes", set(debug["scene"]) == {"test1"}, str(sorted(set(debug['scene']))))

	# ------------------------------------------------------------------ figures
	gt_tracks = data_io.tracks_from_frame(gt, "Actor")
	figures = {
		"trajectory_3d": plots.trajectory_figure(tracks, gt_tracks=gt_tracks, current_frame=90, camera_centers=centers),
		"trajectory_3d_filtered": plots.trajectory_figure(tracks, gt_tracks=gt_tracks, current_frame=30, selected_tracks=list(tracks)[:3], trail_length=10),
		"trajectory_3d_nogt": plots.trajectory_figure(tracks, current_frame=119),
		"trajectory_3d_animated": plots.trajectory_figure(tracks, gt_tracks=gt_tracks, animate=True),
		"trajectory_3d_animated_stride": plots.trajectory_figure(tracks, animate=True, animation_stride=5),
		"projection_xy": plots.projection_figure(tracks, gt_tracks, current_frame=60, plane="XY"),
		"projection_xz": plots.projection_figure(tracks, gt_tracks, current_frame=60, plane="XZ"),
		"projection_yz": plots.projection_figure(tracks, gt_tracks, current_frame=60, plane="YZ"),
		"track_coords": plots.track_coordinate_figure(tracks["A"], "A"),
		"track_speed": plots.track_speed_figure(tracks, selected=list(tracks)[:5]),
		"track_lifetime": plots.track_lifetime_figure(stats),
		"track_count": plots.track_count_figure(pred),
		"per_frame_metrics": plots.per_frame_metrics_figure(report.per_frame),
		"localisation_error": plots.localisation_error_figure(report.per_frame, report.dist_thr),
		"detections_per_frame": plots.detections_per_frame_figure(info.detection_counts),
		"detection_size": plots.detection_size_figure(boxes),
		"camera_layout": plots.camera_layout_figure(centers),
		"loss_curves": plots.loss_curves_figure(epochs),
		"debug_losses": plots.debug_loss_figure(debug),
	}
	for name, figure in figures.items():
		try:
			payload = figure.to_json()
			check(f"figure {name}", len(payload) > 100 and len(figure.data) > 0,
				f"{len(figure.data)} traces, {len(payload) // 1024} KB")
		except Exception as exc:
			check(f"figure {name}", False, f"{type(exc).__name__}: {exc}")

	# Animated figures must carry Plotly frames, a Play button and a scrubber.
	animated = plots.trajectory_figure(tracks, animate=True)
	check("animated figure has frames", len(animated.frames) > 50, f"{len(animated.frames)} frames")
	menus = animated.layout.updatemenus
	sliders = animated.layout.sliders
	check("animated figure has play button", bool(menus) and len(menus[0].buttons) >= 2,
		f"{len(menus[0].buttons) if menus else 0} buttons")
	check("animated scrubber steps", bool(sliders) and len(sliders[0].steps) > 50,
		f"{len(sliders[0].steps) if sliders else 0} steps")
	strided = plots.trajectory_figure(tracks, animate=True, animation_stride=10)
	check("stride reduces frame count", len(strided.frames) < len(animated.frames),
		f"{len(strided.frames)} < {len(animated.frames)}")
	plain = plots.trajectory_figure(tracks, current_frame=119)
	check("non-animated figure has no frames", len(plain.frames) == 0)
	check("non-animated figure marks current frame",
		any("Current frame" in (t.name or "") for t in plain.data), "marker trace present")

	# empty-state figures must not explode
	for name, figure in {
		"empty_3d": plots.trajectory_figure({}),
		"empty_3d_animated": plots.trajectory_figure({}, animate=True),
		"empty_projection": plots.projection_figure({}),
		"empty_lifetime": plots.track_lifetime_figure(stats.iloc[0:0]),
		"empty_per_frame": plots.per_frame_metrics_figure(report.per_frame.iloc[0:0]),
		"empty_camera_layout": plots.camera_layout_figure(centers.iloc[0:0]),
		"empty_loss_curves": plots.loss_curves_figure(epochs.iloc[0:0]),
		"empty_debug": plots.debug_loss_figure(debug.iloc[0:0]),
		"empty_detection_size": plots.detection_size_figure(boxes.iloc[0:0]),
	}.items():
		try:
			figure.to_json()
			check(f"empty-state {name}", True)
		except Exception as exc:
			check(f"empty-state {name}", False, f"{type(exc).__name__}: {exc}")

	# ------------------------------------------------------------ error handling
	for label, thunk in {
		"missing file": lambda: data_io.load_predictions(ROOT / "nope.txt"),
		"wrong header": lambda: data_io.load_predictions(ROOT / "train.log"),
		"missing gt columns": lambda: data_io.load_ground_truth(ROOT / "outputs" / "test1.txt"),
		"bad scene json": lambda: data_io.load_scene_info(ROOT / "train.log"),
		"missing root": lambda: data_io.discover_workspace(ROOT / "nope"),
	}.items():
		try:
			thunk()
			check(f"rejects {label}", False, "no error raised")
		except (data_io.ArtifactError, ValueError):
			check(f"rejects {label}", True)
		except Exception as exc:
			check(f"rejects {label}", False, f"wrong type {type(exc).__name__}")

	print()
	if failures:
		print(f"{len(failures)} check(s) FAILED:")
		for name in failures:
			print(f"  - {name}")
		return 1
	print("all dashboard checks passed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
