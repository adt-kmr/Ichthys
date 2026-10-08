#!/usr/bin/env python3
"""Ichthys dashboard.

A read-mostly visual interface over whatever the pipeline has already written:
3D trajectories, per-track statistics, MOT metrics, scene/detection statistics
and training curves. It also can launch ``infer.py`` / ``evaluate.py`` /
``train.py`` as background jobs and tail their logs.

Run it with::

	streamlit run app.py

or, if the package is installed::

	ichthys-viz

Everything is discovered from a workspace root, so no arguments are required
beyond pointing it at a directory that contains ``outputs/``, ``checkpoints/``,
``synfish/`` or a ``debug.log``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

try:
	import numpy as np
	import pandas as pd
	import streamlit as st
except ImportError as exc:  # pragma: no cover - dependency guard
	raise SystemExit(
		"The Ichthys dashboard needs extra dependencies. Install them with:\n"
		"    pip install -e '.[viz]'\n"
		f"(missing or broken: {exc.name})"
	) from exc

from viz import data_io, jobs, plots
from viz.metrics import compute_mot_report, format_summary

st.set_page_config(page_title="Ichthys", page_icon="🐟", layout="wide")

DEFAULT_ROOT = REPOSITORY_ROOT


def default_root() -> Path:
	"""Workspace root from `--workspace PATH`, if Streamlit passed one through."""
	argv = sys.argv[1:]
	for index, token in enumerate(argv):
		if token in {"--workspace", "-w"} and index + 1 < len(argv):
			return Path(argv[index + 1]).expanduser()
		if token.startswith("--workspace="):
			return Path(token.split("=", 1)[1]).expanduser()
	return DEFAULT_ROOT


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_predictions(path: str, mtime: float) -> pd.DataFrame:
	return data_io.load_predictions(path)


@st.cache_data(show_spinner=False)
def cached_ground_truth(path: str, mtime: float) -> pd.DataFrame:
	return data_io.load_ground_truth(path)


@st.cache_data(show_spinner=False)
def cached_scene_info(path: str, mtime: float):
	return data_io.load_scene_info(path)


@st.cache_data(show_spinner=False)
def cached_scene_boxes(path: str, mtime: float) -> pd.DataFrame:
	return data_io.scene_bbox_overview(path)


@st.cache_data(show_spinner=False)
def cached_scene_frame(path: str, frame: int, mtime: float) -> pd.DataFrame:
	return data_io.scene_detections_frame(path, frame)


@st.cache_data(show_spinner=False)
def cached_text(path: str, mtime: float) -> str:
	return data_io.preview_text_file(path)


@st.cache_data(show_spinner=False)
def cached_workspace(root: str) -> data_io.Workspace:
	return data_io.discover_workspace(root)


@st.cache_data(show_spinner=False)
def cached_report(
	pred: str,
	gt: str,
	dist_thr: Optional[float],
	match_pred_range: bool,
	ignore_empty_frames: bool,
):
	return compute_mot_report(pred, gt, dist_thr, match_pred_range, ignore_empty_frames)


def mtime_of(path: Path) -> float:
	try:
		return float(path.stat().st_mtime)
	except OSError:
		return 0.0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _widget_key(root: Path, name: str) -> str:
	"""Namespace a widget key by workspace root.

	Widget state persists across reruns, so a selectbox whose options came from
	the previous root would hold a value that no longer exists. Mixing the root
	into the key gives every root its own widget state.
	"""
	token = hashlib.md5(str(root).encode("utf-8")).hexdigest()[:8]
	return f"{name}@{token}"


def render_sidebar() -> tuple:
	st.sidebar.title("Ichthys")
	st.sidebar.caption("Self-supervised 3D tracking of schooling fish")

	root_text = st.sidebar.text_input(
		"Workspace root",
		value=st.session_state.get("root", str(default_root())),
		help="Directory scanned for predictions, ground truth, scenes, checkpoints and logs. "
			"Override the initial value with: streamlit run app.py -- --workspace /path/to/outputs",
	)
	uploaded = st.sidebar.file_uploader(
		"or drop files here",
		type=["txt", "csv", "json", "log", "pth"],
		accept_multiple_files=True,
		help="Files land in .ichthys_uploads/ inside the workspace and are picked up automatically.",
	)

	upload_dir = Path(root_text).expanduser() / ".ichthys_uploads"
	if uploaded:
		upload_dir.mkdir(parents=True, exist_ok=True)
		for handle in uploaded:
			target = upload_dir / handle.name
			if not target.exists() or target.stat().st_size != len(handle.getvalue()):
				target.write_bytes(handle.getvalue())
		st.sidebar.success(f"{len(uploaded)} file(s) in {upload_dir}")

	root = Path(root_text).expanduser()
	if not root.exists():
		st.error(f"Workspace root does not exist: {root}")
		st.stop()
	st.session_state["root"] = str(root)

	if st.sidebar.button("Rescan", help="Forget cached file listings and scan again"):
		cached_workspace.clear()
		st.rerun()

	st.sidebar.divider()
	st.sidebar.caption("Point this at the repository, at `synfish/`, or at an `outputs/` directory.")

	try:
		workspace = cached_workspace(str(root))
	except data_io.ArtifactError as exc:
		st.error(str(exc))
		st.stop()

	return root, workspace


def file_selector(
	label: str,
	files: List[Path],
	key: str,
	help_text: str = "",
	container=None,
) -> Optional[Path]:
	if not files:
		(container or st).info(f"No {label} found under the workspace root.")
		return None
	return st.selectbox(label, files, format_func=_relative, key=key, help=help_text)


def _relative(path: Path) -> str:
	"""Format a path relative to the repository when possible.

	This is used as a ``format_func``, so it must be a pure function of the path.
	Streamlit keeps a deepcopy of the previous value in session state, and a
	``format_func`` that looks the value up in a dict built during the current run
	misses that deepcopy and silently renders the wrong label.
	"""
	try:
		return str(path.relative_to(DEFAULT_ROOT))
	except ValueError:
		return str(path)


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------
def render_overview(root: Path, workspace: data_io.Workspace) -> None:
	st.subheader("What is on disk")
	if workspace.is_empty:
		st.warning(
			"No Ichthys artefacts found yet. Run `infer.py`, `evaluate.py` or `train.py`, "
			"or upload a prediction file from the sidebar."
		)

	columns = st.columns(5)
	columns[0].metric("Predictions", len(workspace.predictions))
	columns[1].metric("Ground truth", len(workspace.ground_truth))
	columns[2].metric("Scene JSONs", len(workspace.scenes))
	columns[3].metric("Checkpoints", len(workspace.checkpoints))
	columns[4].metric("Logs", len(workspace.logs))

	if workspace.images:
		total_images = sum(workspace.images.values())
		st.caption(f"{total_images} images across {len(workspace.images)} camera directories.")

	left, right = st.columns(2)
	with left:
		st.markdown("#### Prediction files")
		if workspace.predictions:
			st.dataframe(
				pd.DataFrame({"path": [_relative(p) for p in workspace.predictions]}),
				width="stretch", hide_index=True, height=220,
			)
		else:
			st.caption("none")
	with right:
		st.markdown("#### Ground-truth files")
		if workspace.ground_truth:
			st.dataframe(
				pd.DataFrame({"path": [_relative(p) for p in workspace.ground_truth]}),
				width="stretch", hide_index=True, height=220,
			)
		else:
			st.caption("none")

	st.markdown("#### Checkpoints")
	if workspace.checkpoints:
		st.dataframe(
			pd.DataFrame({
				"path": [_relative(p) for p in workspace.checkpoints],
				"size_MB": [round(p.stat().st_size / 1e6, 1) for p in workspace.checkpoints],
				"modified": [pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M") for p in workspace.checkpoints],
			}),
			width="stretch", hide_index=True, height=220,
		)
	else:
		st.caption("none")

	st.markdown("#### Environment")
	env_rows = []
	for module in ("torch", "numpy", "pandas", "scipy", "motmetrics", "plotly", "pycocotools"):
		try:
			module_obj = __import__(module)
			env_rows.append({"package": module, "version": getattr(module_obj, "__version__", "unknown")})
		except Exception:
			env_rows.append({"package": module, "version": "not installed"})
	try:
		import torch

		env_rows.append({"package": "cuda available", "version": str(torch.cuda.is_available())})
		if torch.cuda.is_available():
			env_rows.append({"package": "gpu", "version": torch.cuda.get_device_name(0)})
	except Exception:
		pass
	st.dataframe(pd.DataFrame(env_rows), width="stretch", hide_index=True, height=240)


# ---------------------------------------------------------------------------
# Shared data selection
# ---------------------------------------------------------------------------
def render_data_picker(root: Path, workspace: data_io.Workspace) -> Optional[dict]:
	"""Sidebar controls for choosing a prediction file (and optional GT)."""
	pred_path = file_selector(
		"Prediction file", workspace.predictions, _widget_key(root, "pred_path"),
		help_text="A track file written by infer.py or sort3d.py.", container=st.sidebar,
	)
	if pred_path is None:
		return None

	pred = cached_predictions(str(pred_path), mtime_of(pred_path))

	with st.sidebar.expander("Optional ground truth", expanded=True):
		gt_files = workspace.ground_truth
		if gt_files:
			default_index = 0
			for index, path in enumerate(gt_files):
				if path.stem.split("-")[0] in pred_path.stem or pred_path.stem in path.stem:
					default_index = index
					break
			gt_path = st.selectbox(
				"GT file", gt_files, index=default_index,
				format_func=_relative, key=_widget_key(root, "gt_path"),
			)
		else:
			gt_path = None
			st.caption("No GT CSVs found under this root.")

	scene_path = None
	if workspace.scenes:
		with st.sidebar.expander("Optional scene (camera overlay)", expanded=False):
			chosen = st.selectbox(
				"Scene JSON", [None] + workspace.scenes,
				format_func=lambda p: "none" if p is None else _relative(p),
				key=_widget_key(root, "scene_overlay"),
			)
			scene_path = chosen

	return {"pred_path": pred_path, "pred": pred, "gt_path": gt_path, "scene_path": scene_path}


def load_gt(gt_path: Optional[Path]) -> Optional[pd.DataFrame]:
	if gt_path is None:
		return None
	try:
		return cached_ground_truth(str(gt_path), mtime_of(gt_path))
	except data_io.ArtifactError as exc:
		st.warning(str(exc))
		return None


# ---------------------------------------------------------------------------
# Tab: 3D view
# ---------------------------------------------------------------------------
def render_3d(root: Path, data: dict) -> None:
	st.subheader("3D trajectories")
	pred = data["pred"]
	gt = load_gt(data["gt_path"])
	pred_tracks = data_io.tracks_from_frame(pred, "object")
	gt_tracks = data_io.tracks_from_frame(gt, "Actor") if gt is not None else None

	frame_min = int(pred["Timestamp"].min())
	frame_max = int(pred["Timestamp"].max())

	st.sidebar.divider()
	show_gt = st.sidebar.checkbox(
		"Overlay ground truth",
		value=bool(gt_tracks),
		disabled=not gt_tracks,
		key=_widget_key(root, "overlay_gt"),
	)
	trail = st.sidebar.selectbox(
		"Trail", ["Full history", "Last 30", "Last 100", "Last frame only"],
		key=_widget_key(root, "trail_mode"),
	)
	trail_length = {"Full history": None, "Last 30": 30, "Last 100": 100, "Last frame only": 1}[trail]

	# Playback runs inside Plotly (client-side) so stepping frames does not cost
	# a full Streamlit rerun; the slider drives a single highlighted frame.
	animate = st.sidebar.checkbox(
		"Animate (in-chart playback)", value=False,
		help="Adds a Play button and frame scrubber to the chart. Playback is client-side, so it stays smooth.",
		key=_widget_key(root, "animate"),
	)
	stride = 1
	if animate and frame_max > frame_min:
		detail = st.sidebar.select_slider(
			"Animation step", options=[1, 2, 5, 10, 20],
			value=1, key=_widget_key(root, "anim_stride"),
			help="Show every Nth frame to keep long sequences responsive.",
		)
		stride = int(detail)

	frame = st.sidebar.slider(
		"Frame", frame_min, frame_max, value=frame_max, key=_widget_key(root, "frame_3d"),
	)

	labels = list(pred_tracks)
	selected = st.sidebar.multiselect(
		"Tracks", labels, default=labels[: min(25, len(labels))],
		help="Leave empty to show every track.", key=_widget_key(root, "track_filter"),
	)

	camera_centers = None
	scene_path = data.get("scene_path")
	if scene_path is not None:
		try:
			camera_centers = cached_scene_info(str(scene_path), mtime_of(scene_path)).camera_centers()
		except Exception:
			camera_centers = None

	figure = plots.trajectory_figure(
		pred_tracks,
		gt_tracks=gt_tracks if show_gt else None,
		current_frame=frame,
		selected_tracks=selected or None,
		camera_centers=camera_centers,
		trail_length=None if animate else trail_length,
		animate=animate,
		animation_stride=stride,
	)
	st.plotly_chart(figure, width="stretch", key=_widget_key(root, "trajectory_3d"))

	columns = st.columns(4)
	columns[0].metric("Tracks", len(pred_tracks))
	columns[1].metric("Predicted points", len(pred))
	columns[2].metric("Frames", f"{frame_min}-{frame_max}")
	active = pred[pred["Timestamp"] == frame]["object"].nunique()
	columns[3].metric(f"Tracks @ frame {frame}", int(active))

	st.caption(
		"Hover any point for coordinates. Dashed grey lines are ground truth; "
		"squares are the calibrated camera positions."
	)

# ---------------------------------------------------------------------------
# Tab: Projections & tracks
# ---------------------------------------------------------------------------
def render_tracks(root: Path, data: dict) -> None:
	st.subheader("Projections and per-track statistics")
	pred = data["pred"]
	pred_tracks = data_io.tracks_from_frame(pred, "object")
	gt = load_gt(data["gt_path"])

	stats = data_io.track_statistics(pred_tracks)
	if stats.empty:
		st.info("No tracks to summarise.")
		return

	st.plotly_chart(
		plots.track_count_figure(pred), width="stretch", key=_widget_key(root, "track_count"),
	)
	st.plotly_chart(
		plots.track_lifetime_figure(stats), width="stretch", key=_widget_key(root, "track_lifetime"),
	)

	left, right = st.columns([1, 1])
	with left:
		st.plotly_chart(
			plots.track_speed_figure(pred_tracks, selected=stats.nlargest(12, "path_length")["track"].tolist()),
			width="stretch", key=_widget_key(root, "track_speed"),
		)
	with right:
		plane = st.selectbox("Projection plane", ["XY", "XZ", "YZ"], key=_widget_key(root, "plane"))
		frame = st.slider(
			"Frame", int(pred["Timestamp"].min()), int(pred["Timestamp"].max()),
			value=int(pred["Timestamp"].max()), key=_widget_key(root, "proj_frame"),
		)
		st.plotly_chart(
			plots.projection_figure(
				pred_tracks, data_io.tracks_from_frame(gt, "Actor") if gt is not None else None,
				current_frame=frame, plane=plane,
			),
			width="stretch", key=_widget_key(root, "projection"),
		)

	st.markdown("#### Track table")
	st.dataframe(stats.round(3), width="stretch", hide_index=True, height=300)
	track_id = st.selectbox(
		"Inspect track", stats["track"].tolist(), key=_widget_key(root, "inspect_track"),
	)
	st.plotly_chart(
		plots.track_coordinate_figure(pred_tracks[track_id], track_id),
		width="stretch", key=_widget_key(root, "track_coords"),
	)

	if gt is not None:
		st.markdown("#### Identity check against ground truth")
		dist_thr = st.number_input(
			"Matching gate", min_value=0.05, max_value=5.0, value=1.0, step=0.05,
			help="Same Euclidean gate evaluate.py uses for 3D matching.",
			key=_widget_key(root, "identity_gate"),
		)
		matches = data_io.match_tracks_to_gt(pred_tracks, gt, float(dist_thr))
		st.dataframe(matches.round(3), width="stretch", hide_index=True, height=300)
		st.caption("`inlier_ratio` is the fraction of a track's observations that fall inside the gate of its nearest GT actor.")


# ---------------------------------------------------------------------------
# Tab: Metrics
# ---------------------------------------------------------------------------
def render_metrics(root: Path, data: dict) -> None:
	st.subheader("MOT metrics")
	if data["gt_path"] is None:
		st.info("Select a ground-truth CSV in the sidebar to score this prediction file.")
		return

	controls = st.columns([2, 2, 2, 1])
	dist_thr = controls[0].number_input(
		"Matching gate", min_value=0.0, max_value=5.0, value=0.0, step=0.05,
		help="0 uses the benchmark default inferred from the GT filename (1.0 for test*-gt.csv).",
		key=_widget_key(root, "metric_gate"),
	)
	match_pred_range = controls[1].checkbox(
		"Match pred range", value=True, help="Recommended for SynFish.",
		key=_widget_key(root, "match_pred_range"),
	)
	ignore_empty = controls[2].checkbox(
		"Ignore empty frames", value=False, key=_widget_key(root, "ignore_empty"),
	)
	if controls[3].button("Score", type="primary"):
		cached_report.clear()

	gate_value = float(dist_thr) if float(dist_thr) > 0 else None
	try:
		with st.spinner("Scoring..."):
			report = cached_report(
				str(data["pred_path"]), str(data["gt_path"]),
				gate_value, bool(match_pred_range), bool(ignore_empty),
			)
	except RuntimeError as exc:
		st.error(str(exc))
		return
	except Exception as exc:
		st.error(f"Evaluation failed: {exc}")
		return

	headline = report.headline
	columns = st.columns(6)
	for column, (key, label, suffix) in zip(columns, [
		("mota", "MOTA", "%"), ("idf1", "IDF1", "%"), ("recall", "Recall", "%"),
		("precision", "Precision", "%"), ("motp", "MOTP", ""), ("num_switches", "IDSW", ""),
	]):
		if key not in headline:
			continue
		value = headline[key]
		shown = f"{value * 100:.1f} {suffix}".strip() if suffix == "%" else (
			f"{value:.3f}" if suffix == "" and key == "motp" else f"{value:.0f}"
		)
		column.metric(label, shown)

	left, right = st.columns([1, 2])
	with left:
		st.markdown("#### Full summary")
		st.dataframe(
			format_summary(report), width="stretch", hide_index=True, height=560,
		)
		st.caption(f"Gate {report.dist_thr:.2f} | frames {report.frame_range[0]}-{report.frame_range[1]}")
	with right:
		st.plotly_chart(
			plots.per_frame_metrics_figure(report.per_frame),
			width="stretch", key=_widget_key(root, "per_frame_metrics"),
		)
		st.plotly_chart(
			plots.localisation_error_figure(report.per_frame, report.dist_thr),
			width="stretch", key=_widget_key(root, "localisation_error"),
		)

	st.download_button(
		"Download summary CSV",
		format_summary(report).to_csv(index=False).encode("utf-8"),
		file_name=f"{data['pred_path'].stem}-metrics.csv",
		mime="text/csv",
	)


# ---------------------------------------------------------------------------
# Tab: Scene data
# ---------------------------------------------------------------------------
def render_scene(root: Path, workspace: data_io.Workspace) -> None:
	st.subheader("Scene and detection statistics")
	scene_path = file_selector("Scene JSON", workspace.scenes, _widget_key(root, "scene_path"))
	if scene_path is None:
		st.info(
			"Scene JSONs are only discovered next to their image directories, "
			"which is how the SynFish release is laid out."
		)
		return

	try:
		info = cached_scene_info(str(scene_path), mtime_of(scene_path))
	except data_io.ArtifactError as exc:
		st.error(str(exc))
		return

	columns = st.columns(5)
	columns[0].metric("Scene", info.name)
	columns[1].metric("Cameras", info.num_cameras)
	columns[2].metric("Frames", info.num_frames)
	columns[3].metric("Detections", info.total_detections)
	counts = info.detection_counts
	columns[4].metric("Detections / frame", round(info.total_detections / max(1, info.num_frames), 1))

	centers = info.camera_centers()
	if not centers.empty:
		st.plotly_chart(
			plots.camera_layout_figure(centers), width="stretch", key=_widget_key(root, "camera_layout"),
		)
		with st.expander("Calibration matrices"):
			rows = []
			for cam_id in info.camera_ids:
				intrinsic = info.intrinsics.get(cam_id)
				extrinsic = info.extrinsics.get(cam_id)
				rows.append({
					"cam": cam_id,
					"fx": float(intrinsic[0][0]) if intrinsic is not None else np.nan,
					"fy": float(intrinsic[1][1]) if intrinsic is not None else np.nan,
					"cx": float(intrinsic[0][2]) if intrinsic is not None else np.nan,
					"cy": float(intrinsic[1][2]) if intrinsic is not None else np.nan,
					"R|T": np.array2string(extrinsic, precision=3, separator=" ") if extrinsic is not None else "",
				})
			st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

	st.plotly_chart(
		plots.detections_per_frame_figure(counts),
		width="stretch", key=_widget_key(root, "detections_per_frame"),
	)

	boxes = cached_scene_boxes(str(scene_path), mtime_of(scene_path))
	if not boxes.empty:
		st.plotly_chart(
			plots.detection_size_figure(boxes), width="stretch", key=_widget_key(root, "detection_size"),
		)

	st.markdown("#### Detections in one frame")
	frame = st.slider(
		"Frame", info.frame_numbers[0], info.frame_numbers[-1],
		value=info.frame_numbers[0], key=_widget_key(root, "scene_frame"),
	)
	frame_detections = cached_scene_frame(str(scene_path), int(frame), mtime_of(scene_path))
	if frame_detections.empty:
		st.caption(f"No detections in frame {frame}.")
	else:
		st.dataframe(frame_detections.round(2), width="stretch", hide_index=True, height=320)

	per_cam = counts.groupby("cam")["detections"].describe() if not counts.empty else pd.DataFrame()
	if not per_cam.empty:
		st.markdown("#### Per-camera detection counts")
		st.dataframe(per_cam.round(2), width="stretch")


# ---------------------------------------------------------------------------
# Tab: Training
# ---------------------------------------------------------------------------
def render_training(root: Path, workspace: data_io.Workspace) -> None:
	st.subheader("Training progress")
	if not workspace.logs:
		st.info(
			"No `.log` files found. Training prints epoch lines to stdout, so pipe it: "
			"`python train.py synfish/train 2>&1 | tee train.log`, or pass `--debug` to get "
			"`<checkpoint_dir>/debug.log`."
		)
		return

	# Prefer a log that actually contains epoch lines or debug blocks.
	def _log_rank(path: Path) -> tuple:
		try:
			text = cached_text(str(path), mtime_of(path))
		except OSError:
			return (2, str(path))
		has_epochs = not data_io.parse_training_log(text).empty
		has_debug = not data_io.parse_debug_log(text).empty
		return (0 if (has_epochs or has_debug) else 1, str(path))

	ordered_logs = sorted(workspace.logs, key=_log_rank)
	log_path = st.selectbox(
		"Log file", ordered_logs, format_func=_relative,
		key=_widget_key(root, "log_path"),
	)
	text = cached_text(str(log_path), mtime_of(log_path))

	st.caption(f"{len(text.splitlines())} lines read from {log_path.name}")

	epochs = data_io.parse_training_log(text)
	debug = data_io.parse_debug_log(text)

	if epochs.empty and debug.empty:
		st.warning("No epoch or debug blocks recognised in this log yet.")
	else:
		if not epochs.empty:
			latest = epochs[epochs["split"] == "train"].sort_values("epoch").iloc[-1]
			columns = st.columns(4)
			columns[0].metric("Epochs logged", int(epochs["epoch"].max()))
			columns[1].metric("Latest train loss", f"{latest['avg_loss']:.5f}")
			val_rows = epochs[epochs["split"] == "val"]
			columns[2].metric("Latest val loss", f"{val_rows.sort_values('epoch').iloc[-1]['avg_loss']:.5f}" if not val_rows.empty else "n/a")
			columns[3].metric("Latest L_temp", f"{latest['temp']:.5f}")
			st.plotly_chart(
				plots.loss_curves_figure(epochs), width="stretch", key=_widget_key(root, "loss_curves"),
			)
			with st.expander("Epoch table"):
				st.dataframe(epochs.round(6), width="stretch", hide_index=True, height=300)

		if not debug.empty:
			st.markdown("#### Per-frame debug blocks")
			st.plotly_chart(
				plots.debug_loss_figure(debug), width="stretch", key=_widget_key(root, "debug_losses"),
			)
			with st.expander("Debug table"):
				st.dataframe(debug.round(6), width="stretch", hide_index=True, height=300)

	with st.expander("Raw log tail"):
		st.code("\n".join(text.splitlines()[-400:]), language="text")


# ---------------------------------------------------------------------------
# Tab: Jobs
# ---------------------------------------------------------------------------
def render_jobs(root: Path) -> None:
	st.subheader("Run and monitor jobs")
	st.caption("Launches this repository's own CLIs as background processes and tails their output.")

	job_name = st.selectbox(
		"Job", list(jobs.JOB_COMMANDS), format_func=lambda name: f"{name} - {jobs.JOB_COMMANDS[name]['summary']}",
		key=_widget_key(root, "job_name"),
	)
	spec = jobs.JOB_COMMANDS[job_name]

	with st.form(_widget_key(root, "job_form")):
		arguments: Dict[str, object] = {}
		labels = {
			"ckpt": "Checkpoint (.pth)",
			"scene_json": "Scene JSON",
			"images_root": "Images root",
			"pred": "Prediction file",
			"gt": "GT CSV",
			"train_folder": "Train folder",
			"input": "Detection stream",
			"output": "Output track file",
		}
		for required in spec["needs"]:
			arguments[required] = st.text_input(
				labels[required], value="", key=f"arg_{job_name}_{required}",
			)

		extra = st.text_input(
			"Extra CLI flags", value="--no_dino" if job_name in {"infer", "train"} else "",
			key=f"arg_{job_name}_extra",
			help="Passed through verbatim, e.g. `--no_dino --fp16`.",
		)
		arguments["extra_args"] = extra.split() if extra.strip() else []

		submitted = st.form_submit_button(f"Run {job_name}", type="primary")

	if submitted:
		try:
			job = jobs.start_job(job_name, arguments, root)
			if job.error:
				st.error(f"Could not start {job_name}: {job.error}")
			else:
				st.success(f"Started `{job_name}` (pid {job.process.pid}), logging to {job.log_path.name}")
		except ValueError as exc:
			st.error(str(exc))

	active = [job for job in jobs.list_jobs() if job.is_running]
	if active:
		st.markdown("#### Running now")
		for job in active:
			st.write(f"`{job.name}` - {job.duration:.0f}s elapsed")
		st.caption("This page does not auto-refresh. Use the button below to poll for progress.")
		if st.button("Refresh job status"):
			st.rerun()

	finished = [job for job in jobs.list_jobs() if not job.is_running]
	if finished:
		st.markdown("#### Recent jobs")
		st.dataframe(
			pd.DataFrame([{
				"job": job.name,
				"status": job.status,
				"seconds": round(job.duration, 1),
				"log": job.log_path.name,
			} for job in finished]),
			width="stretch", hide_index=True,
		)

	for job in jobs.list_jobs():
		with st.expander(f"{job.name} - {job.status} - {job.log_path.name}", expanded=job.is_running):
			st.code(f"$ {' '.join(job.argv)}", language="bash")
			lines = job.tail()
			st.code("\n".join(lines) if lines else "(no output yet)", language="text")
			if job.is_running:
				if st.button("Stop", key=f"stop_{job.log_path.stem}"):
					jobs.stop_job(job.log_path.stem)
					st.rerun()
			if st.button("Refresh this log", key=f"reload_{job.log_path.stem}"):
				st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
	from streamlit.runtime import exists as streamlit_runtime_exists

	if not streamlit_runtime_exists():
		# Reached via `ichthys-viz` or a bare `python app.py`; Streamlit
		# needs to own the script run for widgets and reruns to work.
		raise SystemExit(
			"The dashboard must be run by Streamlit:\n"
			"    streamlit run app.py\n"
			"(optionally add -- --workspace /path/to/outputs)"
		)

	st.title("Ichthys dashboard")
	root, workspace = render_sidebar()
	data = render_data_picker(root, workspace)

	tabs = st.tabs(["Overview", "3D trajectories", "Tracks", "Metrics", "Scene data", "Training", "Jobs"])

	with tabs[0]:
		render_overview(root, workspace)

	with tabs[1]:
		if data is None:
			st.info("Choose a prediction file in the sidebar to populate this tab.")
		else:
			render_3d(root, data)

	with tabs[2]:
		if data is None:
			st.info("Choose a prediction file in the sidebar to populate this tab.")
		else:
			render_tracks(root, data)

	with tabs[3]:
		if data is None:
			st.info("Choose a prediction file in the sidebar to populate this tab.")
		else:
			render_metrics(root, data)

	with tabs[4]:
		render_scene(root, workspace)

	with tabs[5]:
		render_training(root, workspace)

	with tabs[6]:
		render_jobs(root)


if __name__ == "__main__":
	main()
