"""Plotly figure builders for the dashboard.

Kept separate from ``app.py`` so the figures can be built (and smoke-tested)
without a Streamlit runtime. Every builder returns a plain ``go.Figure``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Qualitative palette reused across figures so a track keeps its colour between
# the 3D view, the projections and the per-track plots.
TRACK_COLORS = [
	"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
	"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

AXIS_LABELS = {"X": "X", "Y": "Y", "Z": "Z"}


def _color(index: int) -> str:
	return TRACK_COLORS[index % len(TRACK_COLORS)]


def _empty_figure(message: str) -> go.Figure:
	figure = go.Figure()
	figure.update_layout(
		title=message,
		xaxis={"visible": False},
		yaxis={"visible": False},
		template="plotly_white",
		height=420,
	)
	return figure


# ---------------------------------------------------------------------------
# 3D trajectories
# ---------------------------------------------------------------------------
def trajectory_figure(
	pred_tracks: Dict[str, pd.DataFrame],
	gt_tracks: Optional[Dict[str, pd.DataFrame]] = None,
	current_frame: Optional[int] = None,
	selected_tracks: Optional[Sequence[str]] = None,
	camera_centers: Optional[pd.DataFrame] = None,
	trail_length: Optional[int] = None,
	height: int = 620,
	animate: bool = False,
	animation_stride: int = 1,
) -> go.Figure:
	"""3D trajectories with optional GT overlay and a highlighted current frame.

	Trails are drawn up to ``current_frame`` so the frame slider animates the
	scene forward; ``trail_length`` instead limits the trail to the most recent
	N frames for a comet-trail effect.

	With ``animate=True`` the trails are drawn once in full and the moving markers
	become Plotly animation frames, so playback runs client-side instead of
	requiring a Streamlit rerun per frame.
	"""
	figure = go.Figure()
	if not pred_tracks:
		return _empty_figure("No predicted tracks loaded")

	labels = list(pred_tracks)
	if selected_tracks:
		keep = set(selected_tracks)
		labels = [label for label in labels if label in keep]

	# When animating, always draw the complete trails and move only the markers.
	cutoff = None if animate else current_frame
	highlight_x: List[float] = []
	highlight_y: List[float] = []
	highlight_z: List[float] = []
	highlight_text: List[str] = []
	highlight_colors: List[str] = []
	# frame -> (xs, ys, zs, labels, colors)
	marker_frames: Dict[int, tuple] = {}

	for index, label in enumerate(labels):
		track = pred_tracks[label]
		times = track["Timestamp"].to_numpy()
		xyz = track[["X", "Y", "Z"]].to_numpy(dtype=float)
		color = _color(index)

		if cutoff is not None:
			if trail_length is not None:
				visible = (times <= cutoff) & (times >= cutoff - trail_length)
			else:
				visible = times <= cutoff
		else:
			visible = np.ones(len(times), dtype=bool)

		segment = xyz[visible]
		if len(segment) >= 2:
			figure.add_trace(
				go.Scatter3d(
					x=segment[:, 0],
					y=segment[:, 1],
					z=segment[:, 2],
					mode="lines",
					name=f"Track {label}",
					line={"color": color, "width": 3},
					legendgroup=label,
					hovertemplate=f"Track {label}<br>X=%{{x:.3f}}<br>Y=%{{y:.3f}}<br>Z=%{{z:.3f}}<extra></extra>",
				)
			)
		elif len(segment) == 1:
			figure.add_trace(
				go.Scatter3d(
					x=segment[:, 0], y=segment[:, 1], z=segment[:, 2],
					mode="markers", name=f"Track {label}",
					marker={"color": color, "size": 4}, legendgroup=label,
				)
			)

		if animate:
			per_frame: Dict[int, tuple] = {}
			for frame_key, point in zip(times, xyz):
				bucket = int(frame_key) // max(1, animation_stride) * max(1, animation_stride)
				xs, ys, zs, texts, colors = per_frame.get(bucket, ([], [], [], [], []))
				xs.append(float(point[0]))
				ys.append(float(point[1]))
				zs.append(float(point[2]))
				texts.append(f"Track {label} @ frame {int(frame_key)}")
				colors.append(color)
				per_frame[bucket] = (xs, ys, zs, texts, colors)
			marker_frames.update(per_frame)
		elif current_frame is not None and len(times):
			nearest = int(np.argmin(np.abs(times - current_frame)))
			if abs(int(times[nearest]) - int(current_frame)) <= 1:
				point = xyz[nearest]
				highlight_x.append(float(point[0]))
				highlight_y.append(float(point[1]))
				highlight_z.append(float(point[2]))
				highlight_text.append(f"Track {label} @ frame {int(times[nearest])}")
				highlight_colors.append(color)

	if animate and marker_frames:
		ordered = sorted(marker_frames)
		base = ordered[0]
		xs, ys, zs, texts, colors = marker_frames[base]
		figure.add_trace(
			go.Scatter3d(
				x=xs, y=ys, z=zs,
				mode="markers",
				name="Current frame",
				marker={"color": colors, "size": 7, "line": {"color": "#222", "width": 1}},
				text=texts,
				hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
			)
		)
		animation_frames = [
			go.Frame(
				name=str(frame_key),
				data=[go.Scatter3d(
					x=marker_frames[frame_key][0],
					y=marker_frames[frame_key][1],
					z=marker_frames[frame_key][2],
					mode="markers",
					marker={"color": marker_frames[frame_key][4], "size": 7, "line": {"color": "#222", "width": 1}},
					text=marker_frames[frame_key][3],
					hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
				)],
			)
			for frame_key in ordered
		]
		figure.update_layout(
			updatemenus=[{
				"type": "buttons",
				"showactive": False,
				"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top",
				"buttons": [
					{
						"label": "▶ Play",
						"method": "animate",
						"args": [None, {
							"frame": {"duration": 60, "redraw": False},
							"fromcurrent": True,
							"transition": {"duration": 0, "easing": "linear"},
							"mode": "immediate",
						}],
					},
					{
						"label": "❚❚ Pause",
						"method": "animate",
						"args": [[None], {
							"frame": {"duration": 0, "redraw": False},
							"mode": "immediate",
						}],
					},
				],
			}],
			sliders=[{
				"active": 0,
				"yanchor": "top", "xanchor": "left",
				"x": 0.02, "y": 0.90,
				"currentvalue": {"prefix": "Frame: "},
				"pad": {"b": 10, "t": 60},
				"len": 0.9,
				"steps": [{"label": str(key), "method": "animate", "args": [[str(key)], {"mode": "immediate"}]}
						  for key in ordered],
			}],
		)
		figure.frames = animation_frames
	elif highlight_x:
		figure.add_trace(
			go.Scatter3d(
				x=highlight_x, y=highlight_y, z=highlight_z,
				mode="markers",
				name="Current frame",
				marker={"color": highlight_colors, "size": 7, "line": {"color": "#222", "width": 1}},
				text=highlight_text,
				hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
			)
		)

	if gt_tracks:
		for index, (actor, track) in enumerate(gt_tracks.items()):
			times = track["Timestamp"].to_numpy()
			xyz = track[["X", "Y", "Z"]].to_numpy(dtype=float)
			if current_frame is not None:
				xyz = xyz[times <= current_frame]
			if len(xyz) < 2:
				continue
			figure.add_trace(
				go.Scatter3d(
					x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
					mode="lines",
					name=f"GT {actor}",
					line={"color": "#888", "width": 6, "dash": "dot"},
					opacity=0.35,
					legendgroup="gt",
					showlegend=index < 3,
					hovertemplate=f"GT {actor}<br>X=%{{x:.3f}}<br>Y=%{{y:.3f}}<br>Z=%{{z:.3f}}<extra></extra>",
				)
			)

	if camera_centers is not None and not camera_centers.empty:
		figure.add_trace(
			go.Scatter3d(
				x=camera_centers["X"], y=camera_centers["Y"], z=camera_centers["Z"],
				mode="markers+text",
				name="Cameras",
				marker={"color": "#333", "size": 6, "symbol": "square"},
				text=[f"cam{int(c)}" for c in camera_centers["cam"]],
				textposition="top center",
				hoverinfo="name+text",
			)
		)

	figure.update_layout(
		title="3D trajectories" + (" (use the Play button to animate)" if animate else ""),
		template="plotly_white",
		height=height,
		margin={"l": 0, "r": 0, "t": (90 if animate else 40), "b": 0},
		scene={
			"xaxis": {"title": "X", "backgroundcolor": "#f7f7f7"},
			"yaxis": {"title": "Y", "backgroundcolor": "#f7f7f7"},
			"zaxis": {"title": "Z", "backgroundcolor": "#f7f7f7"},
			"aspectmode": "data",
		},
		legend={"itemsizing": "constant"},
	)
	return figure


# ---------------------------------------------------------------------------
# 2D projections
# ---------------------------------------------------------------------------
def projection_figure(
	pred_tracks: Dict[str, pd.DataFrame],
	gt_tracks: Optional[Dict[str, pd.DataFrame]] = None,
	current_frame: Optional[int] = None,
	plane: str = "XY",
	height: int = 520,
) -> go.Figure:
	"""Scatter of all tracks at ``current_frame`` on one coordinate plane.

	Past positions are drawn faintly so the frame slider reads as motion rather
	than as unrelated dots.
	"""
	axes = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}
	if plane not in axes:
		return _empty_figure(f"Unknown plane {plane!r}")
	x_col, y_col = axes[plane]

	figure = go.Figure()
	if not pred_tracks:
		return _empty_figure("No predicted tracks loaded")

	frame_min, frame_max = _frame_bounds(pred_tracks, gt_tracks)
	frame = current_frame if current_frame is not None else frame_max

	for index, label in enumerate(pred_tracks):
		track = pred_tracks[label]
		color = _color(index)
		times = track["Timestamp"].to_numpy()
		past = track[times < frame]
		if len(past):
			figure.add_trace(
				go.Scatter(
					x=past[x_col], y=past[y_col],
					mode="lines+markers",
					name=f"Track {label}",
					line={"color": color, "width": 1},
					marker={"size": 3, "color": color},
					opacity=0.35,
					legendgroup=label,
					showlegend=False,
				)
			)
		current = track[times == frame]
		if len(current):
			point = current.iloc[0]
			figure.add_trace(
				go.Scatter(
					x=[point[x_col]], y=[point[y_col]],
					mode="markers+text",
					name=f"Track {label}",
					marker={"color": color, "size": 11, "symbol": "circle", "line": {"color": "#222", "width": 0.7}},
					text=[label], textposition="top center",
					legendgroup=label,
					hovertemplate=f"Track {label}<br>{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
				)
			)

	if gt_tracks:
		for actor, track in gt_tracks.items():
			current = track[track["Timestamp"] == frame]
			if not len(current):
				continue
			point = current.iloc[0]
			figure.add_trace(
				go.Scatter(
					x=[point[x_col]], y=[point[y_col]],
					mode="markers",
					name=f"GT {actor}",
					marker={"color": "#000", "size": 15, "symbol": "x-open", "line": {"width": 2}},
					legendgroup="gt",
					showlegend=False,
					hovertemplate=f"GT {actor}<br>{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
				)
			)

	figure.update_layout(
		title=f"{plane} projection @ frame {frame}",
		template="plotly_white",
		height=height,
		xaxis_title=AXIS_LABELS[x_col],
		yaxis_title=AXIS_LABELS[y_col],
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	figure.update_xaxes(range=[frame_min - 0.5, frame_max + 0.5])
	return figure


def _frame_bounds(*track_sets: Optional[Dict[str, pd.DataFrame]]) -> tuple:
	mins: List[int] = []
	maxs: List[int] = []
	for tracks in track_sets:
		if not tracks:
			continue
		for track in tracks.values():
			times = track["Timestamp"].to_numpy()
			if len(times):
				mins.append(int(times.min()))
				maxs.append(int(times.max()))
	if not mins:
		return 0, 1
	return min(mins), max(maxs)


# ---------------------------------------------------------------------------
# Per-track plots
# ---------------------------------------------------------------------------
def track_coordinate_figure(track: pd.DataFrame, label: str) -> go.Figure:
	"""X/Y/Z against frame for a single track, marking interpolated samples."""
	figure = go.Figure()
	times = track["Timestamp"].to_numpy()
	real = ~track["is_interpolated"].to_numpy(dtype=bool) if "is_interpolated" in track else np.ones(len(times), bool)

	for axis in ("X", "Y", "Z"):
		values = track[axis].to_numpy(dtype=float)
		figure.add_trace(
			go.Scatter(
				x=times[real], y=values[real],
				mode="lines+markers", name=axis,
				line={"width": 2}, marker={"size": 5},
				hovertemplate=f"{axis}=%{{y:.4f}} @ frame %{{x}}<extra></extra>",
			)
		)
		if (~real).any():
			figure.add_trace(
				go.Scatter(
					x=times[~real], y=values[~real],
					mode="markers", name=f"{axis} (interp)",
					marker={"size": 5, "symbol": "x", "color": "#999"},
					hovertemplate=f"{axis} interp=%{{y:.4f}} @ frame %{{x}}<extra></extra>",
				)
			)

	figure.update_layout(
		title=f"Track {label} coordinates",
		template="plotly_white",
		height=380,
		xaxis_title="Frame",
		yaxis_title="World coordinate",
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


def track_speed_figure(tracks: Dict[str, pd.DataFrame], selected: Optional[Sequence[str]] = None) -> go.Figure:
	"""Per-frame step size for every track (a proxy for swimming speed)."""
	figure = go.Figure()
	labels = [label for label in tracks if not selected or label in set(selected)]
	for index, label in enumerate(labels[:40]):
		track = tracks[label]
		xyz = track[["X", "Y", "Z"]].to_numpy(dtype=float)
		if len(xyz) < 2:
			continue
		steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
		times = track["Timestamp"].to_numpy()[1:]
		figure.add_trace(
			go.Scatter(
				x=times, y=steps, mode="lines", name=f"Track {label}",
				line={"color": _color(index), "width": 1.5},
				hovertemplate=f"Track {label}<br>step=%{{y:.4f}} @ frame %{{x}}<extra></extra>",
			)
		)
	if not figure.data:
		return _empty_figure("No tracks with enough observations")

	figure.update_layout(
		title="Per-frame displacement (swimming speed)",
		template="plotly_white",
		height=420,
		xaxis_title="Frame",
		yaxis_title="Distance / frame",
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


def track_lifetime_figure(stats: pd.DataFrame) -> go.Figure:
	"""Gantt-style view of which tracks are alive over which frames."""
	if stats.empty:
		return _empty_figure("No track statistics")
	figure = go.Figure()
	for index, row in stats.iterrows():
		figure.add_trace(
			go.Bar(
				x=[int(row["lifetime_frames"])],
				y=[row["track"]],
				orientation="h",
				base=[int(row["first_frame"])],
				name=str(row["track"]),
				marker={"color": _color(index)},
				legendgroup=str(row["track"]),
				showlegend=False,
				hovertemplate=(
					f"Track {row['track']}<br>frames {int(row['first_frame'])}-{int(row['last_frame'])}"
					f"<br>obs={int(row['observations'])}<extra></extra>"
				),
			)
		)
	figure.update_layout(
		title="Track lifetimes",
		template="plotly_white",
		height=max(320, min(120, len(stats)) * 18),
		xaxis_title="Frame",
		yaxis_title="Track",
		barmode="overlay",
		margin={"l": 60, "r": 20, "t": 40, "b": 40},
	)
	return figure


def track_count_figure(frame: pd.DataFrame) -> go.Figure:
	"""How many tracks exist per frame."""
	if frame.empty:
		return _empty_figure("No predictions loaded")
	counts = frame.groupby("Timestamp")["object"].nunique().reset_index(name="tracks")
	figure = go.Figure(
		go.Scatter(
			x=counts["Timestamp"], y=counts["tracks"],
			mode="lines", line={"color": "#1f77b4", "width": 2},
			fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
			hovertemplate="frame %{x}<br>tracks=%{y}<extra></extra>",
		)
	)
	figure.update_layout(
		title="Tracks alive per frame",
		template="plotly_white",
		height=320,
		xaxis_title="Frame",
		yaxis_title="Number of tracks",
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------
def per_frame_metrics_figure(per_frame: pd.DataFrame) -> go.Figure:
	"""Counts plus precision/recall over time, as two stacked subplots."""
	if per_frame.empty:
		return _empty_figure("No per-frame data")

	figure = make_subplots(
		rows=2, cols=1, shared_xaxes=True,
		row_heights=[0.5, 0.5], vertical_spacing=0.12,
		subplot_titles=("Detections per frame", "Precision / recall per frame"),
	)
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["n_gt"], name="GT detections",
			mode="lines", line={"color": "#333", "width": 2},
			hovertemplate="frame %{x}<br>GT=%{y}<extra></extra>",
		), row=1, col=1,
	)
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["n_pred"], name="Predicted detections",
			mode="lines", line={"color": "#1f77b4", "width": 2},
			hovertemplate="frame %{x}<br>Pred=%{y}<extra></extra>",
		), row=1, col=1,
	)
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["n_matched"], name="Matched",
			mode="lines", line={"color": "#2ca02c", "width": 2},
			hovertemplate="frame %{x}<br>Matched=%{y}<extra></extra>",
		), row=1, col=1,
	)
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["precision"], name="Precision",
			mode="lines", line={"color": "#ff7f0e", "width": 2},
			hovertemplate="frame %{x}<br>precision=%{y:.3f}<extra></extra>",
		), row=2, col=1,
	)
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["recall"], name="Recall",
			mode="lines", line={"color": "#1f77b4", "width": 2},
			hovertemplate="frame %{x}<br>recall=%{y:.3f}<extra></extra>",
		), row=2, col=1,
	)
	figure.update_yaxes(title_text="Count", row=1, col=1)
	figure.update_yaxes(title_text="Ratio", range=[0, 1.05], row=2, col=1)
	figure.update_xaxes(title_text="Frame", row=2, col=1)
	figure.update_layout(
		template="plotly_white",
		height=660,
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 20, "t": 60, "b": 40},
	)
	return figure


def localisation_error_figure(per_frame: pd.DataFrame, gate: float) -> go.Figure:
	"""Mean matched 3D distance per frame against the matching gate."""
	if per_frame.empty or per_frame["mean_dist"].isna().all():
		return _empty_figure("No matched frames to measure")
	figure = go.Figure()
	figure.add_trace(
		go.Scatter(
			x=per_frame["frame"], y=per_frame["mean_dist"], name="Mean matched distance",
			mode="lines+markers", line={"color": "#d62728", "width": 2}, marker={"size": 4},
			hovertemplate="frame %{x}<br>mean dist=%{y:.4f}<extra></extra>",
		)
	)
	figure.add_hline(
		y=gate, line={"color": "#888", "dash": "dash"},
		annotation_text=f"gate = {gate:.2f}",
	)
	figure.update_layout(
		title="Localisation error per frame",
		template="plotly_white",
		height=340,
		xaxis_title="Frame",
		yaxis_title="Mean 3D distance",
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


# ---------------------------------------------------------------------------
# Scene plots
# ---------------------------------------------------------------------------
def detections_per_frame_figure(counts: pd.DataFrame) -> go.Figure:
	"""Stacked detection counts per camera over time."""
	if counts.empty:
		return _empty_figure("No detection counts")
	figure = go.Figure()
	for index, (cam, sub) in enumerate(counts.groupby("cam")):
		figure.add_trace(
			go.Scatter(
				x=sub["frame"], y=sub["detections"], name=f"cam{int(cam)}", mode="lines",
				line={"width": 1.5}, stackgroup="one",
				hovertemplate=f"cam{int(cam)}<br>frame %{{x}}<br>detections=%{{y}}<extra></extra>",
			)
		)
	figure.update_layout(
		title="Detections per camera over time",
		template="plotly_white",
		height=360,
		xaxis_title="Frame",
		yaxis_title="Detections",
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


def detection_size_figure(boxes: pd.DataFrame) -> go.Figure:
	"""Box-area distribution per camera plus a size-over-time view."""
	if boxes.empty:
		return _empty_figure("No boxes parsed")
	figure = go.Figure()
	for index, (cam, sub) in enumerate(boxes.groupby("cam")):
		figure.add_trace(
			go.Box(
				y=sub["area"], name=f"cam{int(cam)}", boxpoints="outliers",
				marker={"color": _color(index), "size": 3, "opacity": 0.5},
				line={"color": _color(index)},
				hovertemplate=f"cam{int(cam)}<br>area=%{{y:.1f}}<extra></extra>",
			)
		)
	figure.update_layout(
		title="Detection box area per camera",
		template="plotly_white",
		height=380,
		yaxis_title="Box area (px^2)",
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


def camera_layout_figure(centers: pd.DataFrame) -> go.Figure:
	"""Top-down (X/Z) camera rig layout."""
	if centers.empty:
		return _empty_figure("No camera extrinsics available")
	figure = go.Figure(
		go.Scatter(
			x=centers["X"], y=centers["Z"],
			mode="markers+text",
			marker={"color": "#333", "size": 12, "symbol": "square"},
			text=[f"cam{int(c)}" for c in centers["cam"]],
			textposition="top center",
			hovertemplate="%{text}<br>X=%{x:.2f}<br>Z=%{y:.2f}<extra></extra>",
		)
	)
	figure.update_layout(
		title="Camera rig (top-down X/Z)",
		template="plotly_white",
		height=420,
		xaxis_title="X",
		yaxis_title="Z",
		yaxis={"scaleanchor": "x", "scaleratio": 1},
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------
def loss_curves_figure(epochs: pd.DataFrame) -> go.Figure:
	"""Train/validation loss curves, with the contrastive and temporal terms."""
	if epochs.empty:
		return _empty_figure("No epoch lines found in this log")

	figure = go.Figure()
	series = [
		("avg_loss", "Total loss", True),
		("asso", "L_asso", False),
		("ctr", "L_ctr", False),
		("temp", "L_temp", False),
	]
	for index, (column, label, emphasise) in enumerate(series):
		if column not in epochs.columns:
			continue
		for split, dash, opacity in (("train", "solid", 1.0), ("val", "dot", 0.7)):
			sub = epochs[epochs["split"] == split]
			if sub.empty:
				continue
			figure.add_trace(
				go.Scatter(
					x=sub["epoch"], y=sub[column], mode="lines",
					name=f"{label} ({split})",
					line={
						"color": _color(index),
						"width": 3 if emphasise else 1.6,
						"dash": dash,
					},
					opacity=opacity,
					hovertemplate=f"{label} {split}<br>epoch %{{x}}<br>loss=%{{y:.5f}}<extra></extra>",
				)
			)

	figure.update_layout(
		title="Training losses",
		template="plotly_white",
		height=460,
		xaxis_title="Epoch",
		yaxis_title="Loss",
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
	)
	return figure


def debug_loss_figure(debug: pd.DataFrame) -> go.Figure:
	"""Per-frame loss components and gradient norm from ``debug.log``."""
	if debug.empty:
		return _empty_figure("No debug blocks found in this log")
	figure = go.Figure()
	labels = ["epoch", "scene", "frame"] if {"epoch", "scene", "frame"} <= set(debug.columns) else None
	x_values = debug[labels[0]] if labels else np.arange(len(debug))
	if labels and labels[0] == "frame":
		x_values = debug["frame"]

	for index, column in enumerate(["L_asso_t", "L_ctr_t", "L_temp", "total"]):
		if column not in debug.columns:
			continue
		figure.add_trace(
			go.Scatter(
				x=x_values, y=debug[column], mode="lines+markers",
				name=column, line={"color": _color(index), "width": 1.5}, marker={"size": 4},
				hovertemplate=f"{column}<br>frame %{{x}}<br>value=%{{y:.5f}}<extra></extra>",
			)
		)
	if "grad_norm" in debug.columns:
		figure.add_trace(
			go.Scatter(
				x=x_values, y=debug["grad_norm"], mode="lines", name="grad_norm",
				line={"color": "#111", "width": 1.2, "dash": "dash"}, yaxis="y2",
				hovertemplate="grad_norm<br>frame %{x}<br>value=%{y:.4f}<extra></extra>",
			)
		)
	figure.update_layout(
		title="Per-step debug losses (utils/debug.py)",
		template="plotly_white",
		height=420,
		xaxis_title="Frame",
		yaxis_title="Loss",
		yaxis2={"title": "Gradient norm", "overlaying": "y", "side": "right", "showgrid": False},
		legend={"itemsizing": "constant"},
		margin={"l": 40, "r": 60, "t": 40, "b": 40},
	)
	return figure
