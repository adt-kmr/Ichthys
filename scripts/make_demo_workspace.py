"""Generate a synthetic SynFish-like workspace for exercising the dashboard.

Creates, under a target directory:
    synfish/test/test1.json      scene JSON with calibration + detections
    synfish/test/test1/cam{0,1,2}/frame_XXXX.jpeg   (tiny placeholder files)
    synfish/test/gt-traj/test1-gt.csv
    outputs/test1.txt            prediction file with injected ID switches
    checkpoints/model/best_ckpt.pth
    checkpoints/model/debug.log
    train.log                     train.py stdout lines

Usage:  python scripts/make_demo_workspace.py [target_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NUM_FRAMES = 120
NUM_FISH = 14
SEED = 7


def fish_paths(rng: np.random.Generator) -> np.ndarray:
	"""Smooth random walks, one row per fish: (NUM_FISH, NUM_FRAMES, 3)."""
	steps = rng.normal(scale=0.12, size=(NUM_FISH, NUM_FRAMES, 3))
	# Smooth with a simple 3-tap filter so tracks look like swimming, not jitter.
	smooth = steps.copy()
	smooth[:, 1:-1] = 0.25 * steps[:, :-2] + 0.5 * steps[:, 1:-1] + 0.25 * steps[:, 2:]
	base = rng.uniform(-2.0, 2.0, size=(NUM_FISH, 1, 3))
	return np.cumsum(smooth, axis=1) + base


def label(index: int) -> str:
	letters = []
	while True:
		index, rem = divmod(index, 26)
		letters.append(chr(ord("A") + rem))
		if index == 0:
			break
		index -= 1
	return "".join(reversed(letters))


def project(points: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> tuple:
	"""Project world points into one camera; returns (inside, uv)."""
	camera = (r @ points.T).T + t
	depth = np.clip(camera[:, 2], 1e-6, None)
	uv = (k[:2, :2] @ (camera[:, :2] / depth[:, None]).T).T + k[:2, 2]
	return camera[:, 2] > 0, uv


def main() -> None:
	target = Path(sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / "demo_workspace")
	rng = np.random.default_rng(SEED)

	test_dir = target / "synfish" / "test"
	scene_dir = test_dir / "test1"
	gt_dir = test_dir / "gt-traj"
	outputs = target / "outputs"
	ckpt_dir = target / "checkpoints" / "demo-model"
	for path in (gt_dir, outputs, ckpt_dir):
		path.mkdir(parents=True, exist_ok=True)

	paths = fish_paths(rng)

	# --- cameras: a simple triangle looking at the origin ---------------------
	# Build a right-handed camera basis (x=right, y=up, z=forward) so that
	# z_cam = forward . (p - centre) is positive for points in front of the lens.
	world_up = np.array([0.0, 0.0, 1.0])
	radius = 6.0
	cameras = []
	for index, azimuth in enumerate((0.0, 120.0, 240.0)):
		theta = np.deg2rad(azimuth)
		center = np.array([radius * np.cos(theta), radius * np.sin(theta), 2.0])
		forward = -center / np.linalg.norm(center)
		right = np.cross(world_up, forward)
		right /= np.linalg.norm(right)
		up = np.cross(forward, right)
		r = np.stack([right, up, forward])
		t = -r @ center
		intrinsic = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
		cameras.append({"cam": index, "info": {
			"K": intrinsic.tolist(), "R|T": np.hstack([r, t[:, None]]).tolist(), "frames": [],
		}})
		cam_dir = scene_dir / f"cam{index}"
		cam_dir.mkdir(parents=True, exist_ok=True)

	# --- per-frame detections + placeholder images ---------------------------
	for frame in range(NUM_FRAMES):
		# Drop a couple of fish in some frames so tracks have gaps.
		visible = paths[:, frame, :].copy()
		missing = set()
		if frame == 40:
			missing = {0, 1}
		if frame == 41:
			missing = {1}
		for index, camera in enumerate(cameras):
			intrinsic = np.array(camera["info"]["K"])
			extrinsic = np.array(camera["info"]["R|T"])
			inside, uv = project(visible, intrinsic, extrinsic[:, :3], extrinsic[:, 3])
			annotations = []
			for fish in range(NUM_FISH):
				if fish in missing or not inside[fish]:
					continue
				half = rng.uniform(14.0, 26.0)
				cx, cy = uv[fish]
				annotations.append({
					"bbox": [float(cx - half), float(cy - half * 0.6), float(cx + half), float(cy + half * 0.6)],
					"score": float(rng.uniform(0.55, 0.99)),
				})
			camera["info"]["frames"].append({"frame": frame, "annotations": annotations})
			# Placeholder image bytes; the dashboard never decodes them.
			(scene_dir / f"cam{index}" / f"frame_{frame:04d}.jpeg").write_bytes(b"\xff\xd8\xff\xe0stub")

	(test_dir / "test1.json").write_text(json.dumps(cameras), encoding="utf-8")

	# --- ground truth ---------------------------------------------------------
	gt_lines = ["Actor,Timestamp,X,Y,Z"]
	for fish in range(NUM_FISH):
		for frame in range(NUM_FRAMES):
			x, y, z = paths[fish, frame]
			gt_lines.append(f"fish-{fish},{frame},{x:.6f},{y:.6f},{z:.6f}")
	(gt_dir / "test1-gt.csv").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

	# --- predictions: noisy, with 2 injected identity switches and 1 ghost -----
	pred_lines = ["object,Timestamp,X,Y,Z,group"]
	for fish in range(NUM_FISH):
		for frame in range(NUM_FRAMES):
			if frame in ({40, 41} if fish in (0, 1) else set()):
				continue
			if fish == 5 and frame >= 60:
				name = label(NUM_FISH + fish)
			elif fish == 9 and frame >= 80:
				name = label(2 * NUM_FISH + fish)
			else:
				name = label(fish)
			noise = rng.normal(scale=0.035, size=3)
			x, y, z = paths[fish, frame] + noise
			pred_lines.append(f"{name},{frame},{x:.6f},{y:.6f},{z:.6f},{{cam0-{fish},cam1-{fish},cam2-{fish}}}")
	# A couple of interpolated rows, as infer.py gap filling produces.
	pred_lines.append(f"{label(0)},41,{paths[0,40,0]:.6f},{paths[0,40,1]:.6f},{paths[0,40,2]:.6f},{{interp}}")
	# A short ghost track that should be filtered by min_obs but is not here.
	pred_lines.append("ZZ,10,0.500000,0.500000,0.500000,{{cam0-99}}")
	(outputs / "test1.txt").write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

	# --- checkpoint placeholder ----------------------------------------------
	(ckpt_dir / "best_ckpt.pth").write_bytes(b"PK\x03\x04" + b"\x00" * 4096)
	(ckpt_dir / "epoch_0012.pth").write_bytes(b"PK\x03\x04" + b"\x00" * 2048)

	# --- debug.log in utils/debug.py format -----------------------------------
	debug_lines = []
	for epoch in (11, 12):
		for frame in range(0, 30, 3):
			asso = 0.42 / (1 + 0.08 * epoch) + rng.normal(scale=0.01)
			ctr = 0.31 / (1 + 0.06 * epoch) + rng.normal(scale=0.01)
			temp = 0.18 / (1 + 0.05 * epoch) + rng.normal(scale=0.005)
			total = asso + ctr + temp
			debug_lines += [
				f"Epoch {epoch}, Scene test1, Frame {frame}->{frame + 1}",
				f"Losses: L_asso_t={asso:.4f}, L_asso_tp1={asso * 1.05:.4f}, "
				f"L_ctr_t={ctr:.4f}, L_ctr_tp1={ctr * 1.02:.4f}, L_temp={temp:.4f}, total={total:.4f}",
				"H_t norm: mean=1.0021, max=1.2210",
				"P_t: mean=0.3512, min=0.0004, max=0.9987",
				"H_tp1 norm: mean=1.0033, max=1.2288",
				"P_tp1: mean=0.3488, min=0.0003, max=0.9971",
				"G_t unique labels: [0 1 2 3], C_t mean=0.7412",
				"G_tp1 unique labels: [0 1 2 3], C_tp1 mean=0.7390",
				f"Groups_t: {NUM_FISH}, Groups_tp1: {NUM_FISH}",
				"Matches: [(0, 0), (1, 1), (2, 2)]",
				f"Gradient norm: {1.4 / (1 + 0.1 * epoch):.4f}",
				"--- End of Debug ---\n",
			]
	(ckpt_dir / "debug.log").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")

	# --- train.py stdout log --------------------------------------------------
	train_lines = ["Loading datasets...", "#train scenes: 8"]
	for epoch in range(1, 13):
		avg = 0.95 * (0.82 ** epoch)
		asso = 0.50 * (0.85 ** epoch)
		ctr = 0.28 * (0.80 ** epoch)
		temp = 0.17 * (0.88 ** epoch)
		train_lines.append(
			f"Epoch {epoch} done. Avg loss: {avg:.6f} | mean(L_asso_sum): {asso:.6f} | "
			f"mean(L_ctr_sum): {ctr:.6f} | mean(L_temp): {temp:.6f}"
		)
		if epoch % 4 == 0:
			train_lines.append(
				f"Epoch {epoch} val. Avg loss: {avg * 1.06:.6f} | mean(L_asso_sum): {asso * 1.05:.6f} | "
				f"mean(L_ctr_sum): {ctr * 1.04:.6f} | mean(L_temp): {temp * 1.02:.6f}"
			)
		if epoch in (4, 8, 12):
			train_lines.append(f"Saved best checkpoint checkpoints/demo-model/best_ckpt.pth (train_avg={avg:.6f})")
		else:
			train_lines.append(f"Saved checkpoint checkpoints/demo-model/epoch_{epoch:04d}.pth")
	train_lines.append("Total training time: 6.42 hours")
	train_lines.append("Training finished.")
	(target / "train.log").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

	print(f"Demo workspace written to {target}")
	print(f"  predictions : {outputs / 'test1.txt'}")
	print(f"  ground truth: {gt_dir / 'test1-gt.csv'}")
	print(f"  scene       : {test_dir / 'test1.json'}")
	print(f"  debug log   : {ckpt_dir / 'debug.log'}")
	print(f"  train log   : {target / 'train.log'}")


if __name__ == "__main__":
	main()
