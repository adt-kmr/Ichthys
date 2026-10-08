# Ichthys

Self-supervised 3D multi-object tracking from calibrated multi-view video.

## Idea

Given several synchronized, calibrated cameras and a 2D detector, one subject is observed as several independent boxes — one per camera. Recovering 3D trajectories therefore requires answering two questions:

1. **Cross-view** — which detections across cameras belong to the same subject?
2. **Temporal** — which subject at frame `t` is the same subject at frame `t+1`?

Ichthys answers both with a learned association network over detection tokens, then triangulates matched detections into 3D positions.

The distinguishing property is that supervision is **derived from geometry rather than annotation**. If two detections triangulate to a point that reprojects back inside both original boxes, they must be the same subject. This yields pseudo-labels automatically, so no identity labels and no 3D trajectory annotations are required.

Training optimises three terms:

| Term | Purpose |
| --- | --- |
| `L_asso` | Cross-view grouping |
| `L_ctr` | Contrastive — pulls same-subject embeddings together, pushes different subjects apart |
| `L_temp` | Trains `PredictorM` to forecast an embedding across an occlusion gap |

At inference, when a subject is occluded for several frames, `PredictorM` predicts the embedding it should have now and matches against that prediction — the direct payoff of `L_temp` rather than a heuristic.

The main model is geometry-only: an external detector supplies per-camera, per-frame detections, and this repository performs association, triangulation, and tracking.

## Installation

Python 3.8 or newer is required. Install PyTorch for your CUDA version first if the default package is unsuitable, then install this repository:

```bash
conda create -n ichthys python=3.10 -y
conda activate ichthys
python -m pip install --upgrade pip
python -m pip install -e .
```

The main models use `--no_dino`. Optional DINOv3 experiments expect a DINOv3 checkout at `dinov3/` and weights at `DINOV3Model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`; override both with `--dino_repo_dir` and `--dino_weights`.

## Data layout

Each split contains scene JSON files and equally named image directories. Ground-truth 3D trajectories live in `gt-traj/`:

```text
synfish/
	train/
		train1.json ... train8.json
		train1/ ... train8/
			cam0/frame_0000.jpeg
			cam1/frame_0000.jpeg
			cam2/frame_0000.jpeg
		gt-traj/train1-gt.csv ... train8-gt.csv
	test/
		test1.json ... test6.json
		test1/ ... test6/
			cam0/frame_0000.jpeg
			cam1/frame_0000.jpeg
			cam2/frame_0000.jpeg
		gt-traj/test1-gt.csv ... test6-gt.csv
```

Scene JSON files contain camera intrinsics `K`, extrinsics `R|T`, synchronized frames, bounding boxes, confidence scores, and optional COCO-RLE masks. Mask centroids are used when available; bounding-box centres are the fallback.

GT CSVs are for evaluation only, never for training. They may contain a few frames beyond a scene's detection range, so `--match_pred_range` limits evaluation to frames covered by predictions. Two training scenes also have a one-frame GT/image endpoint mismatch (`train2` has image frame 363 without GT; `train8` has image frame 361 without GT); do not extend tracks or invent GT for these frames.

The bundled detection package uses SAM3 detections. The YOLO26+SAM2 model requires its own matching detections and cannot be reproduced from a SAM3-only package.

## Training

```bash
bash scripts/train_geometry_only.sh \
	/path/to/synfish/train \
	checkpoints/model-name
```

Ground-truth trajectories are not used during training.

## Inference

```bash
python infer.py \
	checkpoints/model-name/best_ckpt.pth \
	/path/to/scene.json \
	/path/to/scene-images \
	--no_dino \
	--output outputs/scene.txt
```

Output columns are `object,Timestamp,X,Y,Z,group`. Tracks are born and killed dynamically, so no object-count prior is required.

## Evaluation

```bash
python evaluate.py \
	--pred outputs/test1.txt \
	--gt /path/to/synfish/test/gt-traj/test1-gt.csv \
	--match_pred_range
```

Reports MOTChallenge-style metrics (MOTA, MOTP, IDF1, IDP, IDR, IDSW, FP, FN, MT/PT/ML, MTBFm) with per-frame matching defined by gated Euclidean distance in 3D.

Full held-out runs:

```bash
bash scripts/evaluate_synfish.sh CHECKPOINT /path/to/synfish/test outputs/synfish
bash scripts/evaluate_zef.sh CHECKPOINT ZEF_ROOT OUTPUT_DIR
```

## Dashboard

An optional Streamlit interface visualises whatever the pipeline has already written. It reads the same files as the CLIs and is never imported by training or inference.

```bash
pip install -e ".[viz]"
streamlit run app.py -- --workspace outputs
```

The workspace root is editable in the sidebar, and prediction / GT / log files can be dropped in via the uploader. Files are identified by content, not name: predictions by their `object,Timestamp,X,Y,Z` header, GT by `Actor,Timestamp,X,Y,Z`.

| Tab | Contents |
| --- | --- |
| Overview | Files discovered, checkpoint inventory, package versions, CUDA availability |
| 3D trajectories | Interactive 3D tracks with GT overlay, camera positions, track filtering, frame scrubbing, in-chart playback |
| Tracks | XY/XZ/YZ projections, track lifetimes, per-frame displacement, nearest-GT identity check |
| Metrics | MOTA / IDF1 / MOTP / IDSW / MTBFm plus per-frame detections, precision, recall, localisation error |
| Scene data | Camera rig layout, calibration, detections over time, box size distributions, per-frame detections |
| Training | Loss curves from a piped log; per-step losses and gradient norms from `--debug` output |
| Jobs | Launch `infer.py` / `evaluate.py` / `train.py` / `sort3d.py` and tail logs live |

The Metrics tab scores in-process by reusing `evaluate.py`'s readers and cost function, so its values and formatting match the CLI exactly. The matching gate is inferred from the GT filename unless overridden. Playback runs client-side inside the chart, so stepping frames does not cost a rerun.

To see the interface without a dataset, generate a small synthetic workspace:

```bash
python scripts/make_demo_workspace.py
streamlit run app.py -- --workspace demo_workspace
```

Dashboard checks:

```bash
python scripts/check_dashboard.py   # loaders, metrics, every figure
python scripts/check_app.py        # renders app.py via Streamlit AppTest
python scripts/check_jobs.py       # runs evaluate.py through the job runner
```

These live in `scripts/` rather than `tests/` because they require the optional `viz` extra, which the default `pytest` run does not install.

## Notes

- `num_cams` is set to 3; the architecture assumes three cameras rather than an arbitrary number.
- Detections are a required input — this repository does not include a detector.
- Validate accuracy on real footage before relying on it, as the reference benchmark is synthetic.