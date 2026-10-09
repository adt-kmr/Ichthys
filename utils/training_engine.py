from __future__ import annotations

import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .association_module import AssociationTransformer
from .association_module_noattn import AssociationNoGlobalAttention
from .debug import append_train_step_debug
from .encode_features import GeometricFeatureEncoder
from .generate_pseudogt import build_pseudo_gt_cache_for_dataset, get_or_make_pseudo_gt
from .grouping import group_and_triangulate
from .loss import info_nce_detection_level, temporal_prediction_loss, weighted_association_loss
from .processdata import load_and_parse_json_folder
from .temporallinking import build_fallback_matches, temporal_linking


@dataclass
class TrainingConfig:
    train_folder: str
    val_folder: str | None = None
    checkpoint_dir: str = "checkpoints"
    total_epochs: int = 400
    warmup_epochs: int = 100
    device: str | None = None
    debug: bool = False
    debug_file: str | None = None
    val_every: int = 999
    selection_metric: str = "train"
    use_dino: bool = True
    no_global_attention: bool = False
    grouping_match_mode: str = "hungarian"
    train_workers: int = 64
    val_workers: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    lambda_asso: float = 1.0
    lambda_ctr: float = 0.5
    lambda_temp: float = 0.25
    tau: float = 0.07
    thr_assoc: float = 0.6
    alpha: float = 0.6
    beta: float = 0.4
    dist_thresh: float = 0.5
    sim_thresh: float = 0.6
    conf_min: float = 0.3
    window_size: int = 10
    encoder_d_model: int = 128
    encoder_hidden_dim: int = 64
    assoc_layers: int = 4
    assoc_heads: int = 4
    predictor_hidden_dim: int = 256
    image_size: tuple[int, int] = (1920, 1080)
    dino_repo_dir: str = "dinov3"
    dino_weights: str = "DINOV3Model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.selection_metric not in {"train", "val"}:
            raise ValueError("selection_metric must be 'train' or 'val'.")
        if self.selection_metric == "val" and self.val_folder is None:
            raise ValueError("selection_metric='val' requires a separate validation folder.")


@dataclass
class EpochMetrics:
    total_loss: float = 0.0
    association_loss: float = 0.0
    contrastive_loss: float = 0.0
    temporal_loss: float = 0.0
    steps: int = 0

    def update(self, *, total: float, association: float, contrastive: float, temporal: float) -> None:
        self.total_loss += total
        self.association_loss += association
        self.contrastive_loss += contrastive
        self.temporal_loss += temporal
        self.steps += 1

    def mean_total(self) -> float:
        return self.total_loss / max(1, self.steps)

    def summary(self) -> tuple[float, float, float, float]:
        denom = max(1, self.steps)
        return (
            self.total_loss / denom,
            self.association_loss / denom,
            self.contrastive_loss / denom,
            self.temporal_loss / denom,
        )


@dataclass
class CachedFrame:
    frame_num: int
    features: torch.Tensor


@dataclass
class FrameState:
    features: torch.Tensor
    hidden: torch.Tensor
    pairwise_probs: torch.Tensor
    cam_ids: torch.Tensor
    boxes: torch.Tensor


@dataclass
class StepArtifacts:
    total_loss: torch.Tensor
    association_t: torch.Tensor
    association_tp1: torch.Tensor
    contrastive_t: torch.Tensor
    contrastive_tp1: torch.Tensor
    temporal: torch.Tensor
    groups_t: list[dict[str, Any]]
    groups_tp1: list[dict[str, Any]]
    matches: list[tuple[Any, ...]]
    pseudo_gt_t: Any
    pseudo_gt_conf_t: Any
    pseudo_gt_tp1: Any
    pseudo_gt_conf_tp1: Any


class PredictorHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class IchthysTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.debug_file = config.debug_file or str(self.checkpoint_dir / "debug.log")
        self.best_score = float("inf")

        self.encoder = self._build_encoder()
        self.assoc_model = self._build_assoc_model()
        self.predictor = PredictorHead(
            d_model=self.config.encoder_d_model,
            hidden_dim=self.config.predictor_hidden_dim,
        ).to(self.config.device)
        self.optimizer, self.optimized_params = self._build_optimizer()

    def run(self) -> None:
        train_dataset, val_dataset = self._load_datasets()
        self._build_pseudo_gt_cache(train_dataset, val_dataset)

        for epoch in tqdm(range(1, self.config.total_epochs + 1), desc="Training epochs", mininterval=5, miniters=1):
            use_ctr = epoch > self.config.warmup_epochs
            use_temp = epoch > self.config.warmup_epochs

            train_metrics = self._run_epoch(
                dataset=train_dataset,
                dataset_root=self.config.train_folder,
                epoch=epoch,
                training=True,
                use_ctr=use_ctr,
                use_temp=use_temp,
            )
            train_avg, train_asso, train_ctr, train_temp = train_metrics.summary()
            print(
                f"Epoch {epoch} done. "
                f"Avg loss: {train_avg:.6f} | "
                f"mean(L_asso_sum): {train_asso:.6f} | "
                f"mean(L_ctr_sum): {train_ctr:.6f} | "
                f"mean(L_temp): {train_temp:.6f}"
            )

            run_val = False
            val_metrics = None
            if val_dataset is not None:
                val_every = max(1, int(self.config.val_every))
                run_val = (epoch % val_every == 0) or (epoch == self.config.total_epochs)

            if val_dataset is not None and run_val:
                val_metrics = self._run_epoch(
                    dataset=val_dataset,
                    dataset_root=self.config.val_folder,
                    epoch=epoch,
                    training=False,
                    use_ctr=use_ctr,
                    use_temp=use_temp,
                )
                val_avg, val_asso, val_ctr, val_temp = val_metrics.summary()
                print(
                    f"Epoch {epoch} val. "
                    f"Avg loss: {val_avg:.6f} | "
                    f"mean(L_asso_sum): {val_asso:.6f} | "
                    f"mean(L_ctr_sum): {val_ctr:.6f} | "
                    f"mean(L_temp): {val_temp:.6f}"
                )
            elif val_dataset is not None:
                print(f"Epoch {epoch}: skipping val (val_every={max(1, int(self.config.val_every))})")

            self._save_checkpoints(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics, validation_ran=run_val)

        print("Training finished.")

    def _build_encoder(self) -> GeometricFeatureEncoder:
        repository_root = Path(__file__).resolve().parents[1]
        dino_repo_dir = Path(self.config.dino_repo_dir)
        dino_weights = Path(self.config.dino_weights)
        if not dino_repo_dir.is_absolute():
            dino_repo_dir = repository_root / dino_repo_dir
        if not dino_weights.is_absolute():
            dino_weights = repository_root / dino_weights

        encoder = GeometricFeatureEncoder(
            d_model=self.config.encoder_d_model,
            hidden_dim=self.config.encoder_hidden_dim,
            num_cams=3,
            image_size=self.config.image_size,
            use_dino=bool(self.config.use_dino),
            dinov3_repo_dir=str(dino_repo_dir),
            weights=str(dino_weights),
            device=self.config.device,
        ).to(self.config.device)

        dino_trainable = False
        if getattr(encoder, "use_dino", False) and hasattr(encoder, "dino_model"):
            dino_trainable = any(parameter.requires_grad for parameter in encoder.dino_model.parameters())
        print(
            "[setup] "
            f"use_dino={getattr(encoder, 'use_dino', False)} "
            f"dino_trainable={dino_trainable}"
        )
        return encoder

    def _build_assoc_model(self) -> nn.Module:
        association_class = AssociationNoGlobalAttention if self.config.no_global_attention else AssociationTransformer
        return association_class(
            in_dim=self.encoder.total_d_model,
            d_model=self.config.encoder_d_model,
            num_layers=self.config.assoc_layers,
            num_heads=self.config.assoc_heads,
            use_rope=True,
        ).to(self.config.device)

    def _build_optimizer(self) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
        if getattr(self.encoder, "use_dino", False):
            params = list(self.encoder.parameters())
        else:
            params = list(getattr(self.encoder, "mlp").parameters()) if hasattr(self.encoder, "mlp") else []
        params += list(self.assoc_model.parameters())
        params += list(self.predictor.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer, params

    def _load_datasets(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        print("Loading datasets...")
        train_dataset = load_and_parse_json_folder(self.config.train_folder, num_workers=self.config.train_workers)
        print(f"#train scenes: {len(train_dataset)}")

        val_dataset = None
        if self.config.val_folder is not None:
            val_dataset = load_and_parse_json_folder(self.config.val_folder, num_workers=self.config.val_workers)
            print(f"#val scenes: {len(val_dataset)}")
        return train_dataset, val_dataset

    def _build_pseudo_gt_cache(
        self,
        train_dataset: list[dict[str, Any]],
        val_dataset: list[dict[str, Any]] | None,
    ) -> None:
        build_pseudo_gt_cache_for_dataset(
            dataset=train_dataset,
            dataset_root=self.config.train_folder,
            name="train",
        )
        if val_dataset is not None and self.config.val_folder is not None:
            build_pseudo_gt_cache_for_dataset(
                dataset=val_dataset,
                dataset_root=self.config.val_folder,
                name="val",
            )

    def _run_epoch(
        self,
        *,
        dataset: list[dict[str, Any]],
        dataset_root: str | None,
        epoch: int,
        training: bool,
        use_ctr: bool,
        use_temp: bool,
    ) -> EpochMetrics:
        if dataset_root is None:
            raise ValueError("dataset_root must be provided for training or validation.")

        metrics = EpochMetrics()
        self.encoder.train(mode=training)
        self.assoc_model.train(mode=training)
        self.predictor.train(mode=training)

        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            for scene in tqdm(dataset, desc=f"Epoch {epoch} {'scenes' if training else 'val'}"):
                cams = scene["cams"]
                frames = scene["frames"]
                scene_name = scene["names"]
                if len(frames) < 2:
                    continue

                camera_ids = self._camera_ids_from_scene(cams)
                self._prepare_scene_encoder(dataset_root, scene_name, frames, camera_ids)
                track_memory: deque[dict[str, Any]] = deque(maxlen=self.config.window_size)
                cached_frame: CachedFrame | None = None

                for frame_t, frame_tp1 in zip(frames[:-1], frames[1:]):
                    state_t, cached_frame = self._frame_state_from_cache_or_encode(
                        dataset_root=dataset_root,
                        scene_name=scene_name,
                        frame=frame_t,
                        cams=cams,
                        camera_ids=camera_ids,
                        cached_frame=cached_frame,
                    )
                    state_tp1, next_cached = self._encode_frame_state(
                        dataset_root=dataset_root,
                        scene_name=scene_name,
                        frame=frame_tp1,
                        cams=cams,
                        camera_ids=camera_ids,
                    )
                    cached_frame = CachedFrame(frame_num=int(frame_tp1["frame"]), features=next_cached.features.detach())

                    if state_t.features.numel() == 0 or state_tp1.features.numel() == 0:
                        continue

                    step = self._compute_step(
                        dataset_root=dataset_root,
                        cams=cams,
                        scene_name=scene_name,
                        epoch=epoch,
                        frame_t=frame_t,
                        frame_tp1=frame_tp1,
                        state_t=state_t,
                        state_tp1=state_tp1,
                        track_memory=track_memory,
                        use_ctr=use_ctr,
                        use_temp=use_temp,
                    )

                    if training:
                        self.optimizer.zero_grad()
                        step.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.optimized_params, self.config.grad_clip_norm)
                        self.optimizer.step()

                    metrics.update(
                        total=float(step.total_loss.detach().cpu().item()),
                        association=float((step.association_t + step.association_tp1).detach().cpu().item()),
                        contrastive=float((step.contrastive_t + step.contrastive_tp1).detach().cpu().item()),
                        temporal=float(step.temporal.detach().cpu().item()),
                    )

                    if use_ctr or use_temp:
                        track_memory.append(
                            {
                                "frame": int(frame_t["frame"]),
                                "groups": [
                                    {
                                        "emb_tensor": group["emb_tensor"].detach(),
                                        "X": group["X"],
                                        "conf": group.get("conf", 0.0),
                                    }
                                    for group in step.groups_t
                                ],
                            }
                        )
        return metrics

    def _prepare_scene_encoder(
        self,
        dataset_root: str,
        scene_name: str,
        frames: list[dict[str, Any]],
        camera_ids: list[int],
    ) -> None:
        if camera_ids:
            self.encoder.num_cams = max(camera_ids) + 1
        if frames and camera_ids:
            try:
                first_frame_num = int(frames[0]["frame"])
                first_base = Path(dataset_root) / scene_name / f"cam{camera_ids[0]}" / f"frame_{first_frame_num:04d}"
                first_image = self._resolve_frame_image(first_base)
                if os.path.exists(first_image):
                    self.encoder.set_image_size_from_path(first_image)
            except Exception:
                pass

    @staticmethod
    def _camera_ids_from_scene(cams: dict[str, Any]) -> list[int]:
        return sorted(
            {
                int(key.split("-")[0].replace("cam", ""))
                for key in cams.keys()
                if str(key).startswith("cam") and "-K" in str(key)
            }
        )

    @staticmethod
    def _resolve_frame_image(base_path: Path) -> str:
        jpeg_path = base_path.with_suffix(".jpeg")
        if jpeg_path.exists():
            return str(jpeg_path)
        jpg_path = base_path.with_suffix(".jpg")
        if jpg_path.exists():
            return str(jpg_path)
        return str(jpeg_path)

    def _frame_image_paths(
        self,
        dataset_root: str,
        scene_name: str,
        frame_num: int,
        camera_ids: list[int],
    ) -> list[str]:
        image_paths = []
        for camera_id in camera_ids:
            base = Path(dataset_root) / scene_name / f"cam{camera_id}" / f"frame_{frame_num:04d}"
            image_paths.append(self._resolve_frame_image(base))
        return image_paths

    def _frame_state_from_cache_or_encode(
        self,
        *,
        dataset_root: str,
        scene_name: str,
        frame: dict[str, Any],
        cams: dict[str, Any],
        camera_ids: list[int],
        cached_frame: CachedFrame | None,
    ) -> tuple[FrameState, CachedFrame]:
        frame_num = int(frame["frame"])
        if cached_frame is not None and cached_frame.frame_num == frame_num:
            state = self._build_frame_state(features=cached_frame.features, frame=frame)
            return state, cached_frame
        state, cache_entry = self._encode_frame_state(
            dataset_root=dataset_root,
            scene_name=scene_name,
            frame=frame,
            cams=cams,
            camera_ids=camera_ids,
        )
        return state, cache_entry

    def _encode_frame_state(
        self,
        *,
        dataset_root: str,
        scene_name: str,
        frame: dict[str, Any],
        cams: dict[str, Any],
        camera_ids: list[int],
    ) -> tuple[FrameState, CachedFrame]:
        frame_num = int(frame["frame"])
        images = self._frame_image_paths(dataset_root, scene_name, frame_num, camera_ids)
        features, _ = self.encoder(frame["detections"], cams, images, return_raw=True)
        features = features.to(self.config.device)
        state = self._build_frame_state(features=features, frame=frame)
        return state, CachedFrame(frame_num=frame_num, features=features.detach())

    def _build_frame_state(self, *, features: torch.Tensor, frame: dict[str, Any]) -> FrameState:
        boxes = self._make_boxes_tensor(frame["detections"])
        cam_ids = torch.tensor(
            [detection["cam_id"] for detection in frame["detections"]],
            dtype=torch.long,
            device=self.config.device,
        )
        hidden, pairwise_probs = self._run_association(features, boxes, cam_ids)
        return FrameState(
            features=features,
            hidden=hidden,
            pairwise_probs=pairwise_probs,
            cam_ids=cam_ids,
            boxes=boxes,
        )

    def _make_boxes_tensor(self, detections: list[dict[str, Any]]) -> torch.Tensor:
        boxes = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            boxes.append(
                [
                    cx / self.encoder.img_w,
                    cy / self.encoder.img_h,
                    w / self.encoder.img_w,
                    h / self.encoder.img_h,
                ]
            )
        return torch.tensor(boxes, dtype=torch.float32, device=self.config.device)

    def _run_association(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        cam_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.assoc_model.input_proj(features)
        if self.assoc_model.use_rope and self.assoc_model.rope is not None and boxes is not None:
            positions = torch.stack([boxes[:, 1], boxes[:, 0]], dim=-1)
            projected = self.assoc_model.rope(projected, positions)

        hidden = self.assoc_model.transformer(projected.unsqueeze(0)).squeeze(0)
        num_detections = hidden.size(0)
        probs = torch.zeros((num_detections, num_detections), dtype=hidden.dtype, device=hidden.device)
        if num_detections == 0:
            return hidden, probs

        cam_matrix_i = cam_ids.view(num_detections, 1).expand(num_detections, num_detections)
        cam_matrix_j = cam_ids.view(1, num_detections).expand(num_detections, num_detections)
        cross_camera_pairs = (cam_matrix_i != cam_matrix_j).nonzero(as_tuple=False)
        if cross_camera_pairs.size(0) == 0:
            return hidden, probs

        hidden_i = hidden.index_select(0, cross_camera_pairs[:, 0])
        hidden_j = hidden.index_select(0, cross_camera_pairs[:, 1])
        pair_features = torch.cat([hidden_i, hidden_j], dim=-1)
        logits = self.assoc_model.pairwise_mlp(pair_features).squeeze(-1)
        probs[cross_camera_pairs[:, 0], cross_camera_pairs[:, 1]] = torch.sigmoid(logits).to(probs.dtype)
        return hidden, probs

    def _compute_step(
        self,
        *,
        dataset_root: str,
        cams: dict[str, Any],
        scene_name: str,
        epoch: int,
        frame_t: dict[str, Any],
        frame_tp1: dict[str, Any],
        state_t: FrameState,
        state_tp1: FrameState,
        track_memory: deque[dict[str, Any]],
        use_ctr: bool,
        use_temp: bool,
    ) -> StepArtifacts:
        pseudo_gt_t, pseudo_gt_conf_t = get_or_make_pseudo_gt(
            detections=frame_t["detections"],
            cams=cams,
            dataset_root=dataset_root,
            scene_name=scene_name,
            frame_num=int(frame_t["frame"]),
        )
        pseudo_gt_tp1, pseudo_gt_conf_tp1 = get_or_make_pseudo_gt(
            detections=frame_tp1["detections"],
            cams=cams,
            dataset_root=dataset_root,
            scene_name=scene_name,
            frame_num=int(frame_tp1["frame"]),
        )

        loss_asso_t = weighted_association_loss(state_t.pairwise_probs, pseudo_gt_t, state_t.cam_ids, C=pseudo_gt_conf_t)
        loss_asso_tp1 = weighted_association_loss(state_tp1.pairwise_probs, pseudo_gt_tp1, state_tp1.cam_ids, C=pseudo_gt_conf_tp1)

        groups_t: list[dict[str, Any]] = []
        groups_tp1: list[dict[str, Any]] = []
        matches: list[tuple[Any, ...]] = []
        loss_ctr_t = torch.tensor(0.0, device=self.config.device)
        loss_ctr_tp1 = torch.tensor(0.0, device=self.config.device)
        loss_temp = torch.tensor(0.0, device=self.config.device)

        if use_ctr or use_temp:
            groups_t = self._group_frame(state_t, frame_t["detections"], cams)
            groups_tp1 = self._group_frame(state_tp1, frame_tp1["detections"], cams)

            self._attach_group_embeddings(groups_t, state_t.hidden)
            self._attach_group_embeddings(groups_tp1, state_tp1.hidden)

            positives_t = self._build_positives(groups_t)
            positives_tp1 = self._build_positives(groups_tp1)
            if use_ctr:
                loss_ctr_t = info_nce_detection_level(state_t.hidden, positives_t, self.config.tau)
                loss_ctr_tp1 = info_nce_detection_level(state_tp1.hidden, positives_tp1, self.config.tau)

            matches = temporal_linking(
                groups_t,
                groups_tp1,
                alpha=self.config.alpha,
                beta=self.config.beta,
                dist_thresh=self.config.dist_thresh,
                sim_thresh=self.config.sim_thresh,
            )
            fallback_matches: list[tuple[Any, ...]] = []
            if use_temp:
                _, fallback_matches = build_fallback_matches(
                    matches=matches,
                    groups_tp1=groups_tp1,
                    track_memory=track_memory,
                    frame_num_tp1=int(frame_tp1["frame"]),
                    dist_thresh=self.config.dist_thresh,
                    sim_thresh=self.config.sim_thresh,
                    gap_dist_scale=0.15,
                    gap_sim_drop=0.04,
                )
                loss_temp = temporal_prediction_loss(
                    predictor=self.predictor,
                    groups_t=groups_t,
                    groups_tp1=groups_tp1,
                    matches=matches,
                    fallback_matches=fallback_matches,
                    device=self.config.device,
                )

        total_loss = (
            self.config.lambda_asso * (loss_asso_t + loss_asso_tp1)
            + self.config.lambda_ctr * (loss_ctr_t + loss_ctr_tp1)
            + self.config.lambda_temp * loss_temp
        )

        if self.config.debug:
            append_train_step_debug(
                debug_file=self.debug_file,
                epoch=epoch,
                scene_name=scene_name,
                t_idx=int(frame_t["frame"]),
                L_asso_t=loss_asso_t,
                L_asso_tp1=loss_asso_tp1,
                L_ctr_t=loss_ctr_t,
                L_ctr_tp1=loss_ctr_tp1,
                L_temp=loss_temp,
                total_loss=total_loss,
                H_t=state_t.hidden,
                P_t=state_t.pairwise_probs,
                H_tp1=state_tp1.hidden,
                P_tp1=state_tp1.pairwise_probs,
                G_t=pseudo_gt_t,
                C_t=pseudo_gt_conf_t,
                G_tp1=pseudo_gt_tp1,
                C_tp1=pseudo_gt_conf_tp1,
                groups_t=groups_t,
                groups_tp1=groups_tp1,
                matches=matches,
                params=self.optimized_params,
            )

        return StepArtifacts(
            total_loss=total_loss,
            association_t=loss_asso_t,
            association_tp1=loss_asso_tp1,
            contrastive_t=loss_ctr_t,
            contrastive_tp1=loss_ctr_tp1,
            temporal=loss_temp,
            groups_t=groups_t,
            groups_tp1=groups_tp1,
            matches=matches,
            pseudo_gt_t=pseudo_gt_t,
            pseudo_gt_conf_t=pseudo_gt_conf_t,
            pseudo_gt_tp1=pseudo_gt_tp1,
            pseudo_gt_conf_tp1=pseudo_gt_conf_tp1,
        )

    def _group_frame(
        self,
        state: FrameState,
        detections: list[dict[str, Any]],
        cams: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            hidden_np = state.hidden.detach().cpu().numpy()
            pairwise_np = state.pairwise_probs.detach().cpu().numpy()

        groups = group_and_triangulate(
            pairwise_np,
            hidden_np,
            detections,
            cams,
            thr_assoc=self.config.thr_assoc,
            match_mode=self.config.grouping_match_mode,
            debug=self.config.debug,
            debug_file=self.debug_file,
        )
        return self._prune_overlapping_groups(groups)

    @staticmethod
    def _prune_overlapping_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not groups:
            return []
        groups_sorted = sorted(groups, key=lambda group: float(group.get("conf", 0.0)), reverse=True)
        used_detections: set[int] = set()
        pruned = []
        for group in groups_sorted:
            members = group.get("members", [])
            if any(member in used_detections for member in members):
                continue
            pruned.append(group)
            used_detections.update(members)
        return pruned

    def _attach_group_embeddings(self, groups: list[dict[str, Any]], hidden: torch.Tensor) -> None:
        for group in groups:
            member_indices = group["members"]
            embedding = hidden[member_indices].mean(dim=0)
            normalized = F.normalize(embedding, p=2, dim=0)
            group["emb_tensor"] = normalized
            group["emb_np"] = normalized.detach().cpu().numpy()

    def _build_positives(self, groups: list[dict[str, Any]]) -> dict[int, list[int]]:
        positives: dict[int, list[int]] = {}
        for group in groups:
            if float(group.get("conf", 0.0)) < self.config.conf_min:
                continue
            members = group["members"]
            for index in members:
                others = [other for other in members if other != index]
                if others:
                    positives.setdefault(index, []).extend(others)
        return positives

    def _save_checkpoints(
        self,
        *,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
        validation_ran: bool,
    ) -> None:
        train_avg = train_metrics.mean_total()
        val_avg = val_metrics.mean_total() if val_metrics is not None else None

        if self.config.selection_metric == "val":
            if not validation_ran or val_avg is None:
                return self._save_periodic_checkpoint(epoch, train_avg, val_avg)
            score = val_avg
            score_name = "val_avg_loss"
        else:
            score = train_avg
            score_name = "avg_loss"

        if score < self.best_score:
            self.best_score = score
            best_path = self.checkpoint_dir / "best_ckpt.pth"
            torch.save(self._checkpoint_payload(epoch, train_avg, val_avg), best_path)
            print(f"Saved best checkpoint {best_path} ({score_name}={score:.6f})")

        self._save_periodic_checkpoint(epoch, train_avg, val_avg)

    def _save_periodic_checkpoint(self, epoch: int, train_avg: float, val_avg: float | None) -> None:
        if (epoch % 100 == 0) or (epoch == self.config.total_epochs):
            epoch_path = self.checkpoint_dir / f"ckpt_epoch_{epoch}.pth"
            torch.save(self._checkpoint_payload(epoch, train_avg, val_avg), epoch_path)
            print(f"Saved checkpoint {epoch_path}")

    def _checkpoint_payload(self, epoch: int, train_avg: float, val_avg: float | None) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "assoc_state_dict": self.assoc_model.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "avg_loss": train_avg,
            "val_avg_loss": val_avg,
            "config": asdict(self.config),
        }