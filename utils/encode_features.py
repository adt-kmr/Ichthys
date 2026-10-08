from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torchvision import io as tvio
import sys, os
from importlib import import_module
from pathlib import Path


def _resolve_dinov3_backbone(repo_dir=None, model_variant="dinov3_vits16"):
    search_roots = []
    if repo_dir:
        repo_path = Path(repo_dir)
        search_roots.append(repo_path)
        if not repo_path.is_absolute():
            search_roots.append((Path(__file__).resolve().parents[1] / repo_path).resolve())

    for search_root in search_roots:
        if search_root.exists() and str(search_root) not in sys.path:
            sys.path.insert(0, str(search_root))

    try:
        backbones_module = import_module("dinov3.hub.backbones")
    except ImportError:
        return None
    return getattr(backbones_module, model_variant, None)

def pixel_to_world_ray(cx, cy, K_inv, R):
    pix_h = np.array([cx, cy, 1.0])
    cam_ray = K_inv @ pix_h
    cam_ray = cam_ray / np.linalg.norm(cam_ray)
    ray_world = R.T @ cam_ray
    ray_world = ray_world / np.linalg.norm(ray_world)
    return ray_world

class GeometricFeatureEncoder(nn.Module):
    _image_cache = {}

    def __init__(self,
                 d_model=128,
                 hidden_dim=64,
                 num_cams=3,
                 image_size=(1920, 1080),
                 use_dino=False,
                 dinov3_repo_dir=None,
                 dinov3_model_variant="dinov3_vits16",
                 weights="path/to/dinov3_vits16.pth",
                 device="cpu",
                 preprocess_workers: int = 8):
        super().__init__()
        self.img_w, self.img_h = image_size
        self.num_cams = num_cams
        self.d_model = d_model
        self.use_dino = use_dino
        self.weights = weights
        self.device = device
        self.preprocess_workers = max(1, int(preprocess_workers))
        # When running on CUDA, we can optionally crop+resize on GPU to reduce CPU load.
        # This does NOT change JPEG decode cost (still CPU), but removes per-detection CPU slicing
        # and CPU resize/interpolate.
        self.use_gpu_crop = True

        self.mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

        if use_dino:
            backbone_ctor = _resolve_dinov3_backbone(dinov3_repo_dir, dinov3_model_variant)
            if backbone_ctor is None:
                raise ImportError(
                    "DINOv3 was requested but could not be imported. "
                    "Set dinov3_repo_dir to a valid checkout or disable DINO with --no_dino."
                )
            self.dino_model = backbone_ctor(pretrained=False)
            self.dino_model.to(device)
            if self.weights and os.path.exists(self.weights):
                state = torch.load(self.weights, map_location="cpu")
                if "model" in state:
                    state = state["model"]
                self.dino_model.load_state_dict(state, strict=False)
                print(f"[DINOv3] Loaded weights from {self.weights}")
            self.dino_model.eval()
            self.dino_feat_dim = self.dino_model.embed_dim
            self.total_d_model = d_model + self.dino_feat_dim
        else:
            self.total_d_model = d_model

    def freeze_dino(self, freeze: bool = True):
        """Freeze/unfreeze DINO backbone parameters.

        When frozen, DINO stays in eval mode and its params are excluded from grads.
        """
        if not self.use_dino:
            return
        for p in self.dino_model.parameters():
            p.requires_grad_(not freeze)
        if freeze:
            self.dino_model.eval()

    def _build_raw_features(self, detections, cams, cam_images=None):
        """Build raw geometric feature vectors (CPU math + normalization).

        Returns:
            raw_feats: torch.Tensor [N, 9] on self.device
            crop_specs: list[(cam_id, x1, y1, x2, y2)] for DINO crops
        """
        feats = []
        crop_specs = []

        # cache inverses per cam
        cam_cache = {}
        for det in detections:
            cid = det["cam_id"]
            if cid not in cam_cache:
                K = np.array(cams[f"cam{cid}-K"]) ; RT = np.array(cams[f"cam{cid}-R|T"])
                cam_cache[cid] = (np.linalg.inv(K), RT[:, :3])

        # If cam_images is not provided, we fall back to scene-level size.
        # This is useful when DINO feats are cached and we want to skip image decode.
        fallback_w = int(getattr(self, "img_w", 0) or 0)
        fallback_h = int(getattr(self, "img_h", 0) or 0)

        for det in detections:
            cam_id = det["cam_id"]
            K_inv, R = cam_cache[cam_id]

            if cam_images is not None and cam_id in cam_images:
                img_t = cam_images[cam_id]
                img_h = int(img_t.shape[-2])
                img_w = int(img_t.shape[-1])
            else:
                img_w = fallback_w
                img_h = fallback_h

            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["segmentation-centroid"]

            w = (x2 - x1) / max(1.0, float(img_w))
            h = (y2 - y1) / max(1.0, float(img_h))
            cx_norm = cx / max(1.0, float(img_w))
            cy_norm = cy / max(1.0, float(img_h))

            cam_norm = cam_id / max(1, self.num_cams - 1)
            score = det.get("det_score", 1.0)
            ray_world = pixel_to_world_ray(cx, cy, K_inv, R)
            feats.append([cx_norm, cy_norm, w, h, cam_norm, score, ray_world[0], ray_world[1], ray_world[2]])

            if self.use_dino and cam_images is not None:
                crop_specs.append(
                    (cam_id,
                     int(max(0, x1)), int(max(0, y1)),
                     int(min(img_w, x2)), int(min(img_h, y2)))
                )

        if not feats:
            raw_feats = torch.empty(0, 9, device=self.device)
            return raw_feats, crop_specs

        raw_feats = torch.tensor(feats, dtype=torch.float32, device=self.device)
        return raw_feats, crop_specs

    def set_image_size(self, image_size):
        """Update (img_w, img_h) used for bbox normalization and crop clamping.

        This is safe to call between scenes when scenes have different resolutions.
        """
        w, h = int(image_size[0]), int(image_size[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image_size={image_size}")
        self.img_w, self.img_h = w, h

    def set_image_size_from_path(self, image_path: str):
        """Infer size from an on-disk image and update encoder settings.

        Uses torchvision image loader so it matches what we use in forward.
        """
        img = self.load_image(image_path)
        # img: [C, H, W]
        h = int(img.shape[-2])
        w = int(img.shape[-1])
        self.set_image_size((w, h))

    @classmethod
    def load_image(cls, path):
        if path not in cls._image_cache:
            cls._image_cache[path] = tvio.read_image(path)  # [C,H,W] uint8
        return cls._image_cache[path]

    def forward(
        self,
        detections,
        cams,
        images_frame_path,
        return_raw: bool = False,
        return_parts: bool = False,
        cached_dino_feats: torch.Tensor | None = None,
    ):
        """Encode detections.

        Args:
            cached_dino_feats: if provided, skip DINO forward and use these features.
                Shape must be [N, dino_feat_dim] aligned with `detections` order.
            return_parts: if True, additionally return geom_feats and dino_feats.
        """
        # If DINO feats are already provided (cached), we can skip decoding images entirely.
        # Geometry features only need normalization by (W,H), which we treat as scene-level constants.
        if cached_dino_feats is not None:
            cam_images = None
        else:
            cam_images = {cam_id: self.load_image(path) for cam_id, path in enumerate(images_frame_path)}

        raw_feats, crop_specs = self._build_raw_features(detections, cams, cam_images)
        if raw_feats.numel() == 0:
            F_out = torch.empty(0, self.total_d_model, device=self.device)
            if return_parts:
                geom_empty = torch.empty(0, self.d_model, device=self.device)
                dino_empty = torch.empty(0, getattr(self, "dino_feat_dim", 0), device=self.device)
                return (F_out, raw_feats, geom_empty, dino_empty) if return_raw else (F_out, geom_empty, dino_empty)
            return (F_out, raw_feats) if return_raw else F_out

        geom_feats = self.mlp(raw_feats)

        dino_feats = None
        if self.use_dino:
            if cached_dino_feats is not None:
                dino_feats = cached_dino_feats.to(self.device)
            elif len(crop_specs) > 0:
                # Prefer GPU crop+resize when available.
                can_gpu_crop = (
                    bool(getattr(self, "use_gpu_crop", True))
                    and isinstance(self.device, str)
                    and self.device.startswith("cuda")
                    and torch.cuda.is_available()
                )

                batch = None

                if can_gpu_crop:
                    try:
                        # Move each camera's full image to GPU once (float in [0,1]).
                        cam_images_gpu = {}
                        for cid, img_cpu in cam_images.items():
                            # img_cpu: uint8 [C,H,W] on CPU
                            cam_images_gpu[cid] = (img_cpu.to(device=self.device, dtype=torch.float32) / 255.0)

                        crops_gpu = []
                        for (cid, x1i, y1i, x2i, y2i) in crop_specs:
                            img_g = cam_images_gpu[cid]
                            # crop: [C, h, w] on GPU
                            crop = img_g[:, y1i:y2i, x1i:x2i]
                            # Guard against empty crops; keep shape consistent by falling back.
                            if crop.numel() == 0 or crop.shape[-1] < 2 or crop.shape[-2] < 2:
                                raise ValueError("empty/degenerate crop")
                            crop = torch.nn.functional.interpolate(
                                crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                            ).squeeze(0)
                            crops_gpu.append(crop)
                        batch = torch.stack(crops_gpu, dim=0)
                    except Exception:
                        batch = None

                # CPU fallback: keep previous behavior (optionally threaded)
                if batch is None:
                    def make_crop(spec):
                        cid, x1i, y1i, x2i, y2i = spec
                        img_t = cam_images[cid]
                        crop = img_t[:, y1i:y2i, x1i:x2i].to(torch.float32) / 255.0
                        crop = torch.nn.functional.interpolate(
                            crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                        ).squeeze(0)
                        return crop
                    if self.preprocess_workers > 1 and len(crop_specs) > 1:
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=self.preprocess_workers) as ex:
                            crops_list = list(ex.map(make_crop, crop_specs))
                    else:
                        crops_list = [make_crop(s) for s in crop_specs]
                    batch = torch.stack(crops_list, dim=0).to(self.device)

                # If DINO is frozen, we run under no_grad; otherwise, allow grads.
                if all(not p.requires_grad for p in self.dino_model.parameters()):
                    with torch.no_grad():
                        dino_feats = self.dino_model(batch)
                else:
                    dino_feats = self.dino_model(batch)
            else:
                dino_feats = torch.empty(geom_feats.shape[0], self.dino_feat_dim, device=self.device)

        if self.use_dino:
            F_out = torch.cat([geom_feats, dino_feats.to(self.device)], dim=1)
        else:
            F_out = geom_feats

        if return_parts:
            if return_raw:
                return F_out, raw_feats, geom_feats, dino_feats
            return F_out, geom_feats, dino_feats
        return (F_out, raw_feats) if return_raw else F_out


