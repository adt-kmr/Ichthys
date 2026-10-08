import torch
import torch.nn as nn
import torch.nn.functional as F

RotaryPositionEmbedding2D = None  # optional; only used if present
_rope_import_err = None
try:
    # Preferred in this repo layout
    from utils.rope_pe import RotaryPositionEmbedding2D  # type: ignore
except Exception as e1:
    _rope_import_err = e1
    try:
        # Fallback when `utils` is treated as a package and we're imported as `utils.*`
        from .rope_pe import RotaryPositionEmbedding2D  # type: ignore
        _rope_import_err = None
    except Exception as e2:
        _rope_import_err = e2
        RotaryPositionEmbedding2D = None


class AssociationTransformer(nn.Module):
    def __init__(self, in_dim, d_model=128, num_layers=4, num_heads=4, dim_ff=256, dropout=0.1,
                 use_rope: bool = True):
        super().__init__()

        self.use_rope = use_rope and (RotaryPositionEmbedding2D is not None)
        if use_rope and RotaryPositionEmbedding2D is None:
            msg = "[AssociationTransformer] RoPE requested but utils/rope_pe.py could not be imported; continuing without RoPE."
            if _rope_import_err is not None:
                msg += f" (import error: {_rope_import_err})"
            print(msg)

        # Project input features to fixed dimension
        self.input_proj = nn.Linear(in_dim, d_model)

        # 2D RoPE
        if self.use_rope:
            self.rope = RotaryPositionEmbedding2D(d_model)
        else:
            self.rope = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pairwise_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, F, cam_ids, ranks=None, boxes=None):  # ranks retained for compatibility (ignored)
        """
        Args:
            F: [N, in_dim] embeddings from feature encoder
            cam_ids: [N] tensor with camera IDs
            ranks: [N] tensor of detection ranks within each camera
            boxes: [N, 4] tensor (cx, cy, w, h), normalized
        Returns:
            P: [N, N] predicted association probabilities
        """
        if len(F) == 0:
            return torch.zeros((0, 0), device=F.device)

        # Project input features
        F = self.input_proj(F)

        # Apply RoPE (only cy,cx from boxes) if available
        if self.use_rope and boxes is not None:
            positions = torch.stack([boxes[:, 1], boxes[:, 0]], dim=-1)  # (cy, cx)
            F = self.rope(F, positions)

        # Transformer reasoning on rotated features
        H = self.transformer(F.unsqueeze(0)).squeeze(0)

        # Step 3: Pairwise prediction
        N = H.size(0)
        H_i = H.unsqueeze(1).expand(N, N, -1)
        H_j = H.unsqueeze(0).expand(N, N, -1)
        pair_feats = torch.cat([H_i, H_j], dim=-1)

        P = self.pairwise_mlp(pair_feats).squeeze(-1)
        P = torch.sigmoid(P)
        return P
