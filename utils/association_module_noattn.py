import torch
import torch.nn as nn

from .association_module import RotaryPositionEmbedding2D, _rope_import_err


class TokenwiseMLP(nn.Module):
    def __init__(self, d_model=128, num_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return inputs + self.net(inputs)


class AssociationNoGlobalAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        d_model=128,
        num_layers=4,
        num_heads=4,
        dim_ff=256,
        dropout=0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        del num_heads

        self.use_rope = use_rope and (RotaryPositionEmbedding2D is not None)
        if use_rope and RotaryPositionEmbedding2D is None:
            message = (
                "[AssociationNoGlobalAttention] RoPE requested but utils/rope_pe.py "
                "could not be imported; continuing without RoPE."
            )
            if _rope_import_err is not None:
                message += f" (import error: {_rope_import_err})"
            print(message)

        self.input_proj = nn.Linear(in_dim, d_model)
        self.rope = RotaryPositionEmbedding2D(d_model) if self.use_rope else None
        self.transformer = TokenwiseMLP(
            d_model=d_model,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, features, cam_ids, ranks=None, boxes=None):
        del cam_ids, ranks
        if len(features) == 0:
            return torch.zeros((0, 0), device=features.device)

        hidden = self.input_proj(features)
        if self.use_rope and boxes is not None:
            positions = torch.stack([boxes[:, 1], boxes[:, 0]], dim=-1)
            hidden = self.rope(hidden, positions)

        hidden = self.transformer(hidden.unsqueeze(0)).squeeze(0)
        num_detections = hidden.size(0)
        hidden_i = hidden.unsqueeze(1).expand(num_detections, num_detections, -1)
        hidden_j = hidden.unsqueeze(0).expand(num_detections, num_detections, -1)
        pair_features = torch.cat([hidden_i, hidden_j], dim=-1)
        return torch.sigmoid(self.pairwise_mlp(pair_features).squeeze(-1))