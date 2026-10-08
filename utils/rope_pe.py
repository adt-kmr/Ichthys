import torch
import torch.nn as nn

class RotaryPositionEmbedding2D(nn.Module):
    """
    Minimal 2D RoPE implementation for detection tokens.

    We split the embedding into two halves:
      - first half encodes vertical (y) position
      - second half encodes horizontal (x) position

    positions: tensor [N, 2] with (cy_norm, cx_norm) in [0,1].

    Reference adapted from standard RoPE formulations (Su et al. 2021) but applied
    independently to y and x axes.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = base
        self.half = dim // 2
        # For each axis we will create frequencies for half of that axis's features.
        # Each axis sub-dim = self.half. Inside that we form pairs -> need even.
        assert self.half % 2 == 0, "Per-axis RoPE dim must be even"
        # inv_freq shape: [self.half/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half, 2).float() / self.half))
        self.register_buffer("inv_freq", inv_freq)  # persistent on device / moves with module

    def _apply_1d_rope(self, feats: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply 1D rotary embedding to one axis half.
        feats: [N, D_axis] where D_axis = self.half (must be even)
        pos:   [N] normalized position in [0,1]
        Returns: rotated feats [N, D_axis]
        """
        N, D = feats.shape
        # angles for every pair (D/2 pairs)
        theta = pos.unsqueeze(1) * self.inv_freq.unsqueeze(0) * torch.pi
        cos = torch.cos(theta)  # [N, D/2]
        sin = torch.sin(theta)  # [N, D/2]
        # split feats into pairs
        feats_pairs = feats.view(N, D // 2, 2)
        x1 = feats_pairs[..., 0]
        x2 = feats_pairs[..., 1]
        # standard rotary transform
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        rotated = torch.stack([rot_x1, rot_x2], dim=-1).view(N, D)
        return rotated

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        tokens: [N, dim]
        positions: [N, 2] -> (cy_norm, cx_norm)
        Returns tokens with RoPE applied per axis.
        """
        if tokens.numel() == 0:
            return tokens
        assert positions.shape[0] == tokens.shape[0], "Position count must match token count"
        # Split embedding
        vert, horiz = torch.split(tokens, [self.half, self.half], dim=-1)
        cy = positions[:, 0].clamp(0.0, 1.0)
        cx = positions[:, 1].clamp(0.0, 1.0)
        vert_rot = self._apply_1d_rope(vert, cy)
        horiz_rot = self._apply_1d_rope(horiz, cx)
        return torch.cat([vert_rot, horiz_rot], dim=-1)
