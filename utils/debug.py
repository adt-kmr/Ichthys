#!/usr/bin/env python3
"""debug.py

Logging helpers for Ichthys training.

Goal: keep `train.py` clean by moving verbose debug formatting + safe file writing here.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def append_train_step_debug(
    *,
    debug_file: str,
    epoch: int,
    scene_name: str,
    t_idx: int,
    # losses
    L_asso_t=None,
    L_asso_tp1=None,
    L_ctr_t=None,
    L_ctr_tp1=None,
    L_temp=None,
    total_loss=None,
    # tensors
    H_t=None,
    P_t=None,
    H_tp1=None,
    P_tp1=None,
    # pseudo-gt
    G_t=None,
    C_t=None,
    G_tp1=None,
    C_tp1=None,
    # groups
    groups_t=None,
    groups_tp1=None,
    matches=None,
    # grads
    params=None,
):
    """Append a training-step debug block to `debug_file`.

    This is intentionally defensive: any formatting failure should not crash training.
    """

    log_lines = []
    log_lines.append(f"Epoch {epoch}, Scene {scene_name}, Frame {t_idx}->{t_idx+1}")

    # 1. Losses
    try:
        log_lines.append(
            "Losses: "
            f"L_asso_t={float(L_asso_t.item()):.4f}, "
            f"L_asso_tp1={float(L_asso_tp1.item()):.4f}, "
            f"L_ctr_t={float(L_ctr_t.item()):.4f}, "
            f"L_ctr_tp1={float(L_ctr_tp1.item()):.4f}, "
            f"L_temp={float(L_temp.item()):.4f}, "
            f"total={float(total_loss.item()):.4f}"
        )
    except Exception:
        log_lines.append("Losses: (could not stringify losses)")

    # 2. Transformer outputs
    try:
        log_lines.append(f"H_t norm: mean={H_t.norm(dim=1).mean():.4f}, max={H_t.norm(dim=1).max():.4f}")
        log_lines.append(f"P_t: mean={P_t.mean():.4f}, min={P_t.min():.4f}, max={P_t.max():.4f}")
        log_lines.append(f"H_tp1 norm: mean={H_tp1.norm(dim=1).mean():.4f}, max={H_tp1.norm(dim=1).max():.4f}")
        log_lines.append(f"P_tp1: mean={P_tp1.mean():.4f}, min={P_tp1.min():.4f}, max={P_tp1.max():.4f}")
    except Exception:
        log_lines.append("Transformer outputs: (could not stringify tensors)")

    # 3. Pseudo-GT stats
    try:
        import numpy as np

        log_lines.append(f"G_t unique labels: {np.unique(G_t)}, C_t mean={np.mean(C_t):.4f}")
        log_lines.append(f"G_tp1 unique labels: {np.unique(G_tp1)}, C_tp1 mean={np.mean(C_tp1):.4f}")
    except Exception:
        log_lines.append("Pseudo-GT: (could not stringify)")

    # 4. Grouping
    try:
        groups_t = groups_t or []
        groups_tp1 = groups_tp1 or []
        log_lines.append(f"Groups_t: {len(groups_t)}, Groups_tp1: {len(groups_tp1)}")
        for i, g in enumerate(groups_t):
            log_lines.append(
                f"Group_t {i}: members={len(g['members'])}, conf={g['conf']:.3f}, X={g['X']}"
            )
        for i, g in enumerate(groups_tp1):
            log_lines.append(
                f"Group_tp1 {i}: members={len(g['members'])}, conf={g['conf']:.3f}, X={g['X']}"
            )
    except Exception:
        log_lines.append("Grouping: (could not stringify)")

    # 5. Temporal linking
    try:
        log_lines.append(f"Matches: {matches}")
    except Exception:
        log_lines.append("Matches: (could not stringify)")

    # 6. Gradients
    try:
        total_grad_norm = 0.0
        if params is not None:
            for p in params:
                if getattr(p, "grad", None) is not None:
                    try:
                        total_grad_norm += p.grad.norm().item() ** 2
                    except Exception:
                        pass
        total_grad_norm = total_grad_norm ** 0.5
        log_lines.append(f"Gradient norm: {total_grad_norm:.4f}")
    except Exception:
        log_lines.append("Gradient norm: (could not compute)")

    log_lines.append("--- End of Debug ---\n")

    try:
        with open(debug_file, "a") as df:
            df.write("\n".join(log_lines) + "\n")
    except Exception as e:
        # last resort: don't crash training
        try:
            print(f"Failed to write debug log to {debug_file}: {e}")
        except Exception:
            pass
