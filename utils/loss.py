#!/usr/bin/env python3
"""
loss.py

Loss functions for Ichthys training:
- Weighted association loss (BCE with pseudo-GT confidences)
- InfoNCE contrastive loss (detection-level)
- Temporal prediction loss (handled in main training loop)
"""
import torch
import torch.nn.functional as F


def weighted_association_loss(P, G, cam_ids, C=None, eps=1e-8):
    """
    Weighted binary cross-entropy loss for pairwise association.
    
    Args:
        P: [N,N] torch tensor of predicted probabilities
        G: [N,N] numpy/tensor {0,1,-1} ground truth (-1 = ignore)
        cam_ids: [N] torch tensor of camera IDs
        C: [N,N] numpy/tensor of confidence weights in [0,1] or None
        eps: numerical stability epsilon
        
    Returns:
        loss: scalar tensor
    """
    device = P.device
    N = P.shape[0]
    # mask invalid: diagonal, same-camera pairs, G == -1
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_cam_mask = (cam_ids.unsqueeze(0) == cam_ids.unsqueeze(1))
    G_tensor = torch.tensor(G, dtype=torch.float32, device=device)
    valid_mask = (G_tensor != -1.0)
    mask = valid_mask & (~diag_mask) & (~same_cam_mask)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    P_valid = P[mask]
    G_valid = G_tensor[mask]
    if C is None:
        loss = F.binary_cross_entropy(P_valid, G_valid)
    else:
        C_tensor = torch.tensor(C, dtype=torch.float32, device=device)
        C_valid = C_tensor[mask]
        # treat positive pairs weighted by confidence; negatives get (1) weight
        pos_mask = (G_valid == 1.0)
        neg_mask = (G_valid == 0.0)
        pos_loss = 0.0
        neg_loss = 0.0
        if pos_mask.sum() > 0:
            ppos = P_valid[pos_mask]
            tpos = G_valid[pos_mask]
            wpos = C_valid[pos_mask]
            pos_loss = -(wpos * (tpos * torch.log(ppos + eps) + (1 - tpos) * torch.log(1 - ppos + eps))).sum() / (wpos.sum() + eps)
        if neg_mask.sum() > 0:
            pneg = P_valid[neg_mask]
            tneg = G_valid[neg_mask]
            # lower weight for negatives (optional)
            wneg = torch.ones_like(pneg)
            neg_loss = -(wneg * (tneg * torch.log(pneg + eps) + (1 - tneg) * torch.log(1 - pneg + eps))).sum() / (wneg.sum() + eps)
        loss = pos_loss + neg_loss
    return loss


def info_nce_detection_level(H, positives_dict, tau=0.07):
    """
    InfoNCE contrastive loss at detection level.
    
    Args:
        H: [N, d] torch tensor of detection embeddings
        positives_dict: dict mapping anchor_idx -> list of positive indices (within same frame)
        tau: temperature parameter
        
    Returns:
        loss: scalar tensor
    """
    if len(positives_dict) == 0:
        return torch.tensor(0.0, device=H.device)
    z = F.normalize(H, p=2, dim=1)  # N x d
    sim = (z @ z.t()) / tau          # N x N
    # numerical stability
    sim_exp = torch.exp(sim - sim.max(dim=1, keepdim=True)[0])
    losses = []
    N = z.shape[0]
    for i, pos_list in positives_dict.items():
        if len(pos_list) == 0:
            continue
        numer = sim_exp[i, pos_list].sum()
        denom = sim_exp[i].sum() - sim_exp[i, i]
        losses.append(-torch.log((numer + 1e-8) / (denom + 1e-8)))
    if len(losses) == 0:
        return torch.tensor(0.0, device=H.device)
    return torch.stack(losses).mean()


def temporal_prediction_loss(
    *,
    predictor,
    groups_t,
    groups_tp1,
    matches,
    fallback_matches=None,
    device=None,
    reduction_eps=1e-8,
):
    """Compute temporal prediction loss using PredictorM.

    This mirrors the inlined logic previously in `train.py`:
      - immediate matches: predictor(emb_t) ~ emb_{t+1}
      - fallback matches: predictor(emb_old) ~ emb_{t+1}
      - each pair weighted by `w`
      - normalize by sum of weights

    Args:
        predictor: torch nn.Module mapping embedding -> embedding
        groups_t: list of groups (frame t), must have 'emb_tensor'
        groups_tp1: list of groups (frame t+1), must have 'emb_tensor'
        matches: list of (i_g, j_g, w)
        fallback_matches: optional list of (emb_old_tensor, emb_new_tensor, w)
        device: torch device; if None uses predictor device
        reduction_eps: epsilon for denom stability

    Returns:
        L_temp: scalar torch tensor
    """
    if device is None:
        try:
            device = next(predictor.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    fallback_matches = fallback_matches or []

    L_temp = torch.tensor(0.0, device=device)
    denom = 0.0

    # immediate matches predictor loss
    for (i_g, j_g, w) in matches:
        emb_i = groups_t[i_g]["emb_tensor"].to(device)
        emb_j = groups_tp1[j_g]["emb_tensor"].to(device)
        pred_j = predictor(emb_i)
        pair_loss = F.mse_loss(pred_j, emb_j) * float(w)
        L_temp += pair_loss
        denom += float(w)

    # fallback matches predictor loss
    for (emb_old, emb_new, w_fb) in fallback_matches:
        pred_new = predictor(emb_old.to(device))
        pair_loss_fb = F.mse_loss(pred_new, emb_new.to(device)) * float(w_fb)
        L_temp += pair_loss_fb
        denom += float(w_fb)

    if denom > 0:
        L_temp = L_temp / (denom + reduction_eps)
    return L_temp
