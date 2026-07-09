import warnings

import torch
from torch.nn import functional as F


def build_focus_routing_masks(
    focus_a,
    focus_b,
    keep_threshold=0.55,
    bref_threshold=0.35,
    gamma=1.0,
    use_soft=False,
    target_size=None,
    valid_mask=None,
):
    """Build focus routing masks.

    focus_a white means A should be kept. focus_b white means B is sharp/reference.
    Returns M_keep, M_bref, M_gen with shape [B, 1, H, W].
    """
    if focus_a is None or focus_b is None:
        warnings.warn("Missing focus_a/focus_b; falling back to uniform valid mask routing.")
        if valid_mask is None:
            if target_size is None:
                raise ValueError("target_size or valid_mask is required when focus maps are missing.")
            device = torch.device("cpu")
            m_keep = torch.ones(1, 1, target_size[0], target_size[1], device=device)
        else:
            m_keep = valid_mask.float()
        m_bref = torch.zeros_like(m_keep)
        m_gen = torch.zeros_like(m_keep)
        return m_keep, m_bref, m_gen

    focus_a = focus_a.float().clamp(0, 1)
    focus_b = focus_b.float().clamp(0, 1)
    if target_size is not None and focus_a.shape[-2:] != tuple(target_size):
        focus_a = F.interpolate(focus_a, size=target_size, mode="bilinear", align_corners=False).clamp(0, 1)
        focus_b = F.interpolate(focus_b, size=target_size, mode="bilinear", align_corners=False).clamp(0, 1)

    if use_soft:
        fa = focus_a.clamp(0, 1) ** gamma
        fb = focus_b.clamp(0, 1) ** gamma
        m_keep = fa
        m_bref = (1 - fa) * fb
        m_gen = (1 - fa) * (1 - fb)
    else:
        m_keep = (focus_a >= keep_threshold).float()
        m_bsharp = (focus_b >= bref_threshold).float()
        m_bref = (1 - m_keep) * m_bsharp
        m_gen = (1 - m_keep) * (1 - m_bsharp)

    if valid_mask is not None:
        valid_mask = valid_mask.float()
        if valid_mask.shape[-2:] != m_keep.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=m_keep.shape[-2:], mode="nearest")
        m_keep = m_keep * valid_mask
        m_bref = m_bref * valid_mask
        m_gen = m_gen * valid_mask
    return m_keep.clamp(0, 1), m_bref.clamp(0, 1), m_gen.clamp(0, 1)


def weighted_mean(error, weight):
    """Mean over [B,C,H,W] error with [B,1,H,W] weight, normalized by channels."""
    if weight.shape[-2:] != error.shape[-2:]:
        weight = F.interpolate(weight.float(), size=error.shape[-2:], mode="nearest")
    weight = weight.to(device=error.device, dtype=error.dtype)
    return (error * weight).sum() / (weight.sum() * error.shape[1]).clamp_min(1.0)


def mask_means(m_keep, m_bref, m_gen):
    return {
        "keep_mean": float(m_keep.detach().float().mean().cpu()),
        "bref_mean": float(m_bref.detach().float().mean().cpu()),
        "gen_mean": float(m_gen.detach().float().mean().cpu()),
    }
