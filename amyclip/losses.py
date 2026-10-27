"""Losses corresponding to the equations reported in Methods Section 2.3."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def multi_positive_loss(
    logits: Tensor,
    positives: Tensor,
    allowed: Tensor,
    smoothing: float = 0.10,
) -> Tensor:
    """Set-positive NLL with smoothing over scientifically allowed candidates."""
    if torch.any(positives & ~allowed):
        raise ValueError("every positive must be included in the allowed mask")
    if not positives.any(dim=1).all() or not allowed.any(dim=1).all():
        raise ValueError("every row requires a positive and an allowed candidate")
    negative_infinity = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~allowed, negative_infinity)
    log_probability = torch.log_softmax(masked, dim=1)
    positive_mass = torch.logsumexp(
        log_probability.masked_fill(~positives, negative_infinity), dim=1
    )
    set_positive_nll = -positive_mass
    safe = torch.where(allowed, log_probability, torch.zeros_like(log_probability))
    smooth_nll = -safe.sum(dim=1) / allowed.sum(dim=1).clamp(min=1)
    return ((1.0 - smoothing) * set_positive_nll + smoothing * smooth_nll).mean()


def symmetric_retrieval_loss(
    logits: Tensor,
    peptide_to_protein_positive: Tensor,
    peptide_to_protein_allowed: Tensor,
    smoothing: float = 0.10,
) -> Tensor:
    forward = multi_positive_loss(
        logits, peptide_to_protein_positive, peptide_to_protein_allowed, smoothing
    )
    reverse = multi_positive_loss(
        logits.T,
        peptide_to_protein_positive.T,
        peptide_to_protein_allowed.T,
        smoothing,
    )
    return 0.5 * (forward + reverse)


def intrinsic_loss(
    logits: Tensor,
    labels: Tensor,
    positive_weight: float = 602.0 / 474.0,
    smoothing: float = 0.05,
) -> Tensor:
    labels = labels.to(logits.dtype)
    targets = labels * (1.0 - 2.0 * smoothing) + smoothing
    weight = torch.tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


def dual_endpoint_loss(retrieval: Tensor, intrinsic: Tensor, weight: float = 0.05) -> Tensor:
    return retrieval + weight * intrinsic
