#!/usr/bin/env python3
"""Dependency-light architecture and loss smoke test."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amyclip.losses import dual_endpoint_loss, intrinsic_loss, symmetric_retrieval_loss
from amyclip.model import AMYCLIP


def main() -> None:
    torch.manual_seed(7)
    model = AMYCLIP()
    peptide = torch.randn(4, 1280)
    protein = torch.randn(4, 1280)
    positive = torch.eye(4, dtype=torch.bool)
    allowed = torch.ones(4, 4, dtype=torch.bool)
    retrieval = symmetric_retrieval_loss(
        model.compatibility_logits(peptide, protein), positive, allowed
    )
    intrinsic = intrinsic_loss(
        model.intrinsic_logits(peptide), torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    total = dual_endpoint_loss(retrieval, intrinsic)
    assert torch.isfinite(total)
    assert model.pep_embedder[-1].out_features == 512
    assert math.isclose(float(model.temperature), 12.5, rel_tol=0.0, abs_tol=1e-5)
    print("PASS: finalized architecture and dual-endpoint loss execute correctly")


if __name__ == "__main__":
    main()
