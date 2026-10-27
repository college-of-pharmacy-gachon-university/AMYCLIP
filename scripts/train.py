#!/usr/bin/env python3
"""Compact training example implementing the finalized dual-endpoint objective.

This script is intended to verify architecture and loss execution on sample
embeddings. The full development data are deliberately not part of the
first-submission repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amyclip.losses import dual_endpoint_loss, intrinsic_loss, symmetric_retrieval_loss
from amyclip.model import AMYCLIP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default=ROOT / "sample_data" / "training_sample_embeddings.npz")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", default=ROOT / "example_results" / "sample_trained_state.pt")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    peptide = torch.from_numpy(data["peptide_embeddings"]).float()
    protein = torch.from_numpy(data["protein_embeddings"]).float()
    labels = torch.from_numpy(data["intrinsic_labels"]).float()
    positives = torch.from_numpy(data["positives"]).bool()
    allowed = torch.from_numpy(data["allowed"]).bool()

    model = AMYCLIP()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )
    model.train()
    for epoch in range(args.epochs):
        logits = model.compatibility_logits(peptide, protein)
        retrieval = symmetric_retrieval_loss(logits, positives, allowed, smoothing=0.10)
        intrinsic = intrinsic_loss(model.intrinsic_logits(peptide), labels, smoothing=0.05)
        loss = dual_endpoint_loss(retrieval, intrinsic, weight=0.05)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(
            f"epoch={epoch + 1} total={loss.item():.6f} "
            f"retrieval={retrieval.item():.6f} intrinsic={intrinsic.item():.6f}"
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved smoke-test state to {args.output}")


if __name__ == "__main__":
    main()
