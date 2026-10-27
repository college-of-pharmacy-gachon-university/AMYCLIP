#!/usr/bin/env python3
"""Create the compact training example from the documented sample sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amyclip.esm import load_esm2, mean_pool_sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ROOT / "sample_data" / "training_sample.csv")
    parser.add_argument("--output", default=ROOT / "sample_data" / "training_sample_embeddings.npz")
    parser.add_argument("--esm-model", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    device = torch.device(args.device)
    model, alphabet = load_esm2(args.esm_model, device)
    peptide = mean_pool_sequences(
        frame["peptide_sequence"].astype(str).tolist(), model, alphabet, device
    ).numpy()
    protein = mean_pool_sequences(
        frame["target_sequence"].astype(str).tolist(), model, alphabet, device
    ).numpy()
    count = len(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        peptide_embeddings=peptide,
        protein_embeddings=protein,
        intrinsic_labels=frame["intrinsic_label"].to_numpy(dtype=np.float32),
        positives=np.eye(count, dtype=bool),
        allowed=np.ones((count, count), dtype=bool),
    )
    print(f"Wrote sample embeddings to {args.output}")


if __name__ == "__main__":
    main()
