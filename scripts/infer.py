#!/usr/bin/env python3
"""Score peptide-target rows with the finalized three-checkpoint ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amyclip.esm import load_esm2, mean_pool_sequences
from amyclip.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", default=ROOT / "checkpoints")
    parser.add_argument("--esm-model", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    frame = pd.read_csv(args.input)
    required = {"peptide_sequence", "target_sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"missing input columns: {sorted(missing)}")

    esm_model, alphabet = load_esm2(args.esm_model, device)
    peptide_embeddings = mean_pool_sequences(
        frame["peptide_sequence"].astype(str).tolist(), esm_model, alphabet, device
    ).to(device)
    target_embeddings = mean_pool_sequences(
        frame["target_sequence"].astype(str).tolist(), esm_model, alphabet, device
    ).to(device)

    manifest_path = Path(args.checkpoint_dir) / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint_records = manifest["checkpoints"]
    checkpoint_paths = [Path(args.checkpoint_dir) / row["file"] for row in checkpoint_records]
    if len(checkpoint_paths) != 3:
        raise SystemExit(f"expected 3 checkpoints, found {len(checkpoint_paths)}")
    compatibility, intrinsic = [], []
    for path in checkpoint_paths:
        model = load_checkpoint(path, device)
        with torch.inference_mode():
            pairwise = model.compatibility_probability(peptide_embeddings, target_embeddings)
            compatibility.append(pairwise.diagonal().cpu())
            intrinsic.append(model.intrinsic_probability(peptide_embeddings).cpu())

    compatibility = torch.stack(compatibility)
    intrinsic = torch.stack(intrinsic)
    for index, record in enumerate(checkpoint_records):
        seed = record["seed"]
        frame[f"compatibility_seed_{seed}"] = compatibility[index].numpy()
        frame[f"intrinsic_seed_{seed}"] = intrinsic[index].numpy()
    frame["compatibility_mean"] = compatibility.mean(0).numpy()
    frame["compatibility_sd"] = compatibility.std(0, unbiased=True).numpy()
    frame["compatibility_conservative"] = (
        compatibility.mean(0) - compatibility.std(0, unbiased=True)
    ).numpy()
    frame["intrinsic_mean"] = intrinsic.mean(0).numpy()
    frame["intrinsic_sd"] = intrinsic.std(0, unbiased=True).numpy()
    frame["intrinsic_conservative"] = (
        intrinsic.mean(0) + intrinsic.std(0, unbiased=True)
    ).numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
