"""Frozen ESM-2 residue-mean embeddings used by AMYCLIP."""

from __future__ import annotations

from pathlib import Path

import torch


def load_esm2(model_path: str | Path | None, device: torch.device):
    import esm

    if model_path:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, alphabet


def mean_pool_sequences(sequences, model, alphabet, device, batch_size: int = 8):
    converter = alphabet.get_batch_converter()
    pooled_batches = []
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        _, _, tokens = converter(
            [(f"sequence_{start + offset}", sequence) for offset, sequence in enumerate(batch)]
        )
        tokens = tokens.to(device)
        with torch.inference_mode():
            output = model(tokens, repr_layers=[33], return_contacts=False)
            representations = output["representations"][33]
            residue_mask = (
                (tokens != alphabet.cls_idx)
                & (tokens != alphabet.eos_idx)
                & (tokens != alphabet.padding_idx)
            )
            pooled = (
                (representations * residue_mask.unsqueeze(-1)).sum(dim=1)
                / residue_mask.sum(dim=1, keepdim=True).clamp(min=1)
            )
        pooled_batches.append(pooled.cpu())
    return torch.cat(pooled_batches, dim=0)
