"""Final AMYCLIP architecture used by the released Run26 checkpoints.

The module mirrors Methods Section 2.3: independent peptide and protein
projections, a shared normalized representation space, fixed-temperature
compatibility logits, and a peptide-only intrinsic amyloidogenicity head.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ProjectionHead(nn.Sequential):
    """1280 -> 1024 -> 512 -> 512 projection used for each sequence type."""

    def __init__(self, input_dim: int = 1280, output_dim: int = 512) -> None:
        super().__init__(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, output_dim),
        )


class AMYCLIP(nn.Module):
    """Final dual-endpoint AMYCLIP network operating on pooled ESM-2 vectors."""

    def __init__(
        self,
        esm_dim: int = 1280,
        projection_dim: int = 512,
        temperature: float = 12.5,
        intrinsic_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.pep_embedder = ProjectionHead(esm_dim, projection_dim)
        self.prop_embedder = ProjectionHead(esm_dim, projection_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature)).log(), requires_grad=False
        )
        self.amyloid_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Dropout(intrinsic_dropout),
            nn.Linear(projection_dim, 1),
        )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def project_peptide(self, peptide_embedding: Tensor) -> Tensor:
        return F.normalize(self.pep_embedder(peptide_embedding), dim=-1)

    def project_protein(self, protein_embedding: Tensor) -> Tensor:
        return F.normalize(self.prop_embedder(protein_embedding), dim=-1)

    def compatibility_logits(
        self, peptide_embedding: Tensor, protein_embedding: Tensor
    ) -> Tensor:
        peptide = self.project_peptide(peptide_embedding)
        protein = self.project_protein(protein_embedding)
        return self.temperature * (peptide @ protein.T)

    def compatibility_probability(
        self, peptide_embedding: Tensor, protein_embedding: Tensor
    ) -> Tensor:
        return torch.sigmoid(
            self.compatibility_logits(peptide_embedding, protein_embedding)
        )

    def intrinsic_logits(self, peptide_embedding: Tensor) -> Tensor:
        peptide = self.project_peptide(peptide_embedding)
        return self.amyloid_head(peptide).squeeze(-1)

    def intrinsic_probability(self, peptide_embedding: Tensor) -> Tensor:
        return torch.sigmoid(self.intrinsic_logits(peptide_embedding))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> AMYCLIP:
    """Load a released PyTorch-Lightning checkpoint into the standalone model."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    model = AMYCLIP(
        esm_dim=int(hyperparameters.get("esm_dim", 1280)),
        projection_dim=int(hyperparameters.get("proj_dim", 512)),
        temperature=float(hyperparameters.get("fixed_temperature", 12.5)),
        intrinsic_dropout=float(hyperparameters.get("head_dropout", 0.30)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()
