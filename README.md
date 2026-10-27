# AMYCLIP

Minimal first-submission source package for the finalized AMYCLIP model.

AMYCLIP maps frozen ESM-2 residue-mean representations through separate peptide
and protein projection heads. It reports two separate outputs: target-conditioned
peptide-protein compatibility and peptide-intrinsic amyloidogenicity. These scores
are model probabilities, not binding affinities or direct evidence of aggregation
inhibition.

## Contents

- `amyclip/model.py`: exact final architecture and strict checkpoint loader.
- `amyclip/losses.py`: symmetric multi-positive retrieval loss and weighted,
  smoothed intrinsic loss corresponding to Methods Section 2.3.
- `scripts/train.py`: executable compact training demonstration on sample frozen
  embeddings.
- `scripts/infer.py`: sequence inference with the final three-checkpoint ensemble.
- `checkpoints/`: the three finalized Run26 checkpoints and SHA-256 manifest.
- `sample_data/`: small execution-only inputs, not the full study datasets.
- `example_results/`: one checkpoint-generated inference result.

Manuscript figures, manuscript-preparation files, full datasets, data-preparation
history, search/generation workflows, and exploratory inference scripts are not
part of this first-submission package.

## Repository hierarchy

```text
AMYCLIP/
├── amyclip/
│   ├── model.py                 # final architecture and checkpoint loader
│   ├── losses.py                # retrieval, intrinsic, and total objectives
│   └── esm.py                   # frozen ESM-2 residue-mean embedding
├── checkpoints/
│   ├── MANIFEST.json            # checkpoint names, seeds, and SHA-256 hashes
│   ├── amyclip_*seed17*.ckpt
│   ├── amyclip_*seed29*.ckpt
│   └── amyclip_*seed42*.ckpt
├── sample_data/
│   ├── inference_input.csv      # input for the scoring example
│   ├── training_sample.csv      # sequences for the training-code check
│   └── training_sample_embeddings.npz
├── example_results/
│   └── inference_output.csv     # expected checkpoint-generated scores
├── scripts/
│   ├── verify_release.py        # verifies hashes and strict checkpoint loading
│   ├── infer.py                 # peptide scoring program
│   ├── compare_example.py       # compares regenerated and expected scores
│   ├── prepare_sample_embeddings.py
│   ├── train.py                 # compact training-code execution check
│   └── self_test.py             # architecture and objective smoke test
├── tests/
│   └── test_model_math.py
├── run_example.sh               # complete checkpoint and scoring reproduction
├── requirements.txt
└── pyproject.toml
```

## Point-by-file matching

| Claimed function | Executable file | Required input | Produced or checked output |
|---|---|---|---|
| Checkpoint verification | `scripts/verify_release.py` | `checkpoints/MANIFEST.json` and the three `.ckpt` files | SHA-256 confirmation and strict loading into `amyclip/model.py` |
| Peptide scoring | `scripts/infer.py` | `sample_data/inference_input.csv`, the three checkpoints, and frozen ESM-2 | `example_results/inference_output_reproduced.csv` |
| Example-result verification | `scripts/compare_example.py` | regenerated output and `example_results/inference_output.csv` | numerical PASS/FAIL comparison |
| Training-code check | `scripts/prepare_sample_embeddings.py` and `scripts/train.py` | `sample_data/training_sample.csv` or the supplied `.npz` embeddings | `example_results/sample_trained_state.pt` |
| Architecture supporting the Methods equations | `amyclip/model.py` | 1,280-dimensional frozen ESM-2 vectors | normalized peptide/protein projections, compatibility logits, and intrinsic logits |
| Objective supporting the Methods equations | `amyclip/losses.py` | logits, positive/allowed masks, and intrinsic labels | symmetric multi-positive loss, weighted intrinsic loss, and their 0.05-weighted total |
| Complete example | `run_example.sh` | local ESM-2 checkpoint path | checkpoint checks, model self-test, regenerated scoring CSV, and comparison result |

The peptide-scoring example uses **two CSV files**: the input
`sample_data/inference_input.csv` and the expected output
`example_results/inference_output.csv`. The scoring implementation is
`scripts/infer.py`. `sample_data/training_sample.csv` is separate and is used only
for the training-code execution check; it is not a third scoring file.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git lfs pull
```

ESM-2 is the frozen `esm2_t33_650M_UR50D` model. Supply a local model file with
`--esm-model`; if omitted, `fair-esm` uses its standard download/cache mechanism.

## Verify the released model files

```bash
python scripts/verify_release.py
pytest -q
```

## Reproduce the included inference example

```bash
python scripts/infer.py \
  --input sample_data/inference_input.csv \
  --output example_results/inference_output_reproduced.csv \
  --checkpoint-dir checkpoints \
  --esm-model /path/to/esm2_t33_650M_UR50D.pt \
  --device cpu
```

The regenerated file should agree numerically with
`example_results/inference_output.csv`, subject only to ordinary floating-point
precision.

The complete verification can also be run as one command:

```bash
bash run_example.sh /path/to/esm2_t33_650M_UR50D.pt
```

## Training-code execution check

The included sample embeddings permit a short execution check of the finalized
architecture and loss. They do not recreate the reported fitted model because the
full development data are outside the first-submission release scope.

```bash
python scripts/prepare_sample_embeddings.py \
  --esm-model /path/to/esm2_t33_650M_UR50D.pt \
  --device cpu
python scripts/train.py --epochs 2
```

## Release scope

This repository is intentionally limited to the first-submission disclosure:
final checkpoints, sample data, final model and training code, and one reproduced
inference result. A revision-stage archive can add the full cited datasets,
data-preparation programs, and all inference/search workflows if requested.
