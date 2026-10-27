# Sample data

The files in this directory are small execution samples only. They are not the
training, development, or protected evaluation datasets used to report the
manuscript results.

- `inference_input.csv` demonstrates sequence-level ensemble scoring.
- `training_sample.csv` documents the sequences represented by the compact
  training example.
- `training_sample_embeddings.npz` contains frozen ESM-2 layer-33 residue-mean
  embeddings for those rows plus the positive and allowed masks needed to execute
  the reported training objective.
