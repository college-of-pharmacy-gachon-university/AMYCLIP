#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash run_example.sh /path/to/esm2_t33_650M_UR50D.pt" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
esm_model="$1"
observed="$repo_dir/example_results/inference_output_reproduced.csv"

python "$repo_dir/scripts/verify_release.py"
python "$repo_dir/scripts/self_test.py"
python "$repo_dir/scripts/infer.py" \
  --input "$repo_dir/sample_data/inference_input.csv" \
  --output "$observed" \
  --checkpoint-dir "$repo_dir/checkpoints" \
  --esm-model "$esm_model" \
  --device cpu
python "$repo_dir/scripts/compare_example.py" \
  --expected "$repo_dir/example_results/inference_output.csv" \
  --observed "$observed"
