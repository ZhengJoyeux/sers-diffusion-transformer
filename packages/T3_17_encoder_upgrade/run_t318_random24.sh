#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${T317_PROJECT:-/home/wqzheng/project_transformer}"
mode="${1:-smoke}"
cd "$project_dir"

case "$mode" in
  check)
    python "$package_dir/verify_t317.py" --project "$project_dir"
    exit 0
    ;;
  smoke)
    epochs=2; warmup=1; heads=8; layers=6; attention_dim=128; ff_dim=256
    limits=(--max-train-batches 8 --max-validation-batches 4 --skip-final-evaluation)
    ;;
  train)
    epochs=50; warmup=3; heads=8; layers=6; attention_dim=128; ff_dim=256
    limits=()
    ;;
  baseline)
    epochs=50; warmup=3; heads=4; layers=4; attention_dim=64; ff_dim=64
    limits=()
    ;;
  *) echo 'Usage: bash run_t317.sh check|smoke|train|baseline' >&2; exit 2 ;;
esac

python - "$package_dir/source_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
for name,expected in json.loads(Path(sys.argv[1]).read_text()).items():
    if hashlib.sha256(Path(name).read_text().encode()).hexdigest()!=expected:
        raise SystemExit(f'Current source differs from uploaded snapshot: {name}; stop and report output.')
print('SOURCE SNAPSHOT: PASS')
PY

run_id="$(date +%Y%m%d_%H%M%S)_$$"
output_dir="outputs/t3_18_random24_${mode}_h${heads}_l${layers}_d${attention_dim}_real_generated_${run_id}"
mkdir -p "$output_dir"
echo "T3.18 output: $project_dir/$output_dir"
echo "Mode=$mode; attention heads=$heads; encoder layers=$layers; attention dim=$attention_dim; epochs=$epochs"

python -u "$package_dir/T3_17_Training.py" \
  --output-directory "$output_dir" \
  --include-generated --maximum-generated-per-condition 24 \
  --epochs "$epochs" --batch-size 16 --early-stopping-patience 50 \
  --learning-rate 0.0001 --minimum-learning-rate 0.000001 \
  --lr-schedule warmup_cosine --warmup-epochs "$warmup" --warmup-start-ratio 0.1 \
  --dim-model 32 --attention-dim "$attention_dim" --dim-ff "$ff_dim" \
  --attention-heads "$heads" --encoder-layers "$layers" --dropout 0.1 \
  --concentration-head-mode ordinal --ordinal-loss-mode corn --ordinal-head-hidden 64 \
  --ordinal-decoding median --checkpoint-selection ordinal --mixture-aware-query-fusion \
  --query-attention-mode peak_guided --peak-prior-mode chemistry_hybrid \
  --peak-top-k 4 --peak-guidance-strength 1.5 \
  --chemistry-core-half-width-cm1 7 --chemistry-auxiliary-max-weight 0.35 \
  --chemistry-shared-core-weight 0.65 --loss-weighting fixed --classification-weight 0.30 --regression-weight 0.70 \
  --num-workers 4 --seed 2026 --device cuda \
  "${limits[@]}" 2>&1 | tee "$output_dir/console.log"
