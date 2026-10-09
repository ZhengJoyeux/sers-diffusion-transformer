#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
python -u "$package_dir/experiment_t322.py" "$@"
