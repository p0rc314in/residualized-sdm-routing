#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$repo_root/runs/data/wikitext103_gpt2_causal_t2048_coverage_v1/manifest.json"
run_root="${RUN_ROOT:-$repo_root/runs/reproduction}"
stage="${1:-all}"

export PYTHONPATH="$repo_root:$repo_root/third_party/runtime${PYTHONPATH:+:$PYTHONPATH}"

prepare() {
  if [[ -f "$manifest" ]]; then
    python3 "$repo_root/scripts/check_prepared_data.py" "$manifest"
  else
    python3 "$repo_root/scripts/prepare_canonical_wikitext103.py" \
      --output "$(dirname "$manifest")"
  fi
}

case "$stage" in
  recall|babylm)
    benchmark="$stage"
    if [[ "$benchmark" == recall ]]; then benchmark="adaptive_recall"; fi
    shift
    python3 "$repo_root/scripts/run_reproductions.py" "$benchmark" "$@"
    ;;
  prepare)
    prepare
    ;;
  train)
    python3 "$repo_root/scripts/run_experiments.py" \
      --manifest "$manifest" \
      --output "$run_root" \
      --gpus "${GPUS:-0}"
    ;;
  verify-results)
    python3 "$repo_root/scripts/verify_results.py"
    ;;
  check-reproduction)
    python3 "$repo_root/scripts/check_reproduction.py" "$run_root"
    ;;
  all)
    prepare
    python3 "$repo_root/scripts/run_experiments.py" \
      --manifest "$manifest" \
      --output "$run_root" \
      --gpus "${GPUS:-0}"
    ;;
  *)
    echo "usage: ./reproduce.sh {prepare|train|check-reproduction|verify-results|all|recall STAGE|babylm STAGE}" >&2
    exit 2
    ;;
esac
