#!/usr/bin/env bash
# Reproduce the Track 2 leaderboard submission (Codabench 995100, combined 21.2379).
#   bash scripts/run_inference.sh /path/to/phase2_test_wavs            # full pipeline, ~4-5 h (template discovery on CPU)
#   bash scripts/run_inference.sh /path/to/phase2_test_wavs --fast     # reuse the released template sets, ~30 min
set -euo pipefail
TEST_DIR=${1:?usage: run_inference.sh <test-wav-dir> [--fast]}
REPO=$(cd "$(dirname "$0")/.." && pwd)
EXTRA=()
if [ "${2:-}" = "--fast" ]; then
  EXTRA=(--templates "$REPO/artifacts/templates_pass1.pkl" "$REPO/artifacts/templates_pass2.pkl")
fi
cd "$REPO/inference"
python infer.py --test-dir "$TEST_DIR" --work "$REPO/work" "${EXTRA[@]}"
