#!/usr/bin/env bash
# Run the pseudo-GPU device pipeline over every lambda GPU test, collecting
# one ProteusPass trace per test, then check the decisions against the
# committed ground truth. Needs no GPU or ROCm: any host clang++ with the
# AMDGPU backend (stock LLVM releases include it) can run it.
#
# Usage: tools/lambda-spec-ci/run_sweep.sh <clang++> <libProteusPass> [logs-dir]
# Exits nonzero on compile failure of a test that has ground-truth entries,
# or on any decision drift.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLANGXX="${1:?usage: run_sweep.sh <clang++> <libProteusPass> [logs-dir]}"
PLUGIN="${2:?usage: run_sweep.sh <clang++> <libProteusPass> [logs-dir]}"
LOGS_DIR="${3:-$REPO_ROOT/build/lambda-spec-logs}"
TRUTH="$REPO_ROOT/tests/gpu/lambda-spec-groundtruth.json"

rm -rf "$LOGS_DIR"
mkdir -p "$LOGS_DIR"

status=0
for SOURCE in "$REPO_ROOT"/tests/gpu/lambda*.cpp; do
  TEST=$(basename "$SOURCE" .cpp)
  WORK_DIR=$(mktemp -d)
  (cd "$WORK_DIR" && PROTEUS_DEBUG_OUTPUT=1 "$CLANGXX" \
      -x hip --offload-arch=gfx90a --offload-device-only \
      -nogpulib -nogpuinc -std=c++17 -O1 \
      -DPROTEUS_ENABLE_HIP=1 \
      -I "$REPO_ROOT/tools/pseudo-gpu/include" \
      -I "$REPO_ROOT/include" \
      -include hip/hip_runtime.h \
      -fpass-plugin="$PLUGIN" \
      -S -emit-llvm -o /dev/null "$SOURCE") \
      > "$WORK_DIR/stdout.txt" 2> "$WORK_DIR/stderr.txt"
  if compgen -G "$WORK_DIR/.proteus-logs/proteus-pass.*.log" > /dev/null; then
    cat "$WORK_DIR"/.proteus-logs/proteus-pass.*.log > "$LOGS_DIR/$TEST.log"
  else
    # No trace at all: fail loudly only for tests the ground truth knows,
    # since some sources are expected not to compile in this pipeline.
    if python3 - "$TRUTH" "$TEST" <<'EOF'
import json, sys
truth = json.load(open(sys.argv[1]))
sys.exit(0 if sys.argv[2] in truth["tests"] else 1)
EOF
    then
      echo "ERROR: $TEST produced no pass trace; compile output:" >&2
      tail -20 "$WORK_DIR/stderr.txt" >&2
      status=1
    else
      echo "note: $TEST produced no trace (not in ground truth); skipping"
    fi
  fi
  rm -rf "$WORK_DIR"
done

[ $status -ne 0 ] && exit $status
exec python3 "$REPO_ROOT/tools/lambda-spec-ci/lambda_spec_ci.py" check \
    --logs "$LOGS_DIR" --truth "$TRUTH"
