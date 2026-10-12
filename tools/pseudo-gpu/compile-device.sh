#!/usr/bin/env bash
# Pseudo-GPU device compilation harness.
#
# Runs the real HIP *device* compilation pipeline (clang -x hip, amdgcn
# target, ProteusPass as a pass plugin) on a machine without ROCm or a GPU,
# so lambda kernel-arg / clobbering analysis changes can be verified at
# compile time from the pass trace (PROTEUS_DEBUG_OUTPUT).
#
# Usage:
#   tools/pseudo-gpu/compile-device.sh [options] <source.cpp> [-- <extra clang args>]
#
# Options:
#   --llvm <prefix>   LLVM install prefix (default: /opt/homebrew/opt/llvm@19;
#                     the pass plugin is picked from build-pseudogpu-llvm<ver>)
#   --build <dir>     Build dir containing src/pass/libProteusPass.dylib
#   --raw-ir <file>   Also emit pre-pass device IR (no plugin) for shape study
#   --log <file>      Copy the pass trace here (default: <source>.proteus-pass.log)
#   -O<n>             Optimization level (default -O1)
#
# Exit status: 0 if compilation reached the end of the Proteus analysis.
# NOTE: upstream clang (unlike ROCm clang) registers -fpass-plugin pipeline
# callbacks twice for HIP device compiles, so ProteusPass runs twice and the
# second run hits the (expected-unique) _jit_bitcode global and calls
# reportFatalError.  That fatal happens after the analysis and manifest are
# complete, so this script treats exactly that error as benign tail noise.
# Every analysis result appears twice in the trace for the same reason.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PREFIX=/opt/homebrew/opt/llvm@19
BUILD_DIR=
RAW_IR=
LOG_OUT=
OPT_LEVEL=-O1
SOURCE=
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llvm) LLVM_PREFIX="$2"; shift 2 ;;
    --build) BUILD_DIR="$2"; shift 2 ;;
    --raw-ir) RAW_IR="$2"; shift 2 ;;
    --log) LOG_OUT="$2"; shift 2 ;;
    -O*) OPT_LEVEL="$1"; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) SOURCE="$1"; shift ;;
  esac
done

if [[ -z "$SOURCE" ]]; then
  echo "error: no source file given" >&2
  exit 2
fi

# The plugin compile runs inside a temp dir (to contain .proteus-logs), so
# user-supplied paths must survive the cd.
case "$SOURCE" in /*) ;; *) SOURCE="$PWD/$SOURCE" ;; esac
if [[ -n "$RAW_IR" ]]; then case "$RAW_IR" in /*) ;; *) RAW_IR="$PWD/$RAW_IR" ;; esac; fi
if [[ -n "$LOG_OUT" ]]; then case "$LOG_OUT" in /*) ;; *) LOG_OUT="$PWD/$LOG_OUT" ;; esac; fi

if [[ -z "$BUILD_DIR" ]]; then
  VER=$("$LLVM_PREFIX/bin/llvm-config" --version | cut -d. -f1)
  BUILD_DIR="$REPO_ROOT/build-pseudogpu-llvm$VER"
fi
PLUGIN="$BUILD_DIR/src/pass/libProteusPass.dylib"
if [[ ! -f "$PLUGIN" ]]; then
  echo "error: $PLUGIN not found; configure with:" >&2
  echo "  cmake -S $REPO_ROOT -B $BUILD_DIR -DLLVM_INSTALL_DIR=$LLVM_PREFIX \\" >&2
  echo "        -DCMAKE_C_COMPILER=$LLVM_PREFIX/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_PREFIX/bin/clang++ \\" >&2
  echo "        -DCMAKE_CXX_FLAGS=-DPROTEUS_ENABLE_HIP=1" >&2
  echo "  cmake --build $BUILD_DIR --target ProteusPass -j 8" >&2
  exit 2
fi

COMMON_ARGS=(
  -x hip --offload-arch=gfx90a --offload-device-only
  -nogpulib -nogpuinc -std=c++17 "$OPT_LEVEL"
  -DPROTEUS_ENABLE_HIP=1
  -I "$REPO_ROOT/tools/pseudo-gpu/include"
  -I "$REPO_ROOT/include"
)

WORK_DIR=$(mktemp -d /tmp/pseudo-gpu.XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ -n "$RAW_IR" ]]; then
  "$LLVM_PREFIX/bin/clang++" "${COMMON_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    -Xclang -disable-llvm-passes -S -emit-llvm -o "$RAW_IR" "$SOURCE" ||
    { echo "error: raw-IR compile failed" >&2; exit 1; }
  echo "raw device IR: $RAW_IR"
fi

STDERR_FILE="$WORK_DIR/stderr.txt"
(cd "$WORK_DIR" && PROTEUS_DEBUG_OUTPUT=1 "$LLVM_PREFIX/bin/clang++" \
  "${COMMON_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  -fpass-plugin="$PLUGIN" \
  -S -emit-llvm -o "$WORK_DIR/device.post.ll" "$SOURCE") 2> "$STDERR_FILE"
STATUS=$?

LOG_FILE=$(ls "$WORK_DIR"/.proteus-logs/proteus-pass.*.log 2>/dev/null | head -1)
if [[ -z "$LOG_FILE" ]]; then
  echo "error: no proteus-pass trace produced; compile failed before the pass:" >&2
  cat "$STDERR_FILE" >&2
  exit 1
fi

if [[ -z "$LOG_OUT" ]]; then
  LOG_OUT="$(basename "${SOURCE%.cpp}").proteus-pass.log"
fi
cp "$LOG_FILE" "$LOG_OUT"

if [[ $STATUS -ne 0 ]]; then
  if ! grep -q "Expected unique name for jit module global variable" "$STDERR_FILE"; then
    echo "error: compile failed with an unexpected error:" >&2
    cat "$STDERR_FILE" >&2
    exit 1
  fi
  echo "note: known benign double-run fatal (_jit_bitcode) ignored;" \
       "results below are from the completed first pass run"
fi

echo "pass trace: $LOG_OUT"
echo
echo "== Lambda kernel-arg analysis decisions (first run) =="
# The trace repeats once per pass run; report each unique line once.
grep -E "Found termination case from function|Kernel arg analysis failed for functor" \
  "$LOG_OUT" | sort -u | sed \
  -e 's/.*Found termination case from function /SPECIALIZE  kernel=/' \
  -e 's/.*Kernel arg analysis failed for functor beginning at */DECLINE     /'
exit 0
