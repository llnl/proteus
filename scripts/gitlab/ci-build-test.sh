#!/bin/bash

set -euo pipefail

echo "### Start ci-build-test $(date)"

WORKDIR="/tmp/proteus-ci-${CI_JOB_ID}"
PROJECT_DIR="$(cd "${CI_PROJECT_DIR}" && pwd -P)"
ENABLE_CODECOV_UPLOAD=0
PYTHON_BIN=python
PYTHON_VERSION=3.12
if [ "${PROTEUS_CODECOV_VERSION_SELECTED:-0}" = "1" ] && [ "${PROTEUS_CODECOV_VARIANT_SELECTED:-0}" = "1" ]; then
  ENABLE_CODECOV_UPLOAD=1
fi
mkdir -p "${WORKDIR}"
WORKDIR="$(cd "${WORKDIR}" && pwd -P)"
cd "${WORKDIR}"

run_ctest() {
  local label="${1}"
  echo "### $(date) START TESTING ${label} ###"
  shift
  env "$@" ctest -j8 -T test --output-on-failure
  echo "### $(date) END TESTING ${label} ###"
}

install_miniforge() {
  local miniforge_dir="${1}"

  mkdir -p "${miniforge_dir}"
  wget -q --tries=5 --retry-connrefused --wait=5 \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    -O "${miniforge_dir}/miniforge.sh"
  bash "${miniforge_dir}/miniforge.sh" -b -u -p "${miniforge_dir}"
  rm "${miniforge_dir}/miniforge.sh"
  source "${miniforge_dir}/etc/profile.d/conda.sh"
  # Create the env by prefix: a named env follows the user's condarc envs_dirs,
  # which concurrent jobs then share and clobber.
  CONDA_ENV_DIR="${miniforge_dir}/envs/proteus"
}

install_coverage_tools() {
  "${PYTHON_BIN}" -m pip install gcovr codecov-cli
}

if [ "${CI_MACHINE}" == "matrix" ]; then
  ml load cmake/3.30
  ml load cuda/12.2.2

  # Print GPU information for debugging.
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

  # Install Clang/LLVM through conda.
  install_miniforge "${WORKDIR}/miniforge3"
  # Use an older version of gcc to avoid issues with detecting Clang as the CUDA
  # compiler.
  conda create -y -q -p "${CONDA_ENV_DIR}" --override-channels -c conda-forge \
      python=${PYTHON_VERSION} clang=${PROTEUS_CI_LLVM_VERSION} clangxx=${PROTEUS_CI_LLVM_VERSION} \
      pip pybind11 clangdev=${PROTEUS_CI_LLVM_VERSION} llvmdev=${PROTEUS_CI_LLVM_VERSION} \
      lit=${PROTEUS_CI_LLVM_VERSION} mlir=${PROTEUS_CI_LLVM_VERSION} gcc=12 gxx=12
  conda activate "${CONDA_ENV_DIR}"

  LLVM_INSTALL_DIR=$(llvm-config --prefix)
  CMAKE_OPTIONS_MACHINE=" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake"
  CMAKE_OPTIONS_MACHINE+=" -DPROTEUS_ENABLE_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_CUDA_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
  # Clang 18 has a bug with CUDA and the GNU C++ standard library extensions:
  # https://github.com/llvm/llvm-project/issues/88695.
  # Fix by explicitly setting CUDA flags.
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_CUDA_FLAGS=-std=c++17"
elif [ "${CI_MACHINE}" == "tioga" ] || [ "${CI_MACHINE}" == "tuolumne" ]; then
  ml load rocm/${PROTEUS_CI_ROCM_VERSION}

  if [ -n "${ROCM_PATH:-}" ]; then
    echo "=> ROCM_PATH ${ROCM_PATH}"
  else
    echo "=> Failed to find ROCM_PATH"
    exit 1
  fi

  LLVM_INSTALL_DIR=${ROCM_PATH}/llvm
  install_miniforge "${WORKDIR}/miniforge3"
  conda create -y -q -p "${CONDA_ENV_DIR}" --override-channels -c conda-forge \
    python=${PYTHON_VERSION} pip pybind11
  conda activate "${CONDA_ENV_DIR}"

  CMAKE_OPTIONS_MACHINE=" -DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake"
  CMAKE_OPTIONS_MACHINE+=" -DPROTEUS_ENABLE_HIP=on"
  CMAKE_OPTIONS_MACHINE+=" -DCMAKE_HIP_ARCHITECTURES=gfx942;gfx90a"
else
  echo "Unsupported machine ${CI_MACHINE}"
  exit 1
fi

GCOV_EXECUTABLE="$LLVM_INSTALL_DIR/bin/llvm-cov gcov"

CMAKE_OPTIONS="-DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Release"
CMAKE_OPTIONS+=" -DCMAKE_C_COMPILER=$LLVM_INSTALL_DIR/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_INSTALL_DIR/bin/clang++"
CMAKE_OPTIONS+=" -DENABLE_DEVELOPER_COMPILER_FLAGS=on"
CMAKE_OPTIONS+=" -DPROTEUS_ENABLE_MLIR=on"
CMAKE_OPTIONS+=" -DPROTEUS_ENABLE_PYTHON=on"
CMAKE_OPTIONS+=" -DENABLE_TESTS=on"
CMAKE_OPTIONS+=" -DENABLE_COVERAGE=$([ "${ENABLE_CODECOV_UPLOAD}" = "1" ] && echo on || echo off)"
CMAKE_OPTIONS+=${CMAKE_OPTIONS_MACHINE}
if [ -n "${PROTEUS_CI_BUILD_SHARED:-}" ]; then
  CMAKE_OPTIONS+=" -DBUILD_SHARED=${PROTEUS_CI_BUILD_SHARED}"
fi

mkdir build
pushd build

cmake "${PROJECT_DIR}" ${CMAKE_OPTIONS} 2>&1 | tee cmake_output.log
if grep -q "Manually-specified variables were not used by the project:" cmake_output.log; then
    echo "Error: Unused variables detected"
    exit 1
fi
make -j

# Test synchronous compilation (default) and kernel clone options.
run_ctest "SYNC COMPILATION KERNEL_CLONE cross-clone" PROTEUS_KERNEL_CLONE=cross-clone
run_ctest "SYNC COMPILATION KERNEL_CLONE link-clone-light" PROTEUS_KERNEL_CLONE=link-clone-light
run_ctest "SYNC COMPILATION KERNEL_CLONE link-clone-prune" PROTEUS_KERNEL_CLONE=link-clone-prune

# Test asynchronous compilation.
run_ctest "(BLOCKING) ASYNC COMPILATION" PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1

# Test also our faster, alternative to HIP RTC codegen.
if [ "${CI_MACHINE}" == "tioga" ] || [ "${CI_MACHINE}" == "tuolumne" ]; then
  run_ctest "SYNC COMPILATION WITH PROTEUS CODEGEN SERIAL" PROTEUS_CODEGEN=serial
  run_ctest "(BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN SERIAL" \
    PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=serial

  if [ "${PROTEUS_CI_ROCM_VERSION}" == "6.4.3" ] \
     || [ "${PROTEUS_CI_ROCM_VERSION}" == "7.1.1" ] \
     || [ "${PROTEUS_CI_ROCM_VERSION}" == "7.2.0" ]; then
    run_ctest "SYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL" PROTEUS_CODEGEN=parallel
    run_ctest "(BLOCKING) ASYNC COMPILATION WITH PROTEUS CODEGEN PARALLEL" \
      PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 PROTEUS_CODEGEN=parallel
  fi
fi

popd

if [ "${ENABLE_CODECOV_UPLOAD}" = "1" ]; then
  if [ -z "${CODECOV_TOKEN:-}" ]; then
    echo "CODECOV_TOKEN must be defined for Codecov uploads"
    exit 1
  fi

  ARTIFACT_DIR="$(cd "${PROJECT_DIR}" && mkdir -p "gitlab-codecov-artifacts/${CI_MACHINE}" && cd "gitlab-codecov-artifacts/${CI_MACHINE}" && pwd -P)"
  install_coverage_tools

  pushd "${PROJECT_DIR}"
  "${PYTHON_BIN}" -m gcovr \
    --root "${PROJECT_DIR}" \
    --filter "${PROJECT_DIR}/src" \
    --filter "${PROJECT_DIR}/include" \
    --exclude-directories "${PROJECT_DIR}/tests" \
    --object-directory "${WORKDIR}/build" \
    --gcov-executable "${GCOV_EXECUTABLE}" \
    --xml-pretty \
    --output "${ARTIFACT_DIR}/coverage.xml" \
    --print-summary \
    "${WORKDIR}/build"

  CODECOV_FLAG="gitlab-${CI_MACHINE}"
  CODECOV_NAME="gitlab-${CI_MACHINE}-gpu"
  CODECOV_PR_ARGS=()
  if [ -n "${GITHUB_PR_NUMBER:-}" ]; then
    CODECOV_PR_ARGS=(-P "${GITHUB_PR_NUMBER}")
  fi

  codecovcli \
    --verbose \
    upload-process \
    --disable-search \
    --fail-on-error \
    -t "${CODECOV_TOKEN}" \
    -f "${ARTIFACT_DIR}/coverage.xml" \
    -F "${CODECOV_FLAG}" \
    -n "${CODECOV_NAME}" \
    -B "${CI_COMMIT_BRANCH}" \
    -C "${CI_COMMIT_SHA}" \
    -r "LLNL/proteus" \
    "${CODECOV_PR_ARGS[@]}" \
    --git-service github 2>&1 | tee "${ARTIFACT_DIR}/codecov-upload.log"
  popd
fi
