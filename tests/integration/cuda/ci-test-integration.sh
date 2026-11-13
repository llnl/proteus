#!/bin/bash

set -e

echo "Run Integration Test ${PROTEUS_CI_INTEGRATION_TEST}"

# LLVM/Clang version installed from conda-forge (replaces the previously assumed
# system install at /usr/lib64/llvm20).
PROTEUS_CI_LLVM_VERSION=20.1.8
PYTHON_VERSION=3.12

# Load cuda module.
ml load cuda/12

install_miniforge() {
  local miniforge_dir="${1}"

  mkdir -p "${miniforge_dir}"
  wget -q --tries=5 --retry-connrefused --wait=5 \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    -O "${miniforge_dir}/miniforge.sh"
  bash "${miniforge_dir}/miniforge.sh" -b -u -p "${miniforge_dir}"
  rm "${miniforge_dir}/miniforge.sh"
  source "${miniforge_dir}/etc/profile.d/conda.sh"
  # Create the environment by prefix. A named environment follows the user's
  # condarc envs_dirs and can select a persistent environment outside this job.
  CONDA_ENV_DIR="${miniforge_dir}/envs/proteus"
}

export TEST_DIR=${CI_PROJECT_DIR}/tests/integration/cuda/${PROTEUS_CI_INTEGRATION_TEST}

mkdir -p /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}
pushd /tmp/proteus-ci-$(basename ${TEST_DIR})-${CI_JOB_ID}

rm -rf build-proteus install-proteus build install miniforge3

# Install Clang/LLVM through conda.
install_miniforge miniforge3
# Use an older version of gcc to avoid issues with detecting Clang as the CUDA
# compiler.
conda create -y -q -p "${CONDA_ENV_DIR}" --override-channels -c conda-forge \
    python=${PYTHON_VERSION} clang=${PROTEUS_CI_LLVM_VERSION} clangxx=${PROTEUS_CI_LLVM_VERSION} \
    pip pybind11 clangdev=${PROTEUS_CI_LLVM_VERSION} llvmdev=${PROTEUS_CI_LLVM_VERSION} \
    lit=${PROTEUS_CI_LLVM_VERSION} mlir=${PROTEUS_CI_LLVM_VERSION} gcc=12 gxx=12
conda activate "${CONDA_ENV_DIR}"

export LLVM_INSTALL_DIR=$(llvm-config --prefix)
echo "LLVM_INSTALL_DIR ${LLVM_INSTALL_DIR}"

# Make the conda LLVM discoverable by the test sub-project's find_package(proteus)
# call. CMake reads CMAKE_PREFIX_PATH from the environment.
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CONDA_PREFIX}/lib/cmake"

bash ${TEST_DIR}/ci-build-and-run.sh

popd

echo "=> Passed ${PROTEUS_CI_INTEGRATION_TEST}"
