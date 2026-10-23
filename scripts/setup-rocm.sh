#!/bin/sh

HOSTN=${HOSTNAME//[0-9]/}

ROCM_VERSION=$1
if [ $# -lt 1 ]; then
    echo "Usage: source setup-rocm.sh <ROCm version>"
    return 0
fi

ml load rocm/${ROCM_VERSION}

LLVM_INSTALL_DIR=${ROCM_PATH}/llvm

mkdir build-${HOSTN}-rocm-${ROCM_VERSION}
pushd build-${HOSTN}-rocm-${ROCM_VERSION}

cmake .. \
-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
-DPROTEUS_ENABLE_HIP=on \
-DCMAKE_INSTALL_PREFIX=../install-${HOSTN}-rocm-${ROCM_VERSION} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
"${@:2}"

popd
