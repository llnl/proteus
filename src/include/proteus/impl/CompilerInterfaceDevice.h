//===-- CompilerInterfaceDevice.h -- JIT entry point for GPU header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILERINTERFACEDEVICE_H
#define PROTEUS_COMPILERINTERFACEDEVICE_H

#if PROTEUS_ENABLE_CUDA

#include "proteus/impl/JitEngineDeviceCUDA.h"
using JitDeviceImplT = proteus::JitEngineDeviceCUDA;

#elif PROTEUS_ENABLE_HIP

#include "proteus/impl/JitEngineDeviceHIP.h"
using JitDeviceImplT = proteus::JitEngineDeviceHIP;

#else
#error                                                                         \
    "CompilerInterfaceDevice requires PROTEUS_ENABLE_CUDA or PROTEUS_ENABLE_HIP"
#endif

// The ABI of __proteus_launch_kernel mirrors device-specific launchKernel and
// depends on the host arch: https://github.com/LLNL/proteus/issues/47.
extern "C" proteus::DeviceTraits<JitDeviceImplT>::DeviceError_t
__proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                        void **KernelArgs, uint64_t ShmemSize, void *Stream);

#endif
