//===-- JitEngineDeviceCUDA.h -- JIT Engine Device for CUDA header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICECUDA_H
#define PROTEUS_JITENGINEDEVICECUDA_H

#include "proteus/impl/JitEngineDevice.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <memory>

namespace proteus {

using namespace llvm;

class JitEngineDeviceCUDA;
template <> struct DeviceTraits<JitEngineDeviceCUDA> {
  using DeviceError_t = cudaError_t;
  using DeviceStream_t = CUstream;
  using KernelFunction_t = CUfunction;
};

class JitEngineDeviceCUDA : public JitEngineDevice<JitEngineDeviceCUDA> {
public:
  static JitEngineDeviceCUDA &instance();

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize);

  void extractModules(BinaryInfo &BinInfo);

  std::unique_ptr<Module> tryExtractKernelModule(BinaryInfo &BinInfo,
                                                 StringRef KernelName,
                                                 LLVMContext &Ctx);

  void codegenPTX(Module &M, StringRef DeviceArch,
                  SmallVectorImpl<char> &PTXStr);

  HashT getModuleHash(BinaryInfo &BinInfo);

  // Returns false when an NVCC fatbinary has no matching shadow IR. Callers
  // must execute the original AOT kernel in that case.
  bool canSpecialize(BinaryInfo &BinInfo);

  // CUDA codegen never runs the Proteus pipeline itself.
  static bool optimizesBeforeCodegen(CodegenOption) { return true; }

  std::unique_ptr<MemoryBuffer>
  codegenObject(Module &M, SmallPtrSetImpl<void *> &GlobalLinkedBinaries,
                const CodeGenerationConfig &CGConfig);

  std::unique_ptr<Dispatcher> createDispatcher();

private:
  JitEngineDeviceCUDA();
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &) = delete;
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &&) = delete;

  void extractLinkedBitcode(LLVMContext &Ctx, CUmodule &CUMod,
                            SmallVector<std::unique_ptr<Module>> &LinkedModules,
                            std::string &ModuleId);
};

} // namespace proteus

#endif
