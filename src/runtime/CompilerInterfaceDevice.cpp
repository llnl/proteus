//===-- CompilerInterfaceDevice.cpp -- JIT library entry point for GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/CompilerInterfaceDevice.h"
#include "proteus/KernelMetadata.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/CompilerInterfaceDeviceInternal.h"
#include "proteus/impl/JitEngineDevice.h"
#include "proteus/impl/JitEngineInfoRegistry.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/ShadowBitcodeRegistry.h"
#include <llvm/ADT/DenseMap.h>

#include <cstring>
#include <utility>
#include <vector>

using namespace proteus;

// NOLINTBEGIN(readability-identifier-naming)

// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void* for HIP
extern "C" __attribute((used)) void __proteus_register_var(void *Handle,
                                                           const void *HostAddr,
                                                           const char *VarName,
                                                           uint64_t VarSize) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  // NOTE: For HIP it works to get the symbol address during the call inside a
  // constructor context, but for CUDA, it fails.  So we save the host address
  // and defer resolving the symbol address when patching the bitcode, which
  // works for both CUDA and HIP.
  JitEngineInfo.registerVar(Handle, HostAddr, VarName, VarSize);
}

extern "C" __attribute__((used)) void
__proteus_register_fatbinary(void *Handle, void *FatbinWrapper,
                             const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinary(Handle, FatbinWrapper, ModuleId);
}

extern "C" __attribute__((used)) void
__proteus_register_fatbinary_end(void *Handle) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinaryEnd(Handle);
}

// Registers device bitcode produced by a shadow Clang compile of an
// NVCC-built translation unit. Called from constructors emitted by
// add_proteus_nvcc(); Bitcode has static storage duration.
extern "C" __attribute__((used)) void
__proteus_register_shadow_bitcode(const char *ModuleId, const void *Bitcode,
                                  uint64_t Size) {
  ShadowBitcodeRegistry::instance().registerBitcode(ModuleId, Bitcode, Size);
}

extern "C" __attribute__((used)) void
__proteus_register_linked_binary(void *FatbinWrapper, const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerLinkedBinary(FatbinWrapper, ModuleId);
}

extern "C" __attribute((used)) void
__proteus_register_function(void *Handle, void *Kernel, char *KernelName,
                            RuntimeConstantInfo **RCInfoArrayPtr,
                            int32_t NumRCs) {
  ArrayRef<RuntimeConstantInfo *> RCInfoArray{RCInfoArrayPtr,
                                              static_cast<size_t>(NumRCs)};
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFunction(Handle, Kernel, KernelName, RCInfoArray);
}

extern "C" __attribute__((used)) void
__proteus_register_lambda_callsite_location(void *Kernel, uint64_t LambdaID,
                                            uint32_t CallsiteIndex,
                                            uint32_t KernelArgIndex,
                                            int64_t Offset,
                                            int32_t StorageType) {
  auto &Jit = JitDeviceImplT::instance();
  Jit.registerLambdaCallsiteLocation(
      Kernel, LambdaID, CallsiteIndex, KernelArgIndex, Offset,
      static_cast<RuntimeConstantType>(StorageType));
}

extern "C" __attribute__((used)) void __proteus_begin_device_lambda_launch() {
  auto &LR = LambdaRegistry::instance();
  LR.beginDeviceLaunch();
}

extern "C" __attribute__((used)) void
__proteus_populate_lambda_registration_function(void *Kernel,
                                                void *RegistrationFunc) {
  auto &LR = LambdaRegistry::instance();
  LR.populateLambdaRegistrationCodeCache(Kernel, RegistrationFunc);
}

extern "C" __attribute__((used)) void
__proteus_push_device_lambda_callsite_constant(uint64_t LambdaID,
                                               uint32_t CallsiteIndex,
                                               int32_t Type, int32_t Pos,
                                               int32_t Offset,
                                               const void *ValuePtr) {
  RuntimeConstant RC{static_cast<RuntimeConstantType>(Type), Pos, Offset};
  switch (static_cast<RuntimeConstantType>(Type)) {
  case RuntimeConstantType::BOOL:
    std::memcpy(&RC.Value.BoolVal, ValuePtr, sizeof(RC.Value.BoolVal));
    break;
  case RuntimeConstantType::INT8:
    std::memcpy(&RC.Value.Int8Val, ValuePtr, sizeof(RC.Value.Int8Val));
    break;
  case RuntimeConstantType::INT32:
    std::memcpy(&RC.Value.Int32Val, ValuePtr, sizeof(RC.Value.Int32Val));
    break;
  case RuntimeConstantType::INT64:
    std::memcpy(&RC.Value.Int64Val, ValuePtr, sizeof(RC.Value.Int64Val));
    break;
  case RuntimeConstantType::FLOAT:
    std::memcpy(&RC.Value.FloatVal, ValuePtr, sizeof(RC.Value.FloatVal));
    break;
  case RuntimeConstantType::DOUBLE:
    std::memcpy(&RC.Value.DoubleVal, ValuePtr, sizeof(RC.Value.DoubleVal));
    break;
  case RuntimeConstantType::PTR:
    std::memcpy(&RC.Value.PtrVal, ValuePtr, sizeof(RC.Value.PtrVal));
    break;
  default:
    reportFatalError("__proteus_push_device_lambda_callsite_constant only "
                     "supports scalar captures");
  }

  auto &LR = LambdaRegistry::instance();
  LR.appendDeviceCallsiteRuntimeConstant(LambdaID, CallsiteIndex, RC);
}

extern "C" __attribute__((used)) void
__proteus_finalize_device_lambda_launch() {
  auto &LR = LambdaRegistry::instance();
  LR.finalizeDeviceLaunch();
}

extern "C" proteus::DeviceTraits<JitDeviceImplT>::DeviceError_t
__proteus_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                        void **KernelArgs, uint64_t ShmemSize, void *Stream) {
  TIMESCOPE("__proteus_launch_kernel");
  auto &LR = LambdaRegistry::instance();
  LR.invokeRegisterLambdaConstants(Kernel, KernelArgs);
  return __proteus_launch_kernel_internal(Kernel, GridDim, BlockDim, KernelArgs,
                                          ShmemSize, Stream);
}

extern "C" void __proteus_enable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.enable();
}

extern "C" void __proteus_disable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.disable();
}

namespace proteus::runtime {

std::optional<KernelMetadata> captureKernelMetadata(const void *Kernel) {
  auto &Jit = JitDeviceImplT::instance();
  auto OptionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!OptionalKernelInfo)
    return std::nullopt;

  auto &KInfo = OptionalKernelInfo.value().get();
  if (!Jit.canSpecializeKernel(KInfo))
    return std::nullopt;

  Jit.extractModuleAndBitcode(KInfo);
  auto StaticHash = Jit.getStaticHash(KInfo);
  auto Bitcode = KInfo.getBitcode().getBuffer();
  if (Bitcode.empty())
    reportFatalError("Proteus captured kernel metadata has empty bitcode");

  auto &GlobalVars = KInfo.getBinaryInfo().getVarNameToGlobalInfo();

  std::vector<char> BitcodeBytes(Bitcode.data(),
                                 Bitcode.data() + Bitcode.size());
  GlobalMetadataMap Globals;
  Globals.reserve(GlobalVars.size());
  for (const auto &[Name, GV] : GlobalVars)
    Globals.try_emplace(Name,
                        GlobalMetadata{GV.HostAddr, GV.DevAddr, GV.VarSize});

  return KernelMetadata(KInfo.getName(), StaticHash.getValue(),
                        std::move(BitcodeBytes), std::move(Globals));
}

} // namespace proteus::runtime

// NOLINTEND(readability-identifier-naming)
