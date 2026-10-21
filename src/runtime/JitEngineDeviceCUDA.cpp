//===-- JitEngineDeviceCUDA.cpp -- JIT Engine Device for CUDA --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/JitEngineDeviceCUDA.h"
#include "proteus/impl/ShadowBitcodeRegistry.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Frontend/DispatcherCUDA.h"
#include "proteus/impl/JitEngineDevice.h"
#include "proteus/impl/Utils.h"
#include "proteus/impl/UtilsCUDA.h"

#include <cuda_runtime.h>

#include <sys/types.h>

using namespace proteus;
using namespace llvm;

JitEngineDeviceCUDA &JitEngineDeviceCUDA::instance() {
  static JitEngineDeviceCUDA Jit{};
  return Jit;
}

bool JitEngineDeviceCUDA::canSpecialize(BinaryInfo &BinInfo) {
  auto &ModuleIds = BinInfo.getModuleIds();
  if (ModuleIds.empty())
    return false;

  // Shadow modules are registered in host memory and need no CUDA module
  // query. If any ID is not shadow-registered, check whether this is an
  // ordinary Clang-CUDA fatbinary carrying the bitcode global.
  CUmodule CUMod = nullptr;
  bool CanSpecialize = true;
  for (const std::string &ModuleId : ModuleIds) {
    if (!ShadowBitcodeRegistry::instance().lookup(ModuleId).empty())
      continue;

    if (!CUMod) {
      FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();
      if (!FatbinWrapper)
        reportFatalError("Expected FatbinWrapper in map");
      proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));
    }

    CUdeviceptr DevPtr;
    size_t Bytes;
    CUresult Result =
        cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str());
    if (Result == CUDA_ERROR_NOT_FOUND) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "No device or shadow bitcode for " << ModuleId
                  << "; using NVCC AOT kernel\n");
      CanSpecialize = false;
      break;
    }
    proteusCuErrCheck(Result);
  }

  if (CUMod)
    proteusCuErrCheck(cuModuleUnload(CUMod));
  return CanSpecialize;
}

void JitEngineDeviceCUDA::extractLinkedBitcode(
    LLVMContext &Ctx, CUmodule &CUMod,
    SmallVector<std::unique_ptr<Module>> &LinkedModules,
    std::string &ModuleId) {
  TIMESCOPE(JitEngineDeviceCUDA, extractLinkedBitcode);
  PROTEUS_DBG(Logger::logs("proteus")
              << "extractLinkedBitcode " << ModuleId << "\n");

  if (!ModuleIdToFatBinary.count(ModuleId))
    reportFatalError("Expected to find module id " + ModuleId + " in map");

  // NVCC fatbinaries carry no bitcode global; a shadow Clang device compile
  // registers the module's bitcode at startup instead. Prefer the fatbin
  // (Clang-CUDA builds), fall back to the shadow registry (NVCC builds).
  // See docs/dev/design/nvcc-interop.md.
  SmallString<4096> DeviceBitcode;
  size_t Bytes = 0;
  CUdeviceptr DevPtr;
  CUresult GetGlobalResult =
      cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str());
  if (GetGlobalResult == CUDA_SUCCESS) {
    DeviceBitcode.reserve(Bytes);
    proteusCuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
  } else {
    StringRef Shadow = ShadowBitcodeRegistry::instance().lookup(ModuleId);
    if (Shadow.empty())
      proteusCuErrCheck(GetGlobalResult);
    PROTEUS_DBG(Logger::logs("proteus")
                << "extractLinkedBitcode: shadow bitcode for " << ModuleId
                << " (" << Shadow.size() << " bytes)\n");
    DeviceBitcode.append(Shadow.begin(), Shadow.end());
    Bytes = Shadow.size();
  }

  Timer T(Config::get().ProteusEnableTimers);
  StringRef Bitcode{DeviceBitcode.data(), Bytes};
  SMDiagnostic Diag;
  // We copy the Bitcode data for lazy parsing.
  // TODO: It could be zero-copy if we use our own backing storage for
  // DeviceBitcode.
  auto M = getLazyIRModule(MemoryBuffer::getMemBufferCopy(Bitcode, ModuleId),
                           Diag, Ctx, true);
  if (!M)
    reportFatalError("Error parsing IR: " + Diag.getMessage());
  M->setModuleIdentifier(ModuleId);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus") << "Parse IR " << ModuleId << " "
                                               << T.elapsed() << " ms\n";)

  LinkedModules.push_back(std::move(M));
}

HashT JitEngineDeviceCUDA::getModuleHash(BinaryInfo &BinInfo) {
  TIMESCOPE(JitEngineDeviceCUDA, getModuleHash);
  if (BinInfo.hasModuleHash())
    return BinInfo.getModuleHash();

  CUmodule CUMod;
  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();
  if (!FatbinWrapper)
    reportFatalError("Expected FatbinWrapper in map");

  auto &LinkedModuleIds = BinInfo.getModuleIds();

  proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  auto ExtractLinkedBitcodeHash = [&CUMod](std::string &ModuleId) {
    CUdeviceptr DevPtr;
    size_t Bytes;
    CUresult GetGlobalResult =
        cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str());
    if (GetGlobalResult != CUDA_SUCCESS) {
      // NVCC build: hash the shadow-registered bitcode instead.
      StringRef Shadow = ShadowBitcodeRegistry::instance().lookup(ModuleId);
      if (Shadow.empty())
        proteusCuErrCheck(GetGlobalResult);
      return hash(Shadow);
    }

    SmallVector<char, 4096> DeviceBitcode;
    DeviceBitcode.reserve(Bytes);
    proteusCuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
    return hash(StringRef{DeviceBitcode.data(), Bytes});
  };

  for (auto &ModuleId : LinkedModuleIds)
    BinInfo.updateModuleHash(ExtractLinkedBitcodeHash(ModuleId));

  proteusCuErrCheck(cuModuleUnload(CUMod));

  return BinInfo.getModuleHash();
}

std::unique_ptr<Module>
JitEngineDeviceCUDA::tryExtractKernelModule(BinaryInfo &, StringRef,
                                            LLVMContext &) {
  // We do not support emitting per-kernel modules during CUDA compilation,
  // hence this returns null to trigger the fallback to extraction of per-TU
  // modules and cloning.
  return nullptr;
}

void JitEngineDeviceCUDA::extractModules(BinaryInfo &BinInfo) {
  TIMESCOPE(JitEngineDeviceCUDA, extractModules);
  CUmodule CUMod;

  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();
  if (!FatbinWrapper)
    reportFatalError("Expected FatbinWrapper in map");

  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = *BinInfo.getLLVMContext();

  auto &LinkedModuleIds = BinInfo.getModuleIds();

  proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  for (auto &ModuleId : LinkedModuleIds)
    extractLinkedBitcode(Ctx, CUMod, LinkedModules, ModuleId);

  proteusCuErrCheck(cuModuleUnload(CUMod));

  BinInfo.setExtractedModules(LinkedModules);
}

JitEngineDeviceCUDA::JitEngineDeviceCUDA() {
  TIMESCOPE(JitEngineDeviceCUDA, JitEngineDeviceCUDA);
  proteusCuErrCheck(cuInit(0));

  CUdevice CUDev;
  CUcontext CUCtx;

  // Try to get the device from an existing context first.
  CUresult CURes = cuCtxGetCurrent(&CUCtx);
  if (CURes == CUDA_SUCCESS && CUCtx) {
    // There is already an active context, use its device.
    proteusCuErrCheck(cuCtxGetDevice(&CUDev));
  } else {
    // No active context, retain the primary context for device 0.
    // NOTE: We intentionally do not call cuDevicePrimaryCtxRelease.
    // The primary context is shared with the host application and will
    // be cleaned up at process exit. Releasing during static destruction
    // risks ordering issues with other CUDA users.

    // TODO: support device selection.
    proteusCuErrCheck(cuDeviceGet(&CUDev, 0));
    proteusCuErrCheck(cuDevicePrimaryCtxRetain(&CUCtx, CUDev));
    proteusCuErrCheck(cuCtxSetCurrent(CUCtx));
  }

  int CCMajor, CCMinor;
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  DeviceArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

  PROTEUS_DBG(Logger::logs("proteus") << "CUDA Arch " << DeviceArch << "\n");

  Dispatch = createDispatcher();
}

std::unique_ptr<MemoryBuffer> JitEngineDeviceCUDA::codegenObject(
    Module &M, SmallPtrSetImpl<void *> &GlobalLinkedBinaries,
    const CodeGenerationConfig &CGConfig) {
  TIMESCOPE(JitEngineDeviceCUDA, codegenObject);
  return proteus::codegenObject(M, DeviceArch, GlobalLinkedBinaries,
                                CGConfig.codeGenOption());
}

std::unique_ptr<Dispatcher> JitEngineDeviceCUDA::createDispatcher() {
  return std::make_unique<DispatcherCUDA>("JitEngineDevice", *this);
}
