//===-- JitEngineDevice.cpp -- Base JIT Engine Device header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICE_H
#define PROTEUS_JITENGINEDEVICE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Init.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/Cloning.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/CompilerAsync.h"
#include "proteus/impl/CompilerSync.h"
#include "proteus/impl/CoreDevice.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/JitEngine.h"
#include "proteus/impl/JitEngineInfoRegistry.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/TransformLambdaSpecialization.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace proteus {

using namespace llvm;

struct FatbinWrapperT {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void **PrelinkedFatbins;
};

class BinaryInfo {
private:
  FatbinWrapperT *FatbinWrapper;
  std::unique_ptr<LLVMContext> Ctx;
  SmallVector<std::string> LinkedModuleIds;
  Module *LinkedModule;
  std::optional<SmallVector<std::unique_ptr<Module>>> ExtractedModules;
  std::optional<HashT> ExtractedModuleHash;
  std::optional<CallGraph> ModuleCallGraph;
  std::unique_ptr<MemoryBuffer> DeviceBinary;
  std::unordered_map<std::string, GlobalVarInfo> VarNameToGlobalInfo;
  std::once_flag Flag;

public:
  BinaryInfo() = default;
  BinaryInfo(FatbinWrapperT *FatbinWrapper,
             SmallVector<std::string> &&LinkedModuleIds)
      : FatbinWrapper(FatbinWrapper), Ctx(std::make_unique<LLVMContext>()),
        LinkedModuleIds(LinkedModuleIds), LinkedModule(nullptr),
        ExtractedModules(std::nullopt), ModuleCallGraph(std::nullopt),
        DeviceBinary(nullptr) {}

  FatbinWrapperT *getFatbinWrapper() const { return FatbinWrapper; }

  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }

  bool hasLinkedModule() const { return (LinkedModule != nullptr); }
  Module &getLinkedModule() {
    TIMESCOPE(BinaryInfo, getLinkedModule);
    if (!LinkedModule) {
      if (!hasExtractedModules())
        reportFatalError("Expected extracted modules");

      Timer T(Config::get().ProteusEnableTimers);
      // Avoid linking when there's a single module by moving it instead and
      // making sure it's materialized for call graph analysis.
      if (ExtractedModules->size() == 1) {
        LinkedModule = ExtractedModules->front().get();
        if (auto E = LinkedModule->materializeAll())
          reportFatalError("Error materializing " + toString(std::move(E)));
      } else {
        // By the LLVM API, linkModules takes ownership of module pointers in
        // ExtractedModules and returns a new unique ptr to the linked module.
        // We update ExtractedModules to contain and own only the generated
        // LinkedModule.
        auto GeneratedLinkedModule =
            proteus::linkModules(*Ctx, std::move(ExtractedModules.value()));
        SmallVector<std::unique_ptr<Module>> NewExtractedModules;
        NewExtractedModules.emplace_back(std::move(GeneratedLinkedModule));
        setExtractedModules(NewExtractedModules);

        LinkedModule = ExtractedModules->front().get();
      }

      PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                           << "getLinkedModule " << T.elapsed() << " ms\n");
    }

    return *LinkedModule;
  }

  bool hasExtractedModules() const { return ExtractedModules.has_value(); }
  const SmallVector<std::reference_wrapper<Module>>
  getExtractedModules() const {
    // This should be called only once when cloning the kernel module to
    // cache.
    SmallVector<std::reference_wrapper<Module>> ModulesRef;
    for (auto &M : ExtractedModules.value())
      ModulesRef.emplace_back(*M);

    return ModulesRef;
  }
  void setExtractedModules(SmallVector<std::unique_ptr<Module>> &Modules) {
    ExtractedModules = std::move(Modules);
  }

  bool hasModuleHash() const { return ExtractedModuleHash.has_value(); }
  HashT getModuleHash() const {
    if (!hasModuleHash())
      reportFatalError("Expected module hash to be set");

    return ExtractedModuleHash.value();
  }
  void setModuleHash(HashT HashValue) { ExtractedModuleHash = HashValue; }
  void updateModuleHash(HashT HashValue) {
    if (ExtractedModuleHash)
      ExtractedModuleHash = hashCombine(ExtractedModuleHash.value(), HashValue);
    else
      ExtractedModuleHash = HashValue;
  }

  CallGraph &getCallGraph() {
    if (!ModuleCallGraph.has_value()) {
      if (!LinkedModule)
        reportFatalError("Expected non-null linked module");
      ModuleCallGraph.emplace(CallGraph(*LinkedModule));
    }
    return ModuleCallGraph.value();
  }

  bool hasDeviceBinary() { return (DeviceBinary != nullptr); }
  MemoryBufferRef getDeviceBinary() {
    if (!hasDeviceBinary())
      reportFatalError("Expected non-null device binary");
    return DeviceBinary->getMemBufferRef();
  }
  void setDeviceBinary(std::unique_ptr<MemoryBuffer> DeviceBinaryBuffer) {
    DeviceBinary = std::move(DeviceBinaryBuffer);
  }

  void addModuleId(const char *ModuleId) {
    LinkedModuleIds.push_back(ModuleId);
  }

  void insertGlobalVar(const char *VarName, const void *HostAddr,
                       const void *DeviceAddr, uint64_t VarSize) {
    auto KV = VarNameToGlobalInfo.emplace(
        VarName, GlobalVarInfo(HostAddr, DeviceAddr, VarSize));

    auto TraceOut = [&KV]() {
      auto GlobalName = KV.first->first;
      auto &GVI = KV.first->second;

      SmallString<128> S;
      raw_svector_ostream OS(S);
      OS << "[GVarInfo]: " << GlobalName << " HAddr:" << GVI.HostAddr
         << " DevAddr:" << GVI.DevAddr << " VarSize:" << GVI.VarSize << "\n";

      return S;
    };

    if (Config::get().traceSpecializations())
      Logger::trace(TraceOut());
  }

  std::unordered_map<std::string, GlobalVarInfo> &getVarNameToGlobalInfo() {
    return VarNameToGlobalInfo;
  }

  auto &getModuleIds() { return LinkedModuleIds; }
};

class JITKernelInfo {
  std::optional<void *> Kernel;
  std::unique_ptr<LLVMContext> Ctx;
  std::string Name;
  ArrayRef<RuntimeConstantInfo *> RCInfoArray;
  std::optional<std::unique_ptr<Module>> ExtractedModule;
  std::optional<std::unique_ptr<MemoryBuffer>> Bitcode;
  std::optional<std::reference_wrapper<BinaryInfo>> BinInfo;
  std::optional<HashT> StaticHash;
  // LambdaCalleeInfo optionally contains a vector with all annotated
  // LambdaFunctorWrapper annotators and thei unique IDs
  std::optional<SmallVector<uint64_t>> LambdaCalleeInfo;
  std::optional<LambdaCallsiteLocationMap> LambdaCallsiteLocationInfo;

  // Recent specializations keyed by the raw launch inputs that feed the
  // specialization hash, so warm launches skip hashing and cache lookups.
  struct LaunchMemoEntry {
    SmallVector<char, 64> Key;
    void *KernelFunc = nullptr;
  };
  std::array<LaunchMemoEntry, 4> LaunchMemo;
  unsigned NextLaunchMemo = 0;

public:
  JITKernelInfo(void *Kernel, BinaryInfo &BinInfo, char const *Name,
                ArrayRef<RuntimeConstantInfo *> RCInfoArray)
      : Kernel(Kernel), Ctx(std::make_unique<LLVMContext>()), Name(Name),
        RCInfoArray(RCInfoArray), ExtractedModule(std::nullopt),
        Bitcode{std::nullopt}, BinInfo(BinInfo), LambdaCalleeInfo(std::nullopt),
        LambdaCallsiteLocationInfo(std::nullopt) {}

  JITKernelInfo() = default;
  void *getKernel() const {
    assert(Kernel.has_value() && "Expected Kernel is inited");
    return Kernel.value();
  }
  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }
  const std::string &getName() const { return Name; }
  ArrayRef<RuntimeConstantInfo *> getRCInfoArray() const { return RCInfoArray; }
  bool hasModule() const { return ExtractedModule.has_value(); }
  Module &getModule() const { return *ExtractedModule->get(); }
  BinaryInfo &getBinaryInfo() const { return BinInfo.value(); }
  void setModule(std::unique_ptr<llvm::Module> Mod) {
    ExtractedModule = std::move(Mod);
  }

  bool hasBitcode() { return Bitcode.has_value(); }
  void setBitcode(std::unique_ptr<MemoryBuffer> ExtractedBitcode) {
    Bitcode = std::move(ExtractedBitcode);
  }
  MemoryBufferRef getBitcode() { return Bitcode.value()->getMemBufferRef(); }

  bool hasStaticHash() const { return StaticHash.has_value(); }
  const HashT getStaticHash() const { return StaticHash.value(); }
  void createStaticHash(HashT ModuleHash) {
    StaticHash = hash(Name);
    StaticHash = hashCombine(StaticHash.value(), ModuleHash);
  }

  void *lookupLaunchMemo(ArrayRef<char> Key) const {
    for (const auto &Entry : LaunchMemo)
      if (Entry.KernelFunc && ArrayRef<char>(Entry.Key) == Key)
        return Entry.KernelFunc;
    return nullptr;
  }

  void insertLaunchMemo(ArrayRef<char> Key, void *KernelFunc) {
    auto &Entry = LaunchMemo[NextLaunchMemo];
    NextLaunchMemo = (NextLaunchMemo + 1) % LaunchMemo.size();
    Entry.Key.assign(Key.begin(), Key.end());
    Entry.KernelFunc = KernelFunc;
  }

  bool hasLambdaCalleeInfo() { return LambdaCalleeInfo.has_value(); }
  const auto &getLambdaCalleeInfo() { return LambdaCalleeInfo.value(); }
  void setLambdaCalleeInfo(SmallVector<uint64_t> &&LambdaInfo) {
    LambdaCalleeInfo = std::move(LambdaInfo);
  }

  bool hasLambdaCallsiteLocationInfo() const {
    return LambdaCallsiteLocationInfo.has_value();
  }
  const auto &getLambdaCallsiteLocationInfo() const {
    return LambdaCallsiteLocationInfo.value();
  }
  void setLambdaCallsiteLocationInfo(LambdaCallsiteLocationMap &&Info) {
    LambdaCallsiteLocationInfo = std::move(Info);
  }
  void addLambdaCallsiteLocation(uint64_t LambdaID, uint32_t CallsiteIndex,
                                 LambdaKernelArgLocation Location) {
    if (!LambdaCallsiteLocationInfo)
      LambdaCallsiteLocationInfo.emplace();

    auto &Callsites = (*LambdaCallsiteLocationInfo)[LambdaID];
    auto It = Callsites.find(CallsiteIndex);
    if (It != Callsites.end()) {
      if (It->second.KernelArgIndex != Location.KernelArgIndex ||
          It->second.Offset != Location.Offset ||
          It->second.StorageType != Location.StorageType)
        reportFatalError("Conflicting lambda callsite location for lambda " +
                         std::to_string(LambdaID));
      return;
    }
    Callsites[CallsiteIndex] = Location;
  }
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : public JitEngine {
public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t
  compileAndRun(JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim,
                void **KernelArgs, uint64_t ShmemSize,
                typename DeviceTraits<ImplT>::DeviceStream_t Stream);

  std::pair<std::unique_ptr<Module>, std::unique_ptr<MemoryBuffer>>
  extractKernelModule(BinaryInfo &BinInfo, StringRef KernelName,
                      LLVMContext &Ctx) {
    TIMESCOPE(JitEngineDevice, extractKernelModule);
    std::unique_ptr<Module> KernelModule =
        static_cast<ImplT &>(*this).tryExtractKernelModule(BinInfo, KernelName,
                                                           Ctx);
    std::unique_ptr<MemoryBuffer> Bitcode = nullptr;

    // If there is no ready-made kernel module from AOT, extract per-TU or the
    // single linked module and clone the kernel module.
    if (!KernelModule) {
      Timer T(Config::get().ProteusEnableTimers);
      if (!BinInfo.hasExtractedModules())
        static_cast<ImplT &>(*this).extractModules(BinInfo);

      std::unique_ptr<Module> KernelModuleTmp = nullptr;
      switch (Config::get().ProteusKernelClone) {
      case proteus::KernelCloneOption::LinkClonePrune: {
        auto &LinkedModule = BinInfo.getLinkedModule();
        KernelModule = llvm::CloneModule(LinkedModule);
        break;
      }
      case proteus::KernelCloneOption::LinkCloneLight: {
        auto &LinkedModule = BinInfo.getLinkedModule();
        KernelModule =
            proteus::cloneKernelFromModules({LinkedModule}, KernelName);
        break;
      }
      case proteus::KernelCloneOption::CrossClone: {
        KernelModule = proteus::cloneKernelFromModules(
            BinInfo.getExtractedModules(), KernelName);
        break;
      }
      default:
        reportFatalError("Unsupported kernel cloning option");
      }

      PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                           << "Cloning "
                           << toString(Config::get().ProteusKernelClone) << " "
                           << T.elapsed() << " ms\n");
    }

    // Internalize and cleanup to simplify the module and prepare it for
    // optimization.
    internalize(*KernelModule, KernelName);
    proteus::runCleanupPassPipeline(*KernelModule);

    // If the module is not in the provided context due to cloning, roundtrip
    // it using bitcode. Re-use the roundtrip bitcode to return it.
    if (&KernelModule->getContext() != &Ctx) {
      SmallVector<char> CloneBuffer;
      raw_svector_ostream OS(CloneBuffer);
      WriteBitcodeToFile(*KernelModule, OS);
      StringRef CloneStr = StringRef(CloneBuffer.data(), CloneBuffer.size());
      auto ExpectedKernelModule =
          parseBitcodeFile(MemoryBufferRef{CloneStr, KernelName}, Ctx);
      if (auto E = ExpectedKernelModule.takeError())
        reportFatalError("Error parsing bitcode: " + toString(std::move(E)));

      KernelModule = std::move(*ExpectedKernelModule);
      Bitcode = MemoryBuffer::getMemBufferCopy(CloneStr);
    } else {
      // Parse the kernel module to create the bitcode since it has not been
      // created by roundtripping.
      SmallVector<char> BitcodeBuffer;
      raw_svector_ostream OS(BitcodeBuffer);
      WriteBitcodeToFile(*KernelModule, OS);
      auto BitcodeStr = StringRef{BitcodeBuffer.data(), BitcodeBuffer.size()};
      Bitcode = MemoryBuffer::getMemBufferCopy(BitcodeStr);
    }

    return std::make_pair(std::move(KernelModule), std::move(Bitcode));
  }

  void extractModuleAndBitcode(JITKernelInfo &KernelInfo) {
    TIMESCOPE(JitEngineDevice, extractModuleAndBitcode);

    if (KernelInfo.hasModule() && KernelInfo.hasBitcode())
      return;

    if (KernelInfo.hasModule())
      reportFatalError("Unexpected KernelInfo has module but not bitcode");

    if (KernelInfo.hasBitcode())
      reportFatalError("Unexpected KernelInfo has bitcode but not module");

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    Timer T(Config::get().ProteusEnableTimers);
    auto [KernelModule, BitcodeBuffer] = extractKernelModule(
        BinInfo, KernelInfo.getName(), *KernelInfo.getLLVMContext());

    if (!KernelModule)
      reportFatalError("Expected non-null kernel module");
    if (!BitcodeBuffer)
      reportFatalError("Expected non-null kernel bitcode");

    KernelInfo.setModule(std::move(KernelModule));
    KernelInfo.setBitcode(std::move(BitcodeBuffer));
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Extract kernel module " << T.elapsed() << " ms\n");
  }

  Module &getModule(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasModule())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasModule())
      reportFatalError("Expected module in KernelInfo");

    return KernelInfo.getModule();
  }

  MemoryBufferRef getBitcode(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasBitcode())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasBitcode())
      reportFatalError("Expected bitcode in KernelInfo");

    return KernelInfo.getBitcode();
  }

  void
  getLambdaJitValues(SmallVector<uint64_t> &LambdaCalleeInfo,
                     LambdaCallsiteRuntimeConstantsMap &LambdaJitValuesMap) {
    TIMESCOPE(JitEngineDevice, getLambdaJitValues);
    LambdaRegistry &LR = LambdaRegistry::instance();
    const auto &LaunchInfo = LR.peekDeviceLaunchInfo();
    if (!LaunchInfo.CallsiteRuntimeConstants.empty()) {
      LambdaCalleeInfo.assign(LaunchInfo.LambdaCalleeInfo.begin(),
                              LaunchInfo.LambdaCalleeInfo.end());
      LambdaJitValuesMap = LaunchInfo.CallsiteRuntimeConstants;
    }
    LR.clearDeviceLaunchInfo();
  }

  // Serializes the launch inputs that feed the specialization hash. Returns
  // false for inputs that are not plain scalars, which take the hashing path.
  bool buildLaunchMemoKey(JITKernelInfo &KernelInfo, void **KernelArgs,
                          dim3 GridDim, dim3 BlockDim,
                          const CodeGenerationConfig &CGConfig,
                          SmallVectorImpl<char> &Key) {
    auto Append = [&Key](const void *Ptr, size_t Size) {
      Key.append(static_cast<const char *>(Ptr),
                 static_cast<const char *>(Ptr) + Size);
    };

    for (const auto *RCInfo : KernelInfo.getRCInfoArray()) {
      RuntimeConstantType Type = RCInfo->ArgInfo.Type;
      if (!isScalarRuntimeConstantType(Type))
        return false;
      size_t Size = Type == RuntimeConstantType::PTR ? sizeof(void *)
                                                     : getSizeInBytes(Type);
      Append(KernelArgs[RCInfo->ArgInfo.Pos], Size);
    }

    Append(&BlockDim, sizeof(BlockDim));
    if (CGConfig.specializeDims() || CGConfig.specializeDimsRange())
      Append(&GridDim, sizeof(GridDim));

    uint64_t PluginGeneration = getJITPassPluginGeneration();
    Append(&PluginGeneration, sizeof(PluginGeneration));

    const auto &LaunchInfo = LambdaRegistry::instance().peekDeviceLaunchInfo();
    for (const auto &[CallsiteIndex, RCs] :
         LaunchInfo.CallsiteRuntimeConstants) {
      Append(&CallsiteIndex, sizeof(CallsiteIndex));
      for (const auto &RC : RCs) {
        if (!isScalarRuntimeConstantType(RC.Type))
          return false;
        Append(&RC.Type, sizeof(RC.Type));
        Append(&RC.Pos, sizeof(RC.Pos));
        Append(&RC.Offset, sizeof(RC.Offset));
        Append(&RC.Value, sizeof(RC.Value));
      }
    }

    return true;
  }

  void registerVar(void *Handle, const char *VarName, const void *HostAddr,
                   uint64_t VarSize) {
    if (!HandleToBinaryInfo.count(Handle))
      reportFatalError("Expected Handle in map");
    BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

    void *DeviceAddr = resolveDeviceGlobalAddr(HostAddr);
    assert(DeviceAddr &&
           "Expected non-null device address for global variable");

    BinInfo.insertGlobalVar(VarName, HostAddr, DeviceAddr, VarSize);
  }

  void registerLinkedBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                         const char *ModuleId);
  void finalizeRegistration();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        ArrayRef<RuntimeConstantInfo *> RCInfoArray);
  void registerLambdaCallsiteLocation(void *Kernel, uint64_t LambdaID,
                                      uint32_t CallsiteIndex,
                                      uint32_t KernelArgIndex, int64_t Offset,
                                      RuntimeConstantType StorageType);

  std::unordered_map<std::string, FatbinWrapperT *> ModuleIdToFatBinary;
  std::unordered_map<const void *, BinaryInfo> HandleToBinaryInfo;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;

  bool containsJITKernelInfo(const void *Func) {
    return JITKernelInfoMap.contains(Func);
  }

  std::optional<std::reference_wrapper<JITKernelInfo>>
  getJITKernelInfo(const void *Func) {
    auto It = JITKernelInfoMap.find(Func);
    if (It == JITKernelInfoMap.end())
      return std::nullopt;
    return It->second;
  }

  HashT getStaticHash(JITKernelInfo &KernelInfo) {
    if (KernelInfo.hasStaticHash())
      return KernelInfo.getStaticHash();

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    if (BinInfo.hasModuleHash()) {
      KernelInfo.createStaticHash(BinInfo.getModuleHash());
      return KernelInfo.getStaticHash();
    }

    HashT ModuleHash = static_cast<ImplT &>(*this).getModuleHash(BinInfo);

    KernelInfo.createStaticHash(BinInfo.getModuleHash());
    return KernelInfo.getStaticHash();
  }

public:
  StringRef getDeviceArch() const { return DeviceArch; }

protected:
  JitEngineDevice() {
    TIMESCOPE(JitEngineDevice, JitEngineDevice);
    auto &JitEngineInfo = JitEngineInfoRegistry::instance();

    for (auto &[Handle, FatbinInfo] : JitEngineInfo.FatbinaryMap) {
      registerFatBinary(
          Handle, reinterpret_cast<FatbinWrapperT *>(FatbinInfo.FatbinWrapper),
          FatbinInfo.ModuleId);

      for (auto &LinkedBin : FatbinInfo.LinkedBinaries)
        registerLinkedBinary(
            Handle, reinterpret_cast<FatbinWrapperT *>(LinkedBin.FatbinWrapper),
            LinkedBin.ModuleId);

      for (auto &Func : FatbinInfo.Functions)
        registerFunction(Handle, Func.Kernel, Func.KernelName,
                         Func.RCInfoArray);

      for (auto &Var : FatbinInfo.Vars)
        registerVar(Var.Handle, Var.VarName, Var.HostAddr, Var.VarSize);
    }

    finalizeRegistration();

    if (Config::get().ProteusAsyncCompilation)
      AsyncCompiler =
          std::make_unique<CompilerAsync>(Config::get().ProteusAsyncThreads);
  }

  ~JitEngineDevice() {
    // Thread joining is handled by CompilerAsync's shutdown guard to ensure it
    // happens before static objects are destroyed. If this destructor does run,
    // joinAllThreads() is idempotent.
    if (AsyncCompiler)
      AsyncCompiler->joinAllThreads();
  }

  Dispatcher &getDispatcher() {
    if (!Dispatch)
      reportFatalError("Dispatcher has not been created by the engine");
    return *Dispatch;
  }
  std::unique_ptr<Dispatcher> Dispatch;
  std::string DeviceArch;

  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
  DenseMap<const void *, LambdaCallsiteLocationMap>
      PendingLambdaCallsiteLocationInfo;
  std::unique_ptr<CompilerAsync> AsyncCompiler;
};

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE(JitEngineDevice, compileAndRun);

  auto &BinInfo = KernelInfo.getBinaryInfo();
  const auto &CGConfig = Config::get().getCGConfig(KernelInfo.getName());

  // Cache statistics count lookups, so tracing them disables the memo.
  SmallVector<char, 64> MemoKey;
  const bool UseLaunchMemo = !Config::get().traceCacheStats() &&
                             !Config::get().traceKernels() &&
                             buildLaunchMemoKey(KernelInfo, KernelArgs, GridDim,
                                                BlockDim, CGConfig, MemoKey);
  if (UseLaunchMemo) {
    if (void *KernelFunc = KernelInfo.lookupLaunchMemo(MemoKey)) {
      LambdaRegistry::instance().clearDeviceLaunchInfo();
      return launchKernelFunction(
          reinterpret_cast<KernelFunction_t>(KernelFunc), GridDim, BlockDim,
          KernelArgs, ShmemSize, Stream);
    }
  }

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(KernelArgs, KernelInfo.getRCInfoArray());

  SmallVector<uint64_t> LambdaCalleeInfoToSpecialize;
  LambdaCallsiteRuntimeConstantsMap LambdaJitValuesMap;
  getLambdaJitValues(LambdaCalleeInfoToSpecialize, LambdaJitValuesMap);
  // Determine the hash based on dimension specialization.  If we do not
  // specialize IR based on grid dimensions, avoid hashing on those to
  // eliminate repeated compilation overhead.
  HashT HashValue =
      hash(getStaticHash(KernelInfo), RCVec, LambdaJitValuesMap, BlockDim.x,
           BlockDim.y, BlockDim.z, hashCodeGenConfig(CGConfig),
           hashRuntimeSpecializationConfig(CGConfig));
  if (CGConfig.specializeDims() || CGConfig.specializeDimsRange())
    HashValue = hash(HashValue, GridDim.x, GridDim.y, GridDim.z);

  Dispatcher &Dispatch = getDispatcher();

  auto Launch = [&](void *KernelFunc) {
    if (UseLaunchMemo)
      KernelInfo.insertLaunchMemo(MemoKey, KernelFunc);
    return static_cast<DeviceError_t>(Dispatch
                                          .launch(KernelFunc, GridDim, BlockDim,
                                                  KernelArgs, ShmemSize, Stream)
                                          .Ret);
  };

  if (void *KernelFunc =
          Dispatch.lookupFunction(KernelInfo.getName(), HashValue))
    return Launch(KernelFunc);

  // NOTE: we don't need a suffix to differentiate kernels, each
  // specialization will be in its own module uniquely identify by HashValue.
  // It exists only for debugging purposes to verify that the jitted kernel
  // executes.
  KernelName Name{KernelInfo.getName(), HashValue};

  auto LoadKernel = [&](CompiledLibrary &Library) {
    Library.VarNameToGlobalInfo = &BinInfo.getVarNameToGlobalInfo();
    Library.RelinkGlobalsByCopy = Config::get().ProteusRelinkGlobalsByCopy;
    return Dispatch.insertFunction(Name, HashValue, Library);
  };

  if (auto CompiledLib = Dispatch.lookupCompiledLibrary(HashValue))
    return Launch(LoadKernel(*CompiledLib));

  MemoryBufferRef KernelBitcode = getBitcode(KernelInfo);
  std::unique_ptr<MemoryBuffer> ObjBuf = nullptr;
  const LambdaCallsiteRuntimeConstantsMap EmptyLambdaCallsiteRuntimeConstants;
  const auto &LambdaCallsiteRuntimeConstants =
      LambdaJitValuesMap.empty() ? EmptyLambdaCallsiteRuntimeConstants
                                 : LambdaJitValuesMap;

  auto CreateTask = [&]() {
    return CompilationTask{
        Dispatch,
        KernelBitcode,
        HashValue,
        Name,
        BlockDim,
        GridDim,
        RCVec,
        LambdaCalleeInfoToSpecialize,
        LambdaCallsiteRuntimeConstants,
        BinInfo.getVarNameToGlobalInfo(),
        /*CodeGenConfig */ CGConfig,
        /*DumpIR*/ Config::get().ProteusDumpLLVMIR,
        /*RelinkGlobalsByCopy*/ Config::get().ProteusRelinkGlobalsByCopy};
  };

  if (Config::get().ProteusAsyncCompilation) {
    //  If there is no compilation pending for the specialization, post the
    //  compilation task to the compiler.
    if (!AsyncCompiler->isCompilationPending(HashValue)) {
      PROTEUS_DBG(Logger::logs("proteus") << "Compile async for HashValue "
                                          << HashValue.toString() << "\n");

      AsyncCompiler->compile(CreateTask());
    }

    // Compilation is pending, try to get the compilation result buffer. If
    // buffer is null, compilation is not done, so execute the AOT version
    // directly.
    ObjBuf = AsyncCompiler->takeCompilationResult(
        HashValue, Config::get().ProteusAsyncTestBlocking);
    if (!ObjBuf) {
      return launchKernelDirect(KernelInfo.getKernel(), GridDim, BlockDim,
                                KernelArgs, ShmemSize, Stream);
    }
  } else {
    // Process through synchronous compilation.
    ObjBuf = CompilerSync::instance().compile(CreateTask());
  }

  if (!ObjBuf)
    reportFatalError("Expected non-null object");

  Dispatch.registerObject(HashValue, ObjBuf->getMemBufferRef());

  CompiledLibrary Library{std::move(ObjBuf)};
  Library.GlobalsRelinked = true;

  return Launch(LoadKernel(Library));
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFatBinary(void *Handle,
                                               FatbinWrapperT *FatbinWrapper,
                                               const char *ModuleId) {
  TIMESCOPE(JitEngineDevice, registerFatBinary);
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register fatbinary Handle " << Handle << " FatbinWrapper "
              << FatbinWrapper << " Binary " << (void *)FatbinWrapper->Binary
              << " ModuleId " << ModuleId << "\n");
  if (FatbinWrapper->PrelinkedFatbins) {
    // This is RDC compilation, just insert the FatbinWrapper and ignore the
    // ModuleId coming from the link.stub.
    HandleToBinaryInfo.try_emplace(Handle, FatbinWrapper,
                                   SmallVector<std::string>{});

    // Initialize GlobalLinkedBinaries with prelinked fatbins.
    void *Ptr = FatbinWrapper->PrelinkedFatbins[0];
    for (int I = 0; Ptr != nullptr;
         ++I, Ptr = FatbinWrapper->PrelinkedFatbins[I]) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "I " << I << " PrelinkedFatbin " << Ptr << "\n");
      GlobalLinkedBinaries.insert(Ptr);
    }
  } else {
    // This is non-RDC compilation, associate the ModuleId of the JIT bitcode
    // in the module with the FatbinWrapper.
    ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
    HandleToBinaryInfo.try_emplace(Handle, FatbinWrapper,
                                   SmallVector<std::string>{ModuleId});
  }
}

template <typename ImplT> void JitEngineDevice<ImplT>::finalizeRegistration() {
  TIMESCOPE(JitEngineDevice, finalizeRegistration);
  PROTEUS_DBG(Logger::logs("proteus") << "Finalize registration\n");
  // Erase linked binaries for which we have LLVM IR code, those binaries are
  // stored in the ModuleIdToFatBinary map.
  for (auto &[ModuleId, FatbinWrapper] : ModuleIdToFatBinary)
    GlobalLinkedBinaries.erase((void *)FatbinWrapper->Binary);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFunction(
    void *Handle, void *Kernel, char *KernelName,
    ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
  PROTEUS_DBG(Logger::logs("proteus") << "Register function " << Kernel
                                      << " To Handle " << Handle << "\n");
  // NOTE: HIP RDC might call multiple times the registerFunction for the same
  // kernel, which has weak linkage, when it comes from different translation
  // units. Either the first or the second call can prevail and should be
  // equivalent. We let the first one prevail.
  if (JITKernelInfoMap.contains(Kernel)) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "Warning: duplicate register function for kernel " +
                       std::string(KernelName)
                << "\n");
    return;
  }

  if (!HandleToBinaryInfo.count(Handle))
    reportFatalError("Expected Handle in map");
  BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

  PROTEUS_DBG(Logger::logs("proteus")
              << "Register function  " << KernelName << " with binary handle "
              << Handle << "\n");

  JITKernelInfoMap[Kernel] =
      JITKernelInfo{Kernel, BinInfo, KernelName, RCInfoArray};
  auto PendingIt = PendingLambdaCallsiteLocationInfo.find(Kernel);
  if (PendingIt != PendingLambdaCallsiteLocationInfo.end()) {
    JITKernelInfoMap[Kernel].setLambdaCallsiteLocationInfo(
        std::move(PendingIt->second));
    PendingLambdaCallsiteLocationInfo.erase(PendingIt);
  }
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLambdaCallsiteLocation(
    void *Kernel, uint64_t LambdaID, uint32_t CallsiteIndex,
    uint32_t KernelArgIndex, int64_t Offset, RuntimeConstantType StorageType) {
  LambdaKernelArgLocation Location{KernelArgIndex, Offset, StorageType};
  auto It = JITKernelInfoMap.find(Kernel);
  if (It == JITKernelInfoMap.end()) {
    PendingLambdaCallsiteLocationInfo[Kernel][LambdaID][CallsiteIndex] =
        Location;
    return;
  }

  It->second.addLambdaCallsiteLocation(LambdaID, CallsiteIndex, Location);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLinkedBinary(void *Handle,
                                                  FatbinWrapperT *FatbinWrapper,
                                                  const char *ModuleId) {
  TIMESCOPE(JitEngineDevice, registerLinkedBinary);
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register linked binary FatBinary " << FatbinWrapper
              << " Binary " << (void *)FatbinWrapper->Binary << " ModuleId "
              << ModuleId << "\n");
  if (!HandleToBinaryInfo.count(Handle))
    reportFatalError("Expected Handle in map");

  HandleToBinaryInfo[Handle].addModuleId(ModuleId);
  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

} // namespace proteus

#endif
