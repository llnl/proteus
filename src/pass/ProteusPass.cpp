//===-- ProteusPass.cpp -- Extact code/runtime info for Proteus JIT --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libProteusPass.dylib -legacy-proteus-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libProteusPass.dylib -passes="proteus-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include "AnnotationHandler.h"
#include "Helpers.h"
#include "KernelArgVisitor.h"
#include "LambdaAnalysis.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRFunction.h"
#include "proteus/impl/Cloning.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/ELF.h>
#include <llvm/Passes/PassBuilder.h>
// Upstream LLVM 22.1.0 moved PassPlugin.h from llvm/Passes to llvm/Plugins,
// while ROCm-packaged LLVM still uses the older include path.
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#elif __has_include(<llvm/Passes/PassPlugin.h>)
#include <llvm/Passes/PassPlugin.h>
#else
#error "Cannot find LLVM PassPlugin.h"
#endif
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileSystem/UniqueID.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <cstdio>
#include <filesystem>
#include <queue>
#include <string>

using namespace llvm;
using namespace proteus;

//-----------------------------------------------------------------------------
// ProteusPass implementation
//-----------------------------------------------------------------------------
namespace {
static cl::opt<bool> ForceProteusAnnotateAll(
    "force-proteus-jit-annotate-all",
    cl::desc("Apply the 'jit' annotation on all GPU kernels"), cl::init(false));

namespace helpers {
#if defined(_WIN32)
#include <tlhelp32.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

static uint32_t getParentProcessId() {
#if defined(_WIN32)
  DWORD pid = GetCurrentProcessId();
  uint32_t ppid = 0;
  HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (h == INVALID_HANDLE_VALUE)
    return 0;
  PROCESSENTRY32 pe;
  pe.dwSize = sizeof(pe);
  if (Process32First(h, &pe)) {
    do {
      if (pe.th32ProcessID == pid) {
        ppid = pe.th32ParentProcessID;
        break;
      }
    } while (Process32Next(h, &pe));
  }
  CloseHandle(h);
  return ppid;
#else
  return getppid();
#endif
}
} // namespace helpers

struct LambdaManifestRecord {
  uint64_t LambdaID;
  uint32_t CallsiteIndex;
  uint32_t KernelArgIndex;
  int64_t Offset;
  RuntimeConstantType StorageType;
};

struct LambdaSchemaRecord {
  uint64_t LambdaID;
  uint32_t Pos;
  uint32_t Offset;
  RuntimeConstantType Type;
};

class ProteusPassImpl {
public:
  ProteusPassImpl(Module &M) : Types(M) {}

  bool run(Module &M, bool IsLTO) {
    AnnotationHandler AnnotHandler{M};
    // We need collect any kernel host stubs to pass to parse annotations, used
    // in forced annotations.
    const auto StubToKernelMap = getKernelHostStubs(M);

    // We force annotate all kernels if the force annotations flag is set and
    // this is not HIP LTO compilation, since LTO constituent modules have
    // already been processed.
    AnnotHandler.parseAnnotations(JitFunctionInfoMap, StubToKernelMap,
                                  (!IsLTO && ForceProteusAnnotateAll));

    // ==================
    // Device compilation
    // ==================
    // For device compilation, just extract the module IR of device code
    // and return.
    if (isDeviceCompilation(M)) {
      DEBUG(Logger::logs("proteus-pass")
            << "=== Pre Original Device Module\n"
            << M << "=== End of Pre Original Device Module\n");
      emitLambdaDeviceCallsiteManifest(M);
      emitJitModuleDevice(M, IsLTO);
      return true;
    }

    // ================
    // Host compilation
    // ================
    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Original Host Module\n"
          << M << "=== End of Pre Original Host Module\n");

    instrumentRegisterLinkedBinary(M);
    instrumentRegisterFatBinary(M);
    instrumentRegisterFatBinaryEnd(M);
    instrumentRegisterVar(M);

    if (hasDeviceLaunchKernelCalls(M)) {
      instrumentLambdaLaunchCallsites(M, StubToKernelMap);
      emitJitLaunchKernelCall(M);
    }

    instrumentRegisterFunction(M);

    // Initialize Proteus CUDA runtime builtins for all CUDA host modules.
    // Both rewritten launches and metadata/global-address queries depend on
    // these runtime function pointers being available at process startup.
    if (isCUDAModule(M))
      emitProteusCUDARuntimeBuiltinsInit(M);

    SmallVector<decltype(JitFunctionInfoMap)::value_type *, 16> JitWorkList;
    for (auto &JFI : JitFunctionInfoMap) {
      Function *JITFn = JFI.first;
      DEBUG(Logger::logs("proteus-pass")
            << "Processing JIT Function " << JITFn->getName() << "\n");
      // Skip host device stubs coming from kernel annotations.
      if (isDeviceKernelHostStub(StubToKernelMap, *JITFn))
        continue;

      JitWorkList.push_back(&JFI);
    }

    // IMPORTANT: A JIT function is rewritten into its dispatch stub before the
    // module of any JIT function that contains it is extracted, so the
    // enclosing module carries the nested dispatch and the inner region
    // specializes on its own runtime constants. Ordering innermost-first is
    // what makes that happen; the stub's mutable bookkeeping globals are
    // cloned by definition (see emitJitModuleHost) so the cloned IR stays
    // valid. See unit tests lambda_def_register_once and lambda_nested.
    for (auto *JFI : sortJitWorkListInnermostFirst(JitWorkList)) {
      emitJitModuleHost(M, *JFI);
      emitJitEntryCall(M, *JFI);
    }

    DEBUG(Logger::logs("proteus-pass")
          << "=== Post Original Host Module\n"
          << M << "=== End Post Original Host Module\n");

    if (verifyModule(M, &errs()))
      reportFatalError("Broken original module found, compilation aborted!");

    return true;
  }

private:
  ProteusTypes Types;

  MapVector<Function *, JitFunctionInfo> JitFunctionInfoMap;
  SmallPtrSet<Function *, 16> ModuleDeviceKernels;
  DenseMap<Value *, Function *> LambdaRegistrationHelpers;

  void runCleanupPassPipeline(Module &M) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager Passes;
    Passes.addPass(GlobalOptPass());
    Passes.addPass(GlobalDCEPass());
    Passes.addPass(StripDeadDebugInfoPass());
    Passes.addPass(StripDeadPrototypesPass());

    Passes.run(M, MAM);

    proteus::pruneDanglingNVVMAnnotations(M);

    // Keep line-table debug info so that recorded device modules retain the
    // kernel's source file and line information.
    stripNonLineTableDebugInfo(M);
  }

  void runOptimizationPassPipeline(Module &M) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager Passes =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    Passes.run(M, MAM);
  }

  static uint64_t getInstructionOrder(const Instruction *I) {
    uint64_t Order = 0;
    for (const BasicBlock &BB : *I->getFunction()) {
      for (const Instruction &Cur : BB) {
        if (&Cur == I)
          return Order;
        ++Order;
      }
    }
    reportFatalError("Instruction not found in parent function");
  }

  static void
  collectWrapperCallsites(Function &FunctorOperatorFunction,
                          SmallVectorImpl<CallBase *> &CBToAnalyze) {
    for (auto *U : FunctorOperatorFunction.users()) {
      if (auto *CB = dyn_cast<CallBase>(U))
        CBToAnalyze.push_back(CB);
    }

    llvm::sort(CBToAnalyze, [](const CallBase *L, const CallBase *R) {
      const Function *LF = L->getFunction();
      const Function *RF = R->getFunction();
      if (LF != RF)
        return LF->getName() < RF->getName();
      return getInstructionOrder(L) < getInstructionOrder(R);
    });
  }

  SmallString<64> getUniqueLambdaManifestFilename(Module &M) {
    auto TmpPath = std::filesystem::temp_directory_path();

    return {TmpPath.string(),
            "/",
            "proteus-lambda-manifest-",
            std::to_string(helpers::getParentProcessId()),
            "-",
            getUniqueFileID(M),
            ".json"};
  }

  static llvm::StringRef getGlobalString(const llvm::GlobalVariable &GV) {
    if (!GV.hasInitializer())
      return {};

    const llvm::Constant *Init = GV.getInitializer();
    if (auto *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(Init)) {
      if (CDS->isCString())
        return CDS->getAsCString();
      if (CDS->isString())
        return CDS->getAsString();
    }

    if (auto *CDA = llvm::dyn_cast<llvm::ConstantDataArray>(Init)) {
      if (CDA->isCString())
        return CDA->getAsCString();
      if (CDA->isString())
        return CDA->getAsString();
    }

    return {};
  }

  void createLambdaDeviceManifestFile(
      Module &M,
      const StringMap<SmallVector<LambdaManifestRecord, 4>> &KernelManifest) {
    if (KernelManifest.empty())
      return;

    SmallString<64> UniqueFilename = getUniqueLambdaManifestFilename(M);

    int FD = -1;
    std::error_code EC = sys::fs::openFileForWrite(UniqueFilename, FD);
    if (EC)
      reportFatalError("Error creating lambda manifest file: " +
                       UniqueFilename.str().str() + ", error: " + EC.message());
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    if (OS.has_error())
      reportFatalError(
          "Error opening lambda manifest file: " + UniqueFilename.str().str() +
          ", error: " + OS.error().message());

    json::Object ManifestInfo;
    json::Array KernelArray;
    for (const auto &KV : KernelManifest) {
      json::Object KernelObject;
      KernelObject["symbol"] = KV.getKey().str();

      json::Array CallsiteArray;
      for (const auto &Record : KV.getValue()) {
        json::Object CallsiteObject;
        CallsiteObject["lambda-id"] = static_cast<int64_t>(Record.LambdaID);
        CallsiteObject["callsite-index"] =
            static_cast<int64_t>(Record.CallsiteIndex);
        CallsiteObject["kernel-arg"] =
            static_cast<int64_t>(Record.KernelArgIndex);
        CallsiteObject["offset"] = Record.Offset;
        CallsiteObject["storage-type"] =
            static_cast<int64_t>(Record.StorageType);
        CallsiteArray.push_back(std::move(CallsiteObject));
      }

      KernelObject["lambda-callsites"] = std::move(CallsiteArray);
      KernelArray.push_back(std::move(KernelObject));
    }

    ManifestInfo["manifest"] = std::move(KernelArray);
    OS << formatv("{0:2}", json::Value(std::move(ManifestInfo)));
    OS.close();
  }

  StringMap<SmallVector<LambdaManifestRecord, 4>>
  parseLambdaManifestFile(Module &M) {
    StringMap<SmallVector<LambdaManifestRecord, 4>> KernelManifest;
    SmallString<64> UniqueFilename = getUniqueLambdaManifestFilename(M);
    if (!sys::fs::exists(UniqueFilename))
      return KernelManifest;

    auto ErrorOrManifestBuf = MemoryBuffer::getFile(UniqueFilename);
    if (!ErrorOrManifestBuf)
      reportFatalError("Error reading lambda manifest file " +
                       UniqueFilename.str().str());

    std::unique_ptr<MemoryBuffer> ManifestBuf = std::move(*ErrorOrManifestBuf);
    auto ExpectedJsonValue = json::parse(ManifestBuf->getBuffer());
    if (auto E = ExpectedJsonValue.takeError())
      reportFatalError("Failed to parse lambda manifest json: " +
                       toString(std::move(E)));

    json::Value ManifestValue = *ExpectedJsonValue;
    json::Object *Manifest = ManifestValue.getAsObject();
    if (!Manifest)
      reportFatalError("Failed to parse lambda manifest object");

    json::Array *KernelArray = Manifest->getArray("manifest");
    if (!KernelArray)
      reportFatalError("Failed to parse lambda manifest kernel array");

    for (auto &Entry : *KernelArray) {
      json::Object *KernelObject = Entry.getAsObject();
      if (!KernelObject)
        reportFatalError("Failed parsing lambda manifest kernel object");

      auto OptionalKernelSym = KernelObject->getString("symbol");
      if (!OptionalKernelSym)
        reportFatalError("Failed parsing lambda manifest kernel symbol");

      json::Array *CallsiteArray = KernelObject->getArray("lambda-callsites");
      if (!CallsiteArray)
        reportFatalError("Failed parsing lambda manifest callsite array");

      auto &Records = KernelManifest[*OptionalKernelSym];
      for (auto &CallsiteEntry : *CallsiteArray) {
        json::Object *CallsiteObject = CallsiteEntry.getAsObject();
        if (!CallsiteObject)
          reportFatalError("Failed parsing lambda manifest callsite object");

        auto LambdaID = CallsiteObject->getInteger("lambda-id");
        auto CallsiteIndex = CallsiteObject->getInteger("callsite-index");
        auto KernelArgIndex = CallsiteObject->getInteger("kernel-arg");
        auto Offset = CallsiteObject->getInteger("offset");
        auto StorageType = CallsiteObject->getInteger("storage-type");
        if (!LambdaID || !CallsiteIndex || !KernelArgIndex || !Offset ||
            !StorageType)
          reportFatalError("Failed parsing lambda manifest callsite payload");

        Records.push_back(LambdaManifestRecord{
            static_cast<uint64_t>(*LambdaID),
            static_cast<uint32_t>(*CallsiteIndex),
            static_cast<uint32_t>(*KernelArgIndex),
            static_cast<int64_t>(*Offset),
            static_cast<RuntimeConstantType>(*StorageType)});
      }
    }

    std::remove(UniqueFilename.c_str());
    return KernelManifest;
  }

  DenseMap<uint64_t, SmallVector<LambdaSchemaRecord, 4>>
  parseLambdaSchemaMetadata(Module &M) {
    DenseMap<uint64_t, SmallVector<LambdaSchemaRecord, 4>> SchemaMap;
    auto *SchemaMD = M.getNamedMetadata(LambdaSchemaMetadataName);
    if (!SchemaMD)
      return SchemaMap;

    for (MDNode *Node : SchemaMD->operands()) {
      if (!Node || Node->getNumOperands() < 4)
        reportFatalError("Malformed lambda schema metadata");

      auto *LambdaCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(0));
      auto *PosCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(1));
      auto *OffsetCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(2));
      auto *TypeCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(3));
      auto *LambdaCI =
          LambdaCAM ? dyn_cast<ConstantInt>(LambdaCAM->getValue()) : nullptr;
      auto *PosCI =
          PosCAM ? dyn_cast<ConstantInt>(PosCAM->getValue()) : nullptr;
      auto *OffsetCI =
          OffsetCAM ? dyn_cast<ConstantInt>(OffsetCAM->getValue()) : nullptr;
      auto *TypeCI =
          TypeCAM ? dyn_cast<ConstantInt>(TypeCAM->getValue()) : nullptr;
      if (!LambdaCI || !PosCI || !OffsetCI || !TypeCI)
        reportFatalError("Malformed lambda schema metadata payload");

      SchemaMap[LambdaCI->getZExtValue()].push_back(LambdaSchemaRecord{
          LambdaCI->getZExtValue(),
          static_cast<uint32_t>(PosCI->getZExtValue()),
          static_cast<uint32_t>(OffsetCI->getZExtValue()),
          static_cast<RuntimeConstantType>(TypeCI->getSExtValue())});
    }

    for (auto &KV : SchemaMap) {
      llvm::sort(KV.second,
                 [](const LambdaSchemaRecord &L, const LambdaSchemaRecord &R) {
                   return L.Pos < R.Pos;
                 });
    }
    return SchemaMap;
  }

  FunctionCallee getJitRegisterLambdaCallsiteLocationFn(Module &M) {
    FunctionType *FnTy =
        FunctionType::get(Types.VoidTy,
                          {Types.PtrTy, Types.Int64Ty, Types.Int32Ty,
                           Types.Int32Ty, Types.Int64Ty, Types.Int32Ty},
                          /*isVarArg=*/false);
    return M.getOrInsertFunction("__proteus_register_lambda_callsite_location",
                                 FnTy);
  }

  void emitLambdaDeviceCallsiteManifest(Module &M) {
    StringMap<SmallVector<LambdaManifestRecord, 4>> KernelManifest;
    SmallVector<std::pair<Function *, uint64_t>, 16> FunctorOperatorMethods;
    findFunctionsWithU64Metadata(M, "proteus.wrapper_call",
                                 FunctorOperatorMethods);
    uint32_t NextCallsiteIndex = 0;
    // Iterate through user-registered functors wrapping lambdas and analyze
    // their callsites.
    for (auto &[FunctorOperator, FunctorId] : FunctorOperatorMethods) {
      SmallVector<CallBase *> CBToAnalyze;
      collectWrapperCallsites(*FunctorOperator, CBToAnalyze);
      if (CBToAnalyze.empty())
        continue;
      auto optionalMeta =
          getU64Metadata(FunctorOperator, "proteus.functor_size");
      if (!optionalMeta)
        reportFatalError("Internal Proteus Error: Missing metadata on functor "
                         "size.  Please report this bug.");
      DenseMap<CallBase *, LambdaKernelArgAnalysis> CallBaseToArgOffset;
      // If analysis fails we log the error in the above function call, and
      // fallback to unspecialized IR at runtime.
      if (!analyzeLambdaUses(M, CallBaseToArgOffset, CBToAnalyze,
                             *optionalMeta))
        continue;

      for (CallBase *CB : CBToAnalyze) {
        auto It = CallBaseToArgOffset.find(CB);
        if (It == CallBaseToArgOffset.end() || !It->second.KernelFunction)
          reportFatalError("Missing lambda kernel-arg analysis result");

        uint32_t CallsiteIndex = NextCallsiteIndex++;
        setLambdaCallsiteMetadata(*CB, FunctorId, CallsiteIndex);
        auto &Records = KernelManifest[It->second.KernelFunction->getName()];
        Records.push_back(LambdaManifestRecord{
            FunctorId, CallsiteIndex, It->second.KernelArgIndex,
            It->second.Offset,
            It->second.ChangedRCLayout.value_or(RuntimeConstantType::NONE)});
      }
    }

    createLambdaDeviceManifestFile(M, KernelManifest);
  }

  CallBase *getStubLaunchCB(Module &M, Function *Stub) {
    CallBase *LaunchKernelCB = nullptr;
    Function *GPULauncher = M.getFunction(LaunchFunctionName);
    for (auto &BB : *Stub)
      for (auto &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I);
            CB && CB->getCalledFunction() == GPULauncher)
          LaunchKernelCB = CB;
      }
    if (!LaunchKernelCB)
      reportFatalError("Error passing device stub pointer to layout analysis");
    return LaunchKernelCB;
  }

  /// Calling conventions for cuda/hiplaunchkernel can vary slightly.  Sometimes
  /// a ptr to ptr is passed, sometimes a ptr to struct.  We need to check how
  /// the arg blob is created to tell the difference.
  RuntimeConstantType getKernelArgLayout(Module &M, Function *Stub) {
    CallBase *LaunchKernelCB = getStubLaunchCB(M, Stub);
    RuntimeConstantType LayoutType = RuntimeConstantType::NONE;
    SmallVector<std::pair<Value *, size_t>> WorkList;
    Value *KernelArgPack =
        LaunchKernelCB->getArgOperand(LaunchKernelCB->arg_size() - 3);
    WorkList.push_back({KernelArgPack, 0});
    SmallDenseSet<Value *> Discovered;
    while (!WorkList.empty()) {
      auto [V, Depth] = WorkList.back();
      WorkList.pop_back();
      if (Discovered.contains(V))
        continue;
      Discovered.insert(V);
      if (AllocaInst *Alloca = dyn_cast<AllocaInst>(V)) {
        for (auto *Usr : Alloca->users())
          WorkList.push_back({Usr, Depth + 1});
        // Ptr to ptr type pack
        if (Depth == 2 && Alloca->getAllocatedType()->isPointerTy())
          return RuntimeConstantType::PTR;
        if (Depth == 2 && Alloca->getAllocatedType()->isStructTy())
          return RuntimeConstantType::NONE;
      } else if (StoreInst *Store = dyn_cast<StoreInst>(V)) {
        WorkList.push_back({Store->getValueOperand(), Depth + 1});
      }
    }
    return LayoutType;
  }

  FunctionCallee getBeginDeviceLambdaLaunchFn(Module &M) {
    auto *FnTy = FunctionType::get(Types.VoidTy, {}, /*isVarArg=*/false);
    return M.getOrInsertFunction("__proteus_begin_device_lambda_launch", FnTy);
  }

  FunctionCallee getPushDeviceLambdaCallsiteConstantFn(Module &M) {
    auto *FnTy = FunctionType::get(Types.VoidTy,
                                   {Types.Int64Ty, Types.Int32Ty, Types.Int32Ty,
                                    Types.Int32Ty, Types.Int32Ty, Types.PtrTy},
                                   /*isVarArg=*/false);
    return M.getOrInsertFunction(
        "__proteus_push_device_lambda_callsite_constant", FnTy);
  }

  FunctionCallee getRegisterLambdaRegisterFunc(Module &M) {
    auto *FnTy = FunctionType::get(Types.VoidTy, {Types.PtrTy, Types.PtrTy},
                                   /*isVarArg=*/false);
    return M.getOrInsertFunction(
        "__proteus_populate_lambda_registration_function", FnTy);
  }

  FunctionCallee getFinalizeDeviceLambdaLaunchFn(Module &M) {
    auto *FnTy = FunctionType::get(Types.VoidTy, {}, /*isVarArg=*/false);
    return M.getOrInsertFunction("__proteus_finalize_device_lambda_launch",
                                 FnTy);
  }

  Value *materializeDeviceLambdaValuePtr(IRBuilder<> &Builder,
                                         Value *KernelArgs,
                                         const LambdaManifestRecord &Manifest,
                                         const LambdaSchemaRecord &Schema) {
    Value *KernelArgsCast = Builder.CreateBitCast(KernelArgs, Types.PtrTy);
    Value *ArgSlotPtr = Builder.CreateInBoundsGEP(
        Types.PtrTy, KernelArgsCast,
        {Builder.getInt32(static_cast<int32_t>(Manifest.KernelArgIndex))});
    Value *ArgStoragePtr = Builder.CreateLoad(Types.PtrTy, ArgSlotPtr);
    Value *ArgStorageI8 = Builder.CreateBitCast(ArgStoragePtr, Types.PtrTy);

    Value *StorageBase = nullptr;
    if (Manifest.StorageType == RuntimeConstantType::NONE) {
      StorageBase =
          Builder.CreateInBoundsGEP(Builder.getInt8Ty(), ArgStorageI8,
                                    {Builder.getInt64(Manifest.Offset)});
    } else {
      Value *IndirectStoragePtr =
          Builder.CreateInBoundsGEP(Builder.getInt8Ty(), ArgStorageI8,
                                    {Builder.getInt64(Manifest.Offset)});
      if (Manifest.StorageType == RuntimeConstantType::PTR) {
        Value *LoadedPtr = Builder.CreateLoad(Types.PtrTy, IndirectStoragePtr);
        StorageBase = Builder.CreateBitCast(LoadedPtr, Types.PtrTy);
      } else if (Manifest.StorageType == RuntimeConstantType::INT64) {
        Value *LoadedBits =
            Builder.CreateLoad(Types.Int64Ty, IndirectStoragePtr);
        Value *LoadedPtr = Builder.CreateIntToPtr(LoadedBits, Types.PtrTy);
        StorageBase = Builder.CreateBitCast(LoadedPtr, Types.PtrTy);
      } else {
        reportFatalError("Unsupported lambda kernel storage type");
      }
    }

    return Builder.CreateInBoundsGEP(Builder.getInt8Ty(), StorageBase,
                                     {Builder.getInt64(Schema.Offset)});
  }

  /// clang-format off
  /// instrumentLambdaLaunchCallsites creates function calls that ultimately
  /// populate the runtime LambdaRegistry using information from the lambda
  /// manifests.  However, it does so indirectly in the following way: (1) Look
  /// at every host stub in the StubToKernelMap.  If the stub's (key in map)
  /// kernel (value in map)
  ///     contains lambda calls (as indicated by the lambda manifests) identify
  ///     the kernel launch site within the IR of the stub
  /// (2) Begin building IR for a registration function for that GPU kernel.
  /// This registration function dynamically
  ///     loads runtime constants from the kernel blob (the void** args passed
  ///     to the launch function). (a) Inject the
  ///     __proteus_begin_device_lambda_launch function call, which tells the
  ///     registry to start
  ///         receiving constants given a kernel
  ///     (b) For each manifest record, inject a call to
  ///     __proteus_push_device_lambda_callsite_constant
  ///         which uses the pointer offset (deduced from analyzeLambdaUses) and
  ///         the runtime constant type (deduced in LambdaAnalysis) to memcpy
  ///         the runtime constant off the kernel blob.  The value pointer
  ///         (indexed from the void** kernel arg pointer) is deduced by
  ///         materializeDeviceLambdaValuePtr.
  ///     (c) Inject __proteus_finalize_device_lambda_launch to finalize
  /// (3) Populate LambdaRegistrationHelpers with a pointer to the new
  /// registration function. The LambdaRegistry holds
  ///     this pointer to the registration function, keyed on the kernel
  ///     address, and calls it during the runtime before lambda specialization
  ///     occurs.
  /// The late propagation of the LambdaRegistry (we populate the registration
  /// functions first, then call them at runtime) occurs because portability
  /// layers like RAJA/Kokkos don't directly use call host stub to launch a
  /// kernel, and instead invoke hipLaunchKernel/cudaLaunchKernel on a
  /// potentially opaque kernel ptr.  The approach outlined above uses the
  /// address of the kernel ptr as the source of truth about runtime constants,
  /// instead of relying on LLVM IR to call the host kernel stub.
  /// clang-format on
  void instrumentLambdaLaunchCallsites(
      Module &M, const DenseMap<Value *, GlobalVariable *> &StubToKernelMap) {
    auto KernelManifest = parseLambdaManifestFile(M);
    auto LambdaSchema = parseLambdaSchemaMetadata(M);
    if (KernelManifest.empty() || LambdaSchema.empty())
      return;
    for (auto It = StubToKernelMap.begin(); It != StubToKernelMap.end(); ++It) {
      auto [Stub, KernelGV] = *It;
      StringRef KernelSym = getGlobalString(*KernelGV);
      if (KernelSym.empty())
        continue;

      auto ManifestIt = KernelManifest.find(KernelSym);
      if (ManifestIt == KernelManifest.end())
        continue;

      Function *StubFn = dyn_cast<Function>(Stub);
      if (!StubFn)
        reportFatalError("Expected stub function for lambda registration "
                         "helper generation");
      auto *LaunchCB = getStubLaunchCB(M, StubFn);

      auto *RegFuncTy = FunctionType::get(Types.VoidTy, {Types.PtrTy}, false);
      // Create a new function that will be called with the kernel args at
      // runtime to populate the LambdaRegistry
      Function *RegFunc = Function::Create(
          RegFuncTy, GlobalValue::InternalLinkage,
          "registerKernel" + demangle(KernelSym) + "LambdaConstants", M);
      RegFunc->arg_begin()->setName("kernel_args");

      BasicBlock *Entry = BasicBlock::Create(M.getContext(), "entry", RegFunc);
      IRBuilder<> Builder(Entry);
      Value *KernelArgs = RegFunc->getArg(0);
      Builder.CreateCall(getBeginDeviceLambdaLaunchFn(M), {});
      RuntimeConstantType KernelArgsStorageType = getKernelArgLayout(M, StubFn);

      for (const auto &Record : ManifestIt->getValue()) {
        auto SchemaIt = LambdaSchema.find(Record.LambdaID);
        if (SchemaIt == LambdaSchema.end())
          continue;

        LambdaManifestRecord EffectiveRecord = Record;
        EffectiveRecord.StorageType = KernelArgsStorageType;
        for (const auto &Schema : SchemaIt->second) {
          // Use the results of the analyzeLambdaUses to deduce the location of
          // the closure ptr within the void** kernel args
          Value *ClosurePtr = materializeDeviceLambdaValuePtr(
              Builder, KernelArgs, EffectiveRecord, Schema);
          Builder.CreateCall(
              getPushDeviceLambdaCallsiteConstantFn(M),
              {Builder.getInt64(Record.LambdaID),
               Builder.getInt32(Record.CallsiteIndex),
               Builder.getInt32(static_cast<int32_t>(Schema.Type)),
               Builder.getInt32(Schema.Pos), Builder.getInt32(Schema.Offset),
               ClosurePtr});
        }
      }
      Builder.CreateCall(getFinalizeDeviceLambdaLaunchFn(M), {});
      Builder.CreateRetVoid();

      LambdaRegistrationHelpers[LaunchCB->getArgOperand(0)] = RegFunc;
    }
  }

  using JitWorkListEntry = decltype(JitFunctionInfoMap)::value_type;

  // JIT functions reachable from F's body, looking through ordinary calls but
  // stopping at another JIT function -- those are the regions nested in F.
  static SmallPtrSet<Function *, 8>
  findNestedJitFunctions(Function &F,
                         const SmallPtrSetImpl<Function *> &JitFunctions) {
    SmallPtrSet<Function *, 8> Nested;
    SmallPtrSet<Function *, 16> Visited;
    SmallVector<Function *, 16> Worklist{&F};

    while (!Worklist.empty()) {
      Function *Current = Worklist.pop_back_val();
      if (!Visited.insert(Current).second)
        continue;

      for (Instruction &I : instructions(*Current)) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;

        Function *Callee = CB->getCalledFunction();
        if (!Callee || Callee->isDeclaration() || Callee == &F)
          continue;

        if (JitFunctions.contains(Callee)) {
          Nested.insert(Callee);
          continue;
        }

        Worklist.push_back(Callee);
      }
    }

    return Nested;
  }

  // Post-order over the nesting relation, so an inner JIT function is always
  // processed before the ones containing it. Recursion through JIT functions
  // has no innermost region, so a cycle keeps its original relative order.
  SmallVector<JitWorkListEntry *, 16> sortJitWorkListInnermostFirst(
      const SmallVectorImpl<JitWorkListEntry *> &JitWorkList) {
    SmallPtrSet<Function *, 16> JitFunctions;
    DenseMap<Function *, JitWorkListEntry *> FnToEntry;
    for (auto *JFI : JitWorkList) {
      JitFunctions.insert(JFI->first);
      FnToEntry[JFI->first] = JFI;
    }

    DenseMap<Function *, SmallPtrSet<Function *, 8>> Nested;
    for (auto *JFI : JitWorkList)
      Nested[JFI->first] = findNestedJitFunctions(*JFI->first, JitFunctions);

    SmallVector<JitWorkListEntry *, 16> Ordered;
    SmallPtrSet<Function *, 16> Done;
    SmallPtrSet<Function *, 16> OnStack;

    std::function<void(Function *)> Visit = [&](Function *F) {
      if (Done.contains(F) || !OnStack.insert(F).second)
        return;

      for (Function *Inner : Nested[F])
        Visit(Inner);

      OnStack.erase(F);
      if (Done.insert(F).second)
        Ordered.push_back(FnToEntry[F]);
    };

    for (auto *JFI : JitWorkList)
      Visit(JFI->first);

    return Ordered;
  }

  void emitJitModuleHost(Module &M,
                         std::pair<Function *, JitFunctionInfo> &JITInfo) {
    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;

    auto JitMod = cloneKernelFromModules(
        {M}, JITFn->getName(), [](const GlobalValue *GV) {
          if (isCoverageGlobal(*GV))
            return true;

          if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV)) {
            // Bookkeeping globals of a nested dispatch stub are per-callsite
            // state, so the enclosing JIT module gets its own definitions. They
            // are mutable and internal, so cloning them as declarations would
            // produce invalid IR.
            if (G->getName().starts_with(".proteus."))
              return true;

            if (!G->isConstant())
              return false;
          }

          return true;
        });

    Function *JitF = JitMod->getFunction(JITFn->getName());
    if (!JitF)
      reportFatalError("Expected JIT function in cloned module");
    JitF->setLinkage(GlobalValue::ExternalLinkage);

    // Internalize functions, besides JIT function, in the module
    // to help global DCE (reduce compilation time), inlining.
    for (Function &JitModF : *JitMod) {
      if (JitModF.isDeclaration())
        continue;

      if (&JitModF == JitF)
        continue;

      // Remove Comdat from functions to be internalized, otherwise they will
      // stay external. The JIT module is not linked with anything else, so
      // Comdat is not needed.
      if (JitModF.hasComdat())
        JitModF.setComdat(nullptr);

      // Internalize other functions in the module.
      JitModF.setLinkage(GlobalValue::InternalLinkage);
    }

    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Passes Host JIT Module\n"
          << *JitMod << "=== End of Pre Passes Host JIT Module\n");

    // Run a global DCE pass and O3 on the JIT module IR to remove unnecessary
    // symbols and reduce the IR to JIT at runtime.
    runCleanupPassPipeline(*JitMod);
    runOptimizationPassPipeline(*JitMod);

    // TODO: Do we want to keep debug info to use for GDB/LLDB
    // interfaces for debugging jitted code?
    StripDebugInfo(*JitMod);

    // Add metadata for the JIT function to store the argument positions for
    // runtime constants.
    emitJitFunctionArgMetadata(*JitMod, JFI, *JitF);

    if (verifyModule(*JitMod, &errs()))
      reportFatalError("Broken JIT module found, compilation aborted!");

    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    DEBUG(Logger::logs("proteus-pass")
          << "=== Final Host JIT Module\n"
          << *JitMod << "=== End of Final Host JIT Module\n");
  }

  // Returns true for globals that are not needed when cloning to extract a
  // module.
  static bool isCoverageGlobal(const GlobalValue &G) {
    StringRef Name = G.getName();
    return Name.starts_with("__llvm_gcov_") ||
           Name.starts_with("__llvm_prf_") || Name.starts_with("__profc_") ||
           Name.starts_with("__profd_") || Name.starts_with("__llvm_cov");
  }

  static bool skipGlobal(const GlobalValue &G) {
    if (G.getName().starts_with("_jit_bitcode") ||
        G.getName().starts_with("__clang_gpu_used_external") ||
        G.getName().starts_with("__hip_cuid") ||
        G.getName().starts_with("llvm.used") ||
        G.getName().starts_with("llvm.compiler.used") ||
        G.getName().starts_with("llvm.global.annotations") ||
        isCoverageGlobal(G))
      return true;

    return false;
  }

  // Emit a uniquely named global variable in a corresponding section that
  // contains the embedded bitcode module.
  void emitModuleDevice(Module &M, Module &EmbedM, StringRef Id,
                        bool HasSourceFileID) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(EmbedM, OS);

    HashT HashValue = hash(StringRef{Bitcode.data(), Bitcode.size()});

    std::string GVName = "_jit_bitcode_" + Id.str() +
                         (HasSourceFileID ? ("_" + getUniqueFileID(M)) : "");
    //  NOTE: HIP compilation supports custom section in the binary to store
    //  the IR. CUDA does not, hence we parse the IR by reading the global
    //  from the device memory.
    Constant *JitModule = ConstantDataArray::get(
        M.getContext(),
        ArrayRef<uint8_t>((const uint8_t *)Bitcode.data(), Bitcode.size()));
    if (M.getNamedGlobal(GVName))
      reportFatalError("Expected unique name for jit module global variable " +
                       GVName);
    auto *GV =
        new GlobalVariable(M, JitModule->getType(), /* isConstant */ true,
                           GlobalValue::ExternalLinkage, JitModule, GVName);
    appendToUsed(M, {GV});
    // We append the hash value to the section name and retrieve it in HIP JIT
    // compilation to avoid hashing at runtime.
    GV->setSection(".jit.bitcode." + Id.str() + getUniqueModuleId(&M) + "." +
                   HashValue.toString());
    DEBUG(Logger::logs("proteus-pass")
          << "Emit jit bitcode GV " << GVName << "\n");
  }

  void emitLinkedKernelModules(Module &LTOModule) {
    LLVMContext Ctx;
    auto LinkedModule = std::make_unique<Module>("linked.module", Ctx);
    LinkedModule->setSourceFileName("linked.module");
    LinkedModule->setDataLayout(LTOModule.getDataLayout());
    LinkedModule->setTargetTriple(LTOModule.getTargetTriple());
    LinkedModule->setModuleInlineAsm(LTOModule.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18 && LLVM_VERSION_MAJOR < 21
    LinkedModule->IsNewDbgInfoFormat = LTOModule.IsNewDbgInfoFormat;
#endif

    // Gather all extracted modules and the pruned LTO module.
    SmallVector<std::unique_ptr<Module>> LinkedModules;
    StringSet KernelSymbols;

    auto ExtractKernelSymbols = [&LTOModule, &KernelSymbols]() {
      for (auto &F : LTOModule) {
        // Skip functions that are not marked for proteus jit.
        if (!F.getMetadata("proteus.jit"))
          continue;

        // Skip non-kernel device functions, for example, device lambda
        // functions.
        if (!isDeviceKernel(&F))
          continue;

        KernelSymbols.insert(F.getName());
      }
    };

    ExtractKernelSymbols();

    SmallPtrSet<GlobalVariable *, 32> RemoveModuleGVs;
    auto ExtractLinkedModules = [&Ctx, &LTOModule, &LinkedModules,
                                 &RemoveModuleGVs]() {
      StringSet DefSet;
      for (auto &GVar : LTOModule.globals()) {
        if (!GVar.hasName())
          continue;

        if (!GVar.getName().starts_with("_jit_bitcode"))
          continue;

        // Found an extracted bitcode global variable, parse the IR and
        // store the names of definitions to prune the LTO module.
        ConstantDataArray *JitModule =
            dyn_cast<ConstantDataArray>(GVar.getInitializer());
        assert(JitModule && "Expected non-null bitcode for JIT module");
        StringRef BitcodeData = JitModule->getAsString();

        // It is important to preserve the unique global variable name as
        // the module name for the parsed module. It will be used later to
        // create unique symbols for linking modules together.
        MemoryBufferRef MBRef{BitcodeData, GVar.getName()};
        auto ExpectedParsedModule = parseBitcodeFile(MBRef, Ctx);
        if (auto E = ExpectedParsedModule.takeError())
          reportFatalError("Error: " + toString(std::move(E)));
        auto ParsedModule = std::move(*ExpectedParsedModule);
        for (auto &G : ParsedModule->global_values()) {
          if (!G.isDeclaration())
            DefSet.insert(G.getName());
        }

        LinkedModules.push_back(std::move(ParsedModule));

        RemoveModuleGVs.insert(&GVar);
      }

      auto ShouldClone = [&DefSet](const GlobalValue *GV) {
        if (GV->hasName()) {
          if (DefSet.contains(GV->getName()))
            return false;

          if (skipGlobal(*GV))
            return false;
        }

        return true;
      };
      // Create pruned LTO module avoiding unneeded globals and globals in
      // the DefSet. Roundtrip it through bitcode parsing to the top-level
      // context.
      ValueToValueMapTy VMap;
      auto PrunedLTOModule = CloneModule(LTOModule, VMap, ShouldClone);
      for (auto &G : PrunedLTOModule->global_values())
        if (G.hasInternalLinkage())
          G.setLinkage(GlobalValue::ExternalLinkage);

      proteus::pruneDanglingNVVMAnnotations(*PrunedLTOModule);

      // Keep line-table debug info so that recorded device modules retain the
      // kernel's source file and line information.
      stripNonLineTableDebugInfo(*PrunedLTOModule);

      if (verifyModule(*PrunedLTOModule, &errs()))
        reportFatalError(
            "Broken pruned lto module found, compilation aborted!");

      SmallVector<char> Bitcode;
      raw_svector_ostream BOS{Bitcode};
      WriteBitcodeToFile(*PrunedLTOModule, BOS);
      MemoryBufferRef MBRef{StringRef{Bitcode.data(), Bitcode.size()},
                            "pruned.lto.module"};
      auto ExpectedPrunedLTOModuleInCtx = parseBitcodeFile(MBRef, Ctx);
      if (auto E = ExpectedPrunedLTOModuleInCtx.takeError())
        reportFatalError("Error parsing pruned lto module " +
                         toString(std::move(E)));
      LinkedModules.push_back(std::move(*ExpectedPrunedLTOModuleInCtx));
    };

    ExtractLinkedModules();
    // Remove global vars with per-TU modules as they are not needed anymore.
    // The final step in this method will create per-kernel modules.
    for (auto *GV : RemoveModuleGVs) {
      removeFromUsedLists(LTOModule, [GV](Constant *C) {
        if (GV == C)
          return true;

        return false;
      });
      LTOModule.eraseGlobalVariable(GV);
    }

    Linker IRLinker(*LinkedModule);
    for (auto &Mod : LinkedModules) {
      if (IRLinker.linkInModule(std::move(Mod)))
        reportFatalError("Linking failed");
    }

    runCleanupPassPipeline(*LinkedModule);

    const char *EnvValue = std::getenv("PROTEUS_PASS_CREATE_KERNEL_MODULES");
    bool CreateKernelModules =
        (EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : true);

    if (CreateKernelModules) {
      for (auto &Sym : KernelSymbols) {
        auto KernelName = Sym.getKey();

        if (!LinkedModule->getFunction(KernelName))
          reportFatalError("Expected kernel function in linked module");

        auto KernelModule = cloneKernelFromModules({*LinkedModule}, KernelName);
        runCleanupPassPipeline(*KernelModule);

        if (verifyModule(*KernelModule, &errs()))
          reportFatalError(
              "Broken original module found, compilation aborted!");

        emitModuleDevice(LTOModule, *KernelModule, KernelName, false);
      }
    } else {
      emitModuleDevice(LTOModule, *LinkedModule, "lto", false);
    }
  }

  void emitJitModuleDevice(Module &M, bool IsLTO) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);

    // For LTO, supported only in HIP RDC, create the per-kernel linked module
    // AOT.
    if (IsLTO) {
      emitLinkedKernelModules(M);
    } else {
      // Emit the TU-wide extracted module.
      ValueToValueMapTy VMap;
      auto ShouldClone = [](const GlobalValue *G) {
        if (skipGlobal(*G))
          return false;

        return true;
      };
      auto EmitM = CloneModule(M, VMap, ShouldClone);
      runCleanupPassPipeline(*EmitM);
      EmitM->setSourceFileName(getCanonicalSourceFileName(M));

      emitModuleDevice(M, *EmitM, "tu", true);
    }
  }

  void emitJitFunctionArgMetadata(Module &JitMod, JitFunctionInfo &JFI,
                                  Function &JitF) {
    LLVMContext &Ctx = JitMod.getContext();
    SmallVector<Metadata *> ConstArgNos;
    for (auto &RCI : JFI.ConstantArgs) {
      Metadata *Meta = ConstantAsMetadata::get(
          ConstantInt::get(Types.Int32Ty, RCI.ArgInfo.Pos));
      ConstArgNos.push_back(Meta);
    }
    MDNode *Node = MDNode::get(Ctx, ConstArgNos);
    JitF.setMetadata("jit_arg_nos", Node);
  }

  FunctionCallee getJitEntryFn(Module &M) {
    // The prototype is
    // __proteus_entry(char *FnName,
    //                 char *IR,
    //                 int IRSize,
    //                 void **Args,
    //                 RuntimeConstantInfo **RCInfoArrayPtr,
    //                 int32_t NumRCs,
    //                 uint64_t *FunctorIdOpt)

    FunctionType *JitEntryFnTy =
        FunctionType::get(Types.PtrTy,
                          {Types.PtrTy, Types.PtrTy, Types.Int32Ty, Types.PtrTy,
                           Types.PtrTy, Types.Int32Ty, Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitEntryFn =
        M.getOrInsertFunction("__proteus_entry", JitEntryFnTy);

    return JitEntryFn;
  }

  FunctionCallee getProteusCreateRuntimeConstantInfoScalarFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_scalar", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty);
  }

  FunctionCallee
  getProteusCreateRuntimeConstantInfoArrayConstSizeFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             int32_t NumElts,
    //             RuntimeConstantType EltType (int32_t))
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_array_const_size", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty);
  }

  FunctionCallee
  getProteusCreateRuntimeConstantInfoArrayRunConstSizeFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             RuntimeConstantType EltType (int32_t),
    //             RuntimeConstantType NumEltsType (int32_t),
    //             int32_t NumEltsPos)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_array_runconst_size",
        Types.PtrTy, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty,
        Types.Int32Ty);
  }

  FunctionCallee getProteusCreateRuntimeConstantInfoObjectFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info_object(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             int32_t Size,
    //             bool PassByValue)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_object", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int1Ty);
  }

  void emitRuntimeConstantInfoScalar(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    FunctionCallee CreateFn = getProteusCreateRuntimeConstantInfoScalarFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Value *RCInfoPtr = Builder.CreateCall(CreateFn, {TypeIdC, ArgNoC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArrayConstSize(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptArrInfo)
      reportFatalError("Expected existing array info");

    FunctionCallee CreateFn =
        getProteusCreateRuntimeConstantInfoArrayConstSizeFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *NumEltsC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->NumElts);
    Constant *EltTypeC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->EltType);
    Value *RCInfoPtr = Builder.CreateCall(CreateFn, {
                                                        TypeIdC,
                                                        ArgNoC,
                                                        NumEltsC,
                                                        EltTypeC,
                                                    });
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArrayRunConstSize(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptArrInfo)
      reportFatalError("Expected array info");

    FunctionCallee CreateFn =
        getProteusCreateRuntimeConstantInfoArrayRunConstSizeFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *EltTypeC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->EltType);
    Constant *NumEltsTypeC = ConstantInt::get(
        Types.Int32Ty, RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
    Constant *NumEltsPosC = ConstantInt::get(
        Types.Int32Ty, RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos);
    Value *RCInfoPtr = Builder.CreateCall(
        CreateFn, {TypeIdC, ArgNoC, EltTypeC, NumEltsTypeC, NumEltsPosC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArray(Module &M,
                                    const RuntimeConstantInfo &RCInfo,
                                    IRBuilderBase &Builder,
                                    ArrayType *RuntimeConstantInfoPtrArrayTy,
                                    GlobalVariable *RuntimeConstantInfoPtrArray,
                                    size_t Idx) {
    if (!RCInfo.OptArrInfo)
      reportFatalError("Expected array info");

    if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
      emitRuntimeConstantInfoArrayRunConstSize(
          M, RCInfo, Builder, RuntimeConstantInfoPtrArrayTy,
          RuntimeConstantInfoPtrArray, Idx);
    } else {
      emitRuntimeConstantInfoArrayConstSize(M, RCInfo, Builder,
                                            RuntimeConstantInfoPtrArrayTy,
                                            RuntimeConstantInfoPtrArray, Idx);
    }
  }

  void emitRuntimeConstantInfoObject(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptObjInfo)
      reportFatalError("Expected object info");

    FunctionCallee CreateFn = getProteusCreateRuntimeConstantInfoObjectFn(M);

    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *SizeC = ConstantInt::get(Types.Int32Ty, RCInfo.OptObjInfo->Size);
    Constant *PassByValueC =
        ConstantInt::get(Types.Int1Ty, RCInfo.OptObjInfo->PassByValue);

    Value *RCInfoPtr =
        Builder.CreateCall(CreateFn, {TypeIdC, ArgNoC, SizeC, PassByValueC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitJitEntryCall(Module &M,
                        std::pair<Function *, JitFunctionInfo> &JITInfo) {

    Function *JITFn = JITInfo.first;
    // Don't rip out the lambda wrapper internals and add an entry call, instead
    // do it inside the lambda call.
    auto *WrapperMeta = JITFn->getMetadata("proteus.wrapper_call");
    if (WrapperMeta)
      return;
    auto *LambdaMeta = JITFn->getMetadata("proteus.registered_lambda");
    ConstantInt *LambdaId = nullptr;
    if (LambdaMeta && LambdaMeta->getNumOperands() > 0) {
      auto *CAM = dyn_cast<ConstantAsMetadata>(LambdaMeta->getOperand(0));
      LambdaId = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
    }

    JitFunctionInfo &JFI = JITInfo.second;
    size_t NumRuntimeConstants = JFI.ConstantArgs.size();

    FunctionCallee JitEntryFn = getJitEntryFn(M);

    // Replaces jit'ed functions in the original module with stubs to call the
    // runtime entry point that compiles and links.
    // Replace jit'ed function with a stub function.
    std::string FnName = JITFn->getName().str();
    JITFn->setName("");
    Function *StubFn = Function::Create(JITFn->getFunctionType(),
                                        JITFn->getLinkage(), FnName, M);
    // We need copy attributes as the can affect affect the ABI, such as
    // 'byval', 'byref' for parameters.
    StubFn->setAttributes(JITFn->getAttributes());
    JITFn->replaceAllUsesWith(StubFn);
    JITFn->eraseFromParent();

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    auto &Ctx = M.getContext();
    IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", StubFn));
    BasicBlock *InitBlock =
        BasicBlock::Create(Ctx, "init_rtconst_info", StubFn);
    BasicBlock *ContinueBlock = BasicBlock::Create(Ctx, "continue", StubFn);

    // Create a static flag to track if we've initialized the constants
    GlobalVariable *RCInfoInitialized = new GlobalVariable(
        M, Types.Int1Ty, false, GlobalValue::InternalLinkage,
        ConstantInt::getFalse(Ctx), ".proteus.rtconst.info.inited." + FnName);

    // Create a static global array to store the runtime constant pointers.
    ArrayType *RuntimeConstantInfoPtrArrayTy =
        ArrayType::get(Types.PtrTy, JFI.ConstantArgs.size());
    GlobalVariable *RuntimeConstantInfoPtrArray = new GlobalVariable(
        M, RuntimeConstantInfoPtrArrayTy, false, GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(RuntimeConstantInfoPtrArrayTy),
        ".proteus.rtconst.array." + FnName);

    Builder.CreateCondBr(Builder.CreateLoad(Types.Int1Ty, RCInfoInitialized),
                         ContinueBlock, InitBlock);

    // Initialize rt constants block.
    Builder.SetInsertPoint(InitBlock);

    for (size_t I = 0; I < NumRuntimeConstants; ++I) {
      auto &RCInfo = JFI.ConstantArgs[I];

      if (RCInfo.ArgInfo.Type == RuntimeConstantType::ARRAY) {
        emitRuntimeConstantInfoArray(M, RCInfo, Builder,
                                     RuntimeConstantInfoPtrArrayTy,
                                     RuntimeConstantInfoPtrArray, I);
      } else if ((RCInfo.ArgInfo.Type == RuntimeConstantType::STATIC_ARRAY) ||
                 (RCInfo.ArgInfo.Type == RuntimeConstantType::VECTOR)) {
        emitRuntimeConstantInfoArrayConstSize(M, RCInfo, Builder,
                                              RuntimeConstantInfoPtrArrayTy,
                                              RuntimeConstantInfoPtrArray, I);
      } else if (RCInfo.ArgInfo.Type == RuntimeConstantType::OBJECT) {
        emitRuntimeConstantInfoObject(M, RCInfo, Builder,
                                      RuntimeConstantInfoPtrArrayTy,
                                      RuntimeConstantInfoPtrArray, I);
      } else if (isScalarRuntimeConstantType(RCInfo.ArgInfo.Type)) {
        emitRuntimeConstantInfoScalar(M, RCInfo, Builder,
                                      RuntimeConstantInfoPtrArrayTy,
                                      RuntimeConstantInfoPtrArray, I);
      } else {
        reportFatalError("Unsupported runtime constant type " +
                         toString(RCInfo.ArgInfo.Type));
      }
    }
    // Mark runtime constants info as initialized.
    Builder.CreateStore(ConstantInt::getTrue(Ctx), RCInfoInitialized);
    // Branch to continue block.
    Builder.CreateBr(ContinueBlock);

    Builder.SetInsertPoint(ContinueBlock);

    // Create globals for the function name, the IR string.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);
    Value *FunctorIdOpt =
        ConstantPointerNull::get(cast<PointerType>(Types.PtrTy));
    if (LambdaId) {
      auto *FunctorIdGlobal = new GlobalVariable(
          M, Types.Int64Ty, /*isConstant=*/true, GlobalValue::InternalLinkage,
          ConstantInt::get(Types.Int64Ty, LambdaId->getZExtValue()),
          ".proteus.registered_lambda." + FnName);
      FunctorIdOpt = FunctorIdGlobal;
    }

    ArrayType *ArgPtrsTy = ArrayType::get(Types.PtrTy, StubFn->arg_size());
    Value *ArgPtrs = nullptr;
    if (!StubFn->arg_empty()) {
      ArgPtrs = Builder.CreateAlloca(ArgPtrsTy);
      // Create an alloca for each argument to store a pointer to the argument,
      // mimicking how arguments are passed for GPU kernels. This is done so
      // that we have a uniform way to read function/kernel arguments in the
      // Proteus runtime for both host and device code.
      SmallVector<AllocaInst *> ArgPtrAllocas;
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *Alloca = Builder.CreateAlloca(StubFn->getArg(ArgI)->getType());
        ArgPtrAllocas.push_back(Alloca);
      }
      // Store each a pointer to the argument value to each alloca.
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            ArgPtrsTy, ArgPtrs, {Builder.getInt32(0), Builder.getInt32(ArgI)});
        // If the argument has the 'byval' or 'byref' attribute, we store the
        // pointer to the argument value directly in the alloca, otherwise we
        // store the pointer-to-pointer. This is done to conform to the ABI and
        // correctly reconstruct the runtime value.
        if (StubFn->getArg(ArgI)->hasByValAttr() ||
            StubFn->getArg(ArgI)->hasByRefAttr()) {
          Builder.CreateStore(StubFn->getArg(ArgI), GEP);
        } else {
          Builder.CreateStore(StubFn->getArg(ArgI), ArgPtrAllocas[ArgI]);
          Builder.CreateStore(ArgPtrAllocas[ArgI], GEP);
        }
      }
    } else {
      ArgPtrs = Constant::getNullValue(Types.PtrTy);
    }

    auto *JitFnPtr = Builder.CreateCall(
        JitEntryFn,
        {FnNameGlobal, StrIRGlobal, Builder.getInt32(JFI.ModuleIR.size()),
         ArgPtrs, RuntimeConstantInfoPtrArray,
         Builder.getInt32(NumRuntimeConstants), FunctorIdOpt});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *CI = Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);

    // We set param attributes for the call to the function pointer returned by
    // the Proteus JIT runtime since they affect the ABI and codegen.
    for (size_t I = 0; I < StubFn->arg_size(); ++I) {
      auto ParamAttrs = StubFn->getAttributes().getParamAttrs(I);
      for (auto A : ParamAttrs)
        CI->addParamAttr(I, A);
    }

    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(CI);
  }

  Value *getStubGV([[maybe_unused]] Value *Operand) {
    // NOTE: when called by isDeviceKernelHostStub, Operand may not be a global
    // variable point to the stub, so we check and return null in that case.
    Value *V = nullptr;
#if PROTEUS_ENABLE_HIP
    // NOTE: Hip creates a global named after the device kernel function that
    // points to the host kernel stub. Because of this, we need to unpeel this
    // indirection to use the host kernel stub for finding the device kernel
    // function name global.
    GlobalVariable *IndirectGV = dyn_cast<GlobalVariable>(Operand);
    V = IndirectGV ? IndirectGV->getInitializer() : nullptr;
#elif PROTEUS_ENABLE_CUDA
    GlobalValue *DirectGV = dyn_cast<GlobalValue>(Operand);
    V = DirectGV ? DirectGV : nullptr;
#endif

    return V;
  }

  DenseMap<Value *, GlobalVariable *> getKernelHostStubs(Module &M) {
    DenseMap<Value *, GlobalVariable *> StubToKernelMap;
    Function *RegisterFunction = nullptr;

    if (!hasDeviceLaunchKernelCalls(M))
      return StubToKernelMap;

    if (!RegisterFunctionName) {
      reportFatalError("getKernelHostStubs only callable with `EnableHIP or "
                       "EnableCUDA set.");
      return StubToKernelMap;
    }
    RegisterFunction = M.getFunction(RegisterFunctionName);

    if (!RegisterFunction)
      return StubToKernelMap;

    constexpr int StubOperand = 1;
    constexpr int KernelOperand = 2;
    for (User *Usr : RegisterFunction->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        GlobalVariable *GV =
            dyn_cast<GlobalVariable>(CB->getArgOperand(KernelOperand));
        assert(GV && "Expected global variable as kernel name operand");
        Value *Key = getStubGV(CB->getArgOperand(StubOperand));
        assert(Key && "Expected valid kernel stub key");
        StubToKernelMap[Key] = GV;
        DEBUG(Logger::logs("proteus-pass")
              << "StubToKernelMap Key: " << Key->getName() << " -> " << *GV
              << "\n");
      }
    return StubToKernelMap;
  }

  bool hasDeviceLaunchKernelCalls(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      return false;
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);

    if (!LaunchKernelFn)
      return false;

    return true;
  }

  FunctionCallee getJitLaunchKernelFn(Module &M) {
    FunctionType *JitLaunchKernelFnTy = nullptr;

    assert(LaunchFunctionName && "Expected valid launch function name");
    Function *LaunchKernelFn = M.getFunction(LaunchFunctionName);
    assert(LaunchKernelFn && "Expected non-null launch kernel function");

    // The ABI of __proteus_launch_kernel mirrors the device-specific
    // launchKernel. Note the ABI can be different depending on the host
    // architecture.
    JitLaunchKernelFnTy = LaunchKernelFn->getFunctionType();

    if (!JitLaunchKernelFnTy)
      reportFatalError(
          "Expected non-null jit entry function type, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

    FunctionCallee JitLaunchKernelFn =
        M.getOrInsertFunction("__proteus_launch_kernel", JitLaunchKernelFnTy);

    return JitLaunchKernelFn;
  }

  void replaceWithJitLaunchKernel(Module &M, CallBase *LaunchKernelCB) {
    FunctionCallee JitLaunchKernelFn = getJitLaunchKernelFn(M);

    // Insert before the launch kernel call instruction.
    IRBuilder<> Builder(LaunchKernelCB);
    CallBase *CallOrInvoke = nullptr;

    SmallVector<Value *> Args = {LaunchKernelCB->arg_begin(),
                                 LaunchKernelCB->arg_end()};

    if (isa<CallInst>(LaunchKernelCB)) {
      CallOrInvoke = Builder.CreateCall(JitLaunchKernelFn, Args);
    } else if (auto *InvokeI = dyn_cast<InvokeInst>(LaunchKernelCB)) {
      CallOrInvoke =
          Builder.CreateInvoke(JitLaunchKernelFn, InvokeI->getNormalDest(),
                               InvokeI->getUnwindDest(), Args);
    }

    if (!CallOrInvoke)
      reportFatalError(
          "Expected non-null jit launch kernel call or invoke, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

    LaunchKernelCB->replaceAllUsesWith(CallOrInvoke);
    LaunchKernelCB->eraseFromParent();
  }

  void emitJitLaunchKernelCall(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      reportFatalError(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);
    if (!LaunchKernelFn)
      reportFatalError(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

    SmallVector<CallBase *> ToBeReplaced;
    for (User *Usr : LaunchKernelFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        // NOTE: We search for calls to the LaunchKernelFn that directly call
        // the kernel through its global value to replace with JIT kernel
        // entries. For cudaLaunchKernel first operand is the stub function,
        // whereas for hipLaunchKernel it is a global variable that points to
        // the stub function. Hence we use GlobalValue instead of
        // GlobalVaraible.
        // TODO: Instrument for indirect launching.

        ToBeReplaced.push_back(CB);
      }

    for (CallBase *CB : ToBeReplaced)
      replaceWithJitLaunchKernel(M, CB);
  }

  FunctionCallee getJitRegisterFatBinaryFn(Module &M) {
    FunctionType *JitRegisterFatbinaryFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy, Types.PtrTy, Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatbinaryFn = M.getOrInsertFunction(
        "__proteus_register_fatbinary", JitRegisterFatbinaryFnTy);

    return JitRegisterFatbinaryFn;
  }

  void instrumentRegisterFatBinary(Module &M) {
    Function *F = nullptr;

    if (!RegisterFatBinaryName)
      return;

    F = M.getFunction(RegisterFatBinaryName);
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryFn = getJitRegisterFatBinaryFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);

      std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
      DEBUG(Logger::logs("proteus-pass")
                << "Instrument register fatbinary bitcode GV " << GVName
                << "\n";);
      auto *Arg = Builder.CreateGlobalString(GVName);

      Builder.CreateCall(JitRegisterFatBinaryFn, {CB, FatbinWrapper, Arg});
    }
  }

  FunctionCallee getJitRegisterFatBinaryEndFn(Module &M) {
    FunctionType *JitRegisterFatBinaryEndFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatBinaryEndFn = M.getOrInsertFunction(
        "__proteus_register_fatbinary_end", JitRegisterFatBinaryEndFnTy);

    return JitRegisterFatBinaryEndFn;
  }

  void instrumentRegisterFatBinaryEnd(Module &M) {
// This is CUDA specific.
#if !PROTEUS_ENABLE_CUDA
    return;
#endif

    Function *F = M.getFunction("__cudaRegisterFatBinaryEnd");
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryEndFn = getJitRegisterFatBinaryEndFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);
      Builder.CreateCall(JitRegisterFatBinaryEndFn, {FatbinWrapper});
    }
  }

  FunctionCallee getJitRegisterLinkedBinaryFn(Module &M) {
    FunctionType *JitRegisterLinkedBinaryFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy, Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisteLinkedBinaryrFn = M.getOrInsertFunction(
        "__proteus_register_linked_binary", JitRegisterLinkedBinaryFnTy);

    return JitRegisteLinkedBinaryrFn;
  }

  void instrumentRegisterLinkedBinary(Module &M) {
// This is CUDA specific.
#if !PROTEUS_ENABLE_CUDA
    return;
#endif

    // Note: we check for __cuda_fatibn_wrapper to avoid emitting for the
    // link.stub. It's not strictly necessary since this module will not have a
    // device bitcode to pull and we skip at runtime.
    if (!M.getGlobalVariable("__cuda_fatbin_wrapper", /*AllowInternal=*/true)) {
      DEBUG(Logger::logs("proteus-pass")
                << "Skip " << M.getSourceFileName() << "\n";)
      return;
    }

    FunctionCallee JitRegisterLinkedBinaryFn = getJitRegisterLinkedBinaryFn(M);

    for (auto &F : M.getFunctionList()) {
      if (!F.getName().starts_with("__cudaRegisterLinkedBinary"))
        continue;

      for (auto *User : F.users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;

        IRBuilder<> Builder(CB);
        std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
        DEBUG(Logger::logs("proteus-pass")
              << "Instrument register linked binary to extract bitcode GV "
              << GVName << "\n");
        auto *Arg = Builder.CreateGlobalString(GVName);
        Builder.CreateCall(JitRegisterLinkedBinaryFn,
                           {CB->getArgOperand(1), Arg});
      }
    }
  }

  FunctionCallee getJitRegisterVarFn(Module &M) {
    // The prototype is
    // __proteus_register_var(void *Handle, const void *HostAddr, const char
    // *VarName, uint64_t VarSize).
    FunctionType *JitRegisterVarFnTy = FunctionType::get(
        Types.VoidTy, {Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.Int64Ty},
        /* isVarArg=*/false);
    FunctionCallee JitRegisterVarFn =
        M.getOrInsertFunction("__proteus_register_var", JitRegisterVarFnTy);

    return JitRegisterVarFn;
  }

  void instrumentRegisterVar(Module &M) {
    Function *RegisterVarFn = nullptr;
    if (!RegisterVarName)
      return;

    RegisterVarFn = M.getFunction(RegisterVarName);
    if (!RegisterVarFn)
      return;

    FunctionCallee JitRegisterVarFn = getJitRegisterVarFn(M);

    for (User *Usr : RegisterVarFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        Value *Handle = CB->getArgOperand(0);
        Value *Symbol = CB->getArgOperand(1);
        auto *GV = dyn_cast<GlobalVariable>(Symbol);
        Value *SymbolName = CB->getArgOperand(2);
        Value *SymbolSize = CB->getArgOperand(5);
        Builder.CreateCall(JitRegisterVarFn,
                           {Handle, GV, SymbolName, SymbolSize});
      }
  }

  FunctionCallee getJitRegisterFunctionFn(Module &M) {
    // The prototype is
    // __proteus_register_function(void *Handle,
    //                             void *Kernel,
    //                             char const *KernelName,
    //                             RuntimeConstantInfo **RCInfoArrayPtr,
    //                             int32_t NumRCs)
    FunctionType *JitRegisterFunctionFnTy = FunctionType::get(
        Types.VoidTy,
        {Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.Int32Ty},
        /* isVarArg=*/false);
    FunctionCallee JitRegisterKernelFn = M.getOrInsertFunction(
        "__proteus_register_function", JitRegisterFunctionFnTy);

    return JitRegisterKernelFn;
  }

  /// instrumentRegisterFunction instruments kernel functions following GPU
  /// runtime registration with __proteus_register_function.
  void instrumentRegisterFunction(Module &M) {
    if (!RegisterFunctionName)
      return;

    Function *RegisterFunction = M.getFunction(RegisterFunctionName);
    if (!RegisterFunction)
      return;

    for (User *RegisterFunctionUser : RegisterFunction->users()) {
      CallBase *RegisterCB = dyn_cast<CallBase>(RegisterFunctionUser);
      if (!RegisterCB)
        continue;

      Function *FunctionToRegister =
          dyn_cast<Function>(getStubGV(RegisterCB->getArgOperand(1)));
      assert(FunctionToRegister &&
             "Expected function passed to register function call");
      if (!JitFunctionInfoMap.contains(FunctionToRegister)) {
        DEBUG(Logger::logs("proteus-pass") << "Not instrumenting device kernel "
                                           << *FunctionToRegister << "\n");
        continue;
      }

      DEBUG(Logger::logs("proteus-pass")
            << "Instrumenting JIT function " << *FunctionToRegister << "\n");
      const auto &JFI = JitFunctionInfoMap[FunctionToRegister];
      size_t NumRuntimeConstants = JFI.ConstantArgs.size();

      ArrayType *RuntimeConstantInfoPtrArrayTy =
          ArrayType::get(Types.PtrTy, NumRuntimeConstants);

      GlobalVariable *RuntimeConstantInfoPtrArray = new GlobalVariable(
          M, RuntimeConstantInfoPtrArrayTy, false, GlobalValue::InternalLinkage,
          ConstantAggregateZero::get(RuntimeConstantInfoPtrArrayTy),
          ".proteus.rtconst.array." + FunctionToRegister->getName());

      IRBuilder<> Builder(RegisterCB->getNextNode());
      for (size_t I = 0; I < NumRuntimeConstants; ++I) {
        auto &RCInfo = JFI.ConstantArgs[I];

        if (RCInfo.ArgInfo.Type == RuntimeConstantType::ARRAY) {
          emitRuntimeConstantInfoArray(M, RCInfo, Builder,
                                       RuntimeConstantInfoPtrArrayTy,
                                       RuntimeConstantInfoPtrArray, I);
        } else if (RCInfo.ArgInfo.Type == RuntimeConstantType::OBJECT) {
          emitRuntimeConstantInfoObject(M, RCInfo, Builder,
                                        RuntimeConstantInfoPtrArrayTy,
                                        RuntimeConstantInfoPtrArray, I);
        } else if (isScalarRuntimeConstantType(RCInfo.ArgInfo.Type)) {
          emitRuntimeConstantInfoScalar(M, RCInfo, Builder,
                                        RuntimeConstantInfoPtrArrayTy,
                                        RuntimeConstantInfoPtrArray, I);
        } else {
          reportFatalError("Unsupported runtime constant type " +
                           toString(RCInfo.ArgInfo.Type));
        }
      }

      Value *NumRCsValue =
          ConstantInt::get(Builder.getInt32Ty(), NumRuntimeConstants);

      FunctionCallee JitRegisterFunction = getJitRegisterFunctionFn(M);

      Builder.CreateCall(JitRegisterFunction,
                         {RegisterCB->getArgOperand(0),
                          RegisterCB->getArgOperand(1),
                          RegisterCB->getArgOperand(2),
                          RuntimeConstantInfoPtrArray, NumRCsValue});

      auto HelperIt =
          LambdaRegistrationHelpers.find(RegisterCB->getArgOperand(1));
      if (HelperIt != LambdaRegistrationHelpers.end()) {
        FunctionCallee RegisterLambdaHelper = getRegisterLambdaRegisterFunc(M);
        Builder.CreateCall(RegisterLambdaHelper,
                           {RegisterCB->getArgOperand(1), HelperIt->second});
      }
    }
  }

  std::optional<RuntimeConstantType> getRCTypeForLLVMType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return RuntimeConstantType::BOOL;
    if (Ty->isIntegerTy(8))
      return RuntimeConstantType::INT8;
    if (Ty->isIntegerTy(32))
      return RuntimeConstantType::INT32;
    if (Ty->isIntegerTy(64))
      return RuntimeConstantType::INT64;
    if (Ty->isFloatTy())
      return RuntimeConstantType::FLOAT;
    if (Ty->isDoubleTy())
      return RuntimeConstantType::DOUBLE;
    if (Ty->isPointerTy())
      return RuntimeConstantType::PTR;
    return std::nullopt;
  }

  // Detect a CUDA module by the presence of the __cuda_register_globals
  // function.
  bool isCUDAModule(Module &M) {
    return M.getFunction("__cuda_register_globals") != nullptr;
  }

  // Add a call to __proteus_cudart_builtins_init to the global constructors of
  // the module to initialize the Proteus CUDA runtime builtins.
  void emitProteusCUDARuntimeBuiltinsInit(Module &M) {
    FunctionCallee InitFn =
        M.getOrInsertFunction("__proteus_cudart_builtins_init",
                              FunctionType::get(Types.VoidTy, false));

    appendToGlobalCtors(M, cast<Function>(InitFn.getCallee()), 65535);
  }
};

// New PM implementation.
struct ProteusPass : PassInfoMixin<ProteusPass> {
  enum class InsertionPoint {
    PipelineStart,
    EarlySimplification,
    FullLTOEarly,
  };

  ProteusPass(InsertionPoint InsertPt) : InsertPt(InsertPt) {}
  InsertionPoint InsertPt;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager & /*AM*/) {
    bool Changed = false;
    switch (InsertPt) {
    case InsertionPoint::PipelineStart:
      Changed = runLambdaAnalysis(M);
      break;
    case InsertionPoint::EarlySimplification: {
      ProteusPassImpl PPI{M};
      Changed = PPI.run(M, false);
      break;
    }
    case InsertionPoint::FullLTOEarly: {
      ProteusPassImpl PPI{M};
      Changed = PPI.run(M, true);
      break;
    }
    }
    if (Changed)
      // TODO: is anything preserved?
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation.
struct LegacyLambdaAnalysisPass : public ModulePass {
  static char ID;
  LegacyLambdaAnalysisPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override { return runLambdaAnalysis(M); }
};

struct LegacyProteusPass : public ModulePass {
  static char ID;
  LegacyProteusPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    ProteusPassImpl PPI{M};
    return PPI.run(M, false);
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
static llvm::PassPluginLibraryInfo getProteusPassPluginInfo() {
  const auto Callback = [](PassBuilder &PB) {
  // Run lambda analysis before inlining obscures register/jit_variable IR.
  // Keep the full Proteus rewrite at the existing later pipeline hooks.
#if LLVM_VERSION_MAJOR >= 20
    PB.registerPipelineStartEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel) {
          MPM.addPass(ProteusPass{ProteusPass::InsertionPoint::PipelineStart});
        });
#else
    PB.registerPipelineStartEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel) {
          MPM.addPass(ProteusPass{ProteusPass::InsertionPoint::PipelineStart});
          return true;
        });
#endif

#if LLVM_VERSION_MAJOR >= 20
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel,
            ThinOrFullLTOPhase LTOPhase) {
          if (LTOPhase != ThinOrFullLTOPhase::None) {
            reportFatalError("Expected registration only for non-LTO");
          }
#else
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel) {
#endif
          MPM.addPass(
              ProteusPass{ProteusPass::InsertionPoint::EarlySimplification});
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(ProteusPass{ProteusPass::InsertionPoint::FullLTOEarly});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "ProteusPass", LLVM_VERSION_STRING,
          Callback};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize ProteusPass when added to the pass pipeline on the
// command line, i.e. via '-passes=proteus-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getProteusPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyLambdaAnalysisPass::ID = 0;
char LegacyProteusPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize the legacy passes when added to the pass pipeline on the command
// line, i.e. via '--legacy-proteus-lambda-analysis --legacy-proteus-pass'.
static RegisterPass<LegacyLambdaAnalysisPass>
    LambdaX("legacy-proteus-lambda-analysis", "Proteus Lambda Analysis",
            false, // This pass doesn't modify the CFG => false
            false  // This pass is not a pure analysis pass => false
    );
static RegisterPass<LegacyProteusPass>
    X("legacy-proteus-pass", "Proteus Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
static RegisterPass<LegacyProteusPass>
    XCompat("legacy-proteuss-pass", "Proteus Pass",
            false, // This pass doesn't modify the CFG => false
            false  // This pass is not a pure analysis pass => false
    );
