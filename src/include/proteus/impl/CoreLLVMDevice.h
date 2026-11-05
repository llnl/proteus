#ifndef PROTEUS_CORE_LLVM_DEVICE_H
#define PROTEUS_CORE_LLVM_DEVICE_H

#if PROTEUS_ENABLE_HIP
#include "proteus/impl/CoreLLVMHIP.h"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/impl/CoreLLVMCUDA.h"
#endif

#if defined(PROTEUS_ENABLE_HIP) || defined(PROTEUS_ENABLE_CUDA)

#include "proteus/impl/CoreDevice.h"
#include "proteus/impl/GlobalVarInfo.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/TransformArgumentSpecialization.h"
#include "proteus/impl/TransformCollapseCasesSpecialization.h"
#include "proteus/impl/TransformLambdaSpecialization.h"
#include "proteus/impl/TransformSharedArray.h"

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace proteus {

inline void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim) {
  auto ReplaceIntrinsicDim = [&](ArrayRef<StringRef> IntrinsicNames,
                                 uint32_t DimValue) {
    auto CollectCallUsers = [](Function &F) {
      SmallVector<CallInst *> CallUsers;
      for (auto *User : F.users()) {
        auto *Call = dyn_cast<CallInst>(User);
        if (!Call)
          continue;
        CallUsers.push_back(Call);
      }

      return CallUsers;
    };

    for (auto IntrinsicName : IntrinsicNames) {

      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction)
        continue;

      auto TraceOut = [](Function *F, Value *C) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[DimSpec] Replace call to " << F->getName() << " with constant "
           << *C << "\n";

        return S;
      };

      for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
        Value *ConstantValue =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), DimValue);
        Call->replaceAllUsesWith(ConstantValue);
        if (Config::get().traceSpecializations())
          Logger::trace(TraceOut(IntrinsicFunction, ConstantValue));
        Call->eraseFromParent();
      }
    }
  };

  ReplaceIntrinsicDim(detail::gridDimXFnName(), GridDim.x);
  ReplaceIntrinsicDim(detail::gridDimYFnName(), GridDim.y);
  ReplaceIntrinsicDim(detail::gridDimZFnName(), GridDim.z);

  ReplaceIntrinsicDim(detail::blockDimXFnName(), BlockDim.x);
  ReplaceIntrinsicDim(detail::blockDimYFnName(), BlockDim.y);
  ReplaceIntrinsicDim(detail::blockDimZFnName(), BlockDim.z);
}

inline void setKernelDimsRange(Module &M, dim3 &GridDim, dim3 &BlockDim) {
  auto AttachRange = [&](ArrayRef<StringRef> IntrinsicNames,
                         uint32_t DimValue) {
    if (DimValue == 0) {
      reportFatalError("Dimension value cannot be zero");
    }

    for (auto IntrinsicName : IntrinsicNames) {
      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction || IntrinsicFunction->use_empty())
        continue;

      auto TraceOut = [](Function *IntrinsicF, uint32_t DimValue) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[DimSpec] Range " << IntrinsicF->getName() << " [0," << DimValue
           << ")\n";
        return S;
      };

      for (auto *U : IntrinsicFunction->users()) {
        auto *Call = dyn_cast<CallInst>(U);
        if (!Call)
          continue;

        auto *RetTy = dyn_cast<IntegerType>(Call->getType());
        if (!RetTy)
          continue;

        unsigned BitWidth = RetTy->getBitWidth();
        ConstantRange Range(APInt(BitWidth, 0), APInt(BitWidth, DimValue));

#if LLVM_VERSION_MAJOR >= 19
        AttrBuilder Builder{M.getContext()};
        Builder.addRangeAttr(Range);
        Call->removeRetAttr(Attribute::Range);
        Call->setAttributes(
            Call->getAttributes().addRetAttributes(M.getContext(), Builder));
#else
        // LLVM 18 (ROCm 6.2.x) does not expose the Range attribute; use range
        // metadata instead.
        LLVMContext &Ctx = M.getContext();
        Metadata *RangeMD[] = {
            ConstantAsMetadata::get(ConstantInt::get(RetTy, 0)),
            ConstantAsMetadata::get(ConstantInt::get(RetTy, DimValue))};
        MDNode *RangeNode = MDNode::get(Ctx, RangeMD);
        Call->setMetadata(LLVMContext::MD_range, RangeNode);
#endif

        if (Config::get().traceSpecializations())
          Logger::trace(TraceOut(IntrinsicFunction, DimValue));
      }
    }
  };

  AttachRange(detail::threadIdxXFnName(), BlockDim.x);
  AttachRange(detail::threadIdxYFnName(), BlockDim.y);
  AttachRange(detail::threadIdxZFnName(), BlockDim.z);

  AttachRange(detail::blockIdxXFnName(), GridDim.x);
  AttachRange(detail::blockIdxYFnName(), GridDim.y);
  AttachRange(detail::blockIdxZFnName(), GridDim.z);
}

inline void replaceGlobalVariablesWithPointers(
    Module &M,
    const std::unordered_map<std::string, GlobalVarInfo> &VarNameToGlobalInfo) {
  TIMESCOPE("proteus::replaceGlobalVariablesWithPointers");
  // Re-link globals to fixed addresses provided by registered
  // variables.
  for (auto RegisterVar : VarNameToGlobalInfo) {
    auto &VarName = RegisterVar.first;
    auto *GV = M.getNamedGlobal(VarName);
    // Skip linking if the GV does not exist in the module.
    if (!GV)
      continue;

    // This will convert constant users of GV to instructions so that we can
    // replace with the GV ptr.
    convertUsersOfConstantsToInstructions({GV});

    Constant *Addr =
        ConstantInt::get(Type::getInt64Ty(M.getContext()), 0xDEADBEEFDEADBEEF);
    auto *CE = ConstantExpr::getIntToPtr(
        Addr, PointerType::get(M.getContext(), GV->getAddressSpace()));
    auto *GVarPtr = new GlobalVariable(
        M, PointerType::get(M.getContext(), GV->getAddressSpace()), false,
        GlobalValue::ExternalLinkage, CE, GV->getName() + "$ptr", nullptr,
        GV->getThreadLocalMode(), GV->getAddressSpace(), true);

    // Find all Constant users that refer to the global variable.
    SmallPtrSet<Value *, 16> ValuesToReplace;
    SmallVector<Value *> Worklist;
    // Seed with the global variable.
    Worklist.push_back(GV);
    ValuesToReplace.insert(GV);
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      for (auto *User : V->users()) {
        if (auto *C = dyn_cast<Constant>(User)) {
          if (ValuesToReplace.insert(C).second)
            Worklist.push_back(C);

          continue;
        }

        // Skip instructions to be handled when replacing.
        if (isa<Instruction>(User))
          continue;

        reportFatalError("Expected Instruction or Constant user for Value: " +
                         toString(*V) + " , User: " + toString(*User));
      }
    }

    for (Value *V : ValuesToReplace) {
      SmallPtrSet<Instruction *, 16> Insts;
      // Find instruction users to replace value.
      for (User *U : V->users()) {
        if (auto *I = dyn_cast<Instruction>(U)) {
          Insts.insert(I);
        }
      }

      // Replace value in instructions.
      for (auto *I : Insts) {
        IRBuilder Builder{I};
        auto *Load = Builder.CreateLoad(GV->getType(), GVarPtr);
        Value *Replacement = Load;
        Type *ExpectedTy = V->getType();
        if (Load->getType() != ExpectedTy)
          Replacement =
              Builder.CreatePointerBitCastOrAddrSpaceCast(Load, ExpectedTy);

        I->replaceUsesOfWith(V, Replacement);
      }
    }
  }

  if (Config::get().ProteusDebugOutput) {
    if (verifyModule(M, &errs()))
      reportFatalError("Broken module found, JIT compilation aborted!");
  }
}

inline void relinkGlobalsObject(
    MemoryBufferRef Object,
    const std::unordered_map<std::string, GlobalVarInfo> &VarNameToGlobalInfo) {
  TIMESCOPE("proteus::relinkGlobalsObject");
  auto ObjOrErr = object::ObjectFile::createObjectFile(Object);
  if (auto E = ObjOrErr.takeError())
    reportFatalError("Cannot create the device elf: " + toString(std::move(E)));
  auto &DeviceObjFile = **ObjOrErr;
  auto *DeviceElf = dyn_cast<object::ELFObjectFileBase>(&DeviceObjFile);
  if (!DeviceElf)
    reportFatalError("Expected ELF object file");

  for (auto &[GlobalName, GVI] : VarNameToGlobalInfo) {
    for (auto &Symbol : DeviceElf->symbols()) {
      auto SymbolNameOrErr = Symbol.getName();
      if (!SymbolNameOrErr)
        continue;
      auto SymbolName = *SymbolNameOrErr;

      if (!(SymbolName == (GlobalName + "$ptr")))
        continue;

      Expected<uint64_t> ValueOrErr = Symbol.getValue();
      if (!ValueOrErr)
        reportFatalError("Expected symbol value");
      uint64_t SymbolValue = *ValueOrErr;

      // Get the section containing the symbol
      auto SectionOrErr = Symbol.getSection();
      if (!SectionOrErr)
        reportFatalError("Cannot retrieve section");
      const auto &Section = *SectionOrErr;
      if (Section == DeviceElf->section_end())
        reportFatalError("Expected sybmol in section");

      // Get the section's address and data
      Expected<StringRef> SectionDataOrErr = Section->getContents();
      if (!SectionDataOrErr)
        reportFatalError("Error retrieving section data");
      StringRef SectionData = *SectionDataOrErr;

      // Calculate offset within the section
      uint64_t SectionAddr = Section->getAddress();
      uint64_t Offset = SymbolValue - SectionAddr;
      if (Offset >= SectionData.size())
        reportFatalError("Expected offset within section size");

      uint64_t *Data = (uint64_t *)(SectionData.data() + Offset);
      if (!GVI.DevAddr)
        reportFatalError("Cannot set global Var " + GlobalName +
                         " without a concrete device address");

      *Data = reinterpret_cast<uint64_t>(GVI.DevAddr);
      break;
    }
  }
}

inline void specializeIR(
    Module &M, StringRef FnName, StringRef Suffix, dim3 &BlockDim,
    dim3 &GridDim, ArrayRef<RuntimeConstant> RCArray,
    const VariantSnapshot &Variants,
    const SmallVector<uint64_t> LambdaCalleeInfo,
    const LambdaCallsiteRuntimeConstantsMap &LambdaCallsiteRuntimeConstants,
    bool SpecializeArgs, bool SpecializeDims, bool SpecializeDimsRange,
    bool SpecializeLaunchBounds, int MinBlocksPerSM) {
  TIMESCOPE("proteus::specializeIR");
  Timer T(Config::get().ProteusEnableTimers);
  Function *F = M.getFunction(FnName);

  assert(F && "Expected non-null function!");
  // Replace argument uses with runtime constants.
  if (SpecializeArgs)
    TransformArgumentSpecialization::transform(M, *F, RCArray);

  // We add a per-function cache for memory ssa so that we don't have to
  // duplicate results. TransformLambdaSpecialization LambdaTransformer;
  for (auto &ID : LambdaCalleeInfo) {
    TransformLambdaSpecialization::transformDeviceKernel(
        M, ID, LambdaCallsiteRuntimeConstants);
  }

  TransformCollapseCasesSpecialization::transform(M, Variants);

  // Run the shared array transform after any value specialization (arguments,
  // captures) to propagate any constants.
  TransformSharedArray::transform(M);

  // Replace uses of blockDim.* and gridDim.* with constants.
  if (SpecializeDims)
    setKernelDims(M, GridDim, BlockDim);
  if (SpecializeDimsRange)
    setKernelDimsRange(M, GridDim, BlockDim);
  F->setName(FnName + Suffix);

  if (SpecializeLaunchBounds) {
    int BlockSize = BlockDim.x * BlockDim.y * BlockDim.z;
    auto TraceOut = [](int BlockSize, int MinBlocksPerSM) {
      SmallString<128> S;
      raw_svector_ostream OS(S);
      OS << "[LaunchBoundSpec] MaxThreads " << BlockSize << " MinBlocksPerSM "
         << MinBlocksPerSM << "\n";

      return S;
    };
    if (Config::get().traceSpecializations())
      Logger::trace(TraceOut(BlockSize, MinBlocksPerSM));
    setLaunchBoundsForKernel(*F, BlockSize, MinBlocksPerSM);
  }

  runCleanupPassPipeline(M);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "specializeIR " << T.elapsed() << " ms\n");
}

} // namespace proteus

#endif

#endif
