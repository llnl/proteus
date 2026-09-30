#ifndef PROTEUS_KERNELARGVISITOR_H
#define PROTEUS_KERNELARGVISITOR_H

#include "Helpers.h"
#include "KernelArgPtrUseVisitor.h"
#include "PointerClobberAnalysis.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

std::optional<ReturnInst *> getRetInst(Function &F) {
  for (auto &BB : F) {
    if (auto *TermInst = dyn_cast<ReturnInst>(BB.getTerminator()))
      return TermInst;
  }
  return std::nullopt;
}

inline int64_t getValueIndicesOffset(const DataLayout &DL, Type *AggTy,
                                     ArrayRef<unsigned> Indices) {
  LLVMContext &Ctx = AggTy->getContext();
  SmallVector<Value *, 8> GEPIndices;
  GEPIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  for (unsigned Idx : Indices)
    GEPIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ctx), Idx));
  return DL.getIndexedOffsetInType(AggTy, GEPIndices);
}

inline std::optional<LambdaPtrUseAnalysis>
getDominatingUse(const DataLayout &DL, Value *ValueNeedingAnalysis,
                 Value *SeenUse, int64_t TargetOffset, CallBase *LambdaCB,
                 const std::shared_ptr<PointerClobberAnalysis> &Clobbers) {
  return runDominatingUseVisitor(DL, ValueNeedingAnalysis, SeenUse,
                                 TargetOffset, LambdaCB, Clobbers);
}

struct LambdaKernelArgAnalysis {
  Function *KernelFunction = nullptr;
  uint32_t KernelArgIndex = 0;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct FunctionAnalysis {
  Value *PtrArgToCB = nullptr;
  uint32_t ArgIndex = 0;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct WorkItem {
  Value *CurVal;
  Value *Src;
};

class LambdaArgVisitor : public InstVisitor<LambdaArgVisitor> {
private:
  CallBase *LambdaCB;
  const DataLayout &DL;
  std::shared_ptr<PointerClobberAnalysis> Clobbers;
  SmallVector<WorkItem> WorkList;
  SmallDenseSet<Value *> Seen;

  int64_t Offset;
  uint32_t KernelArg = 0;
  Function *KernelFunction = nullptr;
  // instructions like inttoptr can change the runtimeconstant type
  // we to read from the Blob
  std::optional<RuntimeConstantType> ChangedRC = std::nullopt;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

  // Constructor used for cloning and merging branches of phi node analysis
  LambdaArgVisitor(Value *Start, Value *LastSeen, CallBase *LambdaCBArg,
                   int64_t Off, const DataLayout &Dl,
                   std::shared_ptr<PointerClobberAnalysis> ClobberResolver)
      : LambdaCB(LambdaCBArg), DL(Dl), Clobbers(std::move(ClobberResolver)),
        Offset(Off) {
    WorkList.push_back({Start, LastSeen});
  }

  std::optional<Value *> getCallBaseIdentityArgOperand(CallBase &CB) {
    auto *CalledFunction = CB.getCalledFunction();
    DEBUG(Logger::logs("proteus-pass")
          << "Checking if function is identity " << *CalledFunction << "\n");
    auto RetInstOpt = getRetInst(*CalledFunction);
    if (!RetInstOpt)
      return std::nullopt;
    auto *RetInst = *RetInstOpt;

    DEBUG(Logger::logs("proteus-pass") << "CB called function return inst "
                                       << *RetInst->getReturnValue() << "\n");
    for (size_t ArgNum = 0; ArgNum < CalledFunction->arg_size(); ++ArgNum) {
      DEBUG(Logger::logs("proteus-pass")
            << "Called Fn arg " << *CalledFunction->getArg(ArgNum) << "\n");
      if (RetInst->getReturnValue() == CalledFunction->getArg(ArgNum))
        return CB.getArgOperand(ArgNum);
    }
    return std::nullopt;
  }

public:
  LambdaKernelArgAnalysis getKernelArgInfo() {
    return LambdaKernelArgAnalysis{KernelFunction, KernelArg, Offset,
                                   ChangedRC};
  }
  // Whenever the analysis encounters an instruction returning a Ptr
  // memory analysis is required to identify a dominating write,
  // which is where the LambdaArgVisitor continues its analysis.
  // This pointer identifies which Ptr use the main analysis used
  // to discover the Ptr needing analysis, so as to prevent cycles.
  Value *MemoryAnalysisPtrUse = nullptr;
  auto back() { return WorkList.back(); }
  void popBack() { WorkList.pop_back(); }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }
  auto getOffset() { return Offset; }

private:
  inline std::optional<LambdaKernelArgAnalysis>
  cloneAndAnalyze(Value *Start, Value *MemoryAnalysisPtrUse,
                  int64_t StartOffset) {
    LambdaArgVisitor Visitor(Start, MemoryAnalysisPtrUse, LambdaCB, StartOffset,
                             DL, Clobbers);
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto [V, AccessedFrom] = Visitor.back();
      Visitor.MemoryAnalysisPtrUse = AccessedFrom;
      Visitor.popBack();
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V))
        Visitor.visitArgument(*A);
      else
        continue;
    }
    if (!Visitor.success() || Visitor.failed())
      return std::nullopt;
    return Visitor.getKernelArgInfo();
  }

  inline std::optional<FunctionAnalysis> analyzeFunction(CallBase &CB,
                                                         int64_t StartOffset) {
    FunctionAnalysis Result;
    auto &F = *CB.getCalledFunction();
    auto RetInstOpt = getRetInst(F);
    if (!RetInstOpt)
      return std::nullopt;
    LambdaArgVisitor Visitor(RetInstOpt.value()->getReturnValue(),
                             MemoryAnalysisPtrUse, LambdaCB, StartOffset, DL,
                             Clobbers);
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto [V, AccessedFrom] = Visitor.back();
      DEBUG(Logger::logs("proteus-pass")
            << "Function analysis visiting " << *V << " with offset "
            << Visitor.getOffset() << "\n");
      Visitor.MemoryAnalysisPtrUse = AccessedFrom;
      Visitor.popBack();
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V)) {
        if (A->getParent() == CB.getCalledFunction()) {
          Result.PtrArgToCB = CB.getArgOperand(A->getArgNo());
          Result.Offset = Visitor.Offset;
          Result.ArgIndex = A->getArgNo();
          DEBUG(Logger::logs("proteus-pass")
                << "Function analysis found termination case "
                << *Result.PtrArgToCB << " with offset " << Visitor.getOffset()
                << "\n");
          return Result;
        }
        Visitor.visitArgument(*A);

      } else
        continue;
    }

    return std::nullopt;
  }

public:
  LambdaArgVisitor(CallBase *LambdaCB, Module &M)
      : LambdaCB(LambdaCB), DL(M.getDataLayout()),
        Clobbers(std::make_shared<PointerClobberResolver>(DL)), Offset(0) {
    auto *ClosurePtr = LambdaCB->getArgOperand(0);
    WorkList.push_back({ClosurePtr, LambdaCB});
  }

  void visitStoreInst(StoreInst &SI) {
    WorkList.push_back({SI.getValueOperand(), &SI});
  }

  void visitCallBase(CallBase &CB) {
    if (!CB.getCalledFunction() || CB.getCalledFunction()->isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Cannot trace indirect or declaration "
               "call "
            << CB << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    // Clone the visitor to determine if this function (a) returns a ptr
    // and (b) which arg determines the value of that ptr, and at which offset
    auto SubAnalysis = analyzeFunction(CB, Offset);
    if (SubAnalysis) {
      WorkList.push_back({SubAnalysis.value().PtrArgToCB, &CB});
      Offset = SubAnalysis.value().Offset;
      return;
    }
    DEBUG(Logger::logs("proteus-pass")
          << "Function analysis of \n"
          << *CB.getCalledFunction() << " failed \n");
    AnalysisFailed = true;
    AnalysisSuccess = false;
  }

  void visitLoadInst(LoadInst &LI) {
    DEBUG(Logger::logs("proteus-pass") << "Load inst analysis \n")
    PointerClobberResult Clobber;
    if (isa<AllocaInst>(getUnderlyingObject(LI.getPointerOperand()))) {
      Clobber = Clobbers->resolve(LI.getPointerOperand(), LI, 0);
      if (Clobber.Kind == PointerClobberKind::Value) {
        WorkList.push_back({Clobber.V, &LI});
        return;
      }
      if (Clobber.Kind == PointerClobberKind::Ambiguous) {
        DEBUG(Logger::logs("proteus-pass")
              << "[Lambda arg analysis]: MemorySSA found ambiguous pointer "
                 "clobbers for "
              << LI << "\n");
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
      if (Clobber.ClobberingInstruction) {
        auto Res = runDominatingUseVisitor(DL, LI.getPointerOperand(), &LI, 0,
                                           nullptr, Clobbers);
        if (!Res) {
          AnalysisFailed = true;
          AnalysisSuccess = false;
          return;
        }
        WorkList.push_back(
            {Res->DominatingWrite,
             Res->ClobberingInstruction ? Res->ClobberingInstruction : &LI});
        Offset -= Res->EnclosingObjectOffsetCorrection;
        return;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: MemorySSA could not resolve the "
               "reaching pointer clobber for "
            << LI << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    // Loading a pointer from a spill slot does not change the offset within
    // the pointee.  Resolve the pointer-sized store at offset zero in the slot,
    // then continue with the original pointee-relative Offset.
    if (isPointerSpillLoad(LI)) {
      ReachingPointerStores Stores = getReachingPointerStores(DL, LI);
      SmallPtrSet<Value *, 8> Visited;
      if (Value *StoredPointer =
              getUniqueReachingPointer(DL, Stores, Visited)) {
        WorkList.push_back({StoredPointer, &LI});
        return;
      }
      if (hasAmbiguousReachingPointers(DL, Stores)) {
        DEBUG(Logger::logs("proteus-pass")
              << "[Lambda arg analysis]: Pointer spill load has ambiguous "
                 "reaching stores: "
              << LI << "\n");
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
      auto Res = getDominatingUse(DL, LI.getPointerOperand(), &LI, 0, LambdaCB,
                                  Clobbers);
      if (!Res) {
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
      WorkList.push_back({Res->DominatingWrite, Res->ClobberingInstruction
                                                    ? Res->ClobberingInstruction
                                                    : &LI});
      return;
    }

    WorkList.push_back({LI.getPointerOperand(), &LI});
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    APInt StepOffset(DL.getIndexTypeSizeInBits(GEP.getType()), 0);
    if (!GEP.accumulateConstantOffset(DL, StepOffset)) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    Offset += StepOffset.getSExtValue();
    WorkList.push_back({GEP.getPointerOperand(), &GEP});
  }

  void visitExtractValueInst(ExtractValueInst &EVI) {
    int64_t EVIOffset = getValueIndicesOffset(
        DL, EVI.getAggregateOperand()->getType(), EVI.getIndices());
    Offset += EVIOffset;
    WorkList.push_back({EVI.getAggregateOperand(), &EVI});
  }

  // The analysis always encounters IVI chains in a backwards direction, meaning
  // we always see the final IVI in a chain of writes. We assert this shape in
  // our analysis, stopping at the location within the aggregate where we know
  // our closure ptr lives
  void visitInsertValueInst(InsertValueInst &IVI) {
    auto *AggregateOperand = IVI.getAggregateOperand();
    auto *Cur = &IVI;

    while (Cur && AggregateOperand) {
      int64_t CurOffset = getValueIndicesOffset(
          DL, Cur->getAggregateOperand()->getType(), Cur->getIndices());
      TypeSize InsertedSize =
          DL.getTypeAllocSize(Cur->getInsertedValueOperand()->getType());
      DEBUG(Logger::logs("proteus-pass")
            << "Curr offset " << CurOffset << "\nOffset " << Offset << "\n");
      if (!InsertedSize.isScalable() && Offset >= CurOffset &&
          static_cast<uint64_t>(Offset - CurOffset) <
              InsertedSize.getFixedValue()) {
        // We are now following the value inserted into this aggregate field,
        // so make the tracked offset relative to that value.  Leaving the
        // aggregate-field offset in place causes it to be counted again when
        // the inserted value came from a GEP into another aggregate.
        Offset -= CurOffset;
        WorkList.push_back({Cur->getInsertedValueOperand(), &IVI});
        return;
      }
      Cur = dyn_cast<InsertValueInst>(AggregateOperand);
      if (Cur)
        AggregateOperand = Cur->getAggregateOperand();
    }
    AnalysisFailed = true;
    AnalysisSuccess = false;
  }

  // todo: these three methods need to be changed to find a dominating store
  void visitAllocaInst(AllocaInst &Alloca) {
    auto Res = getDominatingUse(DL, &Alloca, MemoryAnalysisPtrUse, Offset,
                                LambdaCB, Clobbers);
    if (!Res)
      return;

    WorkList.push_back(
        {Res->DominatingWrite, Res->ClobberingInstruction
                                   ? Res->ClobberingInstruction
                                   : static_cast<Instruction *>(&Alloca)});
    // Res->Offset converts the current allocation-relative byte offset into
    // the coordinate system of DominatingWrite. For a field store it removes
    // the field displacement; for a memory transfer it translates destination
    // displacement into the corresponding source displacement.
    Offset -= Res->Offset;
  }

  void visitBitCastInst(BitCastInst &BC) {
    auto Res = getDominatingUse(DL, &BC, MemoryAnalysisPtrUse, Offset, LambdaCB,
                                Clobbers);
    if (!Res)
      return;
    WorkList.push_back({Res->DominatingWrite, Res->ClobberingInstruction
                                                  ? Res->ClobberingInstruction
                                                  : &BC});
    // Res->Offset converts the current cast-relative byte offset into the
    // coordinate system of DominatingWrite.
    Offset -= Res->Offset;
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    WorkList.push_back({ASC.getPointerOperand(), &ASC});
    auto Res = getDominatingUse(DL, &ASC, MemoryAnalysisPtrUse, Offset,
                                LambdaCB, Clobbers);
    if (!Res)
      return;

    WorkList.push_back({Res->DominatingWrite, Res->ClobberingInstruction
                                                  ? Res->ClobberingInstruction
                                                  : &ASC});
    // Res->Offset converts the current cast-relative byte offset into the
    // coordinate system of DominatingWrite.
    Offset -= Res->Offset;
  }

  void visitIntToPtr(IntToPtrInst &ITP) {
    auto *IntegerVal = ITP.getOperand(0);
    auto *Ptr = dyn_cast<PtrToIntInst>(IntegerVal);
    if (!Ptr) {
      AnalysisSuccess = false;
      AnalysisFailed = true;
      return;
    }
    ChangedRC = convertTypeToRuntimeConstantType(Ptr->getType());

    WorkList.push_back({Ptr->getPointerOperand(), &ITP});
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    auto *MT = dyn_cast<MemTransferInst>(&I); // memcpy/memmove
    if (!MT) {
      // memset doesn't preserve any src->dst relationship we can use
      AnalysisFailed = true;
      return;
    }

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase) {
      AnalysisFailed = true;
      return;
    }

    // Optional safety: only valid if the tracked byte lies within the copied
    // region.
    if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
      uint64_t Len = LenC->getZExtValue();
      if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len) {
        AnalysisFailed = true;
        return;
      }
    }

    WorkList.push_back({SrcBase->stripPointerCasts(), &I});
    Offset = Offset - DstOff + SrcOff;
  }

  void visitIntrinsicInst(IntrinsicInst &) {
    AnalysisFailed = true;
    return;
  }

  void visitTruncInst(TruncInst &TI) {
    WorkList.push_back({TI.getOperand(0), &TI});
  }

  void visitArgument(Argument &A) {
    Function *F = A.getParent();
    DEBUG(Logger::logs("proteus-pass")
          << "Visiting argument with parent function = \n"
          << *F << "\n");
    auto ArgNum = A.getArgNo();
    // termination case:  we have reached the parent calling kernel
    // todo: we could just pass in the kernel pointer here and check equality

    if (F->getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        F->getCallingConv() == CallingConv::PTX_Kernel ||
        (F->hasMetadata("proteus.jit") &&
         !F->hasMetadata("proteus.wrapper_call") &&
         !F->hasMetadata("proteus.registered_lambda"))) {
      DEBUG(Logger::logs("proteus-pass")
            << "Found termination case from function " << F->getName() << "\n");
      DEBUG(Logger::logs("proteus-pass").flush());

      AnalysisSuccess = true;
      KernelArg = ArgNum;
      KernelFunction = F;
      return;
    }

    SmallVector<CallBase *, 2> Callers;
    for (User *U : F->users()) {
      auto *CB = dyn_cast<CallBase>(U);
      if (!CB || CB->getCalledFunction() != F || ArgNum >= CB->arg_size()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[Lambda arg analysis]: Function has an unsupported caller "
                 "use: "
              << *U << "\n");
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
      Callers.push_back(CB);
    }

    // Without a call-context stack, following more than one caller makes the
    // result depend on Value::users() traversal order. Decline instead of
    // selecting whichever kernel argument happens to be visited first.
    if (Callers.size() != 1) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Expected exactly one direct caller of "
            << F->getName() << ", found " << Callers.size() << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    CallBase *Caller = Callers.front();
    DEBUG(Logger::logs("proteus-pass")
          << "Analysis crossed interprocedural boundary at "
          << *Caller->getArgOperand(ArgNum) << "\n");
    WorkList.push_back({Caller->getArgOperand(ArgNum), Caller});
  }

  void visitInstruction(Instruction &I) {
    DEBUG(Logger::logs("proteus-pass")
          << "[Lambda arg analysis]: Unhandled instruction "
          << I.getOpcodeName() << ": " << I << "\n");
    AnalysisFailed = true;
    return;
  }

  void visitPHINode(PHINode &P) {
    if (P.getNumIncomingValues() == 0) {
      AnalysisFailed = true;
      return;
    }

    auto FirstAnalysis =
        cloneAndAnalyze(P.getIncomingValue(0), MemoryAnalysisPtrUse, Offset);
    if (!FirstAnalysis) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    auto BaseSlot = FirstAnalysis->KernelArgIndex;
    auto BaseOffset = FirstAnalysis->Offset;

    for (size_t Idx = 1; Idx < P.getNumIncomingValues(); ++Idx) {
      auto Analysis = cloneAndAnalyze(P.getIncomingValue(Idx),
                                      MemoryAnalysisPtrUse, Offset);
      if (!Analysis || Analysis->KernelArgIndex != BaseSlot ||
          Analysis->Offset != BaseOffset) {
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
    }
    KernelFunction = FirstAnalysis->KernelFunction;
    Offset = BaseOffset;
    KernelArg = BaseSlot;
    ChangedRC = FirstAnalysis->ChangedRCLayout;
    AnalysisSuccess = true;
    AnalysisFailed = false;
  }

  void visitSelectInst(SelectInst &S) {
    // A select can merge call results or loads as well as simple GEPs.  Base
    // pointer equality rejects semantically identical paths in those shapes,
    // so analyze both arms exactly as we do for a PHI and retain the result
    // only when they resolve to one kernel argument and byte offset.
    auto TrueAnalysis =
        cloneAndAnalyze(S.getTrueValue(), MemoryAnalysisPtrUse, Offset);
    auto FalseAnalysis =
        cloneAndAnalyze(S.getFalseValue(), MemoryAnalysisPtrUse, Offset);
    if (!TrueAnalysis || !FalseAnalysis ||
        TrueAnalysis->KernelFunction != FalseAnalysis->KernelFunction ||
        TrueAnalysis->KernelArgIndex != FalseAnalysis->KernelArgIndex ||
        TrueAnalysis->Offset != FalseAnalysis->Offset ||
        TrueAnalysis->ChangedRCLayout != FalseAnalysis->ChangedRCLayout) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Select arms do not resolve to the "
               "same kernel argument and offset: "
            << S << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    KernelFunction = TrueAnalysis->KernelFunction;
    KernelArg = TrueAnalysis->KernelArgIndex;
    Offset = TrueAnalysis->Offset;
    ChangedRC = TrueAnalysis->ChangedRCLayout;
    AnalysisSuccess = true;
    AnalysisFailed = false;
  }
};

inline bool analyzeLambdaUses(
    llvm::Module &M,
    DenseMap<CallBase *, LambdaKernelArgAnalysis> &CallBaseToArgOffset,
    const SmallVector<CallBase *> &CBToAnalyze) {
  DEBUG(Logger::logs("proteus-pass") << "Beginning analysis " << "\n");
  for (auto *FunctorCB : CBToAnalyze) {
    LambdaArgVisitor Visitor(FunctorCB, M);
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto [V, LastSeen] = Visitor.back();
      Visitor.MemoryAnalysisPtrUse = LastSeen;
      Visitor.popBack();
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      DEBUG(Logger::logs("proteus-pass") << "Visiting value " << *V << "\n");
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V))
        Visitor.visitArgument(*A);
      else
        continue;
    }
    if (!Visitor.success() || Visitor.failed()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[WARNING]: Kernel arg analysis failed for functor beginning at "
            << *FunctorCB << "\n");
      return false;
    }
    LambdaKernelArgAnalysis Info = Visitor.getKernelArgInfo();
    if (!Info.KernelFunction)
      return false;
    CallBaseToArgOffset[FunctorCB] = Info;
  }
  return true;
}
} // namespace proteus

#endif
