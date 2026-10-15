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

inline int64_t getValueIndicesOffset(const DataLayout &DL, Type *AggTy,
                                     ArrayRef<unsigned> Indices) {
  LLVMContext &Ctx = AggTy->getContext();
  SmallVector<Value *, 8> GEPIndices;
  GEPIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  for (unsigned Idx : Indices)
    GEPIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ctx), Idx));
  return DL.getIndexedOffsetInType(AggTy, GEPIndices);
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
  Instruction *Src;
};

class LambdaArgVisitor : public InstVisitor<LambdaArgVisitor> {
private:
  CallBase *LambdaCB;
  const DataLayout &DL;
  std::shared_ptr<MemorySSAClobberOracle> ClobberOracle;
  SmallVector<WorkItem> WorkList;
  SmallDenseSet<Value *> Seen;
  // The size of the struct, a pointer to this is passed to the call operator
  uint64_t FunctorSize;
  int64_t Offset;
  uint32_t KernelArg = 0;
  Function *KernelFunction = nullptr;
  // instructions like inttoptr can change the runtimeconstant type
  // we to read from the Blob
  std::optional<RuntimeConstantType> ChangedRC = std::nullopt;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

  // The two flags are a unit: never both true. Every conservative decline
  // routes through here.
  void fail() {
    AnalysisFailed = true;
    AnalysisSuccess = false;
  }
  // Depth of nested clone/function sub-analyses. Cyclic PHI webs and
  // recursive pointer-returning callees otherwise recurse without bound,
  // because each sub-analysis starts with a fresh Seen set.
  int AnalysisDepth = 0;
  static constexpr int MaxAnalysisDepth = 32;

  // Constructor used for cloning and merging branches of phi node analysis
  LambdaArgVisitor(Value *Start, Instruction *LastSeen, CallBase *LambdaCBArg,
                   int64_t Off, const DataLayout &Dl,
                   std::shared_ptr<MemorySSAClobberOracle> Oracle,
                   uint64_t FunctorSizeArg)
      : LambdaCB(LambdaCBArg), DL(Dl), ClobberOracle(std::move(Oracle)),
        FunctorSize(FunctorSizeArg), Offset(Off) {
    WorkList.push_back({Start, LastSeen});
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
  Instruction *MemoryAnalysisPtrUse = nullptr;
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
  cloneAndAnalyze(Value *Start, Instruction *MemoryAnalysisPtrUse,
                  int64_t StartOffset) {
    if (AnalysisDepth >= MaxAnalysisDepth) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Sub-analysis depth limit reached at "
            << *Start << "\n");
      return std::nullopt;
    }
    LambdaArgVisitor Visitor(Start, MemoryAnalysisPtrUse, LambdaCB, StartOffset,
                             DL, ClobberOracle, FunctorSize);
    Visitor.AnalysisDepth = AnalysisDepth + 1;
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
    if (AnalysisDepth >= MaxAnalysisDepth) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Sub-analysis depth limit reached at "
            << CB << "\n");
      return std::nullopt;
    }
    FunctionAnalysis Result;
    auto &F = *CB.getCalledFunction();
    // Analyzing one return of a multi-return callee would specialize every
    // path through it from a single path's provenance. Require exactly one
    // value-returning ReturnInst, matching the forward use visitor.
    const auto Returns = getReturnInstructions(F);
    if (Returns.size() != 1 || !Returns.front()->getReturnValue()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Expected exactly one value-returning "
               "ReturnInst in "
            << F.getName() << "; found " << Returns.size() << "\n");
      return std::nullopt;
    }
    LambdaArgVisitor Visitor(Returns.front()->getReturnValue(),
                             MemoryAnalysisPtrUse, LambdaCB, StartOffset, DL,
                             ClobberOracle, FunctorSize);
    Visitor.AnalysisDepth = AnalysisDepth + 1;
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
  LambdaArgVisitor(CallBase *LambdaCB, Module &M, uint64_t FunctorSize_)
      : LambdaCB(LambdaCB), DL(M.getDataLayout()), FunctorSize(FunctorSize_),
        ClobberOracle(std::make_shared<MemorySSAClobberOracle>()), Offset(0) {

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
      fail();
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
    fail();
  }

  void visitLoadInst(LoadInst &LI) {
    DEBUG(Logger::logs("proteus-pass") << "Load inst analysis \n")
    MemorySSAClobber Clobber = ClobberOracle->query(LI);
    if (Clobber.Kind == MemorySSAClobberKind::LiveOnEntry) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Load reads live-on-entry memory; "
               "continuing from its address\n");
      WorkList.push_back({LI.getPointerOperand(), &LI});
      return;
    }

    if (Clobber.Kind == MemorySSAClobberKind::Definition) {
      int64_t AddressOffset = 0;
      Value *AddressBase = GetPointerBaseWithConstantOffset(
          LI.getPointerOperand(), AddressOffset, DL);
      if (!AddressBase) {
        fail();
        return;
      }

      // MemorySSA tells us which memory definition reaches this load, but a
      // definition may be a store, memory transfer, or a call that writes the
      // tracked storage.  Let LambdaInstUseVisitor interpret that definition
      // and cross any required call boundary.  Start at the underlying object
      // so the forward use walk can see every pointer expression used to
      // reach the selected write.
      int64_t OriginalOffset = Offset;
      int64_t TargetOffset = AddressOffset;
      if (!LI.getType()->isPointerTy())
        TargetOffset += OriginalOffset;
      DEBUG({
        auto &OS = Logger::logs("proteus-pass");
        OS << "[Lambda arg analysis]: Resolving load definition from base "
           << *AddressBase << " at byte offset " << TargetOffset
           << "; MemorySSA selected ";
        if (Clobber.Instruction)
          OS << *Clobber.Instruction;
        else
          OS << "<no instruction>";
        OS << "\n";
      });

      // The entity under resolution is the functor storage for value loads,
      // or the spilled pointer itself for pointer loads -- the pointer's own
      // store size, so partial corruption of a spill slot is range-checked.
      uint64_t QueryEntitySize = LI.getType()->isPointerTy()
                                     ? DL.getTypeStoreSize(LI.getType())
                                     : FunctorSize;
      std::optional<TrackedRange> PostLoadRange;
      if (LI.getType()->isPointerTy())
        PostLoadRange = TrackedRange{OriginalOffset, FunctorSize};
      auto Res =
          getDominatingUse(DL, AddressBase, &LI, TargetOffset, LambdaCB,
                           ClobberOracle, QueryEntitySize, PostLoadRange);
      if (!Res) {
        fail();
        return;
      }

      WorkList.push_back({Res->DominatingWrite, Res->WriteBoundary});
      Offset = OriginalOffset + AddressOffset - Res->Offset;
      return;
    }

    if (Clobber.Kind == MemorySSAClobberKind::Phi) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: MemorySSA selected a MemoryPhi for "
            << LI << "\n");
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: MemorySSA clobber query failed for "
            << LI << "\n");
    }

    // Unknown and MemoryPhi results cannot identify one reaching value.
    fail();
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    APInt StepOffset(DL.getIndexTypeSizeInBits(GEP.getType()), 0);
    if (!GEP.accumulateConstantOffset(DL, StepOffset)) {
      fail();
      return;
    }
    Offset += StepOffset.getSExtValue();
    WorkList.push_back({GEP.getPointerOperand(), MemoryAnalysisPtrUse});
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
    fail();
  }

  // todo: these three methods need to be changed to find a dominating store
  void visitAllocaInst(AllocaInst &Alloca) {
    auto Res = getDominatingUse(DL, &Alloca, MemoryAnalysisPtrUse, Offset,
                                LambdaCB, ClobberOracle, FunctorSize);
    if (!Res)
      return;

    WorkList.push_back({Res->DominatingWrite, Res->WriteBoundary});
    // Res->Offset converts the current allocation-relative byte offset into
    // the coordinate system of DominatingWrite. For a field store it removes
    // the field displacement; for a memory transfer it translates destination
    // displacement into the corresponding source displacement.
    Offset -= Res->Offset;
  }

  void visitBitCastInst(BitCastInst &BC) {
    auto Res = getDominatingUse(DL, &BC, MemoryAnalysisPtrUse, Offset, LambdaCB,
                                ClobberOracle, FunctorSize);
    if (!Res)
      return;
    WorkList.push_back({Res->DominatingWrite, Res->WriteBoundary});
    // Res->Offset converts the current cast-relative byte offset into the
    // coordinate system of DominatingWrite.
    Offset -= Res->Offset;
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // An address-space cast of an SSA argument does not load through the
    // pointer. Continue along the direct use-def edge; there is no memory
    // definition for MemorySSA to resolve.
    if (isa<Argument>(ASC.getPointerOperand())) {
      WorkList.push_back({ASC.getPointerOperand(), &ASC});
      return;
    }

    auto Res = getDominatingUse(DL, &ASC, MemoryAnalysisPtrUse, Offset,
                                LambdaCB, ClobberOracle, FunctorSize);
    if (!Res)
      return;

    WorkList.push_back({Res->DominatingWrite, Res->WriteBoundary});
    // Res->Offset converts the current cast-relative byte offset into the
    // coordinate system of DominatingWrite.
    Offset -= Res->Offset;
  }

  void visitIntToPtr(IntToPtrInst &ITP) {
    auto *IntegerVal = ITP.getOperand(0);
    auto *Ptr = dyn_cast<PtrToIntInst>(IntegerVal);
    if (!Ptr) {
      fail();
      return;
    }
    ChangedRC = convertTypeToRuntimeConstantType(Ptr->getType());

    WorkList.push_back({Ptr->getPointerOperand(), &ITP});
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    auto *MT = dyn_cast<MemTransferInst>(&I); // memcpy/memmove
    if (!MT) {
      // memset doesn't preserve any src->dst relationship we can use
      fail();
      return;
    }

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase) {
      fail();
      return;
    }

    // Only valid if the tracked byte lies within the copied region AND the
    // copy spans the tracked functor's full extent; a shorter copy (the
    // classic wrong-sizeof bug) splices two sources into one functor and the
    // byte-0 witness would lie for the uncovered slots.
    if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
      uint64_t Len = LenC->getZExtValue();
      if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len ||
          uint64_t(Offset - DstOff) + FunctorSize > Len) {
        fail();
        return;
      }
    }

    WorkList.push_back({SrcBase->stripPointerCasts(), &I});
    Offset = Offset - DstOff + SrcOff;
  }

  void visitIntrinsicInst(IntrinsicInst &) {
    fail();
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

    // Only direct callsites of F can supply this argument. Any other use --
    // F passed as a call operand, its address stored -- means the address
    // escapes, so an unseen indirect callsite could also supply the lambda
    // and single-caller reasoning no longer holds.
    SmallVector<CallBase *, 2> CallSites;
    bool AddressEscapes = false;
    for (Use &U : F->uses()) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      if (!CB || !CB->isCallee(&U)) {
        AddressEscapes = true;
        continue;
      }
      if (ArgNum < CB->arg_size())
        CallSites.push_back(CB);
    }

    // This visitor does not carry call context. Following more than one
    // caller would let worklist order choose which actual argument supplies
    // the lambda, so reject that shape conservatively.
    if (AddressEscapes || CallSites.size() != 1) {
      DEBUG(Logger::logs("proteus-pass")
            << "[Lambda arg analysis]: Expected one direct caller for argument "
            << A << "; found " << CallSites.size()
            << (AddressEscapes ? " plus address-escaping uses" : "") << "\n");
      fail();
      return;
    }

    CallBase *CB = CallSites.front();
    DEBUG(Logger::logs("proteus-pass")
          << "Analysis crossed interprocedural boundary at "
          << *CB->getArgOperand(ArgNum) << "\n");
    WorkList.push_back({CB->getArgOperand(ArgNum), CB});
  }

  void visitInstruction(Instruction &I) {
    DEBUG(Logger::logs("proteus-pass")
          << "[Lambda arg analysis]: Unhandled instruction "
          << I.getOpcodeName() << ": " << I << "\n");
    fail();
    return;
  }

  void visitPHINode(PHINode &P) {
    if (P.getNumIncomingValues() == 0) {
      fail();
      return;
    }

    auto FirstAnalysis =
        cloneAndAnalyze(P.getIncomingValue(0), MemoryAnalysisPtrUse, Offset);
    if (!FirstAnalysis) {
      fail();
      return;
    }
    auto BaseSlot = FirstAnalysis->KernelArgIndex;
    auto BaseOffset = FirstAnalysis->Offset;

    for (size_t Idx = 1; Idx < P.getNumIncomingValues(); ++Idx) {
      auto Analysis = cloneAndAnalyze(P.getIncomingValue(Idx),
                                      MemoryAnalysisPtrUse, Offset);
      if (!Analysis || Analysis->KernelArgIndex != BaseSlot ||
          Analysis->Offset != BaseOffset) {
        fail();
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
      fail();
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
    const SmallVector<CallBase *> &CBToAnalyze, uint64_t FunctorStructSize) {
  DEBUG(Logger::logs("proteus-pass") << "Beginning analysis " << "\n");
  for (auto *FunctorCB : CBToAnalyze) {
    LambdaArgVisitor Visitor(FunctorCB, M, FunctorStructSize);
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
