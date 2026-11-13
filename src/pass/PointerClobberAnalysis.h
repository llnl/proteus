#ifndef PROTEUS_POINTERCLOBBERANALYSIS_H
#define PROTEUS_POINTERCLOBBERANALYSIS_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/CaptureTracking.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>

namespace proteus {
using namespace llvm;

inline SmallVector<ReturnInst *> getReturnInstructions(Function &F) {
  SmallVector<ReturnInst *> Result;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        Result.push_back(RI);
      }
    }
  }
  return Result;
}

// Own the function analyses required by MemorySSA. The oracle is used by a
// module pass, so these cannot be obtained from a FunctionAnalysisManager.
// Their lifetimes must also extend beyond MemorySSA because it retains
// references to the alias-analysis results.
class FunctionMemorySSAState {
  const DataLayout &DL;
  DominatorTree DT;
  AssumptionCache AC;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  AAResults AA;
  BasicAAResult BAA;
  TypeBasedAAResult TBAA;
  std::unique_ptr<MemorySSA> MSSA;

  // Prove that a wrapper invocation never writes through its closure
  // argument. MemorySSA otherwise treats an earlier invocation on one field
  // of an aggregate as a possible write to the next field.
  static bool readsArgumentPointeeOnly(Function &F, unsigned ArgNo,
                                       SmallPtrSetImpl<Function *> &Active) {
    if (F.isDeclaration() || ArgNo >= F.arg_size() || !Active.insert(&F).second)
      return false;
    Argument *Formal = F.getArg(ArgNo);
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          const Value *DestBase = getUnderlyingObject(SI->getPointerOperand());
          if (!isa<AllocaInst>(DestBase) ||
              (SI->getValueOperand()->getType()->isPointerTy() &&
               getUnderlyingObject(SI->getValueOperand()) == Formal)) {
            Active.erase(&F);
            return false;
          }
          continue;
        }
        if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
          const Value *DestBase = getUnderlyingObject(MI->getRawDest());
          if (!isa<AllocaInst>(DestBase)) {
            Active.erase(&F);
            return false;
          }
          continue;
        }
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          if (CB->isInlineAsm()) {
            Active.erase(&F);
            return false;
          }
          for (unsigned Index = 0; Index < CB->arg_size(); ++Index) {
            Value *Actual = CB->getArgOperand(Index);
            if (!Actual->getType()->isPointerTy())
              continue;
            const Value *Base = getUnderlyingObject(Actual);
            if (Base == Formal) {
              Function *Callee = CB->getCalledFunction();
              if (!Callee ||
                  !readsArgumentPointeeOnly(*Callee, Index, Active)) {
                Active.erase(&F);
                return false;
              }
            } else if (isa<LoadInst>(Base) || isa<Argument>(Base)) {
              Active.erase(&F);
              return false;
            }
          }
          continue;
        }
        if (I.mayWriteToMemory()) {
          Active.erase(&F);
          return false;
        }
      }
    }
    Active.erase(&F);
    return true;
  }

public:
  explicit FunctionMemorySSAState(Function &F)
      : DL(F.getParent()->getDataLayout()), DT(F), AC(F),
        TLII(Triple(F.getParent()->getTargetTriple())), TLI(TLII, &F), AA(TLI),
#if LLVM_VERSION_MAJOR >= 20
        BAA(DL, F, TLI, AC, &DT), TBAA(/*UsingTypeSanitizer=*/false){
#else
        BAA(DL, F, TLI, AC, &DT) {
#endif
                                      AA.addAAResult(BAA);
  AA.addAAResult(TBAA);
  MSSA = std::make_unique<MemorySSA>(F, &AA, &DT);
}

MemorySSA &
getMemorySSA() {
  return *MSSA;
}

bool instructionMayModify(Instruction &I, const MemoryLocation &Location) {
  if (auto *SI = dyn_cast<StoreInst>(&I);
      SI && Location.Size.hasValue() && Location.Size.isPrecise() &&
      !Location.Size.getValue().isScalable()) {
    int64_t StoreOffset = 0;
    int64_t LocationOffset = 0;
    const Value *StoreBase = GetPointerBaseWithConstantOffset(
        SI->getPointerOperand(), StoreOffset, DL);
    const Value *LocationBase =
        GetPointerBaseWithConstantOffset(Location.Ptr, LocationOffset, DL);
    TypeSize StoreSize = DL.getTypeStoreSize(SI->getValueOperand()->getType());

    if (StoreBase && StoreBase == LocationBase && !StoreSize.isScalable()) {
      uint64_t StoreBytes = StoreSize.getFixedValue();
      uint64_t LocationBytes = Location.Size.getValue().getFixedValue();
      bool Disjoint = StoreOffset <= LocationOffset
                          ? static_cast<uint64_t>(LocationOffset) -
                                    static_cast<uint64_t>(StoreOffset) >=
                                StoreBytes
                          : static_cast<uint64_t>(StoreOffset) -
                                    static_cast<uint64_t>(LocationOffset) >=
                                LocationBytes;
      if (Disjoint)
        return false;
    }
  }
  return isModSet(AA.getModRefInfo(&I, Location));
}

bool callCannotModifyTrackedLocal(CallBase &CB,
                                  const MemoryLocation &Location) {
  // Respect any precise memory effects already understood by alias analysis.
  if (!isModSet(AA.getModRefInfo(&CB, Location)))
    return true;

  // Do not use capture tracking to override an explicit inline-assembly
  // memory clobber. Inline assembly may access memory without exposing the
  // address as an SSA operand, so the absence of a captured pointer is not
  // sufficient proof that the tracked allocation is untouched.
  if (Function *Callee = CB.getCalledFunction()) {
    for (BasicBlock &BB : *Callee) {
      for (Instruction &I : BB) {
        auto *InnerCB = dyn_cast<CallBase>(&I);
        if (InnerCB && InnerCB->isInlineAsm() && I.mayWriteToMemory())
          return false;
      }
    }
  }

  // This additional proof is valid only for function-local allocations. A
  // call cannot access such an allocation unless its address escaped earlier
  // or an alias is supplied by the call itself.
  const Value *Object = getUnderlyingObject(Location.Ptr);
  auto *Alloca = dyn_cast<AllocaInst>(Object);
  if (!Alloca || CB.hasOperandBundles())
    return false;

  // `nocapture` would not be enough here: a callee may modify a pointer
  // argument without retaining it. Require every pointer argument to be
  // disjoint from the tracked location.
  for (const Use &Arg : CB.args()) {
    Value *Actual = Arg.get();
    if (!Actual->getType()->isPointerTy())
      continue;
    if (!AA.isNoAlias(MemoryLocation::getBeforeOrAfter(Actual), Location))
      return false;
  }

  // The current call is excluded because its operands were checked above.
  // Any earlier capture could have made the allocation reachable indirectly.
  return !PointerMayBeCapturedBefore(Alloca, /*ReturnCaptures=*/true,
#if LLVM_VERSION_MAJOR < 21
                                     /*StoreCaptures=*/true,
#endif
                                     &CB, &DT, /*IncludeI=*/false);
}

bool wrapperCallOnlyReadsClosure(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  if (!Callee || !Callee->hasMetadata("proteus.wrapper_call") ||
      CB.arg_size() != 1)
    return false;
  SmallPtrSet<Function *, 8> Active;
  return readsArgumentPointeeOnly(*Callee, 0, Active);
}
}; // namespace proteus

enum class MemorySSAClobberKind {
  // No definition in this function reaches the queried location.
  LiveOnEntry,

  // One MemoryDef reaches the query. ClobberingInstruction identifies the
  // defining instruction, but the oracle deliberately does not interpret its
  // effect.
  Definition,

  // Control-flow paths carry different reaching memory versions.
  Phi,

  // The query could not be represented in MemorySSA.
  Unknown,
};

struct MemorySSAClobber {
  MemorySSAClobberKind Kind = MemorySSAClobberKind::Unknown;
  MemoryAccess *Access = nullptr;
  Instruction *ClobberingInstruction = nullptr;
};

// A deliberately small, function-local MemorySSA oracle. It answers only:
//
//   "Which memory access may clobber this location immediately before this
//    instruction?"
//
// It does not interpret stores, traverse pointer provenance, summarize calls,
// or cross function boundaries. Those decisions belong to LambdaArgVisitor's
// frame/provenance traversal.
class MemorySSAClobberOracle {
  DenseMap<Function *, std::unique_ptr<FunctionMemorySSAState>> States;

  FunctionMemorySSAState &getState(Function &F) {
    auto &State = States[&F];
    if (!State)
      State = std::make_unique<FunctionMemorySSAState>(F);
    return *State;
  }

  static MemoryAccess *getStateBefore(MemorySSA &MSSA, Instruction &Boundary) {
    BasicBlock *BB = Boundary.getParent();
    DominatorTree &DT = MSSA.getDomTree();
    if (!DT.isReachableFromEntry(BB))
      return nullptr;

    if (auto *BoundaryAccess =
            dyn_cast_or_null<MemoryUseOrDef>(MSSA.getMemoryAccess(&Boundary)))
      return BoundaryAccess->getDefiningAccess();

    Instruction *Last = Boundary.getPrevNode();
    while (true) {
      for (Instruction *I = Last; I; I = I->getPrevNode())
        if (auto *Def = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(I)))
          return Def;

      if (MemoryPhi *Phi = MSSA.getMemoryAccess(BB))
        return Phi;
      if (BB == &BB->getParent()->getEntryBlock())
        return MSSA.getLiveOnEntryDef();

      // Without a MemoryPhi, the block inherits the memory state from its
      // immediate dominator, even when it has multiple predecessors. Inspect
      // the dominator's outgoing state, including a memory-defining terminator.
      auto *Node = DT.getNode(BB);
      if (!Node || !Node->getIDom())
        return nullptr;
      BB = Node->getIDom()->getBlock();
      Last = BB->getTerminator();
    }
  }

public:
  MemorySSAClobber query(const MemoryLocation &Location,
                         Instruction &UseBoundary) {
    FunctionMemorySSAState &State = getState(*UseBoundary.getFunction());
    MemorySSA &MSSA = State.getMemorySSA();
    MemoryAccess *Before = getStateBefore(MSSA, UseBoundary);
    if (!Before)
      return {};

    MemorySSAWalker *Walker = MSSA.getWalker();

    // The location-aware walker requires a MemoryDef starting point to be an
    // already-potential clobber. A definition obtained merely because it is
    // the last access before UseBoundary does not satisfy that contract.
    // LLVM 19 happened to walk past non-aliasing starting definitions, while
    // newer walkers may return the starting definition unchanged.
    auto FindClobber = [&](MemoryAccess *Start) {
      while (auto *Def = dyn_cast<MemoryDef>(Start)) {
        Instruction *I = Def->getMemoryInst();
        if (!I || State.instructionMayModify(*I, Location))
          break;
        Start = Def->getDefiningAccess();
      }
      return Walker->getClobberingMemoryAccess(Start, Location);
    };

    MemoryAccess *Clobber = FindClobber(Before);

    // MemorySSA must conservatively model calls whose memory effects have not
    // yet been inferred. Skip such a call only when we can prove that this
    // particular local allocation was unavailable to it, then continue the
    // clobber walk from the state immediately preceding the call.
    while (auto *Def = dyn_cast_or_null<MemoryDef>(Clobber)) {
      auto *CB = dyn_cast_or_null<CallBase>(Def->getMemoryInst());
      if (!CB || (!State.callCannotModifyTrackedLocal(*CB, Location) &&
                  !State.wrapperCallOnlyReadsClosure(*CB)))
        break;
      Clobber = FindClobber(Def->getDefiningAccess());
    }
    if (!Clobber)
      return {};
    if (MSSA.isLiveOnEntryDef(Clobber))
      return {MemorySSAClobberKind::LiveOnEntry, Clobber, nullptr};
    if (auto *Def = dyn_cast<MemoryDef>(Clobber))
      return {MemorySSAClobberKind::Definition, Clobber, Def->getMemoryInst()};
    if (isa<MemoryPhi>(Clobber))
      return {MemorySSAClobberKind::Phi, Clobber, nullptr};
    return {MemorySSAClobberKind::Unknown, Clobber, nullptr};
  }

  MemorySSAClobber query(LoadInst &Load) {
    return query(MemoryLocation::get(&Load), Load);
  }
};

} // namespace proteus

#endif
