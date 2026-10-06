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
#include <llvm/IR/CFG.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>

namespace proteus {
using namespace llvm;

SmallVector<ReturnInst *> getReturnInstructions(Function &F) {
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
  DominatorTree DT;
  AssumptionCache AC;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  AAResults AA;
  BasicAAResult BAA;
  TypeBasedAAResult TBAA;
  std::unique_ptr<MemorySSA> MSSA;

public:
  explicit FunctionMemorySSAState(Function &F)
      : DT(F), AC(F), TLII(Triple(F.getParent()->getTargetTriple())),
        TLI(TLII, &F), AA(TLI),
        BAA(F.getParent()->getDataLayout(), F, TLI, AC, &DT) {
    AA.addAAResult(BAA);
    AA.addAAResult(TBAA);
    MSSA = std::make_unique<MemorySSA>(F, &AA, &DT);
  }

  MemorySSA &getMemorySSA() { return *MSSA; }

  bool callCannotModifyTrackedLocal(CallBase &CB,
                                    const MemoryLocation &Location) {
    // Respect any precise memory effects already understood by alias analysis.
    if (!isModSet(AA.getModRefInfo(&CB, Location)))
      return true;

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
                                       /*StoreCaptures=*/true, &CB, &DT,
                                       /*IncludeI=*/false);
  }
};

enum class MemorySSAClobberKind {
  // No definition in this function reaches the queried location.
  LiveOnEntry,

  // One MemoryDef reaches the query. Instruction identifies it, but the
  // oracle deliberately does not interpret the instruction's effect.
  Definition,

  // Control-flow paths carry different reaching memory versions.
  Phi,

  // The query could not be represented in MemorySSA.
  Unknown,
};

struct MemorySSAClobber {
  MemorySSAClobberKind Kind = MemorySSAClobberKind::Unknown;
  MemoryAccess *Access = nullptr;
  Instruction *Instruction = nullptr;
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

  static MemoryAccess *
  getStateBefore(MemorySSA &MSSA, Instruction &Boundary,
                 SmallPtrSetImpl<BasicBlock *> &VisitedBlocks) {
    if (auto *BoundaryAccess =
            dyn_cast_or_null<MemoryUseOrDef>(MSSA.getMemoryAccess(&Boundary)))
      return BoundaryAccess->getDefiningAccess();

    for (Instruction *I = Boundary.getPrevNode(); I; I = I->getPrevNode())
      if (auto *Def = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(I)))
        return Def;

    BasicBlock *BB = Boundary.getParent();
    if (MemoryPhi *Phi = MSSA.getMemoryAccess(BB))
      return Phi;
    if (!VisitedBlocks.insert(BB).second || pred_empty(BB))
      return MSSA.getLiveOnEntryDef();
    if (BasicBlock *Pred = BB->getSinglePredecessor())
      return getStateBefore(MSSA, *Pred->getTerminator(), VisitedBlocks);
    return MSSA.getLiveOnEntryDef();
  }

public:
  MemorySSAClobber query(const MemoryLocation &Location,
                         Instruction &UseBoundary) {
    FunctionMemorySSAState &State = getState(*UseBoundary.getFunction());
    MemorySSA &MSSA = State.getMemorySSA();
    SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
    MemoryAccess *Before = getStateBefore(MSSA, UseBoundary, VisitedBlocks);
    if (!Before)
      return {};

    MemorySSAWalker *Walker = MSSA.getWalker();
    MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(Before, Location);

    // MemorySSA must conservatively model calls whose memory effects have not
    // yet been inferred. Skip such a call only when we can prove that this
    // particular local allocation was unavailable to it, then continue the
    // clobber walk from the state immediately preceding the call.
    while (auto *Def = dyn_cast_or_null<MemoryDef>(Clobber)) {
      auto *CB = dyn_cast_or_null<CallBase>(Def->getMemoryInst());
      if (!CB || !State.callCannotModifyTrackedLocal(*CB, Location))
        break;
      Clobber =
          Walker->getClobberingMemoryAccess(Def->getDefiningAccess(), Location);
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
