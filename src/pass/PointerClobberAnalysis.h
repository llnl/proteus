#ifndef PROTEUS_POINTERCLOBBERANALYSIS_H
#define PROTEUS_POINTERCLOBBERANALYSIS_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
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

#include "CallWriteAnalysis.h"
#include "ContextualUseDefGraph.h"

#include <functional>
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
  // A pointer loaded before a local allocation's address escapes cannot
  // designate that allocation. BasicAA's whole-function capture answer can
  // miss this when the allocation is passed to the later invocation itself.
  // Restrict this refinement to loaded pointer values: a select/GEP may
  // instead be directly derived from the local allocation.
  const auto *PointerLoad =
      dyn_cast<LoadInst>(getUnderlyingObject(Location.Ptr));
  Value *Destination = nullptr;
  if (auto *SI = dyn_cast<StoreInst>(&I); SI && !SI->isAtomic())
    Destination = SI->getPointerOperand();
  else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
    Destination = MI->getRawDest();
  else if (auto *LI = dyn_cast<LifetimeIntrinsic>(&I))
    Destination = LI->getArgOperand(LI->arg_size() - 1);
  if (PointerLoad && Destination &&
      PointerLoad->getFunction() == I.getFunction()) {
    auto *Local = dyn_cast<AllocaInst>(getUnderlyingObject(Destination));
    if (Local && !PointerMayBeCapturedBefore(Local, /*ReturnCaptures=*/true,
#if LLVM_VERSION_MAJOR < 21
                                             /*StoreCaptures=*/true,
#endif
                                             PointerLoad, &DT,
                                             /*IncludeI=*/false))
      return false;
  }
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

AAResults &getAA() { return AA; }
DominatorTree &getDominatorTree() { return DT; }

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
// CallWriteAnalysis can prove a selected call harmless by inspecting callee
// writes. Recovering replacement values and offsets remains the visitors' job.
class MemorySSAClobberOracle {
  DenseMap<Function *, std::unique_ptr<FunctionMemorySSAState>> States;
  std::unique_ptr<ContextualUseDefGraph> Connections;

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
  ContextualUseDefGraph &getConnections(const DataLayout &DL) {
    if (!Connections)
      Connections = std::make_unique<ContextualUseDefGraph>(
          DL,
          [this](LoadInst &LI) -> Value * {
            if (!LI.getType()->isPointerTy() || LI.isVolatile() ||
                LI.isAtomic() ||
                !isa<AllocaInst>(getUnderlyingObject(LI.getPointerOperand())))
              return nullptr;
            auto Clobber = query(LI);
            auto *SI =
                dyn_cast_or_null<StoreInst>(Clobber.ClobberingInstruction);
            if (Clobber.Kind != MemorySSAClobberKind::Definition || !SI ||
                SI->isVolatile() || SI->isAtomic() ||
                SI->getValueOperand()->getType() != LI.getType() ||
                SI->getPointerOperand()->stripPointerCasts() !=
                    LI.getPointerOperand()->stripPointerCasts())
              return nullptr;
            return SI->getValueOperand();
          },
          [this](Value *A, Value *B, Function &F) {
            return getState(F).getAA().alias(
                       MemoryLocation::getBeforeOrAfter(A),
                       MemoryLocation::getBeforeOrAfter(B)) ==
                   AliasResult::NoAlias;
          });
    return *Connections;
  }

  MemorySSAClobber
  query(const MemoryLocation &Location, Instruction &UseBoundary,
        function_ref<ConnectionResult(CallBase &, const MemoryLocation &)>
            ConnectionQuery = nullptr) {
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

    // Refine a possible call clobber by checking its writes in this call
    // context. Only a complete proof permits continuing before the call.
    auto GetDT = [&](Function &F) -> DominatorTree & {
      return getState(F).getDominatorTree();
    };
    CallWriteAnalysis CallWrites(
        State.getAA(), UseBoundary.getModule()->getDataLayout(), GetDT);
    SmallPtrSet<MemoryAccess *, 16> Active;
    unsigned Budget = 256;
    std::function<MemoryAccess *(MemoryAccess *)> Refine =
        [&](MemoryAccess *Start) -> MemoryAccess * {
      MemoryAccess *Clobber = FindClobber(Start);
      if (!Budget || !Active.insert(Clobber).second)
        return Clobber;
      --Budget;
      MemoryAccess *Result = Clobber;
      if (auto *Def = dyn_cast<MemoryDef>(Clobber)) {
        auto *CB = dyn_cast_or_null<CallBase>(Def->getMemoryInst());
        if (CB) {
          ConnectionResult Connectivity = ConnectionQuery
                                              ? ConnectionQuery(*CB, Location)
                                              : ConnectionResult::Unresolved;
          // A completed disconnection proof from collection can discharge a
          // clobber. Absence from an incomplete candidate set cannot. Retain
          // existing call-effect proofs for cases the graph does not cover.
          if (Connectivity == ConnectionResult::Disconnected ||
              CallWrites.cannotModify(*CB, Location))
            Result = Refine(Def->getDefiningAccess());
        }
      } else if (auto *Phi = dyn_cast<MemoryPhi>(Clobber)) {
        // A call on just one arm can create a MemoryPhi even when inspecting
        // that call proves it harmless. Join only identical reaching accesses;
        // distinct stores, unresolved calls, and cycles remain conservative.
        MemoryAccess *Common = nullptr;
        for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
          MemoryAccess *Incoming = Refine(Phi->getIncomingValue(I));
          if (!Common)
            Common = Incoming;
          else if (Incoming != Common) {
            Common = Phi;
            break;
          }
        }
        if (Common)
          Result = Common;
      }
      Active.erase(Clobber);
      return Result;
    };
    MemoryAccess *Clobber = Refine(Before);
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
