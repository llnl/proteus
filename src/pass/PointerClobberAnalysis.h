#ifndef PROTEUS_POINTERCLOBBERANALYSIS_H
#define PROTEUS_POINTERCLOBBERANALYSIS_H

#include "Helpers.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/CaptureTracking.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Value.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

inline bool offsetCoveredByRange(int64_t TargetOffset, int64_t RangeOffset,
                                 uint64_t RangeSize) {
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Target Offset = " << TargetOffset << "\n");
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Range Offset = " << RangeOffset << "\n");
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Range Size = " << RangeSize << "\n");
  return TargetOffset >= RangeOffset &&
         static_cast<uint64_t>(TargetOffset - RangeOffset) < RangeSize;
}

inline std::optional<uint64_t> getTypeStoreSize(const DataLayout &DL,
                                                Type *Ty) {
  if (!Ty || !Ty->isSized())
    return std::nullopt;
  return static_cast<uint64_t>(DL.getTypeStoreSize(Ty));
}

inline std::optional<MemoryLocation>
getTrackedPointerLocation(const DataLayout &DL, Value *Ptr) {
  if (!Ptr || !Ptr->getType()->isPointerTy())
    return std::nullopt;

  Type *PointeeTy = nullptr;
  if (auto *AI = dyn_cast<AllocaInst>(Ptr))
    PointeeTy = AI->getAllocatedType();

  if (!PointeeTy || !PointeeTy->isSized())
    return MemoryLocation::getBeforeOrAfter(Ptr);

  return MemoryLocation(Ptr,
                        LocationSize::precise(DL.getTypeStoreSize(PointeeTy)));
}

enum class PointerClobberKind { Value, Incoming, Cycle, Ambiguous, Unknown };

// A clobber query either resolves the value stored at a byte offset or returns
// the exact MemorySSA-selected instruction for a provenance visitor to
// interpret. ClobberPointer is the pointer operand through which that
// instruction accesses the tracked storage, and ClobberOffset is relative to
// that operand.
struct PointerClobberResult {
  PointerClobberKind Kind = PointerClobberKind::Unknown;
  llvm::Value *V = nullptr;
  int64_t Offset = 0;
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
  llvm::Instruction *ClobberingInstruction = nullptr;
  llvm::Value *ClobberPointer = nullptr;
  int64_t ClobberOffset = 0;
};

// A write discovered while following the relevant uses of a newly encountered
// pointer definition. Pointer is the operand through which the instruction
// accesses the tracked storage, and TargetOffset is relative to that operand.
struct PointerClobberCandidate {
  llvm::Instruction *I = nullptr;
  llvm::Value *Pointer = nullptr;
  int64_t TargetOffset = 0;
};

using PointerClobberCandidateMap =
    llvm::SmallDenseMap<llvm::Instruction *, PointerClobberCandidate, 8>;

class PointerClobberAnalysis {
public:
  virtual ~PointerClobberAnalysis() = default;

  virtual PointerClobberResult
  resolve(llvm::Value *Ptr, llvm::Instruction &UseBoundary,
          int64_t TargetOffset,
          const PointerClobberCandidateMap *Candidates = nullptr) = 0;
};

struct ReachingPointerStores {
  SmallVector<Value *, 4> Values;
  bool Complete = true;
};

// Return whether LHS and RHS name exactly the same byte address after peeling
// constant-offset pointer arithmetic and casts.
inline bool isSamePointerAddress(const DataLayout &DL, Value *LHS, Value *RHS) {
  int64_t LHSOffset = 0;
  int64_t RHSOffset = 0;
  Value *LHSBase = GetPointerBaseWithConstantOffset(LHS, LHSOffset, DL);
  Value *RHSBase = GetPointerBaseWithConstantOffset(RHS, RHSOffset, DL);
  return LHSBase && RHSBase && LHSBase == RHSBase && LHSOffset == RHSOffset;
}

// Collect the closest pointer-valued store to Address on every CFG path that
// reaches Before. A path with no store, or a revisited block (such as a loop),
// marks the result incomplete so callers conservatively decline rather than
// infer a store that is not guaranteed to reach the load.
inline void collectReachingPointerStores(const DataLayout &DL, BasicBlock *BB,
                                         Instruction *Before, Value *Address,
                                         SmallPtrSetImpl<BasicBlock *> &Visited,
                                         ReachingPointerStores &Result) {
  if (!Visited.insert(BB).second) {
    Result.Complete = false;
    return;
  }

  for (Instruction *I = Before ? Before->getPrevNode() : BB->getTerminator(); I;
       I = I->getPrevNode()) {
    auto *SI = dyn_cast<StoreInst>(I);
    if (SI && SI->getValueOperand()->getType()->isPointerTy() &&
        isSamePointerAddress(DL, SI->getPointerOperand(), Address)) {
      Result.Values.push_back(SI->getValueOperand());
      return;
    }
  }

  if (pred_empty(BB)) {
    Result.Complete = false;
    return;
  }
  for (BasicBlock *Pred : predecessors(BB))
    collectReachingPointerStores(DL, Pred, nullptr, Address, Visited, Result);
}

// Resolve pointer spills by walking backwards from the load through the CFG.
// This finds the nearest store on every incoming path instead of depending on
// the arbitrary order in which Value::users() happens to enumerate writes.
// The result is complete only when every incoming path contributes a store.
inline ReachingPointerStores getReachingPointerStores(const DataLayout &DL,
                                                      LoadInst &LI) {
  ReachingPointerStores Result;
  SmallPtrSet<BasicBlock *, 8> Visited;
  collectReachingPointerStores(DL, LI.getParent(), &LI, LI.getPointerOperand(),
                               Visited, Result);
  return Result;
}

inline Value *getPointerLoadOrigin(const DataLayout &DL, Value *V,
                                   SmallPtrSetImpl<Value *> &Visited);

// Return one source pointer when every reaching store has the same origin.
// Nested pointer-spill loads are recursively resolved; distinct origins or
// cycles are ambiguous and return nullptr.
inline Value *getUniqueReachingPointer(const DataLayout &DL,
                                       const ReachingPointerStores &Stores,
                                       SmallPtrSetImpl<Value *> &Visited) {
  if (!Stores.Complete || Stores.Values.empty())
    return nullptr;

  Value *First = getPointerLoadOrigin(DL, Stores.Values.front(), Visited);
  if (!First)
    return nullptr;
  for (Value *V : drop_begin(Stores.Values)) {
    Value *Origin = getPointerLoadOrigin(DL, V, Visited);
    if (Origin != First)
      return nullptr;
  }
  return First;
}

// Resolve a pointer value through nested pointer-spill loads. Non-load pointer
// values are already origins. Visited prevents cyclic spill graphs from being
// mistaken for a unique source.
inline Value *getPointerLoadOrigin(const DataLayout &DL, Value *V,
                                   SmallPtrSetImpl<Value *> &Visited) {
  auto *LI = dyn_cast<LoadInst>(V);
  if (!LI || !LI->getType()->isPointerTy())
    return V;
  if (!Visited.insert(V).second)
    return nullptr;

  ReachingPointerStores Stores = getReachingPointerStores(DL, *LI);
  return getUniqueReachingPointer(DL, Stores, Visited);
}

// Report ambiguity only for complete reaching-store sets. Incomplete sets can
// still be handled by ordinary backward memory-use analysis.
inline bool hasAmbiguousReachingPointers(const DataLayout &DL,
                                         const ReachingPointerStores &Stores) {
  if (!Stores.Complete || Stores.Values.empty())
    return false;
  SmallPtrSet<Value *, 8> Visited;
  return !getUniqueReachingPointer(DL, Stores, Visited);
}

// A compiler spill is a temporary local slot the compiler uses to save an SSA
// pointer value (for example, `alloca ptr`, followed by `store ptr` and a
// later `load ptr`).  A pointer-valued load is not necessarily such a spill:
// it can instead read an ordinary pointer field from a closure/context
// aggregate.  The reaching-store recovery below is only valid for a local
// `alloca ptr` slot; aggregate fields must be traced backwards through their
// address.
inline bool isPointerSpillLoad(const LoadInst &LI) {
  if (!LI.getType()->isPointerTy())
    return false;
  const Value *Storage = getUnderlyingObject(LI.getPointerOperand());
  auto *Slot = dyn_cast_or_null<AllocaInst>(Storage);
  return Slot && Slot->getAllocatedType()->isPointerTy();
}

// Own the analyses needed to build MemorySSA for one function.  Lambda
// provenance is analyzed by a module pass, so it cannot directly request a
// FunctionAnalysisManager result.  Keeping these objects together also keeps
// every reference held by AAResults and MemorySSA valid.
class FunctionMemorySSAState {
  DominatorTree DT;
  AssumptionCache AC;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  AAResults AA;
  BasicAAResult BAA;
  std::unique_ptr<MemorySSA> MSSA;

public:
  explicit FunctionMemorySSAState(Function &F)
      : DT(F), AC(F), TLII(Triple(F.getParent()->getTargetTriple())),
        TLI(TLII, &F), AA(TLI),
        BAA(F.getParent()->getDataLayout(), F, TLI, AC, &DT) {
    AA.addAAResult(BAA);
    MSSA = std::make_unique<MemorySSA>(F, &AA, &DT);
  }

  MemorySSA &get() { return *MSSA; }
  AAResults &getAA() { return AA; }
  DominatorTree &getDT() { return DT; }
};

// Resolve the pointer definition reaching a pointer-valued load.  MemorySSA
// supplies write order and CFG joins within each function.  Direct calls are
// summarized by resolving the tracked formal argument at every callee return;
// the callee's MemorySSA remains separate from the caller's graph.
class PointerClobberResolver final : public PointerClobberAnalysis {
  const DataLayout &DL;
  DenseMap<Function *, std::unique_ptr<FunctionMemorySSAState>> States;
  SmallPtrSet<LoadInst *, 8> ResolvingOrigins;
  SmallPtrSet<CallBase *, 8> ResolvingCallOrigins;

  static const char *kindName(PointerClobberKind Kind) {
    switch (Kind) {
    case PointerClobberKind::Value:
      return "Value";
    case PointerClobberKind::Incoming:
      return "Incoming";
    case PointerClobberKind::Cycle:
      return "Cycle";
    case PointerClobberKind::Ambiguous:
      return "Ambiguous";
    case PointerClobberKind::Unknown:
      return "Unknown";
    }
    llvm_unreachable("invalid PointerClobberKind");
  }

  FunctionMemorySSAState &getState(Function &F) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter getState; function=" << F.getName()
          << "\n");
    auto &State = States[&F];
    if (!State) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: getState creating analysis state\n");
      State = std::make_unique<FunctionMemorySSAState>(F);
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: getState reusing analysis state\n");
    }
    return *State;
  }

  static PointerClobberResult unresolvedClobber(Instruction &I) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter unresolvedClobber; instruction=" << I
          << "\n");
    // MemorySSA selected I as the reaching clobber, but this analysis cannot
    // model its effect.  This is a fatal ambiguity, not an invitation to use
    // an older definition.  Unknown is reserved for supported instructions
    // (currently memory transfers) that LambdaInstUseVisitor can interpret.
    PointerClobberResult Result{PointerClobberKind::Ambiguous};
    Result.ClobberingInstruction = &I;
    return Result;
  }

  static bool
  mayWriteMemoryTransitively(Function &F,
                             SmallPtrSetImpl<Function *> &ActiveFunctions) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter mayWriteMemoryTransitively; function="
          << F.getName() << "\n");
    if (F.onlyReadsMemory()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: function is read-only\n");
      return false;
    }
    if (F.isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: function is a declaration; assuming it "
               "may write\n");
      return true;
    }
    if (!ActiveFunctions.insert(&F).second) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: recursive function traversal; assuming "
               "it may write\n");
      return true;
    }

    for (Instruction &I : instructions(F)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: may-write loop instruction=" << I << "\n");
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: instruction is not a call; mayWrite="
              << I.mayWriteToMemory() << "\n");
        if (I.mayWriteToMemory()) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: non-call instruction may write; "
                   "returning true\n");
          ActiveFunctions.erase(&F);
          return true;
        }
        continue;
      }

      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: instruction is a call; onlyReadsMemory="
            << CB->onlyReadsMemory() << "\n");
      if (CB->onlyReadsMemory()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: skipping read-only call\n");
        continue;
      }
      Function *Callee = CB->getCalledFunction();
      if (!Callee) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: indirect call may write\n");
        ActiveFunctions.erase(&F);
        return true;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: recursively checking callee="
            << Callee->getName() << "\n");
      if (mayWriteMemoryTransitively(*Callee, ActiveFunctions)) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: callee may write\n");
        ActiveFunctions.erase(&F);
        return true;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: callee is transitively read-only\n");
    }

    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: no transitive write found\n");
    ActiveFunctions.erase(&F);
    return false;
  }

  static bool canProveDisjointFromUncapturedLocal(
      FunctionMemorySSAState &State, CallBase &CB,
      const MemoryLocation &TrackedLocation, Value *TrackedPtr) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter "
             "canProveDisjointFromUncapturedLocal; tracked="
          << *TrackedPtr << "; call=" << CB << "\n");
    Value *Underlying = getUnderlyingObject(TrackedPtr);
    if (!isa<AllocaInst>(Underlying)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: tracked root is not an alloca\n");
      return false;
    }
    if (PointerMayBeCapturedBefore(Underlying, true, true, &CB, &State.getDT(),
                                   true)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: tracked alloca may be captured before "
               "the call\n");
      return false;
    }

    // Capture alone is insufficient: a nocapture pointer argument may still
    // modify the pointee during the call. Require every pointer actual to be
    // disjoint from the tracked allocation. Requiring at least one pointer
    // actual also keeps opaque no-argument calls on the conservative path.
    bool SawPointerArgument = false;
    for (Value *Actual : CB.args()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: disjoint-actual loop value=" << *Actual
            << "\n");
      if (!Actual->getType()->isPointerTy()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: actual is non-pointer; skipping\n");
        continue;
      }
      SawPointerArgument = true;
      bool NoAlias = State.getAA().isNoAlias(
          TrackedLocation, MemoryLocation::getBeforeOrAfter(Actual));
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer actual NoAlias=" << NoAlias
            << "\n");
      if (!NoAlias) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: disjoint-local proof failed on this "
                 "actual\n");
        return false;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: actual is disjoint; continuing loop\n");
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: disjoint-local proof result="
          << SawPointerArgument << "\n");
    return SawPointerArgument;
  }

  bool isKnownNoAlias(Value *Ptr,
                      const SmallPtrSetImpl<Value *> *KnownNoAliasRoots) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter isKnownNoAlias; pointer=" << *Ptr
          << "; roots=" << (KnownNoAliasRoots ? KnownNoAliasRoots->size() : 0)
          << "\n");
    if (!KnownNoAliasRoots) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no propagated no-alias roots\n");
      return false;
    }
    int64_t IgnoredOffset = 0;
    Ptr = GetPointerBaseWithConstantOffset(Ptr, IgnoredOffset, DL);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: stripped pointer=" << *Ptr
          << "; offset=" << IgnoredOffset << "\n");
    if (Value *Origin = getPointerOrigin(Ptr)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolved pointer origin=" << *Origin
            << "\n");
      Ptr = Origin;
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer origin did not resolve\n");
    }
    Value *Root = getUnderlyingObject(Ptr);
    bool Contains = KnownNoAliasRoots->contains(Root);
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: underlying root=" << *Root
         << "; known=" << Contains << "; root set={";
      for (Value *KnownRoot : *KnownNoAliasRoots)
        OS << " " << *KnownRoot << ";";
      OS << " }\n";
    });
    return Contains;
  }

  static bool haveDistinctAllocaRoots(Value *LHS, Value *RHS) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter haveDistinctAllocaRoots; lhs=" << *LHS
          << "; rhs=" << *RHS << "\n");
    Value *LHSRoot = getUnderlyingObject(LHS);
    Value *RHSRoot = getUnderlyingObject(RHS);
    bool Distinct = LHSRoot != RHSRoot && isa<AllocaInst>(LHSRoot) &&
                    isa<AllocaInst>(RHSRoot);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: distinct alloca roots=" << Distinct
          << "; lhs-root=" << *LHSRoot << "; rhs-root=" << *RHSRoot << "\n");
    return Distinct;
  }

  std::optional<std::pair<Value *, int64_t>>
  getReturnedPointerOrigin(Value *V, CallBase &RootCall,
                           SmallPtrSetImpl<Value *> &Active,
                           bool &IsAmbiguous) {
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: enter getReturnedPointerOrigin; value=";
      if (V)
        OS << *V;
      else
        OS << "<null>";
      OS << "; root-call=" << RootCall << "\n";
    });
    if (!V) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned origin has null value\n");
      IsAmbiguous = true;
      return std::nullopt;
    }
    if (!V->getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned value is not a pointer\n");
      IsAmbiguous = true;
      return std::nullopt;
    }
    if (!Active.insert(V).second) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned-origin cycle detected\n");
      IsAmbiguous = true;
      return std::nullopt;
    }

    int64_t LocalOffset = 0;
    Value *Base = GetPointerBaseWithConstantOffset(V, LocalOffset, DL);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: returned-origin base=" << *Base
          << "; local-offset=" << LocalOffset << "\n");
    std::optional<std::pair<Value *, int64_t>> Result;

    if (auto *A = dyn_cast_or_null<Argument>(Base)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base is argument "
            << A->getArgNo() << "\n");
      Function *RootCallee = RootCall.getCalledFunction();
      if (RootCallee && A->getParent() == RootCallee &&
          A->getArgNo() < RootCall.arg_size()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: argument maps to root-call actual\n");
        Result = {{RootCall.getArgOperand(A->getArgNo()), LocalOffset}};
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: argument does not map to root-call "
                 "actual\n");
      }
    } else if (auto *LI = dyn_cast_or_null<LoadInst>(Base)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base is load=" << *LI << "\n");
      SmallPtrSet<Value *, 8> VisitedLoads;
      Value *Origin = getPointerLoadOrigin(DL, LI, VisitedLoads);
      if (Origin && Origin != LI) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: load origin resolved to " << *Origin
              << "\n");
        Result =
            getReturnedPointerOrigin(Origin, RootCall, Active, IsAmbiguous);
        if (Result) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: adding load-local offset\n");
          Result->second += LocalOffset;
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: recursive load-origin resolution "
                   "failed\n");
        }
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: load origin unresolved or unchanged\n");
      }
    } else if (auto *CB = dyn_cast_or_null<CallBase>(Base)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base is nested call=" << *CB
            << "\n");
      bool NestedAmbiguous = false;
      auto Origin = getCallReturnOrigin(*CB, &NestedAmbiguous);
      if (Origin) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: nested call origin resolved to "
              << *Origin->first << "; offset=" << Origin->second << "\n");
        Result = getReturnedPointerOrigin(Origin->first, RootCall, Active,
                                          IsAmbiguous);
        if (Result) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: adding nested-call offsets\n");
          Result->second += LocalOffset + Origin->second;
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: recursive nested-call origin "
                   "resolution failed\n");
        }
      } else if (NestedAmbiguous) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: nested call origin is ambiguous\n");
        IsAmbiguous = true;
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: nested call has no resolved origin\n");
      }
    } else if (auto *Select = dyn_cast_or_null<SelectInst>(Base)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base is select=" << *Select
            << "\n");
      auto TrueOrigin = getReturnedPointerOrigin(Select->getTrueValue(),
                                                 RootCall, Active, IsAmbiguous);
      auto FalseOrigin = getReturnedPointerOrigin(
          Select->getFalseValue(), RootCall, Active, IsAmbiguous);
      if (TrueOrigin && FalseOrigin && *TrueOrigin == *FalseOrigin) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: select arms have identical origins\n");
        Result = TrueOrigin;
        Result->second += LocalOffset;
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: select arms are unresolved or differ\n");
        IsAmbiguous = true;
      }
    } else if (auto *Phi = dyn_cast_or_null<PHINode>(Base)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base is phi=" << *Phi << "\n");
      for (Value *Incoming : Phi->incoming_values()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: returned-phi loop incoming=" << *Incoming
              << "\n");
        auto IncomingOrigin =
            getReturnedPointerOrigin(Incoming, RootCall, Active, IsAmbiguous);
        if (!IncomingOrigin) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: phi incoming origin unresolved\n");
          IsAmbiguous = true;
          Result.reset();
          break;
        }
        if (!Result) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: recording first phi origin\n");
          Result = IncomingOrigin;
        } else if (*Result != *IncomingOrigin) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: phi origins differ\n");
          IsAmbiguous = true;
          Result.reset();
          break;
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: phi origin matches prior origin\n");
        }
      }
      if (Result) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: adding phi-local offset\n");
        Result->second += LocalOffset;
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: phi produced no unique origin\n");
      }
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned base has unsupported value "
               "shape\n");
    }

    Active.erase(V);
    if (!Result) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned-pointer origin unresolved\n");
      IsAmbiguous = true;
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: returned-pointer origin result="
            << *Result->first << "; offset=" << Result->second << "\n");
    }
    return Result;
  }

  std::optional<std::pair<Value *, int64_t>>
  getCallReturnOrigin(CallBase &CB, bool *IsAmbiguous = nullptr) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter getCallReturnOrigin; call=" << CB
          << "\n");
    Function *Callee = CB.getCalledFunction();
    if (!Callee) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: call is indirect\n");
      return std::nullopt;
    }
    if (Callee->isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: callee is a declaration\n");
      return std::nullopt;
    }
    if (!ResolvingCallOrigins.insert(&CB).second) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: recursive call-origin query\n");
      return std::nullopt;
    }

    std::optional<std::pair<Value *, int64_t>> Result;
    bool SawReturn = false;
    bool Valid = true;
    for (BasicBlock &BB : *Callee) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: call-return block loop block="
            << BB.getName() << "\n");
      auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!Ret) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: block does not end in return\n");
        continue;
      }
      if (!Ret->getReturnValue()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: return has no value\n");
        continue;
      }

      SawReturn = true;
      bool ReturnAmbiguous = false;
      SmallPtrSet<Value *, 16> Active;
      auto Candidate = getReturnedPointerOrigin(Ret->getReturnValue(), CB,
                                                Active, ReturnAmbiguous);
      if (!Candidate) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: return origin candidate unresolved\n");
        Valid = false;
        if (IsAmbiguous) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: marking caller ambiguity output\n");
          *IsAmbiguous = true;
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: no ambiguity output was supplied\n");
        }
        break;
      }

      if (!Result) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: recording first return origin\n");
        Result = Candidate;
      } else if (*Result != *Candidate) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: return origins differ\n");
        Valid = false;
        if (IsAmbiguous) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: marking caller ambiguity output\n");
          *IsAmbiguous = true;
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: no ambiguity output was supplied\n");
        }
        break;
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: return origin matches prior return\n");
      }
    }

    ResolvingCallOrigins.erase(&CB);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: getCallReturnOrigin exit; saw-return="
          << SawReturn << "; valid=" << Valid
          << "; has-result=" << Result.has_value() << "\n");
    return SawReturn && Valid ? Result : std::nullopt;
  }

  Value *getPointerOrigin(Value *V) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter getPointerOrigin; value=" << *V
          << "\n");
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer origin value is load\n");
      bool IsAllocaBacked =
          isa<AllocaInst>(getUnderlyingObject(LI->getPointerOperand()));
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: load storage is alloca-backed="
            << IsAllocaBacked << "\n");
      if (IsAllocaBacked && ResolvingOrigins.insert(LI).second) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: resolving load origin with MemorySSA\n");
        PointerClobberResult Clobber = resolve(LI->getPointerOperand(), *LI, 0);
        ResolvingOrigins.erase(LI);
        if (Clobber.Kind == PointerClobberKind::Value) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: load origin resolved to value="
                << *Clobber.V << "\n");
          return Clobber.V;
        }
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemorySSA load-origin result="
              << kindName(Clobber.Kind) << "; preserving load\n");
        // MemorySSA found either no resolvable value or a write that this
        // analysis cannot interpret. Preserve the load as an unresolved
        // provenance step. Falling through to the legacy store scan could skip
        // the reaching write and recover a stale initializer.
        return LI;
      } else if (!IsAllocaBacked) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: load is not alloca-backed; trying "
                 "reaching-store origin\n");
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: recursive load-origin query; trying "
                 "reaching-store origin\n");
      }
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer origin value is not a load\n");
    }
    if (auto *CB = dyn_cast<CallBase>(V)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer origin value is a call\n");
      auto Origin = getCallReturnOrigin(*CB);
      if (Origin && Origin->second == 0) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: following zero-offset call origin="
              << *Origin->first << "\n");
        return getPointerOrigin(Origin->first);
      }
      DEBUG(
          Logger::logs("proteus-pass")
          << "[PTR clobber trace]: call origin absent or has nonzero offset\n");
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer origin value is not a call\n");
    }
    SmallPtrSet<Value *, 8> Visited;
    Value *Origin = getPointerLoadOrigin(DL, V, Visited);
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: getPointerOrigin fallback result=";
      if (Origin)
        OS << *Origin;
      else
        OS << "<null>";
      OS << "\n";
    });
    return Origin;
  }

  std::optional<int64_t>
  getTrackedOffsetFrom(Value *Candidate, Value *Tracked, int64_t TargetOffset,
                       bool *HasAmbiguousCallOrigin = nullptr) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter getTrackedOffsetFrom; candidate="
          << *Candidate << "; tracked=" << *Tracked
          << "; target-offset=" << TargetOffset << "\n");
    bool AmbiguousCallOrigin = false;
    auto GetOriginAndOffset = [&](Value *V) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: enter origin-and-offset lambda; value="
            << *V << "\n");
      int64_t TotalOffset = 0;
      SmallPtrSet<Value *, 8> Visited;
      while (V && Visited.insert(V).second) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: origin-and-offset loop value=" << *V
              << "; accumulated-offset=" << TotalOffset << "\n");
        int64_t StepOffset = 0;
        Value *Base = GetPointerBaseWithConstantOffset(V, StepOffset, DL);
        TotalOffset += StepOffset;
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: origin-and-offset base=" << *Base
              << "; step-offset=" << StepOffset << "\n");
        if (auto *CB = dyn_cast_or_null<CallBase>(Base)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: origin-and-offset base is call\n");
          if (auto Origin = getCallReturnOrigin(*CB, &AmbiguousCallOrigin)) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: following call origin="
                  << *Origin->first << "; offset=" << Origin->second << "\n");
            TotalOffset += Origin->second;
            V = Origin->first;
            continue;
          } else {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: call origin unresolved; ambiguous="
                  << AmbiguousCallOrigin << "\n");
          }
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: origin-and-offset base is not call\n");
        }
        Value *Origin = getPointerOrigin(Base);
        if (!Origin) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: pointer origin is null; stopping\n");
          return std::pair<Value *, int64_t>{Base, TotalOffset};
        }
        if (Origin == Base) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: pointer origin reached fixed point\n");
          return std::pair<Value *, int64_t>{Base, TotalOffset};
        }
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: following pointer origin=" << *Origin
              << "\n");
        V = Origin;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: origin-and-offset loop ended via null or "
               "cycle\n");
      return std::pair<Value *, int64_t>{nullptr, 0};
    };

    auto [CandidateRoot, CandidateOffset] = GetOriginAndOffset(Candidate);
    auto [TrackedRoot, TrackedOffset] = GetOriginAndOffset(Tracked);
    if (HasAmbiguousCallOrigin) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: publishing ambiguous-call-origin="
            << AmbiguousCallOrigin << "\n");
      *HasAmbiguousCallOrigin = AmbiguousCallOrigin;
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no ambiguous-call-origin output\n");
    }
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: origin comparison; candidate-root=";
      if (CandidateRoot)
        OS << *CandidateRoot;
      else
        OS << "<null>";
      OS << "; candidate-offset=" << CandidateOffset << "; tracked-root=";
      if (TrackedRoot)
        OS << *TrackedRoot;
      else
        OS << "<null>";
      OS << "; tracked-offset=" << TrackedOffset << "\n";
    });
    if (!CandidateRoot) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: candidate root unresolved\n");
      return std::nullopt;
    }
    if (CandidateRoot != TrackedRoot) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pointer roots differ\n");
      return std::nullopt;
    }
    int64_t RelativeOffset = TrackedOffset + TargetOffset - CandidateOffset;
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: relative offset=" << RelativeOffset << "\n");
    return RelativeOffset;
  }

  MemoryAccess *getStateBefore(FunctionMemorySSAState &State,
                               Instruction &Boundary,
                               SmallPtrSetImpl<BasicBlock *> &VisitedBlocks) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter getStateBefore; boundary=" << Boundary
          << "\n");
    MemorySSA &MSSA = State.get();
    for (Instruction *I = Boundary.getPrevNode(); I; I = I->getPrevNode()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: state-before instruction loop=" << *I
            << "\n");
      if (auto *Def = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(I))) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: found preceding MemoryDef=" << *Def
              << "\n");
        return Def;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: instruction has no MemoryDef\n");
    }

    BasicBlock *BB = Boundary.getParent();
    if (MemoryPhi *Phi = MSSA.getMemoryAccess(BB)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: using block MemoryPhi=" << *Phi << "\n");
      return Phi;
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: block has no MemoryPhi\n");
    if (!VisitedBlocks.insert(BB).second) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: state-before block cycle; using "
               "live-on-entry\n");
      return MSSA.getLiveOnEntryDef();
    }
    if (pred_empty(BB)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: block has no predecessors; using "
               "live-on-entry\n");
      return MSSA.getLiveOnEntryDef();
    }
    if (BasicBlock *Pred = BB->getSinglePredecessor()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: recursing through single predecessor="
            << Pred->getName() << "\n");
      return getStateBefore(State, *Pred->getTerminator(), VisitedBlocks);
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: multiple predecessors without MemoryPhi; "
             "using live-on-entry\n");
    return MSSA.getLiveOnEntryDef();
  }

  static PointerClobberResult merge(PointerClobberResult LHS,
                                    PointerClobberResult RHS) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter merge; lhs=" << kindName(LHS.Kind)
          << "; rhs=" << kindName(RHS.Kind) << "\n");
    if (LHS.Kind == PointerClobberKind::Cycle) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: lhs is merge identity; choosing rhs\n");
      return RHS;
    }
    if (RHS.Kind == PointerClobberKind::Cycle) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: rhs is merge identity; choosing lhs\n");
      return LHS;
    }
    if (LHS.Kind == PointerClobberKind::Unknown ||
        RHS.Kind == PointerClobberKind::Unknown) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: at least one result is Unknown\n");
      return {PointerClobberKind::Unknown};
    }
    if (LHS.Kind == PointerClobberKind::Ambiguous ||
        RHS.Kind == PointerClobberKind::Ambiguous) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: at least one result is Ambiguous\n");
      return {PointerClobberKind::Ambiguous};
    }
    if (LHS.Kind != RHS.Kind) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: result kinds differ\n");
      return {PointerClobberKind::Ambiguous};
    }
    if (LHS.Kind == PointerClobberKind::Value &&
        (LHS.V != RHS.V || LHS.Offset != RHS.Offset ||
         LHS.ChangedRCLayout != RHS.ChangedRCLayout)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: concrete values or metadata differ\n");
      return {PointerClobberKind::Ambiguous};
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: merge results agree\n");
    return LHS;
  }

  PointerClobberResult
  resolveAccess(FunctionMemorySSAState &State, MemoryAccess *Access,
                const MemoryLocation &Location, Value *TrackedPtr,
                int64_t TargetOffset, SmallPtrSetImpl<MemoryAccess *> &Active,
                const PointerClobberCandidateMap *Candidates,
                const SmallPtrSetImpl<Value *> *KnownNoAliasRoots) {
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: enter resolveAccess; access=";
      if (Access)
        OS << *Access;
      else
        OS << "<null>";
      OS << "; tracked=" << *TrackedPtr << "; target-offset=" << TargetOffset
         << "; active-depth=" << Active.size()
         << "; candidates=" << (Candidates ? Candidates->size() : 0)
         << "; no-alias-roots="
         << (KnownNoAliasRoots ? KnownNoAliasRoots->size() : 0) << "\n";
    });
    MemorySSA &MSSA = State.get();
    if (Access) {
      DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber analysis]: Examining MemorySSA access "
                << *Access << "\n";)
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no MemorySSA access to print\n");
    }
    if (!Access) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: null access means incoming state\n");
      return {PointerClobberKind::Incoming};
    }
    if (MSSA.isLiveOnEntryDef(Access)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber analysis]: Reached live-on-entry without finding "
               "a clobber\n");
      return {PointerClobberKind::Incoming};
    }
    if (!Active.insert(Access).second) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber analysis]: Encountered a MemorySSA cycle at "
            << *Access << "\n");
      return {PointerClobberKind::Cycle};
    }

    PointerClobberResult Result{PointerClobberKind::Unknown};
    if (auto *Phi = dyn_cast<MemoryPhi>(Access)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: access branch=MemoryPhi\n");
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber analysis]: Merging clobbers reaching " << *Phi
            << "\n");
      Result.Kind = PointerClobberKind::Cycle;
      for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryPhi incoming loop index=" << I
              << "; incoming=" << *Phi->getIncomingValue(I) << "\n");
        MemoryAccess *Clobber = MSSA.getWalker()->getClobberingMemoryAccess(
            Phi->getIncomingValue(I), Location);
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryPhi incoming clobber=" << *Clobber
              << "\n");
        PointerClobberResult IncomingResult =
            resolveAccess(State, Clobber, Location, TrackedPtr, TargetOffset,
                          Active, Candidates, KnownNoAliasRoots);
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryPhi incoming result="
              << kindName(IncomingResult.Kind) << "\n");
        Result = merge(Result, IncomingResult);
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryPhi accumulated result="
              << kindName(Result.Kind) << "\n");
      }
    } else if (auto *Def = dyn_cast<MemoryDef>(Access)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: access branch=MemoryDef\n");
      Instruction *I = Def->getMemoryInst();
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber analysis]: Candidate clobbering instruction: "
            << *I << "\n");
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryDef instruction branch=store\n");
        auto StoreSize = getTypeStoreSize(DL, SI->getValueOperand()->getType());
        auto RelativeOffset = getTrackedOffsetFrom(SI->getPointerOperand(),
                                                   TrackedPtr, TargetOffset);
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: store provenance resolved="
              << RelativeOffset.has_value()
              << "; size resolved=" << StoreSize.has_value() << "\n");
        if (!RelativeOffset || !StoreSize) {
          DEBUG(
              Logger::logs("proteus-pass")
              << "[PTR clobber trace]: store branch=unresolved coordinates\n");
          // MemorySSA selected this store as the reaching clobber. Failure to
          // express its pointer in TrackedPtr's coordinates is not evidence
          // that it is disjoint from the tracked location.
          bool IsNoAlias =
              State.getAA().isNoAlias(Location, MemoryLocation::get(SI)) ||
              isKnownNoAlias(SI->getPointerOperand(), KnownNoAliasRoots);
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: unresolved store NoAlias=" << IsNoAlias
                << "\n");
          if (!IsNoAlias) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: unresolved store remains a "
                     "possible clobber\n");
            Result = unresolvedClobber(*SI);
          } else {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: unresolved store is proven "
                     "disjoint; walking before it\n");
            MemoryAccess *Previous =
                MSSA.getWalker()->getClobberingMemoryAccess(
                    Def->getDefiningAccess(), Location);
            Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                   TargetOffset, Active, Candidates,
                                   KnownNoAliasRoots);
          }
        } else if (offsetCoveredByRange(*RelativeOffset, 0, *StoreSize)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: store branch=covers tracked byte\n");
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber analysis]: Found covering clobber: " << *SI
                << "\n");
          if (SI->isAtomic()) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber analysis]: Covering atomic store is "
                     "ambiguous\n");
            Result = {PointerClobberKind::Ambiguous};
          } else {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: covering store is non-atomic\n");
            Value *Stored = SI->getValueOperand();
            if (Stored->getType()->isPointerTy()) {
              DEBUG(
                  Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: resolving stored pointer origin\n");
              Stored = getPointerOrigin(Stored);
            } else {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: stored value is non-pointer\n");
            }
            if (Stored) {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: recording stored value=" << *Stored
                    << "\n");
              Result = {PointerClobberKind::Value, Stored,
                        TargetOffset - *RelativeOffset, std::nullopt};
            } else {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: stored pointer origin is null\n");
            }
            Result.ClobberingInstruction = SI;
            Result.ClobberPointer = SI->getPointerOperand();
            Result.ClobberOffset = *RelativeOffset;
          }
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: store branch=does not cover tracked "
                   "byte\n");
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber analysis]: Store does not cover the tracked "
                   "byte; continuing past "
                << *SI << "\n");
          MemoryAccess *Previous = MSSA.getWalker()->getClobberingMemoryAccess(
              Def->getDefiningAccess(), Location);
          Result =
              resolveAccess(State, Previous, Location, TrackedPtr, TargetOffset,
                            Active, Candidates, KnownNoAliasRoots);
        }
      } else if (auto *CB = dyn_cast<CallBase>(I)) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryDef instruction branch=call\n");
        if (auto *MI = dyn_cast<MemIntrinsic>(CB)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: call branch=memory intrinsic\n");
          auto RelativeOffset =
              getTrackedOffsetFrom(MI->getRawDest(), TrackedPtr, TargetOffset);
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: intrinsic destination provenance "
                   "resolved="
                << RelativeOffset.has_value() << "\n");
          if (!RelativeOffset) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: intrinsic branch=unresolved "
                     "destination coordinates\n");
            // As with stores, only walk past a MemorySSA-selected intrinsic
            // when AA independently proves that its destination is disjoint.
            bool IsNoAlias =
                State.getAA().isNoAlias(Location,
                                        MemoryLocation::getForDest(MI)) ||
                isKnownNoAlias(MI->getRawDest(), KnownNoAliasRoots);
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: unresolved intrinsic NoAlias="
                  << IsNoAlias << "\n");
            if (!IsNoAlias) {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: unresolved intrinsic remains a "
                       "possible clobber\n");
              Result = unresolvedClobber(*MI);
            } else {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: unresolved intrinsic is proven "
                       "disjoint; walking before it\n");
              MemoryAccess *Previous =
                  MSSA.getWalker()->getClobberingMemoryAccess(
                      Def->getDefiningAccess(), Location);
              Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                     TargetOffset, Active, Candidates,
                                     KnownNoAliasRoots);
            }
          } else if (auto *Length = dyn_cast<ConstantInt>(MI->getLength())) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: intrinsic branch=constant length "
                  << Length->getZExtValue() << "\n");
            if (!offsetCoveredByRange(*RelativeOffset, 0,
                                      Length->getZExtValue())) {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: intrinsic does not cover tracked "
                       "byte; walking before it\n");
              MemoryAccess *Previous =
                  MSSA.getWalker()->getClobberingMemoryAccess(
                      Def->getDefiningAccess(), Location);
              Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                     TargetOffset, Active, Candidates,
                                     KnownNoAliasRoots);
            } else if (isa<MemSetInst>(MI)) {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: covering intrinsic is memset; "
                       "result is ambiguous\n");
              Result = {PointerClobberKind::Ambiguous};
            } else {
              DEBUG(Logger::logs("proteus-pass")
                    << "[PTR clobber trace]: covering intrinsic is transfer; "
                       "deferring interpretation to provenance visitor\n");
              // Let LambdaInstUseVisitor translate a covering transfer from
              // destination coordinates to source coordinates.
              Result.ClobberingInstruction = MI;
              Result.ClobberPointer = MI->getRawDest();
              Result.ClobberOffset = *RelativeOffset;
            }
          } else {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: intrinsic branch=dynamic length; "
                     "result is ambiguous\n");
            // A dynamic length may cover the tracked byte.
            Result = {PointerClobberKind::Ambiguous};
          }
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: call branch=general call\n");
          Result =
              resolveCall(State, *Def, *CB, Location, TrackedPtr, TargetOffset,
                          Active, Candidates, KnownNoAliasRoots);
        }
        DEBUG({
          auto &OS = Logger::logs("proteus-pass");
          OS << "[PTR clobber analysis]: Call clobber result "
             << static_cast<unsigned>(Result.Kind) << " from " << *CB;
          if (Result.Kind == PointerClobberKind::Value && Result.V)
            OS << "; resolved value: " << *Result.V;
          OS << "\n";
        });
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: MemoryDef instruction branch="
                 "unsupported\n");
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber analysis]: Unsupported clobbering instruction: "
              << *I << "\n");
        Result = unresolvedClobber(*I);
      }
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: access is neither MemoryPhi nor "
               "MemoryDef; leaving result Unknown\n");
    }

    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber analysis]: Result kind " << kindName(Result.Kind);
      if (Result.Kind == PointerClobberKind::Value && Result.V)
        OS << " with value " << *Result.V;
      OS << "\n";
    });
    Active.erase(Access);
    return Result;
  }

  /// Candidates is passed whenever the CallBase boundary is explicitly known,
  /// for example when a LambdaInstUseVisitor finds a CallBase use of
  /// TrackedPtr. In this case, resolveCall can easily identify which TrackedArg
  /// corresponds.
  PointerClobberResult
  resolveCall(FunctionMemorySSAState &CallerState, MemoryDef &CallDef,
              CallBase &CB, const MemoryLocation &CallerLocation,
              Value *TrackedPtr, int64_t TargetOffset,
              SmallPtrSetImpl<MemoryAccess *> &CallerActive,
              const PointerClobberCandidateMap *Candidates,
              const SmallPtrSetImpl<Value *> *CallerNoAliasRoots) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: enter resolveCall; call=" << CB
          << "; tracked=" << *TrackedPtr << "; target-offset=" << TargetOffset
          << "; candidates=" << (Candidates ? Candidates->size() : 0)
          << "; caller-no-alias-roots="
          << (CallerNoAliasRoots ? CallerNoAliasRoots->size() : 0) << "\n");
    auto ResolvePreCallState = [&]() {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: enter ResolvePreCallState lambda\n");
      MemoryAccess *Previous =
          CallerState.get().getWalker()->getClobberingMemoryAccess(
              CallDef.getDefiningAccess(), CallerLocation);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: pre-call clobber=" << *Previous << "\n");
      return resolveAccess(CallerState, Previous, CallerLocation, TrackedPtr,
                           TargetOffset, CallerActive, Candidates,
                           CallerNoAliasRoots);
    };

    Function *Callee = CB.getCalledFunction();
    if (!Callee) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolveCall branch=indirect call\n");
      ModRefInfo MRI = CallerState.getAA().getModRefInfo(&CB, CallerLocation);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: indirect call modifies tracked location="
            << isModSet(MRI) << "\n");
      return isModSet(MRI) ? unresolvedClobber(CB) : ResolvePreCallState();
    }
    if (Callee->isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolveCall branch=declaration; callee="
            << Callee->getName() << "\n");
      // There is no body to summarize, but AA may still prove from call-site
      // attributes that the call cannot modify this location (for example,
      // readnone target intrinsics). Only that proof permits crossing it.
      ModRefInfo MRI = CallerState.getAA().getModRefInfo(&CB, CallerLocation);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: declaration modifies tracked location="
            << isModSet(MRI) << "\n");
      return isModSet(MRI) ? unresolvedClobber(CB) : ResolvePreCallState();
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: resolveCall branch=defined direct callee="
          << Callee->getName() << "\n");

    // Function attributes are incomplete at this early pipeline point for
    // some target wrappers. A body containing no writes, directly or through
    // any transitive call, is nevertheless safe to cross for every location.
    SmallPtrSet<Function *, 8> ActiveFunctions;
    if (!mayWriteMemoryTransitively(*Callee, ActiveFunctions)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: callee is transitively read-only; "
               "walking before call\n");
      return ResolvePreCallState();
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: callee may write; identifying tracked "
             "formal\n");

    unsigned TrackedArg = Callee->arg_size();
    int64_t CalleeTargetOffset = 0;
    bool HasCandidate = false;
    bool HasKnownUseEdge = false;
    if (Candidates) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: candidate map is present\n");
      auto Candidate = Candidates->find(&CB);
      if (Candidate != Candidates->end()) {
        DEBUG({
          auto &OS = Logger::logs("proteus-pass");
          OS << "[PTR clobber trace]: found candidate for call; pointer=";
          if (Candidate->second.Pointer)
            OS << *Candidate->second.Pointer;
          else
            OS << "<null>";
          OS << "; offset=" << Candidate->second.TargetOffset << "\n";
        });
        HasCandidate = true;
        if (!Candidate->second.Pointer) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: candidate pointer is null\n");
          return unresolvedClobber(CB);
        }
        for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: exact-use-edge argument loop index="
                << I << "; actual=" << *CB.getArgOperand(I) << "\n");
          if (CB.getArgOperand(I) != Candidate->second.Pointer) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: actual does not equal candidate "
                     "pointer\n");
            continue;
          }
          if (HasKnownUseEdge) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: candidate pointer appears in "
                     "multiple arguments\n");
            return {PointerClobberKind::Ambiguous};
          }
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: exact use edge identifies formal " << I
                << "\n");
          HasKnownUseEdge = true;
          TrackedArg = I;
          CalleeTargetOffset = Candidate->second.TargetOffset;
        }
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: candidate map has no entry for call\n");
      }
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no candidate map supplied\n");
    }
    if (HasCandidate && !HasKnownUseEdge) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: candidate did not match any actual\n");
      return unresolvedClobber(CB);
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: candidate matching complete; has-candidate="
          << HasCandidate << "; has-known-edge=" << HasKnownUseEdge << "\n");

    // Candidate-free queries originate directly from LambdaArgVisitor and do
    // not have the def-use edge that led to this call. Recover the formal from
    // pointer provenance only for that path. LambdaInstUseVisitor candidates
    // carry the exact SSA value used by the call and bypass this
    // reconstruction.
    if (!HasCandidate) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: entering provenance-based argument "
               "recovery\n");
      for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: provenance argument loop index=" << I
              << "; actual=" << *CB.getArgOperand(I) << "\n");
        Value *Actual = CB.getArgOperand(I);
        if (!Actual->getType()->isPointerTy()) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: actual is non-pointer; skipping\n");
          continue;
        }
        bool HasAmbiguousCallOrigin = false;
        auto RelativeOffset = getTrackedOffsetFrom(
            Actual, TrackedPtr, TargetOffset, &HasAmbiguousCallOrigin);
        if (HasAmbiguousCallOrigin) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: actual has ambiguous call origin\n");
          return unresolvedClobber(CB);
        }
        if (!RelativeOffset) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: actual does not resolve to tracked "
                   "pointer\n");
          continue;
        }
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: actual aliases tracked pointer; "
                 "relative-offset="
              << *RelativeOffset << "\n");
        if (TrackedArg != Callee->arg_size()) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: multiple actuals alias tracked "
                   "pointer\n");
          return {PointerClobberKind::Ambiguous};
        }
        TrackedArg = I;
        CalleeTargetOffset = *RelativeOffset;
      }
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: exact candidate suppresses provenance "
               "argument recovery\n");
    }
    if (TrackedArg == Callee->arg_size()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no tracked formal was recovered\n");
      // MemorySSA selected this call for CallerLocation. Failure to map the
      // location to a formal argument is not evidence that the call is
      // harmless. Bypass it only when ModRef proves no modification, or when
      // capture analysis and AA together prove that an unescaped local is
      // disjoint from every pointer actual. Nocapture alone is insufficient: a
      // call may modify a pointer argument without retaining it.
      ModRefInfo MRI = CallerState.getAA().getModRefInfo(&CB, CallerLocation);
      bool MayModify = isModSet(MRI);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: unmatched call modifies tracked location="
            << MayModify << "\n");
      bool DisjointLocal =
          MayModify && canProveDisjointFromUncapturedLocal(
                           CallerState, CB, CallerLocation, TrackedPtr);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: unmatched call disjoint-local proof="
            << DisjointLocal << "\n");
      if (MayModify && !DisjointLocal) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: unmatched call remains a possible "
                 "clobber\n");
        return unresolvedClobber(CB);
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: unmatched call is proven harmless; "
               "walking before it\n");
      return ResolvePreCallState();
    }

    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: tracked formal index=" << TrackedArg
          << "; callee-target-offset=" << CalleeTargetOffset << "\n");

    FunctionMemorySSAState &CalleeState = getState(*Callee);
    Value *Formal = Callee->getArg(TrackedArg);
    MemoryLocation CalleeLocation = MemoryLocation::getBeforeOrAfter(Formal);
    SmallPtrSet<Value *, 8> CalleeNoAliasRoots;
    for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no-alias propagation loop index=" << I
            << "; actual=" << *CB.getArgOperand(I) << "\n");
      if (I == TrackedArg) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: skipping tracked argument\n");
        continue;
      }
      if (!CB.getArgOperand(I)->getType()->isPointerTy()) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: skipping non-pointer argument\n");
        continue;
      }
      Value *Actual = CB.getArgOperand(I);
      bool AANoAlias = CallerState.getAA().isNoAlias(
          CallerLocation, MemoryLocation::getBeforeOrAfter(Actual));
      bool DistinctAllocas = haveDistinctAllocaRoots(TrackedPtr, Actual);
      bool PropagatedNoAlias = isKnownNoAlias(Actual, CallerNoAliasRoots);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no-alias evidence; AA=" << AANoAlias
            << "; distinct-allocas=" << DistinctAllocas
            << "; propagated=" << PropagatedNoAlias << "\n");
      if (AANoAlias || DistinctAllocas || PropagatedNoAlias) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: propagating formal as no-alias root="
              << *Callee->getArg(I) << "\n");
        CalleeNoAliasRoots.insert(Callee->getArg(I));
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: no no-alias proof for formal\n");
      }
    }
    PointerClobberResult Summary{PointerClobberKind::Cycle};

    for (BasicBlock &BB : *Callee) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: callee return-block loop block="
            << BB.getName() << "\n");
      auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!Ret) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: block has no return terminator; "
                 "skipping\n");
        continue;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolving callee state at return=" << *Ret
            << "\n");
      SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
      MemoryAccess *ExitState =
          getStateBefore(CalleeState, *Ret, VisitedBlocks);
      MemoryAccess *Clobber =
          CalleeState.get().getWalker()->getClobberingMemoryAccess(
              ExitState, CalleeLocation);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: return reaching clobber=" << *Clobber
            << "\n");
      SmallPtrSet<MemoryAccess *, 16> CalleeActive;
      // Potentially recursive analysis of the clobber
      PointerClobberResult AtReturn = resolveAccess(
          CalleeState, Clobber, CalleeLocation, Formal, CalleeTargetOffset,
          CalleeActive, Candidates, &CalleeNoAliasRoots);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: callee return result="
            << kindName(AtReturn.Kind) << "\n");

      if (AtReturn.Kind == PointerClobberKind::Incoming) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: callee leaves value incoming; "
                 "resuming in caller before call\n");
        MemoryAccess *Previous =
            CallerState.get().getWalker()->getClobberingMemoryAccess(
                CallDef.getDefiningAccess(), CallerLocation);
        AtReturn = resolveAccess(CallerState, Previous, CallerLocation,
                                 TrackedPtr, TargetOffset, CallerActive,
                                 Candidates, CallerNoAliasRoots);
      } else if (AtReturn.Kind == PointerClobberKind::Value) {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: callee return resolved a value\n");
        if (auto *A = dyn_cast<Argument>(AtReturn.V)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: result value is argument "
                << A->getArgNo() << "\n");
          if (A->getParent() == Callee) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: translating callee formal to "
                     "caller actual\n");
            AtReturn.V = CB.getArgOperand(A->getArgNo());
          } else {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber trace]: argument belongs to another "
                     "function\n");
          }
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: result value is not an argument\n");
        }
        // Normalize pointer spills before merging return paths. Distinct SSA
        // loads of the same formal are one semantic value, while loads rooted
        // in different formals must remain distinguishable.
        if (AtReturn.V && AtReturn.V->getType()->isPointerTy()) {
          DEBUG(
              Logger::logs("proteus-pass")
              << "[PTR clobber trace]: normalizing returned pointer origin\n");
          AtReturn.V = getPointerOrigin(AtReturn.V);
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber trace]: returned value is null or "
                   "non-pointer; no normalization\n");
        }
      } else {
        DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber trace]: callee return is neither Incoming nor "
                 "Value\n");
      }
      Summary = merge(Summary, AtReturn);
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: accumulated call summary="
            << kindName(Summary.Kind) << "\n");
    }

    if (Summary.Kind == PointerClobberKind::Cycle) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no return path contributed; converting "
               "Cycle identity to Unknown\n");
      return {PointerClobberKind::Unknown};
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: resolveCall exit result="
          << kindName(Summary.Kind) << "\n");
    return Summary;
  }

public:
  explicit PointerClobberResolver(const DataLayout &DL) : DL(DL) {
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: construct PointerClobberResolver\n");
  }

  PointerClobberResult
  resolve(Value *Ptr, Instruction &UseBoundary, int64_t TargetOffset,
          const PointerClobberCandidateMap *Candidates = nullptr) override {
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber trace]: enter resolve; pointer=";
      if (Ptr)
        OS << *Ptr;
      else
        OS << "<null>";
      OS << "; boundary=" << UseBoundary << "; target-offset=" << TargetOffset
         << "; candidates=" << (Candidates ? Candidates->size() : 0) << "\n";
    });
    if (!Ptr) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolve rejected null pointer\n");
      return {PointerClobberKind::Unknown};
    }
    if (!Ptr->getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: resolve rejected non-pointer value\n");
      return {PointerClobberKind::Unknown};
    }

    FunctionMemorySSAState &State = getState(*UseBoundary.getFunction());
    MemorySSA &MSSA = State.get();
    auto Location = getTrackedPointerLocation(DL, Ptr);
    if (!Location) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: tracked MemoryLocation could not be "
               "constructed\n");
      return {PointerClobberKind::Unknown};
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: tracked MemoryLocation constructed\n");

    MemoryAccess *Before = nullptr;
    if (auto *BoundaryAccess = dyn_cast_or_null<MemoryUseOrDef>(
            MSSA.getMemoryAccess(&UseBoundary))) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: boundary has MemoryUseOrDef="
            << *BoundaryAccess << "\n");
      Before = BoundaryAccess->getDefiningAccess();
    } else {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: boundary has no MemoryUseOrDef; scanning "
               "for prior state\n");
      SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
      Before = getStateBefore(State, UseBoundary, VisitedBlocks);
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: state before boundary=" << *Before << "\n");
    MemoryAccess *Clobber =
        MSSA.getWalker()->getClobberingMemoryAccess(Before, *Location);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: initial clobber query result=" << *Clobber
          << "\n");
    SmallPtrSet<MemoryAccess *, 16> Active;
    PointerClobberResult Result =
        resolveAccess(State, Clobber, *Location, Ptr, TargetOffset, Active,
                      Candidates, nullptr);
    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber trace]: resolveAccess completed with "
          << kindName(Result.Kind) << "\n");
    if (!Candidates) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: no candidates require validation; "
               "returning result\n");
      return Result;
    }
    if (Candidates->empty()) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: empty candidate map requires no "
               "validation; returning result\n");
      return Result;
    }
    if (Result.Kind == PointerClobberKind::Ambiguous) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: ambiguous result remains conservative; "
               "returning it\n");
      return Result;
    }

    if (!Result.ClobberingInstruction) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: result has no instruction to validate "
               "against candidates\n");
      return {PointerClobberKind::Unknown};
    }
    if (Candidates->contains(Result.ClobberingInstruction)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber trace]: selected instruction is in candidate "
               "map; returning result\n");
      return Result;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber analysis]: MemorySSA selected an instruction not "
             "present in the collected relevant-use set: "
          << *Result.ClobberingInstruction << "\n");
    return {PointerClobberKind::Unknown};
  }
};

} // namespace proteus

#endif
