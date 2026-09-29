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

  FunctionMemorySSAState &getState(Function &F) {
    auto &State = States[&F];
    if (!State)
      State = std::make_unique<FunctionMemorySSAState>(F);
    return *State;
  }

  static PointerClobberResult unresolvedClobber(Instruction &I) {
    // MemorySSA selected I as the reaching clobber, but this analysis cannot
    // model its effect.  This is a fatal ambiguity, not an invitation to use
    // an older definition.  Unknown is reserved for supported instructions
    // (currently memory transfers) that LambdaInstUseVisitor can interpret.
    PointerClobberResult Result{PointerClobberKind::Ambiguous};
    Result.ClobberingInstruction = &I;
    return Result;
  }

  std::optional<std::pair<Value *, int64_t>>
  getReturnedPointerOrigin(Value *V, CallBase &RootCall,
                           SmallPtrSetImpl<Value *> &Active,
                           bool &IsAmbiguous) {
    if (!V || !V->getType()->isPointerTy() || !Active.insert(V).second) {
      IsAmbiguous = true;
      return std::nullopt;
    }

    int64_t LocalOffset = 0;
    Value *Base = GetPointerBaseWithConstantOffset(V, LocalOffset, DL);
    std::optional<std::pair<Value *, int64_t>> Result;

    if (auto *A = dyn_cast_or_null<Argument>(Base)) {
      Function *RootCallee = RootCall.getCalledFunction();
      if (RootCallee && A->getParent() == RootCallee &&
          A->getArgNo() < RootCall.arg_size())
        Result = {{RootCall.getArgOperand(A->getArgNo()), LocalOffset}};
    } else if (auto *LI = dyn_cast_or_null<LoadInst>(Base)) {
      SmallPtrSet<Value *, 8> VisitedLoads;
      Value *Origin = getPointerLoadOrigin(DL, LI, VisitedLoads);
      if (Origin && Origin != LI) {
        Result =
            getReturnedPointerOrigin(Origin, RootCall, Active, IsAmbiguous);
        if (Result)
          Result->second += LocalOffset;
      }
    } else if (auto *CB = dyn_cast_or_null<CallBase>(Base)) {
      bool NestedAmbiguous = false;
      auto Origin = getCallReturnOrigin(*CB, &NestedAmbiguous);
      if (Origin) {
        Result = getReturnedPointerOrigin(Origin->first, RootCall, Active,
                                          IsAmbiguous);
        if (Result)
          Result->second += LocalOffset + Origin->second;
      } else if (NestedAmbiguous) {
        IsAmbiguous = true;
      }
    } else if (auto *Select = dyn_cast_or_null<SelectInst>(Base)) {
      auto TrueOrigin = getReturnedPointerOrigin(Select->getTrueValue(),
                                                 RootCall, Active, IsAmbiguous);
      auto FalseOrigin = getReturnedPointerOrigin(
          Select->getFalseValue(), RootCall, Active, IsAmbiguous);
      if (TrueOrigin && FalseOrigin && *TrueOrigin == *FalseOrigin) {
        Result = TrueOrigin;
        Result->second += LocalOffset;
      } else {
        IsAmbiguous = true;
      }
    } else if (auto *Phi = dyn_cast_or_null<PHINode>(Base)) {
      for (Value *Incoming : Phi->incoming_values()) {
        auto IncomingOrigin =
            getReturnedPointerOrigin(Incoming, RootCall, Active, IsAmbiguous);
        if (!IncomingOrigin) {
          IsAmbiguous = true;
          Result.reset();
          break;
        }
        if (!Result)
          Result = IncomingOrigin;
        else if (*Result != *IncomingOrigin) {
          IsAmbiguous = true;
          Result.reset();
          break;
        }
      }
      if (Result)
        Result->second += LocalOffset;
    }

    Active.erase(V);
    if (!Result)
      IsAmbiguous = true;
    return Result;
  }

  std::optional<std::pair<Value *, int64_t>>
  getCallReturnOrigin(CallBase &CB, bool *IsAmbiguous = nullptr) {
    Function *Callee = CB.getCalledFunction();
    if (!Callee || Callee->isDeclaration() ||
        !ResolvingCallOrigins.insert(&CB).second)
      return std::nullopt;

    std::optional<std::pair<Value *, int64_t>> Result;
    bool SawReturn = false;
    bool Valid = true;
    for (BasicBlock &BB : *Callee) {
      auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!Ret || !Ret->getReturnValue())
        continue;

      SawReturn = true;
      bool ReturnAmbiguous = false;
      SmallPtrSet<Value *, 16> Active;
      auto Candidate = getReturnedPointerOrigin(Ret->getReturnValue(), CB,
                                                Active, ReturnAmbiguous);
      if (!Candidate) {
        Valid = false;
        if (IsAmbiguous)
          *IsAmbiguous = true;
        break;
      }

      if (!Result)
        Result = Candidate;
      else if (*Result != *Candidate) {
        Valid = false;
        if (IsAmbiguous)
          *IsAmbiguous = true;
        break;
      }
    }

    ResolvingCallOrigins.erase(&CB);
    return SawReturn && Valid ? Result : std::nullopt;
  }

  Value *getPointerOrigin(Value *V) {
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      if (isa<AllocaInst>(getUnderlyingObject(LI->getPointerOperand())) &&
          ResolvingOrigins.insert(LI).second) {
        PointerClobberResult Clobber = resolve(LI->getPointerOperand(), *LI, 0);
        ResolvingOrigins.erase(LI);
        if (Clobber.Kind == PointerClobberKind::Value)
          return Clobber.V;
        // MemorySSA found either no resolvable value or a write that this
        // analysis cannot interpret. Preserve the load as an unresolved
        // provenance step. Falling through to the legacy store scan could skip
        // the reaching write and recover a stale initializer.
        return LI;
      }
    }
    if (auto *CB = dyn_cast<CallBase>(V)) {
      auto Origin = getCallReturnOrigin(*CB);
      if (Origin && Origin->second == 0)
        return getPointerOrigin(Origin->first);
    }
    SmallPtrSet<Value *, 8> Visited;
    return getPointerLoadOrigin(DL, V, Visited);
  }

  std::optional<int64_t>
  getTrackedOffsetFrom(Value *Candidate, Value *Tracked, int64_t TargetOffset,
                       bool *HasAmbiguousCallOrigin = nullptr) {
    bool AmbiguousCallOrigin = false;
    auto GetOriginAndOffset = [&](Value *V) {
      int64_t TotalOffset = 0;
      SmallPtrSet<Value *, 8> Visited;
      while (V && Visited.insert(V).second) {
        int64_t StepOffset = 0;
        Value *Base = GetPointerBaseWithConstantOffset(V, StepOffset, DL);
        TotalOffset += StepOffset;
        if (auto *CB = dyn_cast_or_null<CallBase>(Base)) {
          if (auto Origin = getCallReturnOrigin(*CB, &AmbiguousCallOrigin)) {
            TotalOffset += Origin->second;
            V = Origin->first;
            continue;
          }
        }
        Value *Origin = getPointerOrigin(Base);
        if (!Origin || Origin == Base)
          return std::pair<Value *, int64_t>{Base, TotalOffset};
        V = Origin;
      }
      return std::pair<Value *, int64_t>{nullptr, 0};
    };

    auto [CandidateRoot, CandidateOffset] = GetOriginAndOffset(Candidate);
    auto [TrackedRoot, TrackedOffset] = GetOriginAndOffset(Tracked);
    if (HasAmbiguousCallOrigin)
      *HasAmbiguousCallOrigin = AmbiguousCallOrigin;
    if (!CandidateRoot || CandidateRoot != TrackedRoot)
      return std::nullopt;
    return TrackedOffset + TargetOffset - CandidateOffset;
  }

  MemoryAccess *getStateBefore(FunctionMemorySSAState &State,
                               Instruction &Boundary,
                               SmallPtrSetImpl<BasicBlock *> &VisitedBlocks) {
    MemorySSA &MSSA = State.get();
    for (Instruction *I = Boundary.getPrevNode(); I; I = I->getPrevNode())
      if (auto *Def = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(I)))
        return Def;

    BasicBlock *BB = Boundary.getParent();
    if (MemoryPhi *Phi = MSSA.getMemoryAccess(BB))
      return Phi;
    if (!VisitedBlocks.insert(BB).second || pred_empty(BB))
      return MSSA.getLiveOnEntryDef();
    if (BasicBlock *Pred = BB->getSinglePredecessor())
      return getStateBefore(State, *Pred->getTerminator(), VisitedBlocks);
    return MSSA.getLiveOnEntryDef();
  }

  static PointerClobberResult merge(PointerClobberResult LHS,
                                    PointerClobberResult RHS) {
    if (LHS.Kind == PointerClobberKind::Cycle)
      return RHS;
    if (RHS.Kind == PointerClobberKind::Cycle)
      return LHS;
    if (LHS.Kind == PointerClobberKind::Unknown ||
        RHS.Kind == PointerClobberKind::Unknown)
      return {PointerClobberKind::Unknown};
    if (LHS.Kind == PointerClobberKind::Ambiguous ||
        RHS.Kind == PointerClobberKind::Ambiguous || LHS.Kind != RHS.Kind)
      return {PointerClobberKind::Ambiguous};
    if (LHS.Kind == PointerClobberKind::Value &&
        (LHS.V != RHS.V || LHS.Offset != RHS.Offset ||
         LHS.ChangedRCLayout != RHS.ChangedRCLayout))
      return {PointerClobberKind::Ambiguous};
    return LHS;
  }

  PointerClobberResult
  resolveAccess(FunctionMemorySSAState &State, MemoryAccess *Access,
                const MemoryLocation &Location, Value *TrackedPtr,
                int64_t TargetOffset, SmallPtrSetImpl<MemoryAccess *> &Active,
                const PointerClobberCandidateMap *Candidates) {
    MemorySSA &MSSA = State.get();
    if (Access) {
      DEBUG(Logger::logs("proteus-pass")
              << "[PTR clobber analysis]: Examining MemorySSA access "
              << *Access << "\n";)
    }
    if (!Access || MSSA.isLiveOnEntryDef(Access)) {
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
            << "[PTR clobber analysis]: Merging clobbers reaching " << *Phi
            << "\n");
      Result.Kind = PointerClobberKind::Cycle;
      for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
        MemoryAccess *Clobber = MSSA.getWalker()->getClobberingMemoryAccess(
            Phi->getIncomingValue(I), Location);
        Result =
            merge(Result, resolveAccess(State, Clobber, Location, TrackedPtr,
                                        TargetOffset, Active, Candidates));
      }
    } else if (auto *Def = dyn_cast<MemoryDef>(Access)) {
      Instruction *I = Def->getMemoryInst();
      DEBUG(Logger::logs("proteus-pass")
            << "[PTR clobber analysis]: Candidate clobbering instruction: "
            << *I << "\n");
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        auto StoreSize = getTypeStoreSize(DL, SI->getValueOperand()->getType());
        auto RelativeOffset = getTrackedOffsetFrom(SI->getPointerOperand(),
                                                   TrackedPtr, TargetOffset);
        if (RelativeOffset && StoreSize &&
            offsetCoveredByRange(*RelativeOffset, 0, *StoreSize)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber analysis]: Found covering clobber: " << *SI
                << "\n");
          if (SI->isAtomic()) {
            DEBUG(Logger::logs("proteus-pass")
                  << "[PTR clobber analysis]: Covering atomic store is "
                     "ambiguous\n");
            Result = {PointerClobberKind::Ambiguous};
          } else {
            Value *Stored = SI->getValueOperand();
            if (Stored->getType()->isPointerTy())
              Stored = getPointerOrigin(Stored);
            if (Stored)
              Result = {PointerClobberKind::Value, Stored,
                        TargetOffset - *RelativeOffset, std::nullopt};
            Result.ClobberingInstruction = SI;
            Result.ClobberPointer = SI->getPointerOperand();
            Result.ClobberOffset = *RelativeOffset;
          }
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "[PTR clobber analysis]: Store does not cover the tracked "
                   "byte; continuing past "
                << *SI << "\n");
          MemoryAccess *Previous = MSSA.getWalker()->getClobberingMemoryAccess(
              Def->getDefiningAccess(), Location);
          Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                 TargetOffset, Active, Candidates);
        }
      } else if (auto *CB = dyn_cast<CallBase>(I)) {
        if (auto *MI = dyn_cast<MemIntrinsic>(CB)) {
          auto RelativeOffset =
              getTrackedOffsetFrom(MI->getRawDest(), TrackedPtr, TargetOffset);
          if (!RelativeOffset) {
            // The intrinsic may read the tracked storage, but it does not
            // write it. Continue with the state preceding the call.
            MemoryAccess *Previous =
                MSSA.getWalker()->getClobberingMemoryAccess(
                    Def->getDefiningAccess(), Location);
            Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                   TargetOffset, Active, Candidates);
          } else if (auto *Length = dyn_cast<ConstantInt>(MI->getLength())) {
            if (!offsetCoveredByRange(*RelativeOffset, 0,
                                      Length->getZExtValue())) {
              MemoryAccess *Previous =
                  MSSA.getWalker()->getClobberingMemoryAccess(
                      Def->getDefiningAccess(), Location);
              Result = resolveAccess(State, Previous, Location, TrackedPtr,
                                     TargetOffset, Active, Candidates);
            } else if (isa<MemSetInst>(MI)) {
              Result = {PointerClobberKind::Ambiguous};
            } else {
              // Let LambdaInstUseVisitor translate a covering transfer from
              // destination coordinates to source coordinates.
              Result.ClobberingInstruction = MI;
              Result.ClobberPointer = MI->getRawDest();
              Result.ClobberOffset = *RelativeOffset;
            }
          } else {
            // A dynamic length may cover the tracked byte.
            Result = {PointerClobberKind::Ambiguous};
          }
        } else {
          Result = resolveCall(State, *Def, *CB, Location, TrackedPtr,
                               TargetOffset, Active, Candidates);
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
              << "[PTR clobber analysis]: Unsupported clobbering instruction: "
              << *I << "\n");
        Result = unresolvedClobber(*I);
      }
    }

    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "[PTR clobber analysis]: Result kind "
         << static_cast<unsigned>(Result.Kind);
      if (Result.Kind == PointerClobberKind::Value && Result.V)
        OS << " with value " << *Result.V;
      OS << "\n";
    });
    Active.erase(Access);
    return Result;
  }

  /// Candidates is passed whenever the CallBase boundary is explicitly known,
  /// for example when a LambdaInstUseVisitor finds a CallBase use of TrackedPtr.
  /// In this case, resolveCall can easily identify which TrackedArg corresponds.
  PointerClobberResult
  resolveCall(FunctionMemorySSAState &CallerState, MemoryDef &CallDef,
              CallBase &CB, const MemoryLocation &CallerLocation,
              Value *TrackedPtr, int64_t TargetOffset,
              SmallPtrSetImpl<MemoryAccess *> &CallerActive,
              const PointerClobberCandidateMap *Candidates) {
    Function *Callee = CB.getCalledFunction();
    if (!Callee || Callee->isDeclaration())
      return unresolvedClobber(CB);

    unsigned TrackedArg = Callee->arg_size();
    int64_t CalleeTargetOffset = 0;
    bool HasCandidate = false;
    bool HasKnownUseEdge = false;
    if (Candidates) {
      auto Candidate = Candidates->find(&CB);
      if (Candidate != Candidates->end()) {
        HasCandidate = true;
        if (!Candidate->second.Pointer)
          return unresolvedClobber(CB);
        for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
          if (CB.getArgOperand(I) != Candidate->second.Pointer)
            continue;
          if (HasKnownUseEdge)
            return {PointerClobberKind::Ambiguous};
          HasKnownUseEdge = true;
          TrackedArg = I;
          CalleeTargetOffset = Candidate->second.TargetOffset;
        }
      }
    }
    if (HasCandidate && !HasKnownUseEdge)
      return unresolvedClobber(CB);

    // Candidate-free queries originate directly from LambdaArgVisitor and do
    // not have the def-use edge that led to this call. Recover the formal from
    // pointer provenance only for that path. LambdaInstUseVisitor candidates
    // carry the exact SSA value used by the call and bypass this
    // reconstruction.
    if (!HasCandidate) {
      for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
        Value *Actual = CB.getArgOperand(I);
        if (!Actual->getType()->isPointerTy())
          continue;
        bool HasAmbiguousCallOrigin = false;
        auto RelativeOffset = getTrackedOffsetFrom(
            Actual, TrackedPtr, TargetOffset, &HasAmbiguousCallOrigin);
        if (HasAmbiguousCallOrigin)
          return unresolvedClobber(CB);
        if (!RelativeOffset)
          continue;
        if (TrackedArg != Callee->arg_size())
          return {PointerClobberKind::Ambiguous};
        TrackedArg = I;
        CalleeTargetOffset = *RelativeOffset;
      }
    }
    if (TrackedArg == Callee->arg_size()) {
      // MemorySSA selected this call for CallerLocation. Failure to map the
      // location to a formal argument is not evidence that the call is
      // harmless. Bypass it only when analysis proves that the call cannot
      // modify the location: either AA reports NoModRef, or the location is a
      // local allocation whose address has not escaped before the call.
      ModRefInfo MRI = CallerState.getAA().getModRefInfo(&CB, CallerLocation);
      Value *Underlying = getUnderlyingObject(TrackedPtr);
      bool IsUncapturedLocal =
          isa<AllocaInst>(Underlying) &&
          !PointerMayBeCapturedBefore(Underlying, true, true, &CB,
                                      &CallerState.getDT(), true);
      if (isModSet(MRI) && !IsUncapturedLocal)
        return unresolvedClobber(CB);
      MemoryAccess *Previous =
          CallerState.get().getWalker()->getClobberingMemoryAccess(
              CallDef.getDefiningAccess(), CallerLocation);
      return resolveAccess(CallerState, Previous, CallerLocation, TrackedPtr,
                           TargetOffset, CallerActive, Candidates);
    }

    FunctionMemorySSAState &CalleeState = getState(*Callee);
    Value *Formal = Callee->getArg(TrackedArg);
    MemoryLocation CalleeLocation = MemoryLocation::getBeforeOrAfter(Formal);
    PointerClobberResult Summary{PointerClobberKind::Cycle};

    for (BasicBlock &BB : *Callee) {
      auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!Ret)
        continue;
      SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
      MemoryAccess *ExitState =
          getStateBefore(CalleeState, *Ret, VisitedBlocks);
      MemoryAccess *Clobber =
          CalleeState.get().getWalker()->getClobberingMemoryAccess(
              ExitState, CalleeLocation);
      SmallPtrSet<MemoryAccess *, 16> CalleeActive;
      // Potentially recursive analysis of the clobber
      PointerClobberResult AtReturn =
          resolveAccess(CalleeState, Clobber, CalleeLocation, Formal,
                        CalleeTargetOffset, CalleeActive, Candidates);

      if (AtReturn.Kind == PointerClobberKind::Incoming) {
        MemoryAccess *Previous =
            CallerState.get().getWalker()->getClobberingMemoryAccess(
                CallDef.getDefiningAccess(), CallerLocation);
        AtReturn =
            resolveAccess(CallerState, Previous, CallerLocation, TrackedPtr,
                          TargetOffset, CallerActive, Candidates);
      } else if (AtReturn.Kind == PointerClobberKind::Value) {
        if (auto *A = dyn_cast<Argument>(AtReturn.V))
          if (A->getParent() == Callee)
            AtReturn.V = CB.getArgOperand(A->getArgNo());
        // Normalize pointer spills before merging return paths. Distinct SSA
        // loads of the same formal are one semantic value, while loads rooted
        // in different formals must remain distinguishable.
        if (AtReturn.V && AtReturn.V->getType()->isPointerTy())
          AtReturn.V = getPointerOrigin(AtReturn.V);
      }
      Summary = merge(Summary, AtReturn);
    }

    return Summary.Kind == PointerClobberKind::Cycle
               ? PointerClobberResult{PointerClobberKind::Unknown}
               : Summary;
  }

public:
  explicit PointerClobberResolver(const DataLayout &DL) : DL(DL) {}

  PointerClobberResult
  resolve(Value *Ptr, Instruction &UseBoundary, int64_t TargetOffset,
          const PointerClobberCandidateMap *Candidates = nullptr) override {
    if (!Ptr || !Ptr->getType()->isPointerTy())
      return {PointerClobberKind::Unknown};

    FunctionMemorySSAState &State = getState(*UseBoundary.getFunction());
    MemorySSA &MSSA = State.get();
    auto Location = getTrackedPointerLocation(DL, Ptr);
    if (!Location)
      return {PointerClobberKind::Unknown};

    MemoryAccess *Before = nullptr;
    if (auto *BoundaryAccess = dyn_cast_or_null<MemoryUseOrDef>(
            MSSA.getMemoryAccess(&UseBoundary)))
      Before = BoundaryAccess->getDefiningAccess();
    else {
      SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
      Before = getStateBefore(State, UseBoundary, VisitedBlocks);
    }
    MemoryAccess *Clobber =
        MSSA.getWalker()->getClobberingMemoryAccess(Before, *Location);
    SmallPtrSet<MemoryAccess *, 16> Active;
    PointerClobberResult Result = resolveAccess(
        State, Clobber, *Location, Ptr, TargetOffset, Active, Candidates);
    if (!Candidates || Candidates->empty() ||
        Result.Kind == PointerClobberKind::Ambiguous)
      return Result;

    if (!Result.ClobberingInstruction)
      return {PointerClobberKind::Unknown};
    if (Candidates->contains(Result.ClobberingInstruction))
      return Result;

    DEBUG(Logger::logs("proteus-pass")
          << "[PTR clobber analysis]: MemorySSA selected an instruction not "
             "present in the collected relevant-use set: "
          << *Result.ClobberingInstruction << "\n");
    return {PointerClobberKind::Unknown};
  }
};

} // namespace proteus

#endif
