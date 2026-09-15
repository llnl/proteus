#ifndef PROTEUS_KERNELARGVISITOR_H
#define PROTEUS_KERNELARGVISITOR_H

#include "Helpers.h"
#include "KernelArgPtrUseVisitor.h"
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

// Own the analyses needed to buildgit  MemorySSA for one function.  Lambda
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
};

enum class PointerClobberKind { Value, Incoming, Cycle, Ambiguous, Unknown };

struct PointerClobberResult {
  PointerClobberKind Kind;
  Value *V = nullptr;
};

// Resolve the pointer definition reaching a pointer-valued load.  MemorySSA
// supplies write order and CFG joins within each function.  Direct calls are
// summarized by resolving the tracked formal argument at every callee return;
// the callee's MemorySSA remains separate from the caller's graph.
class PointerClobberResolver {
  const DataLayout &DL;
  DenseMap<Function *, std::unique_ptr<FunctionMemorySSAState>> States;
  SmallPtrSet<LoadInst *, 8> ResolvingOrigins;

  FunctionMemorySSAState &getState(Function &F) {
    auto &State = States[&F];
    if (!State)
      State = std::make_unique<FunctionMemorySSAState>(F);
    return *State;
  }

  Value *getPointerOrigin(Value *V) {
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      if (isa<AllocaInst>(getUnderlyingObject(LI->getPointerOperand())) &&
          ResolvingOrigins.insert(LI).second) {
        PointerClobberResult Clobber = resolve(*LI);
        ResolvingOrigins.erase(LI);
        if (Clobber.Kind == PointerClobberKind::Value)
          return Clobber.V;
      }
    }
    SmallPtrSet<Value *, 8> Visited;
    return getPointerLoadOrigin(DL, V, Visited);
  }

  bool namesTrackedAddress(Value *Candidate, const Value *Tracked) {
    Value *CandidateOrigin = getPointerOrigin(Candidate);
    Value *TrackedOrigin = getPointerOrigin(const_cast<Value *>(Tracked));
    return CandidateOrigin && TrackedOrigin &&
           isSamePointerAddress(DL, CandidateOrigin, TrackedOrigin);
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
        RHS.Kind == PointerClobberKind::Ambiguous)
      return {PointerClobberKind::Ambiguous};
    if (LHS.Kind != RHS.Kind)
      return {PointerClobberKind::Ambiguous};
    if (LHS.Kind == PointerClobberKind::Value && LHS.V != RHS.V)
      return {PointerClobberKind::Ambiguous};
    return LHS;
  }

  PointerClobberResult resolveAccess(FunctionMemorySSAState &State,
                                     MemoryAccess *Access,
                                     const MemoryLocation &Location,
                                     SmallPtrSetImpl<MemoryAccess *> &Active) {
    MemorySSA &MSSA = State.get();
    DEBUG(if (Access) Logger::logs("proteus-pass")
              << "[Lambda arg analysis]: Resolving MemorySSA access " << *Access
              << "\n";)
    if (!Access || MSSA.isLiveOnEntryDef(Access))
      return {PointerClobberKind::Incoming};
    if (!Active.insert(Access).second)
      return {PointerClobberKind::Cycle};

    PointerClobberResult Result{PointerClobberKind::Unknown};
    if (auto *Phi = dyn_cast<MemoryPhi>(Access)) {
      bool HaveResult = false;
      for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
        MemoryAccess *Incoming = Phi->getIncomingValue(I);
        MemoryAccess *Clobber =
            MSSA.getWalker()->getClobberingMemoryAccess(Incoming, Location);
        PointerClobberResult IncomingResult =
            resolveAccess(State, Clobber, Location, Active);
        if (!HaveResult || Result.Kind == PointerClobberKind::Cycle) {
          Result = IncomingResult;
          HaveResult = true;
        } else {
          Result = merge(Result, IncomingResult);
        }
      }
      if (!HaveResult)
        Result = {PointerClobberKind::Unknown};
    } else if (auto *Def = dyn_cast<MemoryDef>(Access)) {
      Instruction *I = Def->getMemoryInst();
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (!SI->getValueOperand()->getType()->isPointerTy() ||
            !namesTrackedAddress(SI->getPointerOperand(), Location.Ptr)) {
          DEBUG(Logger::logs("proteus-pass")
                << "[Lambda arg analysis]: MemorySSA store does not exactly "
                   "define the tracked pointer: "
                << *SI << "\n");
          Result = {PointerClobberKind::Unknown};
        } else if (Value *Origin = getPointerOrigin(SI->getValueOperand())) {
          Result = {PointerClobberKind::Value, Origin};
        }
      } else if (auto *CB = dyn_cast<CallBase>(I)) {
        Result = resolveCall(State, *Def, *CB, Location, Active);
      }
    } else if (auto *Use = dyn_cast<MemoryUse>(Access)) {
      Result = resolveAccess(State, Use->getDefiningAccess(), Location, Active);
    }

    Active.erase(Access);
    return Result;
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

  PointerClobberResult
  resolveCall(FunctionMemorySSAState &CallerState, MemoryDef &CallDef,
              CallBase &CB, const MemoryLocation &CallerLocation,
              SmallPtrSetImpl<MemoryAccess *> &CallerActive) {
    Function *Callee = CB.getCalledFunction();
    if (!Callee || Callee->isDeclaration())
      return {PointerClobberKind::Unknown};

    unsigned TrackedArg = Callee->arg_size();
    for (unsigned I = 0; I < CB.arg_size() && I < Callee->arg_size(); ++I) {
      Value *Actual = CB.getArgOperand(I);
      if (Actual->getType()->isPointerTy() &&
          namesTrackedAddress(Actual, CallerLocation.Ptr)) {
        if (TrackedArg != Callee->arg_size())
          return {PointerClobberKind::Ambiguous};
        TrackedArg = I;
      }
    }
    if (TrackedArg == Callee->arg_size())
      return {PointerClobberKind::Unknown};

    FunctionMemorySSAState &CalleeState = getState(*Callee);
    Value *Formal = Callee->getArg(TrackedArg);
    MemoryLocation CalleeLocation(
        Formal,
        LocationSize::precise(DL.getPointerTypeSize(Formal->getType())));
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
      PointerClobberResult AtReturn =
          resolveAccess(CalleeState, Clobber, CalleeLocation, CalleeActive);

      if (AtReturn.Kind == PointerClobberKind::Incoming) {
        AtReturn = resolveAccess(CallerState, CallDef.getDefiningAccess(),
                                 CallerLocation, CallerActive);
      } else if (AtReturn.Kind == PointerClobberKind::Value) {
        if (auto *A = dyn_cast<Argument>(AtReturn.V))
          if (A->getParent() == Callee)
            AtReturn.V = CB.getArgOperand(A->getArgNo());
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

  PointerClobberResult resolve(LoadInst &LI) {
    FunctionMemorySSAState &State = getState(*LI.getFunction());
    MemorySSA &MSSA = State.get();
    MemoryAccess *LoadAccess = MSSA.getMemoryAccess(&LI);
    if (!LoadAccess)
      return {PointerClobberKind::Unknown};
    MemoryLocation Location = MemoryLocation::get(&LI);
    MemoryAccess *Clobber =
        MSSA.getWalker()->getClobberingMemoryAccess(LoadAccess, Location);
    SmallPtrSet<MemoryAccess *, 16> Active;
    PointerClobberResult Result =
        resolveAccess(State, Clobber, Location, Active);
    DEBUG(Logger::logs("proteus-pass")
          << "[Lambda arg analysis]: MemorySSA clobber result "
          << static_cast<unsigned>(Result.Kind) << " for " << LI << "\n");
    if (Result.Kind == PointerClobberKind::Cycle)
      return {PointerClobberKind::Unknown};
    return Result;
  }
};

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
  std::shared_ptr<PointerClobberResolver> Clobbers;
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
                   std::shared_ptr<PointerClobberResolver> ClobberResolver)
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
    if (isa<AllocaInst>(getUnderlyingObject(LI.getPointerOperand()))) {
      PointerClobberResult Clobber = Clobbers->resolve(LI);
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
      auto Res = getDominatingUse(DL, LI.getPointerOperand(), &LI, 0, LambdaCB);
      if (!Res) {
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
      WorkList.push_back({Res->DominatingWrite, &LI});
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
    auto Res =
        getDominatingUse(DL, &Alloca, MemoryAnalysisPtrUse, Offset, LambdaCB);
    if (!Res)
      return;

    WorkList.push_back({Res->DominatingWrite, &Alloca});
    // Res->Offset converts the current allocation-relative byte offset into
    // the coordinate system of DominatingWrite. For a field store it removes
    // the field displacement; for a memory transfer it translates destination
    // displacement into the corresponding source displacement.
    Offset -= Res->Offset;
  }

  void visitBitCastInst(BitCastInst &BC) {
    auto Res =
        getDominatingUse(DL, &BC, MemoryAnalysisPtrUse, Offset, LambdaCB);
    if (!Res)
      return;
    WorkList.push_back({Res->DominatingWrite, &BC});
    // Res->Offset converts the current cast-relative byte offset into the
    // coordinate system of DominatingWrite.
    Offset -= Res->Offset;
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    WorkList.push_back({ASC.getPointerOperand(), &ASC});
    auto Res =
        getDominatingUse(DL, &ASC, MemoryAnalysisPtrUse, Offset, LambdaCB);
    if (!Res)
      return;

    WorkList.push_back({Res->DominatingWrite, &ASC});
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

    for (User *U : F->users()) {
      auto *CB = dyn_cast<CallBase>(U);
      if (!CB)
        continue;
      DEBUG(Logger::logs("proteus-pass")
            << "Analysis crossed interprocedural boundary at "
            << *CB->getArgOperand(ArgNum) << "\n");
      WorkList.push_back({CB->getArgOperand(ArgNum), CB});
    }
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
