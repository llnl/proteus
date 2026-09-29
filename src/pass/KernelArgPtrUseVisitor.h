#ifndef PROTEUS_KERNELARGPTRDEFVISITOR_H
#define PROTEUS_KERNELARGPTRDEFVISITOR_H

#include "Helpers.h"
#include "PointerClobberAnalysis.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

// Any instruction creating a new ptr needs use analysis
bool needsDefUseAnalysis(Value *Val) {
  return isa<AddrSpaceCastInst>(Val) || isa<AllocaInst>(Val) ||
         isa<BitCastInst>(Val) || isa<IntToPtrInst>(Val);
}

struct LambdaPtrUseAnalysis {
  Value *DominatingWrite = nullptr;
  // The write selected by MemorySSA. When the backwards analysis continues
  // through DominatingWrite, this is its actual downstream use boundary.
  Instruction *ClobberingInstruction = nullptr;
  int64_t Offset = 0;
  int64_t EnclosingObjectOffsetCorrection = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct CallerFrame {
  Value *ValEnteringCallBase;
  CallBase *CallerCB;
  Function *Callee;
};

// We track use-edges in our analysis.
struct UseEdge {
  Value *CurVal;
  Value *LastVal;
  unsigned Context;
};

inline std::optional<LambdaPtrUseAnalysis> runDominatingUseVisitor(
    const DataLayout &DL, Value *ValueNeedingAnalysis, Value *SeenUse,
    int64_t TargetOffset, CallBase *LambdaCB = nullptr,
    std::shared_ptr<PointerClobberAnalysis> Clobbers = nullptr);

// Given a newly allocated pointer encountered in def-use analysis beginning at
// a lambda callsite, determine which definition dominates that pointer.
class LambdaInstUseVisitor : public InstVisitor<LambdaInstUseVisitor> {
private:
  DominatorTree DTree;
  int64_t Offset = 0;
  // Additional correction when analysis starts from an interior pointer but
  // the selected clobber writes through its enclosing object.
  int64_t CoordinateCorrection = 0;
  // The ValueOffsetMap contains the "live range" of the ptr we're analyzing.
  // For example, let's say that LambdaInstUseVisitor is handed ptr %0 = alloca
  // ptr, and LambdaArgVisitor has already identified that the closure starts at
  // byte
  // 8.  In this case, ValueOffsetMap[%0] = 8.  If we encounter a store like
  // store ptr %2, ptr%0, align 8, we don't care, because its written outside
  // of the range of the closure.
  using ContextID = unsigned;
  using ContextValue = std::pair<Value *, ContextID>;
  // A Value alone is not a sufficient key after crossing a call: the same
  // callee instruction may be visited with different offsets from different
  // call sites. ContextID identifies the complete active call chain.
  DenseMap<ContextValue, int64_t> ValueOffsetMap;
  Value *TrackedBase = nullptr;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  std::shared_ptr<PointerClobberAnalysis> Clobbers;
  // The instruction that consumes the value derived from PtrBegin, and thus
  // the point before which we ask MemorySSA which collected candidate is the
  // reaching clobber. This is normally the lambda call. If PtrBegin belongs to
  // another function, SeenUse is the local consumer on the backwards-analysis
  // edge that brought us to PtrBegin and becomes the boundary instead.
  Instruction *UseBoundary = nullptr;
  Value *ClobberQueryPointer = nullptr;
  int64_t ClobberQueryOffset = 0;
  PointerClobberCandidateMap ClobberCandidates;
  struct CallContext {
    CallBase *Caller = nullptr;
    ContextID Parent = 0;
  };
  SmallVector<CallContext, 4> CallContexts{{nullptr, 0}};
  DenseMap<std::pair<CallBase *, ContextID>, ContextID> CallContextIDs;
  struct DeferredPointerMerge {
    Instruction *Merge;
    SmallVector<Value *, 4> Incoming;
    ContextID Context;
  };
  SmallVector<DeferredPointerMerge, 4> DeferredPointerMerges;
  SmallDenseSet<std::pair<Instruction *, ContextID>, 4> DeferredMergeSet;
  SmallVector<UseEdge> WorkList;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  ContextID CurrentContext = 0;
  SmallDenseSet<ContextValue, 16> Seen;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

public:
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We
  // need to track where the calling LambdaArgVisitor came in from, so that our
  // analysis does not
  LambdaInstUseVisitor(
      Value *PtrBegin, Value *SeenUse, CallBase *LambdaCB, const DataLayout &Dl,
      int64_t TargetOff,
      std::shared_ptr<PointerClobberAnalysis> ClobberAnalysis = nullptr)
      : TrackedBase(PtrBegin), DL(Dl), Clobbers(std::move(ClobberAnalysis)),
        ClobberQueryPointer(PtrBegin), ClobberQueryOffset(TargetOff) {
    UseBoundary = dyn_cast_or_null<Instruction>(LambdaCB);
    auto *StartInstruction = dyn_cast<Instruction>(PtrBegin);
    if (!UseBoundary || !StartInstruction ||
        UseBoundary->getFunction() != StartInstruction->getFunction())
      UseBoundary = dyn_cast<Instruction>(SeenUse);
    WorkList.push_back({PtrBegin, nullptr, 0});
    // A pointer-transform on the backwards provenance path may also have an
    // earlier store as a user.  Visit that transform so those writes remain
    // visible, but stop before re-entering the lambda invocation itself.
    if (!isa<GetElementPtrInst, BitCastInst, AddrSpaceCastInst>(SeenUse))
      Seen.insert({SeenUse, 0});
    if (LambdaCB)
      Seen.insert({LambdaCB, 0});
    ValueOffsetMap[{PtrBegin, 0}] = TargetOff;
  }
  auto back() { return WorkList.back(); }
  auto popBack() {
    auto Result = WorkList.back();
    Def = Result.LastVal;
    CurrentContext = Result.Context;
    WorkList.pop_back();
    return Result;
  }
  auto getLastDef() { return Def; }
  bool seen(Value *Val) { return Seen.contains({Val, CurrentContext}); }
  void markAsSeen(Value *Val) { Seen.insert({Val, CurrentContext}); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  bool retryDeferredPointerMerges() {
    if (DeferredPointerMerges.empty())
      return false;

    SmallVector<size_t, 4> ReadyIndices;
    for (size_t I = 0; I < DeferredPointerMerges.size(); ++I) {
      auto &Deferred = DeferredPointerMerges[I];
      if (llvm::all_of(Deferred.Incoming, [this, &Deferred](Value *V) {
            return ValueOffsetMap.contains({V, Deferred.Context});
          }))
        ReadyIndices.push_back(I);
    }

    if (ReadyIndices.empty()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Pointer merge inputs could not all "
               "be reached\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return false;
    }

    for (auto It = ReadyIndices.rbegin(); It != ReadyIndices.rend(); ++It) {
      DeferredPointerMerge Deferred = std::move(DeferredPointerMerges[*It]);
      DeferredPointerMerges.erase(DeferredPointerMerges.begin() + *It);
      DeferredMergeSet.erase({Deferred.Merge, Deferred.Context});
      Seen.erase({Deferred.Merge, Deferred.Context});
      pushBack(Deferred.Merge, nullptr, Deferred.Context);
    }
    return true;
  }

  auto getAnalysisResult() { return Result; }

  void pushBack(Value *NextVal, Value *CurVal) {
    pushBack(NextVal, CurVal, CurrentContext);
  }
  void pushBack(Value *NextVal, Value *CurVal, ContextID Context) {
    WorkList.push_back(UseEdge{NextVal, CurVal, Context});
  }

  bool hasOffset(Value *V, ContextID Context) const {
    return ValueOffsetMap.contains({V, Context});
  }
  bool hasOffset(Value *V) const { return hasOffset(V, CurrentContext); }
  int64_t getOffset(Value *V, ContextID Context) const {
    return ValueOffsetMap.lookup({V, Context});
  }
  int64_t getOffset(Value *V) const { return getOffset(V, CurrentContext); }
  void setOffset(Value *V, int64_t Value, ContextID Context) {
    ValueOffsetMap[{V, Context}] = Value;
  }
  void setOffset(Value *V, int64_t Value) {
    setOffset(V, Value, CurrentContext);
  }

  ContextID getCallContext(CallBase &CB) {
    auto Key = std::make_pair(&CB, CurrentContext);
    auto [It, Inserted] = CallContextIDs.try_emplace(Key, CallContexts.size());
    if (Inserted)
      CallContexts.push_back({&CB, CurrentContext});
    return It->second;
  }

  void offsetValueMapFailure(Value *V) {
    AnalysisFailed = true;
    AnalysisSuccess = false;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to absence of " << *V
          << " in offset tracking map, this is an internal compiler bug\n");
  }

  void addClobberCandidate(Instruction &I, Value *Pointer,
                           int64_t TargetOffset) {
    if (!ClobberCandidates
             .try_emplace(&I,
                          PointerClobberCandidate{&I, Pointer, TargetOffset})
             .second)
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Collected possible clobber " << I
          << "\n");
  }

  // Candidate collection is complete before this query runs. MemorySSA may
  // select only an instruction reached through the definition's relevant-use
  // graph; traversal order therefore cannot choose a writer.
  void resolveCollectedClobber() {
    if (ClobberCandidates.empty()) {
      AnalysisFailed = true;
      return;
    }

    auto *PtrInstruction = dyn_cast_or_null<Instruction>(ClobberQueryPointer);
    if (!Clobbers || !UseBoundary || !PtrInstruction ||
        UseBoundary->getFunction() != PtrInstruction->getFunction()) {
      AnalysisFailed = true;
      return;
    }

    PointerClobberResult Clobber =
        Clobbers->resolve(ClobberQueryPointer, *UseBoundary, ClobberQueryOffset,
                          &ClobberCandidates);
    if (Clobber.Kind == PointerClobberKind::Value) {
      Result = {.DominatingWrite = Clobber.V,
                .ClobberingInstruction = Clobber.ClobberingInstruction,
                .Offset = Clobber.Offset,
                .EnclosingObjectOffsetCorrection = CoordinateCorrection,
                .ChangedRCLayout = Clobber.ChangedRCLayout};
      AnalysisSuccess = Clobber.V != nullptr;
      AnalysisFailed = !AnalysisSuccess;
      return;
    }
    if (Clobber.Kind == PointerClobberKind::Ambiguous) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    auto *MT = dyn_cast_or_null<MemTransferInst>(Clobber.ClobberingInstruction);
    if (!MT || !Clobber.ClobberPointer) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
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
    Result = {.DominatingWrite = SrcBase,
              .ClobberingInstruction = Clobber.ClobberingInstruction,
              .Offset = DstOff - SrcOff + CoordinateCorrection,
              .EnclosingObjectOffsetCorrection = CoordinateCorrection,
              .ChangedRCLayout = std::nullopt};
    AnalysisSuccess = true;
    AnalysisFailed = false;
  }

  void pushPointerUsers(Value *V) {
    for (User *Usr : V->users()) {
      if (Seen.contains({Usr, CurrentContext}))
        continue;
      pushBack(Usr, V);
    }
  }

  void propagatePointerMerge(Value &Merged, ArrayRef<Value *> Incoming) {
    std::optional<int64_t> MergedOffset;
    for (Value *V : Incoming) {
      auto It = ValueOffsetMap.find({V, CurrentContext});
      if (It == ValueOffsetMap.end()) {
        DEBUG(Logger::logs("proteus-pass")
              << "    [PTR use analysis]: Deferring pointer merge with an "
                 "untracked incoming value: "
              << Merged << "\n");
        auto Key = std::make_pair(cast<Instruction>(&Merged), CurrentContext);
        if (DeferredMergeSet.insert(Key).second)
          DeferredPointerMerges.push_back(
              {Key.first, SmallVector<Value *, 4>(Incoming), CurrentContext});
        return;
      }
      if (!MergedOffset)
        MergedOffset = It->second;
      else if (*MergedOffset != It->second) {
        DEBUG(Logger::logs("proteus-pass")
              << "    [PTR use analysis]: Pointer merge combines different "
                 "tracked offsets: "
              << Merged << "\n");
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
    }

    if (!MergedOffset) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    setOffset(&Merged, *MergedOffset);
    pushPointerUsers(&Merged);
  }

  void visitStoreInst(StoreInst &SI) {
    Value *Stored = SI.getValueOperand();
    Value *StoreBase = SI.getPointerOperand();

    // A pointer argument is commonly spilled in an unoptimized or optnone
    // callee before it is used.  Follow the slot's loads as carrying the same
    // pointee-relative offset instead of interpreting this as a write to the
    // tracked pointee.
    if (Stored == Def && Stored->getType()->isPointerTy()) {
      if (!hasOffset(Stored)) {
        offsetValueMapFailure(Stored);
        return;
      }
      setOffset(StoreBase, getOffset(Stored));
      for (User *Usr : StoreBase->users())
        if (Usr != &SI && !Seen.contains({Usr, CurrentContext}))
          pushBack(Usr, StoreBase);
      return;
    }

    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!hasOffset(StoreBase)) {
      offsetValueMapFailure(StoreBase);
      return;
    }

    if (!StoreSize ||
        !offsetCoveredByRange(getOffset(StoreBase), 0, *StoreSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to offset " << getOffset(StoreBase)
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    addClobberCandidate(SI, StoreBase, getOffset(StoreBase));
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Ignoring non-pointer read " << LI
            << "\n");
      return;
    }
    if (!hasOffset(LI.getPointerOperand())) {
      offsetValueMapFailure(LI.getPointerOperand());
      return;
    }
    setOffset(&LI, getOffset(LI.getPointerOperand()));
    pushPointerUsers(&LI);
  }

  void visitCallBase(CallBase &CB) {
    // Lifetime markers describe the validity of an allocation, not a write to
    // its contents.  Following their declaration as if it were an ordinary
    // callee makes an otherwise valid search fail b.efore reaching a store.
    if (auto *II = dyn_cast<IntrinsicInst>(&CB)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end)
        return;
    }

    Value *DefBeforeCB = getLastDef();
    if (!DefBeforeCB || !hasOffset(DefBeforeCB)) {
      offsetValueMapFailure(DefBeforeCB ? DefBeforeCB : TrackedBase);
      return;
    }

    if (!CB.onlyReadsMemory())
      addClobberCandidate(CB, DefBeforeCB, getOffset(DefBeforeCB));

    Function *F = CB.getCalledFunction();
    if (!F || F->isDeclaration())
      return;

    bool FoundArg = false;
    for (size_t ArgI = 0; ArgI < F->arg_size(); ++ArgI) {
      DEBUG(Logger::logs("proteus-pass") << "    ARG " << ArgI << " VAL "
                                         << *CB.getArgOperand(ArgI) << "\n");
      if (CB.getArgOperand(ArgI) != DefBeforeCB)
        continue;

      FoundArg = true;
      Argument *ArgToTrack = F->getArg(ArgI);
      ContextID CalleeContext = getCallContext(CB);
      setOffset(ArgToTrack, getOffset(DefBeforeCB), CalleeContext);
      DEBUG(Logger::logs("proteus-pass")
            << "    Looking at uses of " << *ArgToTrack << "\n");
      for (User *Usr : ArgToTrack->users())
        pushBack(Usr, ArgToTrack, CalleeContext);
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    Beginning analysis within " << *F << "\n");
    if (!FoundArg) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Call does not pass the tracked "
               "pointer on any callee argument: "
            << CB << "\n");
      // This use does not forward the tracked pointer into the callee. It can
      // still be a candidate through aliasing, which MemorySSA will decide.
      return;
    }
  }

  void visitReturnInst(ReturnInst &RI) {
    Value *Returned = RI.getReturnValue();
    if (!Returned || Returned != Def || !Returned->getType()->isPointerTy() ||
        !hasOffset(Returned))
      return;

    if (CurrentContext == 0)
      return;

    CallContext Context = CallContexts[CurrentContext];
    CallBase *Caller = Context.Caller;
    if (!Caller || !Caller->getType()->isPointerTy() ||
        Caller->getCalledFunction() != RI.getFunction())
      return;
    setOffset(Caller, getOffset(Returned), Context.Parent);
    for (User *Usr : Caller->users())
      if (!Seen.contains({Usr, Context.Parent}))
        pushBack(Usr, Caller, Context.Parent);
  }

  void visitSelectInst(SelectInst &SI) {
    SmallVector<Value *, 2> Incoming{SI.getTrueValue(), SI.getFalseValue()};
    propagatePointerMerge(SI, Incoming);
  }

  void visitPHINode(PHINode &Phi) {
    SmallVector<Value *, 4> Incoming(Phi.incoming_values());
    propagatePointerMerge(Phi, Incoming);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    // We don't want to use GetPointerBaseWithConstantOffset here.
    // We actually don't want the true "pointer base" here.  I.E. if we are
    // analyzing the dominating store to %4 = addrspacecast ptr addrspace(5) %3
    // to ptr where %3 = alloca %class.anon.1, align 8, addrspace(5)
    // GetPointerBaseWithConstantOffset gets us 3, which we (a) don't know about
    // and (b) don't care about, we just care about the SSA to %4.
    APInt StepOffset(DL.getIndexTypeSizeInBits(GEP.getType()), 0);
    if (!GEP.accumulateConstantOffset(DL, StepOffset)) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    int64_t GEPOffset = StepOffset.getSExtValue();
    DEBUG(Logger::logs("proteus-pass")
          << "    " << "Computed GEP offset " << GEPOffset << "\n");
    Value *GEPBase = GEP.getPointerOperand();
    if (!GEPBase)
      return;

    // When analysis begins at a field GEP, the reaching write can target the
    // enclosing object (for example, a copy constructor). Translate the
    // field-relative target back into the base object's coordinates before
    // looking for that write.
    if (!Def && hasOffset(&GEP)) {
      setOffset(GEPBase, getOffset(&GEP) + GEPOffset);
      CoordinateCorrection -= GEPOffset;
      ClobberQueryPointer = GEPBase;
      ClobberQueryOffset = getOffset(GEPBase);
      pushPointerUsers(GEPBase);
      return;
    }

    if (!hasOffset(GEPBase)) {
      offsetValueMapFailure(GEPBase);
      return;
    }

    auto ResultSize = getTypeStoreSize(DL, GEP.getResultElementType());
    DEBUG(if (ResultSize) Logger::logs("proteus-pass")
              << "    GEP size = " << *ResultSize << "\n";)
    if (ResultSize &&
        !offsetCoveredByRange(getOffset(GEPBase), GEPOffset, *ResultSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found GEP applicable to offset=" << getOffset(GEPBase)
          << " ; " << GEP << "\n");
    // We found a GEP, now we need to track the GEP itself, so the TargetOffset
    // is now zero again
    setOffset(&GEP, getOffset(GEPBase) - GEPOffset);
    Offset += GEPOffset;
    DEBUG(Logger::logs("proteus-pass") << "    " << "Setting map K " << GEP
                                       << " : " << getOffset(&GEP) << "\n");
    pushPointerUsers(&GEP);
  }

  // todo: these three methods may need to be changed to find a dominating store
  // particularly for the case of mutable lambdas.
  void visitAllocaInst(AllocaInst &Alloca) {
    // This analysis should only ever encounter an AllocaInst as the first
    // instruction We assert this below and log a failure otherwise
    if (Def) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      DEBUG(Logger::logs("proteus-pass")
            << "    Dominating use analysis somehow reached AllocaInst from "
               "non-null def\n");
      return;
    }
    if (!hasOffset(&Alloca)) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      DEBUG(Logger::logs("proteus-pass")
            << "    Value offset map not correctly initialized with "
               "AllocaInst\n");
      return;
    }

    pushPointerUsers(&Alloca);
  }

  // TODO(bowen) come up with a unit test for an analysis starting with
  // a BC
  void visitBitCastInst(BitCastInst &BC) {
    // If the last Def is nullptr, we have just begun the use analysis.
    // In this case, respect the constructor's offset for the pointer operand.
    if (!Def)
      setOffset(BC.getOperand(0), Offset);
    // AddrSpaceCast does not change the offset we track.
    setOffset(&BC, getOffset(BC.getOperand(0)));
    pushPointerUsers(&BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // The constructor automatically populates the map with ASC's offset
    // if its not present we need to rely on the pointer operand's offset
    if (!hasOffset(&ASC)) {
      if (!hasOffset(ASC.getPointerOperand())) {
        offsetValueMapFailure(ASC.getPointerOperand());
        return;
      }
      // AddrSpaceCast does not change the offset we track.
      setOffset(&ASC, getOffset(ASC.getPointerOperand()));
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Setting offset " << ASC << " = "
          << getOffset(&ASC));

    pushPointerUsers(&ASC);
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      if (Def != MS->getRawDest())
        return;
      if (!hasOffset(Def)) {
        offsetValueMapFailure(Def);
        return;
      }
      auto *Len = dyn_cast<ConstantInt>(MS->getLength());
      if (Len && !offsetCoveredByRange(getOffset(Def), 0, Len->getZExtValue()))
        return;
      addClobberCandidate(I, Def, getOffset(Def));
      return;
    }

    auto *MT = cast<MemTransferInst>(&I); // memcpy/memmove

    // A transfer defines the tracked memory only when the use edge reached it
    // through the destination operand.  Reaching the same intrinsic through
    // its source is merely a read.
    if (Def != MT->getRawDest())
      return;

    if (!hasOffset(Def)) {
      offsetValueMapFailure(Def);
      return;
    }

    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    if (!Len) {
      addClobberCandidate(I, Def, getOffset(Def));
      return;
    }
    if (!offsetCoveredByRange(getOffset(Def), 0, Len->getZExtValue()))
      return;
    addClobberCandidate(I, Def, getOffset(Def));
  }

  void visitInstruction(Instruction &I) {
    // Non-memory terminal uses (comparisons, returns, debug operations) cannot
    // clobber the tracked storage. Pointer-producing or memory-writing uses
    // would hide part of the relevant-use graph, so reject those
    // conservatively.
    if (I.getType()->isPointerTy() || I.mayWriteToMemory()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Unsupported relevant use "
            << I.getOpcodeName() << ": " << I << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
    }
  }
};

inline std::optional<LambdaPtrUseAnalysis>
runDominatingUseVisitor(const DataLayout &DL, Value *ValueNeedingAnalysis,
                        Value *SeenUse, int64_t TargetOffset,
                        CallBase *LambdaCB,
                        std::shared_ptr<PointerClobberAnalysis> Clobbers) {
  DEBUG(Logger::logs("proteus-pass")
        << "Beginning PtrUse analysis with offset = " << TargetOffset << "\n");

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, SeenUse, LambdaCB, DL,
                               TargetOffset, std::move(Clobbers));
  // Analysis loop
  while (!Visitor.failed()) {
    while (!Visitor.empty() && !Visitor.failed()) {
      auto *V = Visitor.popBack().CurVal;
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: Visiting ptr use " << *V << "\n");
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
    }
    if (Visitor.empty() && !Visitor.retryDeferredPointerMerges())
      break;
  }
  if (!Visitor.failed())
    Visitor.resolveCollectedClobber();
  if (!Visitor.success() || Visitor.failed()) {
    DEBUG(
        Logger::logs("proteus-pass")
        << "  [PTR use analysis] [WARNING]: Dominating use analysis FAILED for "
        << *ValueNeedingAnalysis << " <-- " << *SeenUse << "\n");
    return std::nullopt;
  }
  LambdaPtrUseAnalysis Info = Visitor.getAnalysisResult();
  if (!Info.DominatingWrite)
    return std::nullopt;
  DEBUG(Logger::logs("proteus-pass")
        << "  [PTR USE ANALYSIS]: Computed offset " << Info.Offset << "\n");
  return Info;

  return std::nullopt;
}
} // namespace proteus

#endif
