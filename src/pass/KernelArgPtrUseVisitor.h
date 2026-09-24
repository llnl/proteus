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
};

inline std::optional<LambdaPtrUseAnalysis> runDominatingUseVisitor(
    const DataLayout &DL, Value *ValueNeedingAnalysis, Value *SeenUse,
    int64_t TargetOffset, CallBase *LambdaCB = nullptr,
    std::shared_ptr<PointerClobberAnalysis> Clobbers = nullptr);

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

// Given a newly allocated ptr encountered in def-use analysis beginning at a
// Lambda callsite, we need to determine which definition dominates that ptr.
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
  DenseMap<Value *, int64_t> ValueOffsetMap;
  Value *TrackedBase = nullptr;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  std::shared_ptr<PointerClobberAnalysis> Clobbers;
  Instruction *UseBoundary = nullptr;
  Value *ClobberQueryPointer = nullptr;
  int64_t ClobberQueryOffset = 0;
  SmallVector<PointerClobberCandidate, 8> ClobberCandidates;
  SmallPtrSet<Instruction *, 8> SeenClobberCandidates;
  SmallVector<UseEdge> WorkList;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  SmallDenseSet<Value *> Seen;
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
    WorkList.push_back({PtrBegin, nullptr});
    // A pointer-transform on the backwards provenance path may also have an
    // earlier store as a user.  Visit that transform so those writes remain
    // visible, but stop before re-entering the lambda invocation itself.
    if (!isa<GetElementPtrInst, BitCastInst, AddrSpaceCastInst>(SeenUse))
      Seen.insert(SeenUse);
    if (LambdaCB)
      Seen.insert(LambdaCB);
    ValueOffsetMap[PtrBegin] = TargetOff;
  }
  auto back() { return WorkList.back(); }
  auto popBack() {
    auto Result = WorkList.back();
    Def = Result.LastVal;
    WorkList.pop_back();
    return Result;
  }
  auto getLastDef() { return Def; }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  auto getAnalysisResult() { return Result; }

  // Keep track of Function frame
  void pushBack(Value *NextVal, Value *CurVal) {
    WorkList.push_back(UseEdge{NextVal, CurVal});
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
    if (!SeenClobberCandidates.insert(&I).second)
      return;
    ClobberCandidates.push_back({&I, Pointer, TargetOffset});
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
                          ClobberCandidates);
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
      if (Seen.contains(Usr))
        continue;
      pushBack(Usr, V);
    }
  }

  void visitStoreInst(StoreInst &SI) {
    Value *Stored = SI.getValueOperand();
    Value *StoreBase = SI.getPointerOperand();

    // A pointer argument is commonly spilled in an unoptimized or optnone
    // callee before it is used.  Follow the slot's loads as carrying the same
    // pointee-relative offset instead of interpreting this as a write to the
    // tracked pointee.
    if (Stored == Def && Stored->getType()->isPointerTy()) {
      if (!ValueOffsetMap.contains(Stored)) {
        offsetValueMapFailure(Stored);
        return;
      }
      ValueOffsetMap[StoreBase] = ValueOffsetMap[Stored];
      for (User *Usr : StoreBase->users())
        if (Usr != &SI && !Seen.contains(Usr))
          pushBack(Usr, StoreBase);
      return;
    }

    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!ValueOffsetMap.contains(StoreBase)) {
      offsetValueMapFailure(StoreBase);
      return;
    }

    if (!StoreSize ||
        !offsetCoveredByRange(ValueOffsetMap[StoreBase], 0, *StoreSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to offset " << ValueOffsetMap[&SI]
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    addClobberCandidate(SI, StoreBase, ValueOffsetMap[StoreBase]);
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Ignoring non-pointer read " << LI
            << "\n");
      return;
    }
    if (!ValueOffsetMap.contains(LI.getPointerOperand())) {
      offsetValueMapFailure(LI.getPointerOperand());
      return;
    }
    ValueOffsetMap[&LI] = ValueOffsetMap[LI.getPointerOperand()];
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
    if (!DefBeforeCB || !ValueOffsetMap.contains(DefBeforeCB)) {
      offsetValueMapFailure(DefBeforeCB ? DefBeforeCB : TrackedBase);
      return;
    }

    if (!CB.onlyReadsMemory())
      addClobberCandidate(CB, DefBeforeCB, ValueOffsetMap[DefBeforeCB]);

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
      ValueOffsetMap[ArgToTrack] = ValueOffsetMap[DefBeforeCB];
      DEBUG(Logger::logs("proteus-pass")
            << "    Looking at uses of " << *ArgToTrack << "\n");
      for (User *Usr : ArgToTrack->users())
        pushBack(Usr, ArgToTrack);
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
    if (!Def && ValueOffsetMap.contains(&GEP)) {
      ValueOffsetMap[GEPBase] = ValueOffsetMap[&GEP] + GEPOffset;
      CoordinateCorrection -= GEPOffset;
      ClobberQueryPointer = GEPBase;
      ClobberQueryOffset = ValueOffsetMap[GEPBase];
      pushPointerUsers(GEPBase);
      return;
    }

    if (!ValueOffsetMap.contains(GEPBase)) {
      offsetValueMapFailure(GEPBase);
      return;
    }

    auto ResultSize = getTypeStoreSize(DL, GEP.getResultElementType());
    DEBUG(if (ResultSize) Logger::logs("proteus-pass")
              << "    GEP size = " << *ResultSize << "\n";)
    if (ResultSize &&
        !offsetCoveredByRange(ValueOffsetMap[GEPBase], GEPOffset, *ResultSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found GEP applicable to offset=" << ValueOffsetMap[GEPBase]
          << " ; " << GEP << "\n");
    // We found a GEP, now we need to track the GEP itself, so the TargetOffset
    // is now zero again
    ValueOffsetMap[&GEP] = ValueOffsetMap[GEPBase] - GEPOffset;
    Offset += GEPOffset;
    DEBUG(Logger::logs("proteus-pass")
          << "    " << "Setting map K " << GEP << " : " << ValueOffsetMap[&GEP]
          << "\n");
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
    if (!ValueOffsetMap.contains(&Alloca)) {
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
      ValueOffsetMap[BC.getOperand(0)] = Offset;
    // AddrSpaceCast does not change the offset we track.
    ValueOffsetMap[&BC] = ValueOffsetMap[BC.getOperand(0)];
    pushPointerUsers(&BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // The constructor automatically populates the map with ASC's offset
    // if its not present we need to rely on the pointer operand's offset
    if (!ValueOffsetMap.contains(&ASC)) {
      if (!ValueOffsetMap.contains(ASC.getPointerOperand())) {
        offsetValueMapFailure(ASC.getPointerOperand());
        return;
      }
      // AddrSpaceCast does not change the offset we track.
      ValueOffsetMap[&ASC] = ValueOffsetMap[ASC.getPointerOperand()];
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Setting offset " << ASC << " = "
          << ValueOffsetMap[&ASC]);

    pushPointerUsers(&ASC);
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      if (Def != MS->getRawDest())
        return;
      if (!ValueOffsetMap.contains(Def)) {
        offsetValueMapFailure(Def);
        return;
      }
      auto *Len = dyn_cast<ConstantInt>(MS->getLength());
      if (Len &&
          !offsetCoveredByRange(ValueOffsetMap[Def], 0, Len->getZExtValue()))
        return;
      addClobberCandidate(I, Def, ValueOffsetMap[Def]);
      return;
    }

    auto *MT = cast<MemTransferInst>(&I); // memcpy/memmove

    // A transfer defines the tracked memory only when the use edge reached it
    // through the destination operand.  Reaching the same intrinsic through
    // its source is merely a read.
    if (Def != MT->getRawDest())
      return;

    if (!ValueOffsetMap.contains(Def)) {
      offsetValueMapFailure(Def);
      return;
    }

    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    if (!Len) {
      addClobberCandidate(I, Def, ValueOffsetMap[Def]);
      return;
    }
    if (!offsetCoveredByRange(ValueOffsetMap[Def], 0, Len->getZExtValue()))
      return;
    addClobberCandidate(I, Def, ValueOffsetMap[Def]);
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
    else
      continue;
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
