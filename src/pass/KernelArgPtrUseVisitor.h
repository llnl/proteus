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
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct ClobberQuery {
  Function *Frame = nullptr;
  Value *TrackedPtr = nullptr;
  Instruction *UseBoundary = nullptr;
  int64_t TargetOffset = 0;

  bool operator==(const ClobberQuery &Other) const {
    return Frame == Other.Frame && TrackedPtr == Other.TrackedPtr &&
           UseBoundary == Other.UseBoundary &&
           TargetOffset == Other.TargetOffset;
  }
};

struct ClobberQueryDenseMapInfo {
  static ClobberQuery getEmptyKey() {
    return {DenseMapInfo<Function *>::getEmptyKey(),
            DenseMapInfo<Value *>::getEmptyKey(),
            DenseMapInfo<Instruction *>::getEmptyKey(), 0};
  }

  static ClobberQuery getTombstoneKey() {
    return {DenseMapInfo<Function *>::getTombstoneKey(),
            DenseMapInfo<Value *>::getTombstoneKey(),
            DenseMapInfo<Instruction *>::getTombstoneKey(), 0};
  }

  static unsigned getHashValue(const ClobberQuery &Query) {
    return static_cast<unsigned>(hash_combine(
        Query.Frame, Query.TrackedPtr, Query.UseBoundary, Query.TargetOffset));
  }

  static bool isEqual(const ClobberQuery &LHS, const ClobberQuery &RHS) {
    return LHS == RHS;
  }
};

// We track use-edges in our analysis, but call them clobbers because
// each one is a potential clobber.
struct PotentialClobber {
  Value *CurVal = nullptr;
  // this is the use-edge, and potentially different from the use boundary
  // below.  For example, the use boundary could be a lambda CB but the
  // LastVal could be a GEP in this case:
  // %1 = alloca ptr
  // %2 = getelementptr %1, 0, 1
  // store %0, %2
  // call lambda(%1)
  Value *LastVal = nullptr;
  // this is the boundary at which we actually care what the use is, in
  // the example above it would be call lambda(%1)
  Instruction *UseBoundary = nullptr;
  Function *Frame = nullptr;
};

inline std::optional<MemoryLocation>
getTrackedPointerLocation(const DataLayout &DL, Value *Ptr) {
  if (!Ptr || !Ptr->getType()->isPointerTy())
    return std::nullopt;

  Type *PointeeTy = nullptr;
  if (auto *AI = dyn_cast_or_null<AllocaInst>(Ptr))
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
  // The ValueOffsetMap contains the "live range" of the ptr we're analyzing.
  // For example, let's say that LambdaInstUseVisitor is handed ptr %0 = alloca
  // ptr, and LambdaArgVisitor has already identified that the closure starts at
  // byte
  // 8.  In this case, ValueOffsetMap[%0] = 8.  If we encounter a store like
  // store ptr %2, ptr%0, align 8, we don't care, because its written outside
  // of the range of the closure.
  DenseMap<Value *, int64_t> ValueOffsetMap;
  DenseMap<ClobberQuery, MemorySSAClobber, ClobberQueryDenseMapInfo> QueryCache;
  std::shared_ptr<MemorySSAClobberOracle> ClobberOracle;
  Value *TrackedBase = nullptr;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  SmallVector<PotentialClobber> WorkList;
  PotentialClobber Current;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  SmallDenseSet<Value *> Seen;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;
  // This vector contains all the potentially clobbering instructions for
  // the pointer who's write we are tracking in question.
  SmallVector<Instruction *> ClobberCandidates;

public:
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We
  // need to track where the calling LambdaArgVisitor came in from, so that our
  // analysis does not
  LambdaInstUseVisitor(Value *PtrBegin, Instruction *SeenUse,
                       CallBase *LambdaCB, const DataLayout &Dl,
                       int64_t TargetOff,
                       std::shared_ptr<MemorySSAClobberOracle> Oracle)
      : ClobberOracle(std::move(Oracle)), TrackedBase(PtrBegin), DL(Dl) {
    ClobberQuery Query{SeenUse->getFunction(), PtrBegin, SeenUse, TargetOff};
    queryMemorySSA(Query);
    WorkList.push_back({PtrBegin, nullptr, SeenUse, SeenUse->getFunction()});
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
    Current = WorkList.back();
    Def = Current.LastVal;
    WorkList.pop_back();
    return Current;
  }
  auto getLastDef() { return Def; }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  auto getAnalysisResult() { return Result; }

  // Keep track of Function frame
  void pushBack(Value *NextVal, Value *CurVal, Function *Frame,
                Instruction *UseBoundary) {
    WorkList.push_back({NextVal, CurVal, UseBoundary, Frame});
  }

  MemorySSAClobber queryMemorySSA(const ClobberQuery &Query) {
    auto Cached = QueryCache.find(Query);
    if (Cached != QueryCache.end())
      return Cached->second;

    MemorySSAClobber Result;
    if (!Query.Frame || !Query.TrackedPtr || !Query.UseBoundary ||
        Query.UseBoundary->getFunction() != Query.Frame) {
      QueryCache.try_emplace(Query, Result);
      return Result;
    }

    auto Location = getTrackedPointerLocation(DL, Query.TrackedPtr);
    if (Location)
      Result = ClobberOracle->query(*Location, *Query.UseBoundary);
    QueryCache.try_emplace(Query, Result);
    return Result;
  }

  bool isSelectedClobber(Instruction &Candidate) {
    ClobberQuery Query{Current.Frame, TrackedBase, Current.UseBoundary};
    MemorySSAClobber Clobber = queryMemorySSA(Query);
    bool Selected = Clobber.Kind == MemorySSAClobberKind::Definition &&
                    Clobber.Instruction == &Candidate;
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "    [PTR use analysis]: MemorySSA "
         << (Selected ? "selected " : "did not select ") << Candidate;
      if (!Selected && Clobber.Instruction)
        OS << "; selected clobber is " << *Clobber.Instruction;
      OS << "\n";
    });
    return Selected;
  }

  void offsetValueMapFailure(Value *V) {
    AnalysisFailed = true;
    AnalysisSuccess = false;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to absence of " << *V
          << " in offset tracking map, this is an internal compiler bug\n");
  }

  // WorkList is LIFO.  Enqueue possible writers last so they are inspected
  // before an older store reached through a GEP.  Otherwise a memcpy/memmove
  // call can be skipped merely because the initializer happens to appear
  // earlier in Value::users().
  void pushBackPointerUsers(Value *V) {
    SmallVector<User *, 4> PossibleWriters;
    for (User *Usr : V->users()) {
      if (Seen.contains(Usr))
        continue;
      auto *CB = dyn_cast_or_null<CallBase>(Usr);
      if (CB && !isa<DbgInfoIntrinsic>(CB) && !CB->onlyReadsMemory()) {
        PossibleWriters.push_back(Usr);
        continue;
      }
      // All these users don't cross an interprocedural boundary (only CB does),
      // so just copy UseBoundary/Frame data
      pushBack(Usr, V, Current.Frame, Current.UseBoundary);
    }
    for (User *Usr : PossibleWriters)
      pushBack(Usr, V, Current.Frame, Current.UseBoundary);
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
          pushBack(Usr, StoreBase, Current.Frame, Current.UseBoundary);
      return;
    }

    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!ValueOffsetMap.contains(StoreBase)) {
      offsetValueMapFailure(StoreBase);
      return;
    }

    // Def-use traversal discovers every store through the tracked pointer.
    // Only the MemorySSA definition reaching the use boundary can provide the
    // value consumed there; older and non-reaching stores are not candidates.
    if (!isSelectedClobber(SI))
      return;

    if (!StoreSize ||
        !offsetCoveredByRange(ValueOffsetMap[StoreBase], 0, *StoreSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to offset " << ValueOffsetMap[&SI]
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    AnalysisFailed = false;
    AnalysisSuccess = true;
    Result = {.DominatingWrite = SI.getValueOperand(),
              .Offset = Offset,
              .ChangedRCLayout = std::nullopt};
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Expected a pointer load, got " << LI
            << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    if (!ValueOffsetMap.contains(LI.getPointerOperand())) {
      offsetValueMapFailure(LI.getPointerOperand());
      return;
    }
    ValueOffsetMap[&LI] = ValueOffsetMap[LI.getPointerOperand()];
    pushBackPointerUsers(&LI);
  }

  void visitCallBase(CallBase &CB) {
    // Lifetime markers describe the validity of an allocation, not a write to
    // its contents.  Following their declaration as if it were an ordinary
    // callee makes an otherwise valid search fail before reaching a store.
    if (auto *II = dyn_cast<IntrinsicInst>(&CB)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end)
        return;
    }

    if (!isSelectedClobber(CB))
      return;

    Function *CalledFunction = CB.getCalledFunction();
    if (!CalledFunction || CalledFunction->isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Cannot trace indirect or declaration "
               "call "
            << CB << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    Value *DefBeforeCB = getLastDef();
    if (!DefBeforeCB || !ValueOffsetMap.contains(DefBeforeCB)) {
      offsetValueMapFailure(DefBeforeCB ? DefBeforeCB : TrackedBase);
      return;
    }

    const auto ReturnInstVec = proteus::getReturnInstructions(*CalledFunction);
    if (ReturnInstVec.size() != 1) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Call " << CB
            << "\nContains multiple ReturnInsts" << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    auto *Return = ReturnInstVec.front();
    bool FoundArg = false;
    for (size_t ArgI = 0; ArgI < CalledFunction->arg_size(); ++ArgI) {
      DEBUG(Logger::logs("proteus-pass") << "    ARG " << ArgI << " VAL "
                                         << *CB.getArgOperand(ArgI) << "\n");
      if (CB.getArgOperand(ArgI) != DefBeforeCB)
        continue;

      FoundArg = true;
      Argument *ArgToTrack = CalledFunction->getArg(ArgI);
      ValueOffsetMap[ArgToTrack] = ValueOffsetMap[DefBeforeCB];
      DEBUG(Logger::logs("proteus-pass")
            << "    Looking at uses of " << *ArgToTrack << "\n");

      for (User *Usr : ArgToTrack->users())
        pushBack(Usr, ArgToTrack, CalledFunction, Return);
    }

    if (!FoundArg) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Call does not pass the tracked "
               "pointer on any callee argument: "
            << CB << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
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
    pushBackPointerUsers(&GEP);
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

    pushBackPointerUsers(&Alloca);
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
    pushBackPointerUsers(&BC);
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

    pushBackPointerUsers(&ASC);
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    // Reaching an intrinsic through its source is a read, not a definition of
    // the tracked memory.
    if (Def != I.getRawDest())
      return;

    // As with stores, the def-use walk can encounter memory writes that do not
    // reach the use boundary.  Interpret only MemorySSA's selected definition.
    if (!isSelectedClobber(I))
      return;

    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      if (!ValueOffsetMap.contains(Def)) {
        offsetValueMapFailure(Def);
        return;
      }
      auto *Len = dyn_cast<ConstantInt>(MS->getLength());
      if (Len &&
          !offsetCoveredByRange(ValueOffsetMap[Def], 0, Len->getZExtValue()))
        return;
      // A covering memset destroys the tracked provenance, and a dynamic
      // length cannot be proven not to cover it.
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Memset may overwrite the tracked "
               "byte: "
            << I << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    auto *MT = cast<MemTransferInst>(&I); // memcpy/memmove

    if (!ValueOffsetMap.contains(Def)) {
      offsetValueMapFailure(Def);
      return;
    }

    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    if (!Len) {
      // We cannot prove that a dynamic-sized transfer defines the tracked byte.
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Dynamic-length transfer cannot prove "
               "provenance for the tracked byte: "
            << I << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    if (!offsetCoveredByRange(ValueOffsetMap[Def], 0, Len->getZExtValue()))
      return;

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase) {
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: Failure due to nullptr dst/src " << "\n");
      AnalysisFailed = true;
      return;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "  [PTR use analysis]: Completed instrinsic analysis " << "\n");
    // LambdaArgVisitor applies Result.Offset by subtracting it from its
    // current, destination-relative offset.  The value carried across a
    // transfer is therefore the difference between the destination and
    // source bases, rather than the source's absolute offset.  For example,
    // copying a field at byte 8 into a field at byte 24 needs a correction of
    // 16, so the caller turns 24 into 8.
    int64_t OffsetCorrection = DstOff - SrcOff;
    AnalysisSuccess = true;
    AnalysisFailed = false;
    Result = {.DominatingWrite = SrcBase,
              .Offset = OffsetCorrection,
              .ChangedRCLayout = std::nullopt};
  }

  void visitInstruction(Instruction &I) {
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Unhandled instruction "
          << I.getOpcodeName() << ": " << I << "\n");
    AnalysisFailed = true;
    AnalysisSuccess = false;
  }
};

inline std::optional<LambdaPtrUseAnalysis>
getDominatingUse(const DataLayout &DL, Value *ValueNeedingAnalysis,
                 Instruction *UseofPtr, int64_t TargetOffset,
                 CallBase *LambdaCB,
                 std::shared_ptr<MemorySSAClobberOracle> ClobberOracle) {
  DEBUG(Logger::logs("proteus-pass")
        << "Beginning PtrUse analysis with offset = " << TargetOffset << "\n");

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, UseofPtr, LambdaCB, DL,
                               TargetOffset, std::move(ClobberOracle));
  // Analysis loop
  while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
    auto CurWorkItem = Visitor.popBack();
    auto *V = CurWorkItem.CurVal;
    // Prevent loops/infinite recursion
    if (Visitor.seen(V))
      continue;
    Visitor.markAsSeen(V);

    // Analyze the instruction
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (auto *LastI = dyn_cast_or_null<Instruction>(CurWorkItem.LastVal);
          LastI && LastI->getFunction() != I->getFunction())
        DEBUG(Logger::logs("proteus-pass")
              << "    Crossing interprocedural boundary from  "
              << LastI->getFunction()->getName() << " ----> "
              << I->getFunction()->getName() << "\n"
              << "At value " << *I << "\n");
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: Visiting ptr use " << *V << "\n");
      Visitor.visit(*I);
    } else
      continue;
  }
  if (!Visitor.success() || Visitor.failed()) {
    DEBUG(
        Logger::logs("proteus-pass")
        << "  [PTR use analysis] [WARNING]: Dominating use analysis FAILED for "
        << *ValueNeedingAnalysis << " <-- " << *UseofPtr << "\n");
    return std::nullopt;
  }
  LambdaPtrUseAnalysis Info = Visitor.getAnalysisResult();
  if (!Info.DominatingWrite)
    return std::nullopt;
  DEBUG(Logger::logs("proteus-pass")
        << "  [PTR USE ANALYSIS]: Computed offset " << Info.Offset << "\n");
  return Info;
}
} // namespace proteus

#endif
