#ifndef PROTEUS_KERNELARGPTRDEFVISITOR_H
#define PROTEUS_KERNELARGPTRDEFVISITOR_H

#include "Helpers.h"
#include "PointerClobberAnalysis.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>
#include <algorithm>

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
  // This is the current function the analysis in examining.
  Function *Frame = nullptr;
  // This is the ptr argument, alloca, bitcast, addrspacecast defined ptr that triggered the
  // analysis. E.G. %1 in the example below
  // %1 = alloca ptr
  // %2 = getelementptr %1, 0, 1
  // store %0, %2
  // call lambda(%1)
  Value *TrackedPtr = nullptr;
  // This is the instruction at which we care about the value of the TrackedPtr.  Above,
  // it would be call lambda(%1)
  Instruction *UseBoundary = nullptr;
  int64_t OffsetOfWriteToBasePtr = 0;
  uint64_t SizeOfWriteToBasePtr = 0;

  bool operator==(const ClobberQuery &Other) const {
    return Frame == Other.Frame && TrackedPtr == Other.TrackedPtr &&
           UseBoundary == Other.UseBoundary &&
           OffsetOfWriteToBasePtr == Other.OffsetOfWriteToBasePtr;
  }
};

// struct CallBaseCandidate {
//   CallBase *CB;
//   int64_t ArgContainingPtr;
// };

// struct MemIntrinsicCandidate {
//   MemIntrinsic *MI;
//   Value *SrcPtr;

// };

// struct StoreCandidate {
//   StoreInst *SI;

// };

// Whenever the analysis encounters a CallBase, MemIntrinsic, or Store
// one of these structs in generated, combining the context of the ClobberQuery
// with the Instruction itself.
struct UseDefWriteCandidate {
  // std::variant<StoreCandidate, MemIntrinsicCandidate, CallBaseCandidate> WriteCandidate;
  Instruction *WriteCandidate;
  ClobberQuery ClobberInfo;
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
        Query.Frame, Query.TrackedPtr, Query.UseBoundary, Query.OffsetOfWriteToBasePtr));
  }

  static bool isEqual(const ClobberQuery &LHS, const ClobberQuery &RHS) {
    return LHS == RHS;
  }
};

// The basic unit of work on this analysis is a use--an edge connecting two
// values. We also track their parent function and the useboundary pertaining
// to them.  This use boundary could be an alloca ptr that triggered this analysis
// or it could be a ReturnInst in a called function.
struct UseAnalysisWorkItem {
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
  Value *TrackedBase = nullptr;
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
  SmallVector<UseAnalysisWorkItem> WorkList;
  DenseMap<CallBase *, SmallVector<UseAnalysisWorkItem, 4>>
      DeferredCallBaseWorkItems;
  UseAnalysisWorkItem Current;
  SmallVector<UseDefWriteCandidate> PotentialClobbers;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  SmallDenseSet<Value *> Seen;

  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

public:
// Data about offset comes from the use-def graph traversal, but information about clobbering
// comes from the MemorySSA IR.  This phase of the analysis attempts to reconcile the results
// of the two
inline Value *resolveCollectedClobbers() {
  if (PotentialClobbers.empty()) {
    AnalysisFailed = true;
    AnalysisSuccess = false;
    DEBUG(Logger::logs("proteus-pass") << "Analysis failed to produce any write candidates\n");
    return nullptr;
  }
  MemorySSAClobber SSAResult;
  if (PotentialClobbers.size() == 1) {
    auto &Candidate = PotentialClobbers.front();
    if (auto *SI = dyn_cast<StoreInst>(Candidate.WriteCandidate))
      SSAResult = ClobberOracle->query(MemoryLocation::get(SI),
                                       *Candidate.ClobberInfo.UseBoundary);
    else
      SSAResult = queryMemorySSA(Candidate.ClobberInfo);
  } else {
    SSAResult = queryMemorySSA(PotentialClobbers.begin()->ClobberInfo);
  }
  auto *ClobberingInst = SSAResult.Instruction;
  if (!ClobberingInst) {
    DEBUG(Logger::logs("proteus-pass")
          << "MemorySSA did not identify a clobbering instruction\n");
    AnalysisFailed = true;
    AnalysisSuccess = false;
    return nullptr;
  }
  if (PotentialClobbers.size() == 1) {
    // check if the clobber is identical to the write from the analysis, if not
    // we are in trouble
    if (PotentialClobbers[0].WriteCandidate != ClobberingInst) {
      DEBUG(Logger::logs("proteus-pass") << "Analysis found write at " << *PotentialClobbers[0].WriteCandidate <<
    "\nBut MemorySSA found " << *ClobberingInst);
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return nullptr;
    }
    return PotentialClobbers[0].WriteCandidate;
  }
  // We check that all the potential clobbers share a TrackedPtr, Frame, and UseBoundary.
  // This shape is baked into the analysis, because MemorySSA queries must occur within
  // the same procedure. We do a sanity check here and fail if it doesn't hold true (internal
  // compiler error).
  bool AllTrackedPointersSame = std::all_of(PotentialClobbers.begin(), PotentialClobbers.end(), [&](const UseDefWriteCandidate& Q) {
    return Q.ClobberInfo.TrackedPtr == PotentialClobbers[0].ClobberInfo.TrackedPtr
      && Q.ClobberInfo.Frame == PotentialClobbers[0].ClobberInfo.Frame
      && Q.ClobberInfo.UseBoundary == PotentialClobbers[0].ClobberInfo.UseBoundary;
  });

  if (!AllTrackedPointersSame) {
    clobberCandidatesFailure();
    return nullptr;
  }
  auto *ClobberingStoreOrNull = dyn_cast<StoreInst>(ClobberingInst);

  bool AreAllStores = PotentialClobbers.size() > 1 && ClobberingInst && ClobberingStoreOrNull &&
    std::all_of(PotentialClobbers.begin(), PotentialClobbers.end(), [&](const UseDefWriteCandidate& Q) {
      return dyn_cast<StoreInst>(Q.WriteCandidate);
    });
  if (!AreAllStores) {
    DEBUG(Logger::logs("proteus-pass") << "Detected mixed StoreInst/CallBase/MemIntrinsics in list of candidate clobbers:\n");
    int CandidateIdx = 0;
    for (auto& Candidate : PotentialClobbers) {
      ++CandidateIdx;
      DEBUG(Logger::logs("proteus-pass") << "Candidate " << CandidateIdx << " = " << *Candidate.WriteCandidate);
    }

    AnalysisFailed = true;
    AnalysisSuccess = false;
    return nullptr;
  }

  // Multiple stores to one exact destination are successive definitions of
  // the same bytes, not aggregate construction from heterogeneous sources.
  // MemorySSA has already selected the definition reaching UseBoundary, so
  // discard the shadowed stores before comparing source provenance.
  int64_t ClobberingDestOffset = 0;
  Value *ClobberingDestBase = GetPointerBaseWithConstantOffset(
      ClobberingStoreOrNull->getPointerOperand(), ClobberingDestOffset, DL);
  DEBUG(Logger::logs("proteus-pass") << "Computed base ptr " << *ClobberingDestBase << " and offset " << ClobberingDestOffset << " for instruction " << *ClobberingStoreOrNull <<"\n");
  bool StoresHaveSameDestination = ClobberingDestBase &&
      std::all_of(PotentialClobbers.begin(), PotentialClobbers.end(),
                  [&](const UseDefWriteCandidate &Candidate) {
                    // Note we have already exited if the cast below would be null
                    auto *SI = cast<StoreInst>(Candidate.WriteCandidate);
                    int64_t DestOffset = 0;
                    Value *DestBase = GetPointerBaseWithConstantOffset(
                        SI->getPointerOperand(), DestOffset, DL);
                    DEBUG(Logger::logs("proteus-pass") << "Computed base ptr " << *DestBase << " and offset " << DestOffset << " for instruction " << *SI <<"\n");
                    return DestBase == ClobberingDestBase &&
                           DestOffset == ClobberingDestOffset;
                  });
  if (StoresHaveSameDestination)
    return ClobberingStoreOrNull;

  // clang-format off
  // four possible cases for stores to the aggregate:
  // Stores = all stores INCLUSIVE of the clobbering store
  // all stores have same base ptr, distinct offsets:
  //     return the base ptr with 0 offset from analysis
  // one store has different base ptr, distinct offsets:
  //     means one slot in the lambda was written to from someplace else,
  //     lambda is filled from heterogenous sources, fail
  // all stores have same base ptr, share offsets:
  //     the clobber should be the definitive answer for this case,
  //     analysis should return that
  // one store has different base ptr, some offsets shared:
  //     fail
  // clang-format on
  int64_t ClobberingOffset = 0;
  auto *ClobberingBasePtr = GetPointerBaseWithConstantOffset(ClobberingStoreOrNull->getValueOperand(), ClobberingOffset, DL);
  SmallDenseMap<int64_t, SmallVector<std::pair<Value*, StoreInst*>>> OffsetToBasePtrStoreMap;
  SmallDenseMap<StoreInst*, SmallVector<std::pair<Value*, int64_t>>> StoreInstToOffsetBasePtrMap;
  OffsetToBasePtrStoreMap[ClobberingOffset].push_back({ClobberingBasePtr, ClobberingStoreOrNull});
  // These two booleans create the axes of cases.
  bool StoresHaveSameBasePtr = std::all_of(PotentialClobbers.begin(), PotentialClobbers.end(), [&](const UseDefWriteCandidate& Q) {
    auto *SI = dyn_cast<StoreInst>(Q.WriteCandidate);
    auto* ValueOp = SI->getValueOperand();
    int64_t ValueOffset = 0;
    auto *SIValueBasePtr = GetPointerBaseWithConstantOffset(ValueOp, ValueOffset, DL);
    OffsetToBasePtrStoreMap[ValueOffset].push_back({SIValueBasePtr, SI});
    StoreInstToOffsetBasePtrMap[SI].push_back({SIValueBasePtr, ValueOffset});
    return SIValueBasePtr == ClobberingBasePtr;
  });

  bool StoresHaveDifferentOffsets = std::all_of(OffsetToBasePtrStoreMap.begin(), OffsetToBasePtrStoreMap.end(), [&](const auto &It) {
    return It.second.size() ==1;
  });
  // This is the easiest possible case: basically we've found the LLVM IR shape where we load a bunch
  // of values in to the aggregate pointer representing lambda storage. The Clobbering analysis just
  // picks the last write to an element in the aggregate lambda storage.
  if (StoresHaveDifferentOffsets && StoresHaveSameBasePtr) {
    return ClobberingStoreOrNull;
  }
  // We can still conservatively pass in this case: if there are two writes to the same field in the lambda
  // but we know one of them clobbers, we can still return the common base ptr as the source of truth for
  // the lambda.
  if (!StoresHaveDifferentOffsets && StoresHaveSameBasePtr) {
    for (const auto &Entry : OffsetToBasePtrStoreMap) {
      const auto &CurVec = Entry.second;
      if (CurVec.size() == 1)
        continue;
      auto It = std::find_if(CurVec.begin(), CurVec.end(), [&](std::pair<Value*, StoreInst*> Arg) {
        return Arg.second == ClobberingStoreOrNull;
      });
      if (It != CurVec.end())
        return ClobberingStoreOrNull;
    }
  }
  // What we can't have is a data occupying slots in the lambda pointer coming in from miscellaneous sources.
  if (!StoresHaveSameBasePtr) {
    DEBUG(Logger::logs("proteus-pass") << "Ptr use analysis error: found heterogenous data sources in lambda storage slots. This is illegal and should be detected by the frontend. \n");
    for (const auto& Clobber : PotentialClobbers) {
      DEBUG(Logger::logs("proteus-pass") << "Candidate at offset = " << Clobber.ClobberInfo.OffsetOfWriteToBasePtr << " : " << *Clobber.WriteCandidate);
    }
  }


// ; Function Attrs: convergent mustprogress norecurse nounwind
// define protected amdgpu_kernel void @_ZN17MockRajaInterface13globalWrapperIZN17MockMfemInterface6forallIN7proteus6detail20LambdaFunctorWrapperILm6894195003319168199EZ14integralKernelILi3ELi4EEvdiiEUliiE_EEEEviiOT_EUliE_EEvS9_m(ptr addrspace(4) noundef byref(%class.anon.1) align 8 %0, i64 noundef %1) #3 comdat !dbg !1989 !proteus.jit !1805 {
//   %3 = alloca %class.anon.1, align 8, addrspace(5)
//   %4 = alloca %"struct.MockRajaInterface::Privatizer.4", align 8, addrspace(5)
//   %5 = addrspacecast ptr addrspace(5) %3 to ptr
//   %6 = addrspacecast ptr addrspace(5) %4 to ptr
//   call void @llvm.memcpy.p0.p4.i64(ptr align 8 %5, ptr addrspace(4) align 8 %0, i64 24, i1 false)
//     #dbg_declare(ptr addrspace(5) %3, !1993, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(%class.anon.1)), !1999)
//     #dbg_value(i64 %1, !1994, !DIExpression(DIOpArg(0, i64)), !2000)
//   call void @llvm.lifetime.start.p5(i64 24, ptr addrspace(5) %4) #14, !dbg !2001
//     #dbg_declare(ptr addrspace(5) %4, !1995, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(%"struct.MockRajaInterface::Privatizer.4")), !2002)
//   %7 = call %"struct.MockRajaInterface::Privatizer.4" @_ZN17MockRajaInterface15threadPrivatizeIZN17MockMfemInterface6forallIN7proteus6detail20LambdaFunctorWrapperILm6894195003319168199EZ14integralKernelILi3ELi4EEvdiiEUliiE_EEEEviiOT_EUliE_EENS_10PrivatizerIS9_EERKS9_(ptr noundef nonnull align 8 dereferenceable(24) %5) #16, !dbg !2003
//   %8 = extractvalue %"struct.MockRajaInterface::Privatizer.4" %7, 0, !dbg !2003
//   %9 = extractvalue %class.anon.1 %8, 0, !dbg !2003
//   store i32 %9, ptr %6, align 8, !dbg !2003
//   %10 = extractvalue %class.anon.1 %8, 1, 0, 0, !dbg !2003
//   %11 = getelementptr inbounds %class.anon.1, ptr %6, i32 0, i32 1, i32 0, i32 0, !dbg !2003
//   store i32 %10, ptr %11, align 8, !dbg !2003
//   %12 = extractvalue %class.anon.1 %8, 1, 0, 1, !dbg !2003
//   %13 = getelementptr inbounds %class.anon.1, ptr %6, i32 0, i32 1, i32 0, i32 1, !dbg !2003
//   store i32 %12, ptr %13, align 4, !dbg !2003
//   %14 = extractvalue %class.anon.1 %8, 1, 0, 2, !dbg !2003
//   %15 = getelementptr inbounds %class.anon.1, ptr %6, i32 0, i32 1, i32 0, i32 2, !dbg !2003
//   store double %14, ptr %15, align 8, !dbg !2003
    // todo: In the shape above, we have multiple stores to the same closureptr.
    // Problem: SSA selects the store double %14, ptr %15, align 8, but all are valid
    // In this case, its easy to check the StoreInst's valueptr's GetPointerBaseWithConstantOffset
    // And verify we are just copying values from one aggregate into another.
    // If two candidates have a different base ptr but same offset into TrackedPtr, and one clobbers the other,
    // just trust the one that clobbers.
    //
    // In general: all candidates will be StoreInst, CallBase, or MemIntrinsic
    // We might have TrackedPtr = %6
    // %10 = getelementptr inbounds %class.anon.1, ptr %6, i32 0, i32 0, i32 0, i32 0
    // %11 = getelementptr inbounds %class.anon.1, ptr %6, i32 0, i32 1, i32 0, i32 0
    // store int32 %9, ptr %10 align 8
    // call void writeslot(%11)
    // Here we need to check that the CB is writing from the same fundamental ptr as the store but across
    // the interprocedural boundary, or at least reject this shape
  return nullptr;
}
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
    WorkList.push_back({PtrBegin, nullptr, SeenUse, SeenUse->getFunction(), PtrBegin});
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
                Instruction *UseBoundary, Value *TrackedBase) {
    WorkList.push_back({NextVal, CurVal, UseBoundary, Frame, TrackedBase});
  }

  bool cacheCallBaseWorkItems(CallBase &CB, Value *DefBeforeCB) {
    Function *CalledFunction = CB.getCalledFunction();
    if (!CalledFunction || CalledFunction->isDeclaration()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Cannot trace indirect or declaration "
               "call "
            << CB << "\n");
      return false;
    }

    const auto ReturnInstVec = proteus::getReturnInstructions(*CalledFunction);
    if (ReturnInstVec.size() != 1) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Call " << CB
            << "\nContains multiple ReturnInsts\n");
      return false;
    }

    if (!DefBeforeCB || !ValueOffsetMap.contains(DefBeforeCB)) {
      offsetValueMapFailure(DefBeforeCB ? DefBeforeCB : TrackedBase);
      return false;
    }

    auto &Deferred = DeferredCallBaseWorkItems[&CB];
    bool FoundArg = false;
    for (size_t ArgI = 0; ArgI < CalledFunction->arg_size(); ++ArgI) {
      if (CB.getArgOperand(ArgI) != DefBeforeCB)
        continue;

      FoundArg = true;
      Argument *ArgToTrack = CalledFunction->getArg(ArgI);
      ValueOffsetMap[ArgToTrack] = ValueOffsetMap[DefBeforeCB];
      for (User *Usr : ArgToTrack->users())
        Deferred.push_back(
            {Usr, ArgToTrack, ReturnInstVec.front(), CalledFunction, ArgToTrack});
    }

    if (!FoundArg)
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Call does not pass the tracked pointer "
               "on any callee argument: "
            << CB << "\n");
    return FoundArg;
  }

  void continueFromResolvedClobber(Value *Resolved) {
    DEBUG({
      auto &OS = Logger::logs("proteus-pass");
      OS << "    [PTR use analysis]: Resolving collected clobbers; candidates="
         << PotentialClobbers.size();
      if (Resolved)
        OS << "; selected=" << *Resolved;
      else
        OS << "; selected=<none>";
      OS << "\n";
    });

    auto *ResolvedInst = dyn_cast_or_null<Instruction>(Resolved);
    if (!ResolvedInst) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Resolution phase failed: selected "
               "clobber is not an instruction\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    auto CandidateIt = llvm::find_if(PotentialClobbers,
                                     [ResolvedInst](const auto &Candidate) {
                                       return Candidate.WriteCandidate ==
                                              ResolvedInst;
                                     });
    if (CandidateIt == PotentialClobbers.end()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Resolution phase failed: MemorySSA "
               "selection is absent from collected candidates: "
            << *ResolvedInst << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    if (auto *SI = dyn_cast<StoreInst>(ResolvedInst)) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Exiting visitor with value from "
               "resolved store: "
            << *SI << "; offset="
            << CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr << "\n");
      Result = {.DominatingWrite = SI->getValueOperand(),
                .Offset = CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr,
                .ChangedRCLayout = std::nullopt};
      AnalysisFailed = false;
      AnalysisSuccess = true;
      return;
    }

    if (auto *MT = dyn_cast<MemTransferInst>(ResolvedInst)) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Exiting visitor with source from "
               "resolved memory transfer: "
            << *MT << "; offset-correction="
            << CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr << "\n");
      Result = {.DominatingWrite = MT->getRawSource(),
                .Offset = CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr,
                .ChangedRCLayout = std::nullopt};
      AnalysisFailed = false;
      AnalysisSuccess = true;
      return;
    }

    auto *CB = dyn_cast<CallBase>(ResolvedInst);
    auto Deferred = CB ? DeferredCallBaseWorkItems.find(CB)
                       : DeferredCallBaseWorkItems.end();
    if (!CB || Deferred == DeferredCallBaseWorkItems.end() ||
        Deferred->second.empty()) {
      DEBUG({
        auto &OS = Logger::logs("proteus-pass");
        OS << "    [PTR use analysis]: Resolution phase failed: selected "
              "clobber cannot resume interprocedural traversal: "
           << *ResolvedInst;
        if (CB)
          OS << "; cached-work-items="
             << (Deferred == DeferredCallBaseWorkItems.end()
                     ? 0
                     : Deferred->second.size());
        OS << "\n";
      });
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Resuming visitor across selected call "
             "into "
          << CB->getCalledFunction()->getName()
          << "; deferred-work-items=" << Deferred->second.size() << "\n");
    PotentialClobbers.clear();
    WorkList.append(Deferred->second.begin(), Deferred->second.end());
    DeferredCallBaseWorkItems.erase(Deferred);
  }

  void resolveCollectedClobbersAndContinue() {
    continueFromResolvedClobber(resolveCollectedClobbers());
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

  bool isSelectedClobber(Instruction &Candidate, Function *Frame,
                         Instruction *UseBoundary) {
    ClobberQuery Query{Frame, TrackedBase, UseBoundary};
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

  void clobberCandidatesFailure() {
    AnalysisFailed = true;
    AnalysisSuccess = false;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to frame/tracked ptr mismatch"
          << " in Clobber candidates map\n");
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
      pushBack(Usr, V, Current.Frame, Current.UseBoundary, Current.TrackedBase);
    }
    for (User *Usr : PossibleWriters)
      pushBack(Usr, V, Current.Frame, Current.UseBoundary, Current.TrackedBase);
  }

  void visitStoreInst(StoreInst &SI) {
    Value *Stored = SI.getValueOperand();
    Value *StoreBase = SI.getPointerOperand();
    // This analysis tracks which write to the original pointer triggering this result,
    // so a use that reads from the prior value is not valid here.
    if (Stored == Current.LastVal)
      return;
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
          pushBack(Usr, StoreBase, Current.Frame, Current.UseBoundary,
                   Current.TrackedBase);
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
    // if (!isSelectedClobber(SI, Current.Frame, Current.UseBoundary))
    //   return;

    if (!StoreSize ||
        !offsetCoveredByRange(ValueOffsetMap[StoreBase], 0, *StoreSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to offset " << ValueOffsetMap[&SI]
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    PotentialClobbers.push_back(UseDefWriteCandidate {
      .WriteCandidate = &SI,
      .ClobberInfo = {.Frame = Current.Frame,
                      .TrackedPtr = Current.TrackedBase,
                      .UseBoundary = Current.UseBoundary,
                      .OffsetOfWriteToBasePtr = Offset,
                      },
    });
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Skipping tracking non-pointer load " << LI
            << "\n");
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

    Value *DefBeforeCB = getLastDef();
    if (!cacheCallBaseWorkItems(CB, DefBeforeCB)) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    PotentialClobbers.push_back(
        {.WriteCandidate = &CB,
         .ClobberInfo = {.Frame = Current.Frame,
                         .TrackedPtr = Current.TrackedBase,
                         .UseBoundary = Current.UseBoundary,
                         .OffsetOfWriteToBasePtr = Offset}});
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
          << "  [PTR use analysis]: Completed intrinsic analysis " << "\n");
    // LambdaArgVisitor applies Result.Offset by subtracting it from its
    // current, destination-relative offset.  The value carried across a
    // transfer is therefore the difference between the destination and
  // source bases, rather than the source's absolute offset.  For example,
    // copying a field at byte 8 into a field at byte 24 needs a correction of
    // 16, so the caller turns 24 into 8.
    int64_t OffsetCorrection = DstOff - SrcOff;
    PotentialClobbers.push_back({
    .WriteCandidate = &I,
    .ClobberInfo = {.Frame = Current.Frame,
                   .TrackedPtr = Current.TrackedBase,
                   .UseBoundary = Current.UseBoundary,
                   .OffsetOfWriteToBasePtr = OffsetCorrection,
                   .SizeOfWriteToBasePtr = Len->getZExtValue()
                  }});
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
  while (!Visitor.success() && !Visitor.failed()) {
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
      }
    }

    if (!Visitor.success() && !Visitor.failed())
      Visitor.resolveCollectedClobbersAndContinue();
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
