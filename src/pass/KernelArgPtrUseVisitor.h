#ifndef PROTEUS_KERNELARGPTRDEFVISITOR_H
#define PROTEUS_KERNELARGPTRDEFVISITOR_H

#include "Helpers.h"
#include "PointerClobberAnalysis.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <algorithm>
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>

#include <limits>
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

inline bool offsetCoveredByRange(int64_t TargetOffset, int64_t RangeOffset,
                                 uint64_t RangeSize) {
  return TargetOffset >= RangeOffset &&
         static_cast<uint64_t>(TargetOffset - RangeOffset) < RangeSize;
}

inline constexpr unsigned PointerIntervalBitWidth = 128;

inline bool rangesOverlap(int64_t FirstOffset, uint64_t FirstSize,
                          int64_t SecondOffset, uint64_t SecondSize) {
  if (FirstSize == 0 || SecondSize == 0)
    return false;
  APInt FirstBegin(PointerIntervalBitWidth, FirstOffset, true);
  APInt FirstEnd = FirstBegin + FirstSize;
  APInt SecondBegin(PointerIntervalBitWidth, SecondOffset, true);
  APInt SecondEnd = SecondBegin + SecondSize;
  return FirstBegin.slt(SecondEnd) && SecondBegin.slt(FirstEnd);
}

inline std::optional<uint64_t> getTypeStoreSize(const DataLayout &DL,
                                                Type *Ty) {
  if (!Ty || !Ty->isSized())
    return std::nullopt;
  return static_cast<uint64_t>(DL.getTypeStoreSize(Ty));
}

struct LambdaPtrUseAnalysis {
  Value *DominatingWrite = nullptr;
  // The instruction that directly consumes DominatingWrite. Keeping this a
  // real use-def edge gives LambdaArgVisitor a valid boundary for its next
  // MemorySSA query.
  Instruction *WriteBoundary = nullptr;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

// The tracked entity begins Offset bytes from Value and occupies Extent bytes.
// Keeping these together prevents pointer transforms from accidentally using a
// range belonging to a different level of indirection.
struct TrackedRange {
  int64_t Offset = 0;
  uint64_t Extent = 0;

  bool operator==(const TrackedRange &Other) const {
    return Offset == Other.Offset && Extent == Other.Extent;
  }

  bool operator!=(const TrackedRange &Other) const { return !(*this == Other); }
};

inline raw_ostream &operator<<(raw_ostream &OS, const TrackedRange &Range) {
  return OS << "{offset=" << Range.Offset << ", extent=" << Range.Extent << "}";
}

struct ClobberQuery {
  // This is the current function the analysis in examining.
  Function *Frame = nullptr;
  // This is the ptr argument, alloca, bitcast, addrspacecast defined ptr that
  // triggered the analysis. E.G. %1 in the example below %1 = alloca ptr %2 =
  // getelementptr %1, 0, 1 store %0, %2 call lambda(%1)
  Value *TrackedPtr = nullptr;
  // This is the instruction at which we care about the value of the TrackedPtr.
  // Above, it would be call lambda(%1)
  Instruction *UseBoundary = nullptr;
  // Per-candidate byte correction consumed by LambdaArgVisitor
  // (Offset -= Result.Offset); it is NOT part of the query identity.
  int64_t OffsetOfWriteToBasePtr = 0;
  // The exact byte range this query resolves, relative to TrackedPtr. This is
  // part of the query identity: two questions at the same pointer and boundary
  // but with different extents can have different reaching writes.
  TrackedRange Range;

  bool operator==(const ClobberQuery &Other) const {
    return Frame == Other.Frame && TrackedPtr == Other.TrackedPtr &&
           UseBoundary == Other.UseBoundary && Range == Other.Range;
  }
};

// Whenever the analysis encounters a CallBase, MemIntrinsic, or Store
// one of these structs in generated, combining the context of the ClobberQuery
// with the Instruction itself.
struct UseDefWriteCandidate {
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
    return static_cast<unsigned>(hash_combine(Query.Frame, Query.TrackedPtr,
                                              Query.UseBoundary,
                                              Query.Range.Offset,
                                              Query.Range.Extent));
  }

  static bool isEqual(const ClobberQuery &LHS, const ClobberQuery &RHS) {
    return LHS == RHS;
  }
};

// A pointer-returning helper is traversed by the same use visitor as its
// caller.  This records where that traversal should resume when it reaches the
// helper's ReturnInst.  Chaining continuations handles nested helpers without
// introducing a second provenance walk.
struct ReturnContinuation {
  CallBase *Call = nullptr;
  Function *CallerFrame = nullptr;
  Instruction *CallerUseBoundary = nullptr;
  Value *CallerTrackedBase = nullptr;
  std::shared_ptr<ReturnContinuation> Parent;
  unsigned PendingWorkItems = 0;
  bool SawTrackedWrite = false;
  bool Completed = false;
  std::optional<TrackedRange> ReturnedRange;
};

// The basic unit of work on this analysis is a use--an edge connecting two
// values. We also track their parent function and the useboundary pertaining
// to them.  This use boundary could be an alloca ptr that triggered this
// analysis or it could be a ReturnInst in a called function.
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
  std::shared_ptr<ReturnContinuation> ReturnTo;
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
  // ValueRangeMap contains the byte range of the entity relative to each
  // pointer value we're analyzing.
  // For example, let's say that LambdaInstUseVisitor is handed ptr %0 = alloca
  // ptr, and LambdaArgVisitor has already identified that the closure starts at
  // byte
  // 8. In this case, ValueRangeMap[%0] begins at byte 8. If we encounter a
  // store like
  // store ptr %2, ptr%0, align 8, we don't care, because its written outside
  // of the range of the closure.
  DenseMap<Value *, TrackedRange> ValueRangeMap;
  DenseMap<ClobberQuery, MemorySSAClobber, ClobberQueryDenseMapInfo> QueryCache;
  std::shared_ptr<MemorySSAClobberOracle> ClobberOracle;
  Value *TrackedBase = nullptr;
  // A pointer load changes the coordinate system from the pointer's storage to
  // the entity reached through the loaded pointer. LambdaArgVisitor supplies
  // that post-load range when those two extents differ.
  std::optional<TrackedRange> LoadedPointerRange;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  SmallVector<UseAnalysisWorkItem> WorkList;
  DenseMap<CallBase *, SmallVector<UseAnalysisWorkItem, 4>>
      DeferredCallBaseWorkItems;
  UseAnalysisWorkItem Current;
  SmallVector<UseDefWriteCandidate> PotentialClobbers;
  SmallVector<SmallVector<UseDefWriteCandidate, 4>, 2>
      ParentClobberCandidateStack;
  SmallPtrSet<CallBase *, 4> ProvenReturnedAliasCalls;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  DenseSet<std::pair<Value *, const ReturnContinuation *>> Seen;
  SmallDenseSet<Value *> NeverVisit;
  // Owns every ReturnContinuation created by this visitor so the raw
  // addresses used as Seen keys remain valid and unique for its lifetime.
  SmallVector<std::shared_ptr<ReturnContinuation>, 8> OwnedContinuations;
  // A pointer-valued load is an alias edge only when we have seen a store of
  // the tracked pointer into local storage and MemorySSA proves that store
  // reaches the load. Pointer fields loaded from a closure are captured data,
  // not aliases of the closure itself.
  SmallPtrSet<LoadInst *, 8> ProvenPointerReloads;

  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;
  uint64_t ProcessedWorkItems = 0;
  static constexpr uint64_t MaxWorkItems = 1 << 17;

  bool storeCoversTrackedRange(const StoreInst &Store) const {
    auto StoreSize = getTypeStoreSize(DL, Store.getValueOperand()->getType());
    auto StoredRange = ValueRangeMap.find(Store.getPointerOperand());
    if (!StoreSize || StoredRange == ValueRangeMap.end() ||
        StoredRange->second.Extent == 0)
      return false;

    APInt StoreBegin(PointerIntervalBitWidth, 0, true);
    APInt StoreEnd = StoreBegin + *StoreSize;
    APInt RangeBegin(PointerIntervalBitWidth, StoredRange->second.Offset, true);
    APInt RangeEnd = RangeBegin + StoredRange->second.Extent;
    return StoreBegin.sle(RangeBegin) && StoreEnd.sge(RangeEnd);
  }

public:
  // Data about offset comes from the use-def graph traversal, but information
  // about clobbering comes from the MemorySSA IR.  This phase of the analysis
  // attempts to reconcile the results of the two
  inline Value *resolveCollectedClobbers() {
    if (PotentialClobbers.empty()) {
      fail();
      DEBUG(Logger::logs("proteus-pass")
            << "Analysis failed to produce any write candidates\n");
      return nullptr;
    }
    DEBUG(Logger::logs("proteus-pass") << "Resolving collected clobbers\n");
    for (const auto& Candidate : PotentialClobbers)
      DEBUG(Logger::logs("proteus-pass") << "CANDIDATE = " << *Candidate.WriteCandidate <<"\n");
    // This is mostly a sanity check.
    // All candidates must share a Frame, TrackedPtr, and UseBoundary: the
    // reconciliation below asks one function-local question -- which write
    // reaches the tracked byte at the boundary. This shape is baked into the
    // analysis; a mismatch is an internal error and fails conservatively.
    const ClobberQuery &CommonInfo = PotentialClobbers.front().ClobberInfo;
    bool AllQueriesSame =
        llvm::all_of(PotentialClobbers, [&](const UseDefWriteCandidate &Q) {
          return Q.ClobberInfo.TrackedPtr == CommonInfo.TrackedPtr &&
                 Q.ClobberInfo.Frame == CommonInfo.Frame &&
                 Q.ClobberInfo.UseBoundary == CommonInfo.UseBoundary &&
                 Q.ClobberInfo.Range == CommonInfo.Range;
        });
    if (!AllQueriesSame) {
      // If we get here this is an internal compiler error.
      clobberCandidatesFailure();
      return nullptr;
    }

    // Anchor the reconciliation query at the tracked byte itself whenever a
    // pointer denoting it exists, else at the smallest covering candidate
    // store, else at the tracked base object. The choice is deterministic --
    // unlike candidate discovery order -- and soundness does not depend on
    // it: acceptance requires MemorySSA's selection to be one of the
    // collected candidates, and every store/transfer candidate covers the
    // tracked byte, so an imprecise anchor can only yield a decline.
    MemorySSAClobber SSAResult;
    if (auto PreciseLocation = getPreciseTrackedLocation(CommonInfo)) {
      SSAResult = queryLocation(*PreciseLocation, *CommonInfo.UseBoundary);
    } else if (StoreInst *AnchorStore = findSmallestCoveringStore()) {
      SSAResult = queryLocation(MemoryLocation::get(AnchorStore),
                                *CommonInfo.UseBoundary);
    } else {
      SSAResult = queryMemorySSA(CommonInfo);
    }
    auto *ClobberingInst = SSAResult.Instruction;
    if (!ClobberingInst) {
      DEBUG(Logger::logs("proteus-pass")
            << "MemorySSA did not identify a clobbering instruction\n");
      fail();
      return nullptr;
    }
    if (PotentialClobbers.size() == 1) {
      // check if the clobber is identical to the write from the analysis, if
      // not we are in trouble
      if (PotentialClobbers[0].WriteCandidate != ClobberingInst) {
        DEBUG(Logger::logs("proteus-pass")
              << "Analysis found write at "
              << *PotentialClobbers[0].WriteCandidate
              << "\nBut MemorySSA found " << *ClobberingInst);
        fail();
        return nullptr;
      }
      if (auto *Store = dyn_cast<StoreInst>(ClobberingInst);
          Store && !storeCoversTrackedRange(*Store)) {
        DEBUG(Logger::logs("proteus-pass")
              << "MemorySSA selected a partial store for a whole-range "
                 "clobber query; declining: "
              << *Store << "\n");
        fail();
        return nullptr;
      }
      return PotentialClobbers[0].WriteCandidate;
    }
    auto *ClobberingStoreOrNull = dyn_cast<StoreInst>(ClobberingInst);

    bool AreAllStores =
        PotentialClobbers.size() > 1 && ClobberingInst &&
        ClobberingStoreOrNull &&
        llvm::all_of(PotentialClobbers, [&](const UseDefWriteCandidate &Q) {
          return dyn_cast<StoreInst>(Q.WriteCandidate);
        });
    if (!AreAllStores) {
      auto Selected = llvm::find_if(
          PotentialClobbers, [ClobberingInst](const auto &Candidate) {
            return Candidate.WriteCandidate == ClobberingInst;
          });
      if (Selected != PotentialClobbers.end()) {
        if (auto *Store = dyn_cast<StoreInst>(ClobberingInst);
            Store && !storeCoversTrackedRange(*Store)) {
          DEBUG(Logger::logs("proteus-pass")
                << "MemorySSA selected a partial store from mixed clobber "
                   "candidates; declining: "
                << *Store << "\n");
          fail();
          return nullptr;
        }
        return ClobberingInst;
      }

      DEBUG(Logger::logs("proteus-pass")
            << "Detected mixed StoreInst/CallBase/MemIntrinsics in list of "
               "candidate clobbers:\n");
      int CandidateIdx = 0;
      for (auto &Candidate : PotentialClobbers) {
        ++CandidateIdx;
        DEBUG(Logger::logs("proteus-pass")
              << "Candidate " << CandidateIdx << " = "
              << *Candidate.WriteCandidate);
      }

      fail();
      return nullptr;
    }

    // Source provenance is meaningful only after proving that the stores form
    // a valid destination partition. Exact duplicate intervals are successive
    // definitions of the same bytes, and disjoint intervals may be aggregate
    // fields. Partially overlapping intervals can splice bytes from different
    // values and must be rejected.
    for (size_t I = 0; I < PotentialClobbers.size(); ++I) {
      auto *First = cast<StoreInst>(PotentialClobbers[I].WriteCandidate);
      int64_t FirstOffset = 0;
      Value *FirstBase = GetPointerBaseWithConstantOffset(
          First->getPointerOperand(), FirstOffset, DL);
      auto FirstSize =
          getTypeStoreSize(DL, First->getValueOperand()->getType());
      if (!FirstBase || !FirstSize) {
        DEBUG(Logger::logs("proteus-pass")
              << "Cannot determine store destination interval: " << *First
              << "\n");
        return nullptr;
      }

      for (size_t J = I + 1; J < PotentialClobbers.size(); ++J) {
        auto *Second = cast<StoreInst>(PotentialClobbers[J].WriteCandidate);
        int64_t SecondOffset = 0;
        Value *SecondBase = GetPointerBaseWithConstantOffset(
            Second->getPointerOperand(), SecondOffset, DL);
        auto SecondSize =
            getTypeStoreSize(DL, Second->getValueOperand()->getType());
        if (!SecondBase || !SecondSize || FirstBase != SecondBase) {
          DEBUG(Logger::logs("proteus-pass")
                << "Cannot prove compatible store destinations: " << *First
                << " and " << *Second << "\n");
          return nullptr;
        }

        APInt FirstBegin(PointerIntervalBitWidth, FirstOffset, true);
        APInt FirstEnd = FirstBegin + *FirstSize;
        APInt SecondBegin(PointerIntervalBitWidth, SecondOffset, true);
        APInt SecondEnd = SecondBegin + *SecondSize;
        bool Overlap = FirstBegin.slt(SecondEnd) &&
                       SecondBegin.slt(FirstEnd);
        bool Identical = FirstBegin == SecondBegin && FirstEnd == SecondEnd;
        if (Overlap && !Identical) {
          DEBUG(Logger::logs("proteus-pass")
                << "Rejecting partially overlapping store destinations: "
                << *First << " and " << *Second << "\n");
          return nullptr;
        }
      }
    }

    // Multiple stores to one exact destination are successive definitions of
    // the same bytes, not aggregate construction from heterogeneous sources.
    // MemorySSA has already selected the definition reaching UseBoundary, so
    // discard the shadowed stores before comparing source provenance.
    int64_t ClobberingDestOffset = 0;
    Value *ClobberingDestBase = GetPointerBaseWithConstantOffset(
        ClobberingStoreOrNull->getPointerOperand(), ClobberingDestOffset, DL);
    DEBUG(Logger::logs("proteus-pass")
          << "Computed base ptr " << *ClobberingDestBase << " and offset "
          << ClobberingDestOffset << " for instruction "
          << *ClobberingStoreOrNull << "\n");
    bool StoresHaveSameDestination =
        ClobberingDestBase &&
        llvm::all_of(
            PotentialClobbers, [&](const UseDefWriteCandidate &Candidate) {
              // Note we have already exited if the cast below would be null
              auto *SI = cast<StoreInst>(Candidate.WriteCandidate);
              int64_t DestOffset = 0;
              Value *DestBase = GetPointerBaseWithConstantOffset(
                  SI->getPointerOperand(), DestOffset, DL);
              DEBUG(Logger::logs("proteus-pass")
                    << "Computed base ptr " << *DestBase << " and offset "
                    << DestOffset << " for instruction " << *SI << "\n");
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
    auto *ClobberingBasePtr = getValueBaseWithConstantOffset(
        ClobberingStoreOrNull->getValueOperand(), ClobberingOffset, DL);
    if (!ClobberingBasePtr) {
      DEBUG(Logger::logs("proteus-pass")
            << "Cannot recover the selected store's source provenance; "
               "declining: "
            << *ClobberingStoreOrNull << "\n");
      return nullptr;
    }
    int64_t ClobberingDestOffsetForDelta = 0;
    if (!GetPointerBaseWithConstantOffset(
            ClobberingStoreOrNull->getPointerOperand(),
            ClobberingDestOffsetForDelta, DL))
      return nullptr;
    // One source base is not enough: the source-to-destination translation
    // must be one affine map, i.e. (source offset - destination offset) must
    // be constant across the candidates. Without this, a field-swapping copy
    // from a single source (dest slot 0 <- src slot 1, dest slot 1 <- src
    // slot 0) passes the same-base check and specializes transposed slots.
    const int64_t ClobberingDelta =
        ClobberingOffset - ClobberingDestOffsetForDelta;
    SmallDenseMap<int64_t, SmallVector<std::pair<Value *, StoreInst *>>>
        OffsetToBasePtrStoreMap;
    // These two booleans create the axes of cases.
    DEBUG(Logger::logs("proteus-pass") << "Clobbering query ptr candidate = " << *ClobberingBasePtr<<"\n");
    bool StoresHaveSameBasePtr =
        llvm::all_of(PotentialClobbers, [&](const UseDefWriteCandidate &Q) {
          auto *SI = dyn_cast<StoreInst>(Q.WriteCandidate);
          auto *ValueOp = SI->getValueOperand();
          int64_t ValueOffset = 0;
          auto *SIValueBasePtr =
              getValueBaseWithConstantOffset(ValueOp, ValueOffset, DL);
          if (!SIValueBasePtr) {
            DEBUG(Logger::logs("proteus-pass")
                  << "Cannot recover store source provenance; declining: "
                  << *SI << "\n");
            return false;
          }
          OffsetToBasePtrStoreMap[ValueOffset].push_back({SIValueBasePtr, SI});
          DEBUG(Logger::logs("proteus-pass") << "Base ptr candidate = " << *SIValueBasePtr<<"\n");
          int64_t DestOffset = 0;
          if (!GetPointerBaseWithConstantOffset(SI->getPointerOperand(),
                                                DestOffset, DL))
            return false;
          if (ValueOffset - DestOffset != ClobberingDelta) {
            DEBUG(Logger::logs("proteus-pass")
                  << "Store breaks the affine source-to-destination "
                     "translation (delta "
                  << (ValueOffset - DestOffset) << " vs " << ClobberingDelta
                  << "); declining: " << *SI << "\n");
            return false;
          }
          return SIValueBasePtr == ClobberingBasePtr;
        });

    bool StoresHaveDifferentOffsets =
        llvm::all_of(OffsetToBasePtrStoreMap,
                     [&](const auto &It) { return It.second.size() == 1; });
    // This is the easiest possible case: basically we've found the LLVM IR
    // shape where we load a bunch of values in to the aggregate pointer
    // representing lambda storage. The Clobbering analysis just picks the last
    // write to an element in the aggregate lambda storage.
    if (StoresHaveDifferentOffsets && StoresHaveSameBasePtr) {
      return ClobberingStoreOrNull;
    }
    // We can still conservatively pass in this case: if there are two writes to
    // the same field in the lambda but we know one of them clobbers, we can
    // still return the common base ptr as the source of truth for the lambda.
    if (!StoresHaveDifferentOffsets && StoresHaveSameBasePtr) {
      for (const auto &Entry : OffsetToBasePtrStoreMap) {
        const auto &CurVec = Entry.second;
        if (CurVec.size() == 1)
          continue;
        auto It = std::find_if(CurVec.begin(), CurVec.end(),
                               [&](std::pair<Value *, StoreInst *> Arg) {
                                 return Arg.second == ClobberingStoreOrNull;
                               });
        if (It != CurVec.end())
          return ClobberingStoreOrNull;
      }
    }
    // What we can't have is a data occupying slots in the lambda pointer coming
    // in from miscellaneous sources.
    if (!StoresHaveSameBasePtr) {
      DEBUG(Logger::logs("proteus-pass")
            << "Ptr use analysis error: found heterogenous data sources in "
               "lambda storage slots. This is illegal and should be detected "
               "by the frontend. \n");
      for (const auto &Clobber : PotentialClobbers) {
        DEBUG(Logger::logs("proteus-pass")
              << "Candidate at offset = "
              << Clobber.ClobberInfo.OffsetOfWriteToBasePtr << " : "
              << *Clobber.WriteCandidate);
      }
    }

    return nullptr;
  }
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We
  // need to track where the calling LambdaArgVisitor came in from, so that our
  // analysis does not
  LambdaInstUseVisitor(Value *PtrBegin, Instruction *SeenUse,
                       CallBase *LambdaCB, const DataLayout &Dl,
                       int64_t TargetOff,
                       std::shared_ptr<MemorySSAClobberOracle> Oracle,
                       uint64_t EntitySize,
                       std::optional<TrackedRange> PostLoadRange)
      : ClobberOracle(std::move(Oracle)), TrackedBase(PtrBegin),
        LoadedPointerRange(PostLoadRange), DL(Dl) {
    // queryMemorySSA() constructs its MemoryLocation from this range. Populate
    // it before the first query so the cache cannot record a zero-byte query.
    ValueRangeMap[PtrBegin] = {TargetOff, EntitySize};
    WorkList.push_back({PtrBegin, nullptr, SeenUse, SeenUse->getFunction(),
                        PtrBegin, nullptr});
    // A pointer-transform on the backwards provenance path may also have an
    // earlier store as a user.  Visit that transform so those writes remain
    // visible, but stop before re-entering the lambda invocation itself.
    if (!isa<GetElementPtrInst, BitCastInst, AddrSpaceCastInst>(SeenUse))
      NeverVisit.insert(SeenUse);
    if (LambdaCB)
      NeverVisit.insert(LambdaCB);
  }
  auto popBack() {
    // Bound total work so no IR shape -- however adversarial -- can make the
    // analysis run unbounded; exceeding the budget declines specialization.
    if (++ProcessedWorkItems > MaxWorkItems) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Work-item budget (" << MaxWorkItems
            << ") exhausted; failing conservatively\n");
      fail();
    }
    Current = WorkList.back();
    Def = Current.LastVal;
    WorkList.pop_back();
    if (Current.ReturnTo)
      --Current.ReturnTo->PendingWorkItems;
    return Current;
  }
  auto getLastDef() { return Def; }
  bool seen(Value *Val) {
    return NeverVisit.contains(Val) ||
           Seen.contains({Val, Current.ReturnTo.get()});
  }
  void markAsSeen(Value *Val) { Seen.insert({Val, Current.ReturnTo.get()}); }
  void markAsUnseen(Value *Val) { Seen.erase({Val, Current.ReturnTo.get()}); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  auto getAnalysisResult() { return Result; }

  // Keep track of Function frame
  void pushBack(Value *NextVal, Value *CurVal, Function *Frame,
                Instruction *UseBoundary, Value *TrackedBase) {
    if (Current.ReturnTo)
      ++Current.ReturnTo->PendingWorkItems;
    WorkList.push_back(
        {NextVal, CurVal, UseBoundary, Frame, TrackedBase, Current.ReturnTo});
  }

  void pushBack(Value *NextVal, Value *CurVal, Function *Frame,
                Instruction *UseBoundary, Value *TrackedBase,
                std::shared_ptr<ReturnContinuation> ReturnTo) {
    if (ReturnTo)
      ++ReturnTo->PendingWorkItems;
    WorkList.push_back({NextVal, CurVal, UseBoundary, Frame, TrackedBase,
                        std::move(ReturnTo)});
  }

  void noteTrackedWrite() {
    for (auto Continuation = Current.ReturnTo; Continuation;
         Continuation = Continuation->Parent)
      Continuation->SawTrackedWrite = true;
  }

  void finishCurrentWorkItem() {
    auto ReturnTo = Current.ReturnTo;
    if (!ReturnTo || ReturnTo->Completed || ReturnTo->PendingWorkItems != 0 ||
        !ReturnTo->ReturnedRange)
      return;

    ReturnTo->Completed = true;
    if (ReturnTo->SawTrackedWrite) {
      // Supporting a helper that both writes through and returns the tracked
      // pointer requires reconciling two frame-local clobber queries. Reject
      // that shape rather than treating the call as transparent.
      fail();
      return;
    }

    CallBase *Call = ReturnTo->Call;
    ValueRangeMap[Call] = *ReturnTo->ReturnedRange;
    ProvenReturnedAliasCalls.insert(Call);
    // A query made before the helper was traversed may have cached the call
    // itself as MemorySSA's conservative clobber.
    QueryCache.clear();

    for (User *Usr : Call->users())
      pushBack(Usr, Call, ReturnTo->CallerFrame, ReturnTo->CallerUseBoundary,
               ReturnTo->CallerTrackedBase, ReturnTo->Parent);
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

    if (!DefBeforeCB || !ValueRangeMap.contains(DefBeforeCB)) {
      rangeMapFailure(DefBeforeCB ? DefBeforeCB : TrackedBase);
      return false;
    }

    auto &Deferred = DeferredCallBaseWorkItems[&CB];
    bool FoundArg = false;
    for (size_t ArgI = 0; ArgI < CalledFunction->arg_size(); ++ArgI) {
      if (CB.getArgOperand(ArgI) != DefBeforeCB)
        continue;

      Argument *ArgToTrack = CalledFunction->getArg(ArgI);
      // ValueRangeMap is not call-context scoped: a second entry into this
      // callee with a different range would silently retarget facts computed
      // for the first entry. Equal ranges recompute identical facts
      // and are permitted.
      auto ExistingRange = ValueRangeMap.find(ArgToTrack);
      if (ExistingRange != ValueRangeMap.end() &&
          ExistingRange->second != ValueRangeMap[DefBeforeCB]) {
        DEBUG(Logger::logs("proteus-pass")
              << "    [PTR use analysis]: Conflicting re-entry ranges for "
                 "formal "
              << *ArgToTrack << ": " << ExistingRange->second << " vs "
              << ValueRangeMap[DefBeforeCB] << "\n");
        return false;
      }
      FoundArg = true;
      ValueRangeMap[ArgToTrack] = ValueRangeMap[DefBeforeCB];
      for (User *Usr : ArgToTrack->users())
        Deferred.push_back({Usr, ArgToTrack, ReturnInstVec.front(),
                            CalledFunction, ArgToTrack, Current.ReturnTo});
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
      fail();
      return;
    }

    auto CandidateIt =
        llvm::find_if(PotentialClobbers, [ResolvedInst](const auto &Candidate) {
          return Candidate.WriteCandidate == ResolvedInst;
        });
    if (CandidateIt == PotentialClobbers.end()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Resolution phase failed: MemorySSA "
               "selection is absent from collected candidates: "
            << *ResolvedInst << "\n");
      fail();
      return;
    }

    if (auto *SI = dyn_cast<StoreInst>(ResolvedInst)) {
      // When this store resolves a CallBase selected in the parent frame, it
      // must account for the parent's entire query. A helper-local store to
      // one field cannot stand in for the other bytes still supplied by the
      // deferred parent writes.
      if (!ParentClobberCandidateStack.empty()) {
        auto StoreSize = getTypeStoreSize(DL, SI->getValueOperand()->getType());
        auto StoredRange = ValueRangeMap.find(SI->getPointerOperand());
        if (!storeCoversTrackedRange(*SI)) {
          DEBUG({
            auto &OS = Logger::logs("proteus-pass");
            OS << "    [PTR use analysis]: Callee store covers only part of "
                  "the parent clobber query; declining: store-size="
               << (StoreSize ? *StoreSize : 0) << "; query-at-store=";
            if (StoredRange != ValueRangeMap.end())
              OS << StoredRange->second;
            else
              OS << "<unknown>";
            OS << "; store=" << *SI << "\n";
          });
          fail();
          return;
        }
      }
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Exiting visitor with value from "
               "resolved store: "
            << *SI << "; offset="
            << CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr << "\n");
      Result = {.DominatingWrite = SI->getValueOperand(),
                .WriteBoundary = SI,
                .Offset = CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr,
                .ChangedRCLayout = std::nullopt};
      AnalysisFailed = false;
      AnalysisSuccess = true;
      return;
    }

    if (auto *MT = dyn_cast<MemTransferInst>(ResolvedInst)) {
      int64_t SourceOffset = 0;
      Value *SourceBase = GetPointerBaseWithConstantOffset(MT->getRawSource(),
                                                           SourceOffset, DL);
      if (!SourceBase) {
        fail();
        return;
      }
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Exiting visitor with source from "
               "resolved memory transfer: "
            << *MT << "; offset-correction="
            << CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr << "\n");
      Result = {.DominatingWrite = MT->getRawSource(),
                .WriteBoundary = MT,
                .Offset = CandidateIt->ClobberInfo.OffsetOfWriteToBasePtr +
                          SourceOffset,
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
      fail();
      return;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Resuming visitor across selected call "
             "into "
          << CB->getCalledFunction()->getName()
          << "; deferred-work-items=" << Deferred->second.size() << "\n");

    SmallVector<UseDefWriteCandidate, 4> ParentCandidates;
    for (UseDefWriteCandidate Candidate : PotentialClobbers) {
      if (Candidate.WriteCandidate == CB)
        continue;
      // If the selected call proves not to write the tracked byte, the next
      // query must ask for the value immediately before that call.
      Candidate.ClobberInfo.UseBoundary = CB;
      ParentCandidates.push_back(Candidate);
    }
    ParentClobberCandidateStack.push_back(std::move(ParentCandidates));
    PotentialClobbers.clear();
    // Route through the accounting that popBack undoes: appending directly
    // would decrement PendingWorkItems without the matching increment and
    // wrap the unsigned counter, wedging the continuation.
    for (const UseAnalysisWorkItem &Item : Deferred->second) {
      if (Item.ReturnTo)
        ++Item.ReturnTo->PendingWorkItems;
      WorkList.push_back(Item);
    }
    DeferredCallBaseWorkItems.erase(Deferred);
  }

  void resolveCollectedClobbersAndContinue() {
    if (PotentialClobbers.empty() && !ParentClobberCandidateStack.empty() &&
        !ParentClobberCandidateStack.back().empty()) {
      PotentialClobbers = std::move(ParentClobberCandidateStack.back());
      ParentClobberCandidateStack.pop_back();
    }
    continueFromResolvedClobber(resolveCollectedClobbers());
  }

  // Query the oracle for Location at UseBoundary. A pointer-returning call
  // that this visitor has traversed through to a ReturnInst without finding a
  // write is an alias edge, not the memory definition we seek; ask the same
  // question immediately before it.
  MemorySSAClobber queryLocation(const MemoryLocation &Location,
                                 Instruction &UseBoundary) {
    MemorySSAClobber Result = ClobberOracle->query(Location, UseBoundary);
    while (Result.Kind == MemorySSAClobberKind::Definition &&
           ProvenReturnedAliasCalls.contains(
               dyn_cast_or_null<CallBase>(Result.Instruction))) {
      Result = ClobberOracle->query(Location, *Result.Instruction);
    }
    return Result;
  }

  // Resolve Query.Range exactly when traversal found a pointer denoting its
  // first byte, and use a covering prefix of TrackedPtr otherwise.
  MemoryLocation getQueryLocation(const ClobberQuery &Query) const {
    if (auto Precise = getPreciseTrackedLocation(Query))
      return *Precise;
    auto Covering = getTrackedPointerLocation(DL, Query.TrackedPtr);
    // Bound the covering fallback by the tracked entity's extent: the prefix
    // [0, remaining offset + size) still covers the entity but excludes
    // writes past it, so e.g. an envelope's suffix store can no longer be
    // selected as the clobber and spuriously decline a correct shape
    // (lambda_size_test, kernelEnvelopeSuffix).
    TrackedRange Range = Query.Range;
    if (Range.Offset >= 0) {
      uint64_t CoverBytes = static_cast<uint64_t>(Range.Offset) + Range.Extent;
      // Keep a smaller already-precise covering location (whole allocation).
      if (!Covering || !Covering->Size.hasValue() ||
          !Covering->Size.isPrecise() || Covering->Size.getValue() > CoverBytes)
        return MemoryLocation(Query.TrackedPtr,
                              LocationSize::precise(CoverBytes));
    }
    if (Covering)
      return *Covering;
    return MemoryLocation::getBeforeOrAfter(Query.TrackedPtr);
  }

  MemorySSAClobber queryMemorySSA(const ClobberQuery &Query) {
    auto Cached = QueryCache.find(Query);
    if (Cached != QueryCache.end())
      return Cached->second;

    MemorySSAClobber Result;
    if (!Query.Frame || !Query.TrackedPtr ||
        !Query.TrackedPtr->getType()->isPointerTy() || !Query.UseBoundary ||
        Query.UseBoundary->getFunction() != Query.Frame) {
      QueryCache.try_emplace(Query, Result);
      return Result;
    }

    Result = queryLocation(getQueryLocation(Query), *Query.UseBoundary);
    QueryCache.try_emplace(Query, Result);
    return Result;
  }

  // The deterministic fallback anchor for the reconciliation query when no
  // pointer denoting exactly the tracked byte exists: the smallest covering
  // candidate store (ties broken by lowest destination offset). Candidates
  // with equal size and offset denote the same location, so further
  // tie-breaking is immaterial.
  StoreInst *findSmallestCoveringStore() const {
    StoreInst *Best = nullptr;
    uint64_t BestSize = std::numeric_limits<uint64_t>::max();
    int64_t BestOffset = std::numeric_limits<int64_t>::max();
    for (const UseDefWriteCandidate &Candidate : PotentialClobbers) {
      auto *SI = dyn_cast<StoreInst>(Candidate.WriteCandidate);
      if (!SI)
        continue;
      auto Size = getTypeStoreSize(DL, SI->getValueOperand()->getType());
      if (!Size)
        continue;
      int64_t DestOffset = 0;
      if (!GetPointerBaseWithConstantOffset(SI->getPointerOperand(), DestOffset,
                                            DL))
        continue;
      if (*Size < BestSize || (*Size == BestSize && DestOffset < BestOffset)) {
        Best = SI;
        BestSize = *Size;
        BestOffset = DestOffset;
      }
    }
    return Best;
  }

  // Return a MemoryLocation denoting exactly the tracked byte, when an
  // already-visited pointer addresses it. Invariant this relies on:
  // visitGetElementPtrInst only inserts covering GEPs into ValueRangeMap,
  // and a remaining offset of zero means the pointer consumed the entire
  // remaining displacement, i.e. it addresses the tracked byte itself. The
  // smallest result size gives the tightest range at that address;
  // equal-size candidates denote the same location, so ties are immaterial.
  // NOTE: deliberately NOT widened to the tracked entity's extent -- interior
  // slot stores are legitimate parallel copies in slot-wise assembly (the
  // mfem shape), and extent-wide visibility would require per-piece
  // reconciliation to accept them. Interior-write soundness is a frontend
  // contract; see the functor-extent discussion in
  // docs/dev/reviews/clobbering-aware-use-analysis.md.
  std::optional<MemoryLocation>
  getPreciseTrackedLocation(const ClobberQuery &Query) const {
    if (Query.Range.Extent == 0)
      return std::nullopt;
    Value *TrackedObject = getUnderlyingObject(Query.TrackedPtr);
    Value *BestPtr = nullptr;
    uint64_t BestSize = std::numeric_limits<uint64_t>::max();

    for (const auto &[Ptr, Range] : ValueRangeMap) {
      if (Range.Offset != 0 || !Ptr->getType()->isPointerTy() ||
          getUnderlyingObject(Ptr) != TrackedObject)
        continue;

      if (auto *I = dyn_cast<Instruction>(Ptr)) {
        if (I->getFunction() != Query.Frame)
          continue;
      } else if (auto *Arg = dyn_cast<Argument>(Ptr)) {
        if (Arg->getParent() != Query.Frame)
          continue;
      } else {
        continue;
      }

      auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
      if (!GEP)
        continue;
      auto Size = getTypeStoreSize(DL, GEP->getResultElementType());
      if (!Size || *Size >= BestSize)
        continue;
      BestPtr = Ptr;
      BestSize = *Size;
    }

    if (!BestPtr)
      return std::nullopt;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Using precise MemorySSA location: ptr="
          << *BestPtr << "; size=" << Query.Range.Extent << "\n");
    return MemoryLocation(BestPtr,
                          LocationSize::precise(Query.Range.Extent));
  }

  // The two flags are a unit: never both true. Every conservative decline
  // routes through here.
  void fail() {
    AnalysisFailed = true;
    AnalysisSuccess = false;
  }

  void rangeMapFailure(Value *V) {
    fail();
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to absence of " << *V
          << " in range tracking map, this is an internal compiler bug\n");
  }

  void clobberCandidatesFailure() {
    fail();
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to frame/tracked ptr "
             "mismatch"
          << " in Clobber candidates map\n");
  }

  // WorkList is LIFO.  Enqueue possible writers last so they are inspected
  // before an older store reached through a GEP.  Otherwise a memcpy/memmove
  // call can be skipped merely because the initializer happens to appear
  // earlier in Value::users().
  void pushBackPointerUsers(Value *V) {
    SmallVector<User *, 4> PossibleWriters;
    for (User *Usr : V->users()) {
      if (seen(Usr))
        continue;
      // Loads are not generic pointer-use edges. visitStoreInst explicitly
      // enqueues the loads proven to reload a tracked pointer spill.
      if (isa<LoadInst>(Usr))
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
    // A pointer argument is commonly spilled in an unoptimized or optnone
    // callee before it is used. Only follow a load from a local pointer slot
    // when MemorySSA proves that this exact store reaches the load. This keeps
    // a stale spill from propagating past a later overwrite.
    if (Stored == Current.LastVal && Stored->getType()->isPointerTy()) {
      auto *Storage = dyn_cast<AllocaInst>(getUnderlyingObject(StoreBase));
      if (!Storage)
        return;
      if (!ValueRangeMap.contains(Stored)) {
        rangeMapFailure(Stored);
        return;
      }
      ValueRangeMap[StoreBase] = ValueRangeMap[Stored];
      for (User *Usr : StoreBase->users()) {
        auto *LI = dyn_cast<LoadInst>(Usr);
        if (!LI || seen(LI))
          continue;
        MemorySSAClobber ReachingDefinition = ClobberOracle->query(*LI);
        if (ReachingDefinition.Kind != MemorySSAClobberKind::Definition ||
            ReachingDefinition.Instruction != &SI)
          continue;
        ProvenPointerReloads.insert(LI);
        pushBack(LI, StoreBase, Current.Frame, Current.UseBoundary,
                 Current.TrackedBase);
      }
      return;
    }

    // This analysis tracks writes to the original pointer. A non-spill store
    // that merely reads the prior value is not such a write.
    if (Stored == Current.LastVal)
      return;

    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!ValueRangeMap.contains(StoreBase)) {
      rangeMapFailure(StoreBase);
      return;
    }
    TrackedRange StoreRange = ValueRangeMap[StoreBase];

    if (!StoreSize ||
        !rangesOverlap(StoreRange.Offset, StoreRange.Extent, 0, *StoreSize))
      return;
    // Derive the correction from this use-def path. Offset is visitor-global
    // and may include unrelated sibling GEPs or GEPs visited in a returned-
    // alias helper; it therefore cannot describe this particular store.
    if (!ValueRangeMap.contains(TrackedBase)) {
      rangeMapFailure(TrackedBase);
      return;
    }
    int64_t WriteOffset = ValueRangeMap[TrackedBase].Offset - StoreRange.Offset;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to range " << StoreRange
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    noteTrackedWrite();
    PotentialClobbers.push_back(UseDefWriteCandidate{
        .WriteCandidate = &SI,
        .ClobberInfo =
            {
                .Frame = Current.Frame,
                .TrackedPtr = Current.TrackedBase,
                .UseBoundary = Current.UseBoundary,
                .OffsetOfWriteToBasePtr = WriteOffset,
                .Range = ValueRangeMap.lookup(Current.TrackedBase),
            },
    });
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Skipping tracking non-pointer load "
            << LI << "\n");
      return;
    }
    if (!ProvenPointerReloads.contains(&LI)) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Stopping at unproven pointer load "
            << LI << "\n");
      return;
    }
    if (!ValueRangeMap.contains(LI.getPointerOperand())) {
      rangeMapFailure(LI.getPointerOperand());
      return;
    }
    TrackedRange InputRange = ValueRangeMap[LI.getPointerOperand()];
    // The caller-supplied post-load range changes coordinates only when
    // dereferencing the original tracked storage. A callee may spill and
    // reload its pointer argument; that reload remains an alias of the same
    // aggregate and must preserve the incoming range.
    bool LoadsFromOriginalTrackedObject =
        getUnderlyingObject(LI.getPointerOperand()) ==
        getUnderlyingObject(TrackedBase);
    ValueRangeMap[&LI] =
        LoadsFromOriginalTrackedObject && LoadedPointerRange
            ? *LoadedPointerRange
            : InputRange;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Pointer load changes tracked range from "
          << InputRange << " to " << ValueRangeMap[&LI] << "\n");
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

    // A pointer-returning helper is another segment of this same use walk.
    // Enter it through the formal corresponding to the use-def edge that
    // reached this call. visitReturnInst restores the caller context and
    // continues from the CallBase result. This deliberately uses the normal
    // visitor worklist instead of independently tracing returned provenance.
    bool EnteredReturnedAlias = false;
    if (CB.getType()->isPointerTy() && DefBeforeCB &&
        ValueRangeMap.contains(DefBeforeCB)) {
      Function *Callee = CB.getCalledFunction();
      if (Callee && !Callee->isDeclaration()) {
        // Reject recursion through pointer-returning helpers: each entry
        // mints a fresh ReturnContinuation, and Seen is keyed on
        // (Value, continuation), so a recursive helper would re-enqueue its
        // formal's users without bound.
        bool Recursive = Callee == Current.Frame;
        for (auto Continuation = Current.ReturnTo; Continuation && !Recursive;
             Continuation = Continuation->Parent)
          Recursive = Continuation->CallerFrame == Callee ||
                      (Continuation->Call &&
                       Continuation->Call->getCalledFunction() == Callee);
        if (Recursive) {
          DEBUG(Logger::logs("proteus-pass")
                << "    [PTR use analysis]: Rejecting recursive "
                   "pointer-returning helper "
                << Callee->getName() << "\n");
          fail();
          return;
        }
        auto Returns = proteus::getReturnInstructions(*Callee);
        if (Returns.size() == 1) {
          for (unsigned ArgI = 0;
               ArgI < CB.arg_size() && ArgI < Callee->arg_size(); ++ArgI) {
            if (CB.getArgOperand(ArgI) != DefBeforeCB)
              continue;

            Argument *Formal = Callee->getArg(ArgI);
            // Same conflict rule as cacheCallBaseWorkItems: ValueRangeMap is
            // not call-context scoped, so a re-entry with a different range
            // must decline rather than retarget in-flight facts.
            auto ExistingRange = ValueRangeMap.find(Formal);
            if (ExistingRange != ValueRangeMap.end() &&
                ExistingRange->second != ValueRangeMap[DefBeforeCB]) {
              DEBUG(Logger::logs("proteus-pass")
                    << "    [PTR use analysis]: Conflicting re-entry ranges "
                       "for formal "
                    << *Formal << ": " << ExistingRange->second << " vs "
                    << ValueRangeMap[DefBeforeCB] << "\n");
              fail();
              return;
            }
            ValueRangeMap[Formal] = ValueRangeMap[DefBeforeCB];
            auto ReturnTo = std::make_shared<ReturnContinuation>(
                ReturnContinuation{&CB, Current.Frame, Current.UseBoundary,
                                   Current.TrackedBase, Current.ReturnTo, 0,
                                   false, false, std::nullopt});
            // Seen is keyed on raw continuation addresses; keep every
            // continuation alive for the visitor's lifetime so a freed
            // allocation cannot be reused and alias stale Seen entries.
            OwnedContinuations.push_back(ReturnTo);
            for (User *Usr : Formal->users())
              pushBack(Usr, Formal, Callee, Returns.front(), Formal, ReturnTo);
            EnteredReturnedAlias = true;
          }
        }
      }
    }

    // The call result is an alias edge, not itself a candidate write. Any
    // writes performed by the helper are encountered while visiting the
    // formal's users and are handled by the ordinary StoreInst/CallBase
    // visitors.
    if (EnteredReturnedAlias)
      return;

    if (!cacheCallBaseWorkItems(CB, DefBeforeCB)) {
      fail();
      return;
    }

    noteTrackedWrite();
    PotentialClobbers.push_back(
        {.WriteCandidate = &CB,
         .ClobberInfo = {
             .Frame = Current.Frame,
             .TrackedPtr = Current.TrackedBase,
             .UseBoundary = Current.UseBoundary,
             // Unused for calls: a selected CallBase resumes interprocedural
             // traversal instead of producing a value+correction.
             .OffsetOfWriteToBasePtr = 0,
             .Range = ValueRangeMap.lookup(Current.TrackedBase)}});
  }

  void visitPHINode(PHINode &Phi) {
    std::optional<TrackedRange> IncomingRange;
    for (Value *Incoming : Phi.incoming_values()) {
      auto It = ValueRangeMap.find(Incoming);
      if (It == ValueRangeMap.end()) {
        // Another incoming use may still be on the visitor worklist. Permit
        // that edge to schedule this PHI again.
        markAsUnseen(&Phi);
        return;
      }
      if (IncomingRange && *IncomingRange != It->second) {
        fail();
        return;
      }
      IncomingRange = It->second;
    }
    if (!IncomingRange) {
      fail();
      return;
    }
    ValueRangeMap[&Phi] = *IncomingRange;
    pushBackPointerUsers(&Phi);
  }

  void visitSelectInst(SelectInst &Select) {
    auto TrueIt = ValueRangeMap.find(Select.getTrueValue());
    auto FalseIt = ValueRangeMap.find(Select.getFalseValue());
    if (TrueIt == ValueRangeMap.end() || FalseIt == ValueRangeMap.end()) {
      markAsUnseen(&Select);
      return;
    }
    if (TrueIt->second != FalseIt->second) {
      fail();
      return;
    }
    ValueRangeMap[&Select] = TrueIt->second;
    pushBackPointerUsers(&Select);
  }

  void visitReturnInst(ReturnInst &RI) {
    if (!Current.ReturnTo || RI.getReturnValue() != Current.LastVal ||
        !Current.LastVal || !ValueRangeMap.contains(Current.LastVal)) {
      fail();
      return;
    }

    auto ReturnTo = Current.ReturnTo;
    CallBase *Call = ReturnTo->Call;
    TrackedRange ReturnedRange = ValueRangeMap[Current.LastVal];
    auto Existing = ValueRangeMap.find(Call);
    if (Existing != ValueRangeMap.end() && Existing->second != ReturnedRange) {
      fail();
      return;
    }
    if (ReturnTo->ReturnedRange && *ReturnTo->ReturnedRange != ReturnedRange) {
      fail();
      return;
    }
    ReturnTo->ReturnedRange = ReturnedRange;
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
      fail();
      return;
    }

    int64_t GEPOffset = StepOffset.getSExtValue();
    // LLVM may lower aggregate field materialization to a byte-wise GEP under
    // opaque pointers. Preserve that shape when the allocation still provides
    // local aggregate provenance, but reject byte arithmetic through an
    // argument or another root whose storage type is unavailable here.
    Value *UnderlyingObject = getUnderlyingObject(GEP.getPointerOperand());
    if (GEPOffset != 0 && GEP.getSourceElementType()->isIntegerTy(8) &&
        !isa<AllocaInst>(UnderlyingObject)) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Rejecting byte-wise pointer offset "
            << GEP << "\n");
      fail();
      return;
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    " << "Computed GEP offset " << GEPOffset << "\n");
    Value *GEPBase = GEP.getPointerOperand();
    if (!GEPBase)
      return;

    if (!ValueRangeMap.contains(GEPBase)) {
      rangeMapFailure(GEPBase);
      return;
    }

    TrackedRange BaseRange = ValueRangeMap[GEPBase];
    auto ResultSize = getTypeStoreSize(DL, GEP.getResultElementType());
    DEBUG(if (ResultSize) Logger::logs("proteus-pass")
              << "    GEP size = " << *ResultSize << "\n";)
    bool TypeErasedGEP = GEP.getSourceElementType()->isIntegerTy(8);
    if ((!TypeErasedGEP && ResultSize &&
         !rangesOverlap(BaseRange.Offset, BaseRange.Extent, GEPOffset,
                        *ResultSize)) ||
        (TypeErasedGEP &&
         !offsetCoveredByRange(GEPOffset, BaseRange.Offset,
                               BaseRange.Extent)))
      return;
    DEBUG(Logger::logs("proteus-pass") << "    Found GEP applicable to range="
                                       << BaseRange << " ; " << GEP << "\n");
    // A GEP changes the pointer-relative offset but not the extent of the
    // tracked entity. This also supports constant type-erased byte GEPs: their
    // i8 element type describes pointer arithmetic, not entity size.
    ValueRangeMap[&GEP] = {BaseRange.Offset - GEPOffset, BaseRange.Extent};
    DEBUG(Logger::logs("proteus-pass") << "    " << "Setting map K " << GEP
                                       << " : " << ValueRangeMap[&GEP] << "\n");
    pushBackPointerUsers(&GEP);
  }

  // todo: these three methods may need to be changed to find a dominating store
  // particularly for the case of mutable lambdas.
  void visitAllocaInst(AllocaInst &Alloca) {
    // This analysis should only ever encounter an AllocaInst as the first
    // instruction We assert this below and log a failure otherwise
    if (Def) {
      fail();
      DEBUG(Logger::logs("proteus-pass")
            << "    Dominating use analysis somehow reached AllocaInst from "
               "non-null def\n");
      return;
    }
    if (!ValueRangeMap.contains(&Alloca)) {
      fail();
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
    if (!Def) {
      if (!ValueRangeMap.contains(&BC)) {
        rangeMapFailure(&BC);
        return;
      }
      ValueRangeMap[BC.getOperand(0)] = ValueRangeMap[&BC];
    }
    // AddrSpaceCast does not change the offset we track.
    ValueRangeMap[&BC] = ValueRangeMap[BC.getOperand(0)];
    pushBackPointerUsers(&BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // The constructor automatically populates the map with ASC's offset
    // if its not present we need to rely on the pointer operand's offset
    if (!ValueRangeMap.contains(&ASC)) {
      if (!ValueRangeMap.contains(ASC.getPointerOperand())) {
        rangeMapFailure(ASC.getPointerOperand());
        return;
      }
      // AddrSpaceCast does not change the offset we track.
      ValueRangeMap[&ASC] = ValueRangeMap[ASC.getPointerOperand()];
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Setting range " << ASC << " = "
          << ValueRangeMap[&ASC]);

    pushBackPointerUsers(&ASC);
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    // Reaching an intrinsic through its source is a read, not a definition of
    // the tracked memory.
    if (Def != I.getRawDest())
      return;

    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      if (!ValueRangeMap.contains(Def)) {
        rangeMapFailure(Def);
        return;
      }
      auto *Len = dyn_cast<ConstantInt>(MS->getLength());
      // The memset destroys provenance if it touches ANY byte of the tracked
      // entity [Remaining, Remaining + size), not only its first byte; an
      // interior-slot memset previously slipped through here.
      TrackedRange Range = ValueRangeMap[Def];
      if (Len && !(Range.Offset < static_cast<int64_t>(Len->getZExtValue()) &&
                   Range.Offset + static_cast<int64_t>(Range.Extent) > 0))
        return;
      // An overlapping memset destroys the tracked provenance, and a dynamic
      // length cannot be proven not to overlap it.
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Memset may overwrite the tracked "
               "byte: "
            << I << "\n");
      fail();
      return;
    }

    // A memmove permits overlapping source and destination ranges. Resolving
    // its destination through the source therefore requires proving which
    // definition reaches the source at this instruction. Reject it until the
    // analysis performs that second clobber query.
    if (isa<MemMoveInst>(&I)) {
      fail();
      return;
    }

    auto *MT = cast<MemTransferInst>(&I); // memcpy

    if (!ValueRangeMap.contains(Def)) {
      rangeMapFailure(Def);
      return;
    }
    TrackedRange Range = ValueRangeMap[Def];

    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    if (!Len) {
      // We cannot prove that a dynamic-sized transfer defines the tracked byte.
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Dynamic-length transfer cannot prove "
               "provenance for the tracked byte: "
            << I << "\n");
      fail();
      return;
    }
    if (!offsetCoveredByRange(Range.Offset, 0, Len->getZExtValue()))
      return;
    // The transfer defines the tracked byte; it must then cover the entity's
    // full extent. A shorter copy -- the classic wrong-sizeof production bug,
    // legal C++ with no casts -- splices two sources into one functor, and
    // the byte-0 witness would lie for the uncovered slots.
    if (static_cast<uint64_t>(Range.Offset) + Range.Extent >
        Len->getZExtValue()) {
      DEBUG(Logger::logs("proteus-pass")
            << "    [PTR use analysis]: Transfer covers the tracked byte but "
               "not the tracked entity's full extent;\n "
               "Using range "
            << Range << " from def " << *Def << "\n"
            << "Transfer size = " << Len->getZExtValue() << "\n"
            << " declining: " << I << "\n");
      fail();
      return;
    }

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase) {
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: Failure due to nullptr dst/src " << "\n");
      fail();
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
    noteTrackedWrite();
    PotentialClobbers.push_back(
        {.WriteCandidate = &I,
         .ClobberInfo = {
             .Frame = Current.Frame,
             .TrackedPtr = Current.TrackedBase,
             .UseBoundary = Current.UseBoundary,
             .OffsetOfWriteToBasePtr = OffsetCorrection,
             .Range = ValueRangeMap.lookup(Current.TrackedBase)}});
  }

  void visitInstruction(Instruction &I) {
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Unhandled instruction "
          << I.getOpcodeName() << ": " << I << "\n");
    fail();
  }
};

inline std::optional<LambdaPtrUseAnalysis>
getDominatingUse(const DataLayout &DL, Value *ValueNeedingAnalysis,
                 Instruction *UseofPtr, int64_t TargetOffset,
                 CallBase *LambdaCB,
                 std::shared_ptr<MemorySSAClobberOracle> ClobberOracle,
                 uint64_t TrackedEntitySize,
                 std::optional<TrackedRange> PostLoadRange = std::nullopt) {
  DEBUG(Logger::logs("proteus-pass")
        << "Beginning PtrUse analysis with offset = " << TargetOffset
        << " entity size = " << TrackedEntitySize << "\n");

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, UseofPtr, LambdaCB, DL,
                               TargetOffset, std::move(ClobberOracle),
                               TrackedEntitySize, PostLoadRange);
  // Analysis loop
  while (!Visitor.success() && !Visitor.failed()) {
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto CurWorkItem = Visitor.popBack();
      auto *V = CurWorkItem.CurVal;
      // Prevent loops/infinite recursion. Completion accounting must still
      // run: a duplicate may be the final pending item of a continuation.
      if (Visitor.seen(V)) {
        Visitor.finishCurrentWorkItem();
        continue;
      }
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
      Visitor.finishCurrentWorkItem();
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
