#ifndef PROTEUS_CALLWRITEANALYSIS_H
#define PROTEUS_CALLWRITEANALYSIS_H

#include "Helpers.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/CaptureTracking.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/MathExtras.h>

namespace proteus {

// Prove that every write in a called function is disjoint from a location in
// its caller. The result belongs to this call context, not to the function.
// Unknown destinations/effects fail without a separate isolation proof. This
// analysis never recovers replacement closure values and never assumes
// distinct formals do not alias.
class CallWriteAnalysis {
  AAResults &CallerAA;
  const DataLayout &DL;
  function_ref<DominatorTree &(Function &)> GetDT;
  SmallVector<CallBase *, 8> Calls;
  unsigned RemainingInstructions = 0;
  SmallPtrSet<Function *, 8> ProjectingReturns;
  SmallVector<CallBase *, 8> ReturnContexts;
  SmallPtrSet<Value *, 8> JoiningPointers;
  Function *Caller = nullptr;
  bool PrivateObject = false;

  // Escape proof, separate from the write proof: writing an object need not
  // expose its address, while storing that address elsewhere can expose it
  // without writing the object. Follow optnone's private pointer spills.
  bool doesNotEscape(Value *Ptr, Instruction *Before, unsigned Depth = 0) {
    if (Depth >= 16)
      return false;
    SmallVector<Value *, 8> Pending{Ptr};
    SmallPtrSet<Value *, 16> Seen;
    while (!Pending.empty()) {
      Value *V = Pending.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (Use &U : V->uses()) {
        if (!RemainingInstructions)
          return false;
        --RemainingInstructions;
        auto *I = dyn_cast<Instruction>(U.getUser());
        if (!I)
          return false;
        if (Before &&
            (I == Before || !isPotentiallyReachable(I, Before, nullptr,
                                                    &GetDT(*I->getFunction()))))
          continue;
        if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
            isa<AddrSpaceCastInst>(I) || isa<PHINode>(I) ||
            isa<SelectInst>(I)) {
          Pending.push_back(I);
          continue;
        }
        if (isa<LoadInst>(I) || isa<ICmpInst>(I) || isa<DbgInfoIntrinsic>(I))
          continue;
        if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->getValueOperand() != V)
            continue;
          auto *Slot = dyn_cast<AllocaInst>(
              SI->getPointerOperand()->stripPointerCasts());
          if (!Slot || !Slot->getAllocatedType()->isPointerTy() ||
              PointerMayBeCaptured(Slot, /*ReturnCaptures=*/true,
#if LLVM_VERSION_MAJOR < 21
                                   /*StoreCaptures=*/true,
#endif
                                   /*MaxUsesToExplore=*/64))
            return false;
          SmallVector<Value *, 4> SlotAliases{Slot};
          SmallPtrSet<Value *, 8> SeenSlots;
          while (!SlotAliases.empty()) {
            Value *Alias = SlotAliases.pop_back_val();
            if (!SeenSlots.insert(Alias).second)
              continue;
            for (User *SU : Alias->users()) {
              if (!RemainingInstructions)
                return false;
              --RemainingInstructions;
              if (isa<BitCastInst>(SU) || isa<AddrSpaceCastInst>(SU))
                SlotAliases.push_back(SU);
              else if (auto *LI = dyn_cast<LoadInst>(SU)) {
                if (getSpilledPointer(*LI) != V)
                  return false;
                Pending.push_back(LI);
              } else if (SU != SI && !isa<DbgInfoIntrinsic>(SU) &&
                         !isa<LifetimeIntrinsic>(SU))
                return false;
            }
          }
          continue;
        }
        auto *CB = dyn_cast<CallBase>(I);
        if (!CB || CB->hasOperandBundles() || CB->isInlineAsm() ||
            !CB->isArgOperand(&U))
          return false;
        unsigned ArgNo = CB->getArgOperandNo(&U);
        if (CB->doesNotCapture(ArgNo))
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee || Callee->isDeclaration() || Callee->isInterposable() ||
            ArgNo >= Callee->arg_size() ||
            !doesNotEscape(Callee->getArg(ArgNo), nullptr, Depth + 1))
          return false;
      }
    }
    return true;
  }

  bool hasPrivateOrigin(Value *Ptr, Instruction &Before, unsigned Depth = 0) {
    if (Depth >= 16)
      return false;
    int64_t Offset = 0;
    Ptr = simplifyPointer(Ptr, Offset);
    if (auto *Local = dyn_cast_or_null<AllocaInst>(Ptr))
      return Local->getFunction() == Before.getFunction() &&
             doesNotEscape(Local, &Before);
    auto *A = dyn_cast_or_null<Argument>(Ptr);
    if (!A || A->getParent() != Before.getFunction())
      return false;
    if (!doesNotEscape(A, &Before))
      return false;
    if (A->hasByValAttr())
      return true;
    if (!A->getParent()->hasLocalLinkage())
      return false; // Callers outside this module could supply exposed storage.
    // Another incoming pointer can alias A and expose the same allocation.
    // A def-use scan of A alone is not an escape proof for that allocation.
    // Conservatively check every pointer formal until caller-context AA can
    // prove which of them are disjoint.
    for (Argument &Other : A->getParent()->args())
      if (&Other != A && Other.getType()->isPointerTy() &&
          !Other.hasByValAttr() && !doesNotEscape(&Other, &Before))
        return false;
    // An incoming pointer is not inherently private. Establish its origin
    // through one known caller; never assume a kernel pointer is unaliased.
    CallBase *CallerSite = nullptr;
    for (Use &U : A->getParent()->uses()) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      if (!CB || !CB->isCallee(&U) || CallerSite)
        return false;
      CallerSite = CB;
    }
    return CallerSite &&
           hasPrivateOrigin(CallerSite->getArgOperand(A->getArgNo()),
                            *CallerSite, Depth + 1);
  }

  // Handle the single-store pointer spills emitted by optnone. No memory
  // history is guessed: the slot cannot escape or have another writer, and
  // its store must dominate the load. All other pointer loads stay unknown.
  Value *getSpilledPointer(LoadInst &Load) {
    if (!Load.getType()->isPointerTy() || Load.isVolatile() || Load.isAtomic())
      return nullptr;
    auto *Slot =
        dyn_cast<AllocaInst>(Load.getPointerOperand()->stripPointerCasts());
    if (!Slot || !Slot->getAllocatedType()->isPointerTy())
      return nullptr;
    StoreInst *Store = nullptr;
    SmallVector<Value *, 8> Pending{Slot};
    SmallPtrSet<Value *, 8> Seen;
    while (!Pending.empty()) {
      Value *V = Pending.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (User *U : V->users()) {
        if (!RemainingInstructions)
          return nullptr;
        --RemainingInstructions;
        if (isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U)) {
          Pending.push_back(U);
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() != V || SI->isVolatile() ||
              SI->isAtomic() ||
              !SI->getValueOperand()->getType()->isPointerTy() ||
              (Store && Store != SI))
            return nullptr;
          Store = SI;
          continue;
        }
        if (auto *LI = dyn_cast<LoadInst>(U);
            LI && LI->getPointerOperand() == V && LI->getType()->isPointerTy())
          continue;
        if (!isa<DbgInfoIntrinsic>(U) && !isa<LifetimeIntrinsic>(U))
          return nullptr;
      }
    }
    if (!Store || !GetDT(*Load.getFunction()).dominates(Store, &Load))
      return nullptr;
    return Store->getValueOperand();
  }

  // Recover pointer identity only. Writes performed by these helpers are
  // still checked by inspectCall; returning an alias is not a no-write proof.
  Value *simplifyPointer(Value *Ptr, int64_t &Offset) {
    while (RemainingInstructions) {
      --RemainingInstructions;
      int64_t Step = 0;
      Ptr = GetPointerBaseWithConstantOffset(Ptr, Step, DL);
      if (AddOverflow(Offset, Step, Offset))
        return nullptr;
      if (auto *LI = dyn_cast<LoadInst>(Ptr)) {
        if (Value *Stored = getSpilledPointer(*LI)) {
          Ptr = Stored;
          continue;
        }
      }
      if (auto *A = dyn_cast<Argument>(Ptr)) {
        auto Context =
            llvm::find_if(llvm::reverse(ReturnContexts), [&](CallBase *CB) {
              return CB->getCalledFunction() == A->getParent();
            });
        if (Context != ReturnContexts.rend()) {
          Ptr = (*Context)->getArgOperand(A->getArgNo());
          continue;
        }
      }
      if (isa<PHINode>(Ptr) || isa<SelectInst>(Ptr)) {
        if (!JoiningPointers.insert(Ptr).second)
          return nullptr;
        SmallVector<Value *, 4> Inputs;
        if (auto *Phi = dyn_cast<PHINode>(Ptr))
          for (Value *Incoming : Phi->incoming_values())
            Inputs.push_back(Incoming);
        else {
          auto *Select = cast<SelectInst>(Ptr);
          Inputs = {Select->getTrueValue(), Select->getFalseValue()};
        }
        Value *Common = nullptr;
        int64_t CommonOffset = 0;
        for (Value *Input : Inputs) {
          int64_t InputOffset = Offset;
          Value *Base = simplifyPointer(Input, InputOffset);
          if (!Base ||
              (Common && (Common != Base || CommonOffset != InputOffset))) {
            JoiningPointers.erase(Ptr);
            return nullptr;
          }
          Common = Base;
          CommonOffset = InputOffset;
        }
        JoiningPointers.erase(Ptr);
        Offset = CommonOffset;
        return Common;
      }
      auto *CB = dyn_cast<CallBase>(Ptr);
      if (!CB || !CB->getType()->isPointerTy())
        return Ptr;
      Function *F = CB->getCalledFunction();
      if (!F || F->isDeclaration() || F->isInterposable() ||
          CB->hasOperandBundles() || ProjectingReturns.size() >= 16 ||
          !ProjectingReturns.insert(F).second)
        return nullptr;
      ReturnInst *Return = nullptr;
      for (BasicBlock &BB : *F) {
        if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          if (Return) {
            ProjectingReturns.erase(F);
            return nullptr;
          }
          Return = RI;
        }
      }
      // Substitute actual arguments before joining returned aliases. Two
      // different formals may denote the same slot at this particular call.
      ReturnContexts.push_back(CB);
      Value *Returned = Return && Return->getReturnValue()
                            ? simplifyPointer(Return->getReturnValue(), Offset)
                            : nullptr;
      ReturnContexts.pop_back();
      ProjectingReturns.erase(F);
      if (!Returned)
        return nullptr;
      Ptr = Returned;
    }
    return nullptr;
  }

  bool writeIsDisjoint(const MemoryLocation &Write,
                       const MemoryLocation &Tracked) {
    Value *Ptr = const_cast<Value *>(Write.Ptr);
    int64_t Offset = 0;
    unsigned Frame = Calls.size();
    // Project a destination up the inspected callees into the original
    // caller. We do not climb above that caller to refine its argument aliases.
    while (RemainingInstructions) {
      --RemainingInstructions;
      Ptr = simplifyPointer(Ptr, Offset);
      if (!Ptr)
        return false;
      if (isa<GlobalVariable>(Ptr)) {
        if (PrivateObject)
          return true;
        break;
      }
      Function *F = Frame ? Calls[Frame - 1]->getCalledFunction() : Caller;
      if (auto *I = dyn_cast<Instruction>(Ptr)) {
        if (I->getFunction() != F)
          return false;
        if (Frame && isa<AllocaInst>(I))
          return true;
        if (auto *LI = dyn_cast<LoadInst>(I)) {
          if (Value *Stored = getSpilledPointer(*LI)) {
            Ptr = Stored;
            continue;
          }
        }
        if (Frame)
          return false;
        break;
      }
      if (auto *A = dyn_cast<Argument>(Ptr)) {
        if (A->getParent() != F)
          return false;
        if (!Frame)
          break;
        CallBase *CB = Calls[--Frame];
        if (A->getArgNo() >= CB->arg_size())
          return false;
        Ptr = CB->getArgOperand(A->getArgNo());
        continue;
      }
      return false;
    }
    if (!RemainingInstructions)
      return false;

    // Both pointers now belong to the original caller (or denote globals).
    // A base-wide query can prove distinct objects without materializing GEPs
    // in the IR. For a shared base, compare precise constant byte intervals.
    if (CallerAA.isNoAlias(MemoryLocation::getBeforeOrAfter(Ptr), Tracked))
      return true;
    int64_t TrackedOffset = 0;
    const Value *TrackedBase =
        simplifyPointer(const_cast<Value *>(Tracked.Ptr), TrackedOffset);
    if (TrackedBase != Ptr || !Write.Size.hasValue() ||
        !Write.Size.isPrecise() || !Tracked.Size.hasValue() ||
        !Tracked.Size.isPrecise() || Write.Size.getValue().isScalable() ||
        Tracked.Size.getValue().isScalable())
      return false;
    APInt WriteBegin(128, Offset, true);
    APInt TrackedBegin(128, TrackedOffset, true);
    return (WriteBegin + Write.Size.getValue().getFixedValue())
               .sle(TrackedBegin) ||
           (TrackedBegin + Tracked.Size.getValue().getFixedValue())
               .sle(WriteBegin);
  }

  bool inspectCall(CallBase &CB, const MemoryLocation &Tracked) {
    if (CB.hasOperandBundles() || CB.isInlineAsm())
      return false;
    if (CB.onlyReadsMemory())
      return true;
    // No code reached from this call can name an unexposed allocation when
    // no argument can reach it. This also excludes callback/synchronization
    // effects: publication would have failed the separate escape proof.
    if (PrivateObject && llvm::all_of(CB.args(), [&](const Use &Arg) {
          return !Arg->getType()->isPointerTy() ||
                 writeIsDisjoint(MemoryLocation::getBeforeOrAfter(Arg.get()),
                                 Tracked);
        }))
      return true;
    Function *F = CB.getCalledFunction();
    if (!F || F->isDeclaration() || F->isInterposable() || Calls.size() >= 16 ||
        (!Calls.empty() && F == Calls.front()->getFunction()) ||
        llvm::any_of(Calls, [F](CallBase *Outer) {
          return Outer->getCalledFunction() == F;
        }))
      return false;
    Calls.push_back(&CB);
    bool Safe = true;
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (!RemainingInstructions) {
          Safe = false;
          break;
        }
        --RemainingInstructions;
        if (!I.mayWriteToMemory())
          continue;
        if (auto *MI = dyn_cast<MemIntrinsic>(&I))
          Safe = writeIsDisjoint(MemoryLocation::getForDest(MI), Tracked);
        else if (auto *LI = dyn_cast<LifetimeIntrinsic>(&I))
          Safe = writeIsDisjoint(MemoryLocation::getBeforeOrAfter(
                                     LI->getArgOperand(LI->arg_size() - 1)),
                                 Tracked);
        else if (auto *Inner = dyn_cast<CallBase>(&I))
          Safe = inspectCall(*Inner, Tracked);
        else if (auto *SI = dyn_cast<StoreInst>(&I); SI && !SI->isAtomic())
          Safe = writeIsDisjoint(MemoryLocation::get(SI), Tracked);
        else
          // Ordered atomic accesses and fences may synchronize writes to
          // other locations. Disjoint operands alone do not discharge them.
          Safe = false;
        if (!Safe) {
          DEBUG(Logger::logs("proteus-pass")
                << "[Call write analysis]: Cannot prove memory effect harmless "
                   "for "
                << *Tracked.Ptr << " in " << F->getName() << ": " << I << "\n");
          break;
        }
      }
      if (!Safe)
        break;
    }
    Calls.pop_back();
    return Safe;
  }

public:
  CallWriteAnalysis(AAResults &AA, const DataLayout &Layout,
                    function_ref<DominatorTree &(Function &)> GetDT)
      : CallerAA(AA), DL(Layout), GetDT(GetDT) {}

  bool cannotModify(CallBase &CB, const MemoryLocation &Tracked) {
    RemainingInstructions = 4096;
    Caller = CB.getFunction();
    PrivateObject = false;
    if (auto *MI = dyn_cast<MemIntrinsic>(&CB))
      return writeIsDisjoint(MemoryLocation::getForDest(MI), Tracked);
    PrivateObject = hasPrivateOrigin(const_cast<Value *>(Tracked.Ptr), CB);
    // A nested call can use the isolation proof only if the root call cannot
    // publish the object's address before that nested call executes.
    if (PrivateObject) {
      Function *F = CB.getCalledFunction();
      for (unsigned I = 0; I < CB.arg_size(); ++I) {
        Value *Arg = CB.getArgOperand(I);
        if (Arg->getType()->isPointerTy() &&
            !writeIsDisjoint(MemoryLocation::getBeforeOrAfter(Arg), Tracked) &&
            (!F || F->isDeclaration() || F->isInterposable() ||
             I >= F->arg_size() || !doesNotEscape(F->getArg(I), nullptr))) {
          PrivateObject = false;
          break;
        }
      }
    }
    DEBUG(Logger::logs("proteus-pass")
          << "[Call write analysis]: private-object=" << PrivateObject
          << "; remaining-budget=" << RemainingInstructions
          << "; tracked=" << *Tracked.Ptr << "; call=" << CB << "\n");
    bool Safe = inspectCall(CB, Tracked);
    DEBUG(Logger::logs("proteus-pass")
          << "[Call write analysis]: "
          << (Safe ? "Proven harmless: " : "Possible clobber: ") << CB << "\n");
    return Safe;
  }
};

} // namespace proteus

#endif
