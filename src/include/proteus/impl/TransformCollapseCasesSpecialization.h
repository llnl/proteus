// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROTEUS_TRANSFORM_COLLAPSE_CASES_SPECIALIZATION_H
#define PROTEUS_TRANSFORM_COLLAPSE_CASES_SPECIALIZATION_H

#include "proteus/impl/Config.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/VariantIR.h"
#include <algorithm>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Utils/Local.h>

namespace proteus {
class TransformCollapseCasesSpecialization {
  // Evaluate only scalar SSA expressions whose inputs are known. Other loads,
  // calls, PHIs, and conditions remain dynamic. Depth bounds compile time.
  static llvm::ConstantInt *evaluate(llvm::Value *V, llvm::CallInst *Marker,
                                     uint64_t Raw, unsigned Depth = 0) {
    using namespace llvm;
    if (V == Marker)
      return cast<ConstantInt>(ConstantInt::get(V->getType(), Raw));
    if (auto *C = dyn_cast<ConstantInt>(V))
      return C;
    if (Depth > 32)
      return nullptr;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !I->getType()->isIntegerTy())
      return nullptr;
    auto Eval = [&](Value *Operand) {
      return evaluate(Operand, Marker, Raw, Depth + 1);
    };
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      if (auto *C = Eval(Cast->getOperand(0)))
        return dyn_cast<ConstantInt>(
            ConstantFoldCastOperand(Cast->getOpcode(), C, Cast->getType(),
                                    I->getModule()->getDataLayout()));
    } else if (auto *Cmp = dyn_cast<ICmpInst>(I)) {
      auto *L = Eval(Cmp->getOperand(0));
      auto *R = Eval(Cmp->getOperand(1));
      if (L && R)
        return ConstantInt::getBool(
            I->getContext(), ICmpInst::compare(L->getValue(), R->getValue(),
                                               Cmp->getPredicate()));
    } else if (auto *Op = dyn_cast<BinaryOperator>(I)) {
      auto *L = Eval(Op->getOperand(0));
      auto *R = Eval(Op->getOperand(1));
      if (L && R)
        return dyn_cast<ConstantInt>(ConstantFoldBinaryOpOperands(
            Op->getOpcode(), L, R, I->getModule()->getDataLayout()));
    }
    return nullptr;
  }

  static void constrain(llvm::CallInst *Marker,
                        const std::vector<uint64_t> &Values) {
    using namespace llvm;
    Function &F = *Marker->getFunction();
    DominatorTree DT(F);
    SmallVector<Instruction *> Work;
    for (auto &I : instructions(F))
      if (isa<SwitchInst>(I) || isa<ICmpInst>(I))
        Work.push_back(&I);

    // Compute all facts before editing the CFG or comparisons used by DT.
    std::vector<std::pair<Instruction *, std::vector<uint64_t>>> Facts;
    for (auto *I : Work) {
      std::vector<uint64_t> Local = Values;
      for (auto &BB : F) {
        auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
        if (!Br || !Br->isConditional() || !DT.dominates(Marker, Br))
          continue;
        for (unsigned Edge = 0; Edge != 2; ++Edge) {
          if (!DT.dominates(BasicBlockEdge(&BB, Br->getSuccessor(Edge)),
                            I->getParent()))
            continue;
          Local.erase(std::remove_if(Local.begin(), Local.end(),
                                     [&](uint64_t V) {
                                       auto *C = evaluate(Br->getCondition(),
                                                          Marker, V);
                                       return C && C->isZero() != (Edge == 1);
                                     }),
                      Local.end());
        }
      }
      Facts.emplace_back(I, std::move(Local));
    }
    for (auto &[I, Local] : Facts) {
      if (Local.empty())
        continue;
      Value *Selector =
          isa<SwitchInst>(I) ? cast<SwitchInst>(I)->getCondition() : I;
      SmallVector<ConstantInt *> Results;
      for (auto V : Local) {
        auto *C = evaluate(Selector, Marker, V);
        if (!C) {
          Results.clear();
          break;
        }
        Results.push_back(C);
      }
      if (Results.empty())
        continue;
      if (auto *Switch = dyn_cast<SwitchInst>(I)) {
        unsigned OriginalCases = Switch->getNumCases();
        bool NeedsDefault = false;
        for (auto *C : Results)
          NeedsDefault |= Switch->findCaseValue(C) == Switch->case_default();
        for (auto It = Switch->case_begin(); It != Switch->case_end();) {
          if (llvm::is_contained(Results, It->getCaseValue())) {
            ++It;
            continue;
          }
          It->getCaseSuccessor()->removePredecessor(Switch->getParent());
          It = Switch->removeCase(It);
        }
        if (!NeedsDefault) {
          Switch->getDefaultDest()->removePredecessor(Switch->getParent());
          auto *Unreachable =
              BasicBlock::Create(F.getContext(), "variants.unreachable", &F);
          new UnreachableInst(F.getContext(), Unreachable);
          Switch->setDefaultDest(Unreachable);
        }
        // Branch weights describe the original dispatch, including default.
        Switch->setMetadata(LLVMContext::MD_prof, nullptr);
        if (OriginalCases != Switch->getNumCases()) {
          if (Config::get().traceSpecializations()) {
            SmallString<128> S;
            raw_svector_ostream OS(S);
            OS << "[CollapseCasesSpec] Switch cases " << OriginalCases << " -> "
               << Switch->getNumCases() << "\n";
            Logger::trace(OS.str());
          }
        }
      } else if (llvm::all_of(Results, [&](ConstantInt *C) {
                   return C == Results.front();
                 })) {
        I->replaceAllUsesWith(Results.front());
      }
    }
    // Give subsequent LLVM passes the same exhaustive fact for expressions
    // outside our deliberately small evaluator (e.g. after further inlining).
    IRBuilder<> B(Marker->getNextNode());
    Value *Allowed = B.getFalse();
    for (auto V : Values)
      Allowed = B.CreateOr(
          Allowed,
          B.CreateICmpEQ(Marker, ConstantInt::get(Marker->getType(), V)));
    B.CreateAssumption(Allowed);
  }

public:
  static void transform(llvm::Module &M, const VariantSnapshot &Snapshot) {
    using namespace llvm;
    auto Markers = variantMarkers(M);
    if (Markers.empty())
      return;
    FunctionAnalysisManager FAM;
    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    SmallPtrSet<Function *, 8> Prepared;
    for (auto *Call : Markers) {
      auto *F = Call->getFunction();
      if (!Prepared.insert(F).second)
        continue;
      // Scalarize local spills, including AMDGPU private-to-generic pointer
      // casts, so a local variable does not hide the marker's SSA users.
      // Run directly to support extracted optnone functions too.
      auto PA = SROAPass(SROAOptions::PreserveCFG).run(*F, FAM);
      FAM.invalidate(*F, PA);
    }
    for (auto *Call : Markers) {
      auto Site = variantSite(*Call);
      auto It = Snapshot.find(Site.Tag);
      Value *Replacement = Call->getArgOperand(0);
      size_t Count = It == Snapshot.end() ? 0 : It->second.size();
      if (Count && Count <= Site.Budget) {
        if (Count == 1)
          Replacement = ConstantInt::get(Call->getType(), It->second.front());
        else
          constrain(Call, It->second);
      }
      if (Config::get().traceSpecializations()) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[CollapseCasesSpec] Tag " << Site.Tag << " Values " << Count
           << " Budget " << Site.Budget << " "
           << (Count && Count <= Site.Budget ? "specialized" : "generic")
           << "\n";
        Logger::trace(OS.str());
      }
      Call->replaceAllUsesWith(Replacement);
      Call->eraseFromParent();
    }
  }
};
} // namespace proteus
#endif
