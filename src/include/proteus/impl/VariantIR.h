// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROTEUS_VARIANT_IR_H
#define PROTEUS_VARIANT_IR_H

#include "proteus/Error.h"
#include "proteus/impl/VariantRegistry.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace proteus {
// Extraction runs before Clang's always-inliner. Expose the marker directly
// at the source expression so its SSA users include the controlling branch.
inline void inlineVariantWrappers(llvm::Module &M) {
  std::vector<llvm::CallInst *> Calls;
  for (auto &F : M)
    for (auto &I : llvm::instructions(F))
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I))
        if (auto *Callee = Call->getCalledFunction())
          if (Callee->getName().starts_with("_ZN7proteus12jit_variantsI") &&
              !Callee->isDeclaration())
            Calls.push_back(Call);
  for (auto *Call : Calls) {
    llvm::InlineFunctionInfo IFI;
    if (!llvm::InlineFunction(*Call, IFI).isSuccess())
      reportFatalError("Could not inline jit_variants expression marker");
  }
}

inline bool isVariantMarker(const llvm::Function &F) {
  return F.getName().contains("_ZN7proteus6detail21__jit_variants_markerI");
}

inline std::vector<llvm::CallInst *> variantMarkers(llvm::Module &M) {
  std::vector<llvm::CallInst *> Calls;
  for (auto &F : M)
    for (auto &I : llvm::instructions(F))
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I))
        if (auto *Callee = Call->getCalledFunction())
          if (isVariantMarker(*Callee))
            Calls.push_back(Call);
  return Calls;
}

inline VariantSite variantSite(const llvm::CallInst &Call) {
  if (Call.arg_size() != 5 || !Call.getType()->isIntegerTy())
    reportFatalError("Malformed jit_variants marker");
  uint64_t Args[4];
  for (unsigned I = 0; I != 4; ++I) {
    auto *C = llvm::dyn_cast<llvm::ConstantInt>(Call.getArgOperand(I + 1));
    if (!C)
      reportFatalError("jit_variants identity and budget must be constant");
    Args[I] = C->getZExtValue();
  }
  if (Args[2] != Call.getType()->getIntegerBitWidth())
    reportFatalError("jit_variants scalar width mismatch");
  return {Args[0], Args[1], static_cast<unsigned>(Args[2]),
          static_cast<unsigned>(Args[3])};
}

inline VariantManifest variantManifest(llvm::Module &M) {
  VariantManifest Manifest;
  for (auto *Call : variantMarkers(M))
    Manifest.push_back(variantSite(*Call));
  return Manifest;
}

// An opaque declaration keeps each marker intact through extraction cleanup,
// device linking, and inlining of its callers. Constant arguments are the
// embedded manifest, so they cannot become detached from cloned call sites.
inline void preserveVariantMarkers(llvm::Module &M) {
  for (auto &F : M) {
    if (!isVariantMarker(F))
      continue;
    F.deleteBody();
    F.setLinkage(llvm::GlobalValue::ExternalLinkage);
    F.setComdat(nullptr);
    F.setAttributes(llvm::AttributeList{});
  }
}

inline void lowerVariantMarkers(llvm::Module &M) {
  for (auto *Call : variantMarkers(M)) {
    Call->replaceAllUsesWith(Call->getArgOperand(0));
    Call->eraseFromParent();
  }
}
} // namespace proteus
#endif
