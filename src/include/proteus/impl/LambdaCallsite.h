#ifndef PROTEUS_LAMBDA_CALLSITE_H
#define PROTEUS_LAMBDA_CALLSITE_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>

#include <optional>

namespace proteus {

using namespace llvm;

struct LambdaKernelArgLocation {
  uint32_t KernelArgIndex;
  int64_t Offset;
  RuntimeConstantType StorageType = RuntimeConstantType::NONE;
};

using LambdaKernelArgLocationVec = SmallVector<LambdaKernelArgLocation, 4>;
using LambdaKernelArgLocationMap =
    DenseMap<uint64_t, LambdaKernelArgLocationVec>;
using LambdaCallsiteLocationMap =
    DenseMap<uint64_t, DenseMap<uint32_t, LambdaKernelArgLocation>>;
using LambdaCallsiteRuntimeConstants = SmallVector<RuntimeConstant, 8>;
using LambdaCallsiteRuntimeConstantsMap =
    DenseMap<uint64_t, LambdaCallsiteRuntimeConstants>;

inline constexpr char LambdaCallsiteMetadataName[] = "proteus.lambda_callsite";
inline constexpr char LambdaSchemaMetadataName[] = "proteus.lambda_schema";

inline void setLambdaCallsiteMetadata(CallBase &CB, uint64_t LambdaID,
                                      uint32_t CallsiteIndex) {
  LLVMContext &Ctx = CB.getContext();
  auto *LambdaMD = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt64Ty(Ctx), LambdaID));
  auto *IndexMD = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(Ctx), CallsiteIndex));
  CB.setMetadata(LambdaCallsiteMetadataName,
                 MDNode::get(Ctx, {LambdaMD, IndexMD}));
}

inline std::optional<std::pair<uint64_t, uint32_t>>
getLambdaCallsiteMetadata(const CallBase &CB) {
  auto *Node = CB.getMetadata(LambdaCallsiteMetadataName);
  if (!Node || Node->getNumOperands() < 2)
    return std::nullopt;

  auto *LambdaCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(0));
  auto *IndexCAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(1));
  auto *LambdaCI =
      LambdaCAM ? dyn_cast<ConstantInt>(LambdaCAM->getValue()) : nullptr;
  auto *IndexCI =
      IndexCAM ? dyn_cast<ConstantInt>(IndexCAM->getValue()) : nullptr;
  if (!LambdaCI || !IndexCI)
    return std::nullopt;

  return std::make_pair(LambdaCI->getZExtValue(),
                        static_cast<uint32_t>(IndexCI->getZExtValue()));
}

inline std::optional<uint32_t>
getLambdaCallsiteIndex(const CallBase &CB, uint64_t ExpectedLambdaID) {
  auto Metadata = getLambdaCallsiteMetadata(CB);
  if (!Metadata || Metadata->first != ExpectedLambdaID)
    return std::nullopt;
  return Metadata->second;
}

inline LambdaKernelArgLocationVec getOrderedLambdaKernelArgLocations(
    const DenseMap<uint32_t, LambdaKernelArgLocation> &CallsiteLocations) {
  LambdaKernelArgLocationVec Ordered;
  if (CallsiteLocations.empty())
    return Ordered;

  SmallVector<uint32_t, 8> Indices;
  Indices.reserve(CallsiteLocations.size());
  for (const auto &KV : CallsiteLocations)
    Indices.push_back(KV.first);
  llvm::sort(Indices);

  Ordered.reserve(Indices.size());
  for (uint32_t Index : Indices) {
    auto It = CallsiteLocations.find(Index);
    if (It == CallsiteLocations.end())
      continue;
    Ordered.push_back(It->second);
  }

  return Ordered;
}

} // namespace proteus

#endif
