#ifndef PROTEUS_POINTERCLOBBERANALYSIS_H
#define PROTEUS_POINTERCLOBBERANALYSIS_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>

#include <cstdint>
#include <optional>

namespace proteus {

enum class PointerClobberKind {
  Value,
  Incoming,
  NoClobber,
  Cycle,
  Ambiguous,
  Unknown
};

// A clobber query either resolves the value stored at a byte offset or returns
// the exact MemorySSA-selected instruction for a provenance visitor to
// interpret. ClobberPointer is the pointer operand through which that
// instruction accesses the tracked storage, and ClobberOffset is relative to
// that operand.
struct PointerClobberResult {
  PointerClobberKind Kind = PointerClobberKind::Unknown;
  llvm::Value *V = nullptr;
  int64_t Offset = 0;
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
  llvm::Instruction *ClobberingInstruction = nullptr;
  llvm::Value *ClobberPointer = nullptr;
  int64_t ClobberOffset = 0;
};

// A write discovered while following the relevant uses of a newly encountered
// pointer definition. Pointer is the operand through which the instruction
// accesses the tracked storage, and TargetOffset is relative to that operand.
struct PointerClobberCandidate {
  llvm::Instruction *I = nullptr;
  llvm::Value *Pointer = nullptr;
  int64_t TargetOffset = 0;
};

class PointerClobberAnalysis {
public:
  virtual ~PointerClobberAnalysis() = default;

  virtual PointerClobberResult
  resolve(llvm::Value *Ptr, llvm::Instruction &UseBoundary,
          int64_t TargetOffset,
          llvm::ArrayRef<PointerClobberCandidate> Candidates = {}) = 0;
};

} // namespace proteus

#endif
