//===-- LaunchMemo.h -- Per-kernel memo of recent specializations --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_LAUNCH_MEMO_H
#define PROTEUS_LAUNCH_MEMO_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/CoreDevice.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

namespace proteus {

using namespace llvm;

// Recent specializations of one kernel, keyed by the raw launch inputs that
// feed the specialization hash. Keys have a fixed per-kernel size.
class LaunchMemo {
public:
  static constexpr unsigned NumSlots = 4;

  // Disabled.
  LaunchMemo() = default;

  // Disabled if any runtime constant is not a scalar.
  LaunchMemo(ArrayRef<RuntimeConstantInfo *> RCInfoArray, bool KeyGridDim)
      : KeyGridDim(KeyGridDim) {
    if (!all_of(RCInfoArray, [](const RuntimeConstantInfo *RCInfo) {
          return isScalarRuntimeConstantType(RCInfo->ArgInfo.Type);
        }))
      return;

    PrefixLen =
        sizeof(dim3) + (KeyGridDim ? sizeof(dim3) : 0) + sizeof(uint64_t);
    for (const auto *RCInfo : RCInfoArray) {
      RuntimeConstantType Type = RCInfo->ArgInfo.Type;
      uint32_t Size = Type == RuntimeConstantType::PTR ? sizeof(void *)
                                                       : getSizeInBytes(Type);
      Fields.push_back({static_cast<uint32_t>(RCInfo->ArgInfo.Pos), Size});
      PrefixLen += Size;
    }
    Enabled = true;
  }

  LaunchMemo(LaunchMemo &&) = default;
  LaunchMemo &operator=(LaunchMemo &&) = default;

  bool isEnabled() const { return Enabled; }

  // Serializes the launch inputs. The result is valid until this thread's
  // next encode().
  ArrayRef<char>
  encode(void **KernelArgs, dim3 GridDim, dim3 BlockDim,
         uint64_t PluginGeneration,
         const LambdaRegistry::DeviceLaunchInfo &LaunchInfo) const {
    assert(Enabled && "Expected enabled launch memo");

    size_t Len = PrefixLen;
    for (const auto &[CallsiteIndex, RCs] : LaunchInfo.CallsiteRuntimeConstants)
      Len += sizeof(uint64_t) + RCs.size() * LambdaRCSize;
    if (KeyLen != 0 && Len != KeyLen)
      reportFatalError("Launch memo key size changed");

    auto &Scratch = scratch();
    if (Scratch.size() < Len)
      Scratch.resize_for_overwrite(Len);

    char *Out = Scratch.data();
    for (const ArgField &F : Fields)
      Out = putField(Out, KernelArgs[F.Pos], F.Size);
    Out = put(Out, BlockDim);
    if (KeyGridDim)
      Out = put(Out, GridDim);
    Out = put(Out, PluginGeneration);
    for (const auto &[CallsiteIndex, RCs] :
         LaunchInfo.CallsiteRuntimeConstants) {
      Out = put(Out, static_cast<uint64_t>(CallsiteIndex));
      for (const auto &RC : RCs) {
        assert(isScalarRuntimeConstantType(RC.Type) &&
               "Expected scalar lambda runtime constant");
        Out = put(Out, RC.Type);
        Out = put(Out, RC.Pos);
        Out = put(Out, RC.Offset);
        Out = put(Out, RC.Value);
      }
    }
    assert(Out == Scratch.data() + Len && "Launch memo key size mismatch");

    return {Scratch.data(), Len};
  }

  // Returns the memoized kernel function, or nullptr.
  void *lookup(ArrayRef<char> Key) const {
    for (unsigned I = 0; I < NumSlots; ++I)
      if (Funcs[I] && std::memcmp(slot(I), Key.data(), KeyLen) == 0)
        return Funcs[I];
    return nullptr;
  }

  // Replaces the oldest entry.
  void insert(ArrayRef<char> Key, void *KernelFunc) {
    assert(Enabled && "Expected enabled launch memo");
    if (KeyLen == 0) {
      KeyLen = Key.size();
      Keys.reset(new char[NumSlots * size_t(KeyLen)]);
    } else if (Key.size() != KeyLen) {
      reportFatalError("Launch memo key size changed");
    }

    std::memcpy(slot(Next), Key.data(), KeyLen);
    Funcs[Next] = KernelFunc;
    Next = (Next + 1) % NumSlots;
  }

private:
  struct ArgField {
    uint32_t Pos;
    uint32_t Size;
  };

  // Type, Pos, Offset, Value.
  static constexpr size_t LambdaRCSize = sizeof(RuntimeConstantType) +
                                         2 * sizeof(int32_t) +
                                         sizeof(RuntimeConstantValue);

  static SmallVectorImpl<char> &scratch() {
    static thread_local SmallVector<char, 128> Scratch;
    return Scratch;
  }

  template <typename T> static char *put(char *Out, const T &V) {
    std::memcpy(Out, &V, sizeof(T));
    return Out + sizeof(T);
  }

  static char *putField(char *Out, const void *In, uint32_t Size) {
    switch (Size) {
    case 1:
      std::memcpy(Out, In, 1);
      break;
    case 4:
      std::memcpy(Out, In, 4);
      break;
    case 8:
      std::memcpy(Out, In, 8);
      break;
    case 16:
      std::memcpy(Out, In, 16);
      break;
    default:
      std::memcpy(Out, In, Size);
    }
    return Out + Size;
  }

  char *slot(unsigned I) const { return Keys.get() + size_t(I) * KeyLen; }

  SmallVector<ArgField, 6> Fields;
  // NumSlots * KeyLen, set on first insert.
  std::unique_ptr<char[]> Keys;
  // nullptr marks an empty slot.
  std::array<void *, NumSlots> Funcs{};
  // Kernel RCs, dims, plugin generation.
  uint32_t PrefixLen = 0;
  // 0 until the first insert.
  uint32_t KeyLen = 0;
  uint8_t Next = 0;
  bool KeyGridDim = false;
  bool Enabled = false;
};

} // namespace proteus

#endif
