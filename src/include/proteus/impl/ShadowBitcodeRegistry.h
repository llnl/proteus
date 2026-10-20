//===-- ShadowBitcodeRegistry.h -- Shadow device bitcode registry --------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When NVCC is the AOT compiler, its fatbinary carries SASS/PTX only, so the
// runtime cannot recover LLVM bitcode from the loaded CUmodule. A shadow
// device-only Clang compile of the same translation unit runs the Proteus
// device pipeline and produces the bitcode instead; the build embeds it in
// the host binary and registers it here at program startup, keyed by the
// same module ID the host pass recorded at the fatbinary registration site.
// The CUDA engine consults this registry when a module ID's bitcode global
// is absent from the CUmodule. See docs/dev/design/nvcc-interop.md.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_SHADOWBITCODEREGISTRY_H
#define PROTEUS_SHADOWBITCODEREGISTRY_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include <llvm/ADT/StringRef.h>

namespace proteus {

class ShadowBitcodeRegistry {
public:
  static ShadowBitcodeRegistry &instance() {
    static ShadowBitcodeRegistry Registry;
    return Registry;
  }

  // Called from emitted static constructors; Bitcode must have static storage
  // duration (the registry does not copy it).
  void registerBitcode(const char *ModuleId, const void *Bitcode,
                       uint64_t Size) {
    std::lock_guard<std::mutex> Lock(Mutex);
    Modules.emplace(std::string(ModuleId),
                    llvm::StringRef(static_cast<const char *>(Bitcode), Size));
  }

  // Returns the registered bitcode for ModuleId, or an empty StringRef.
  llvm::StringRef lookup(llvm::StringRef ModuleId) {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Modules.find(ModuleId.str());
    if (It == Modules.end())
      return {};
    return It->second;
  }

  bool empty() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Modules.empty();
  }

private:
  ShadowBitcodeRegistry() = default;
  ShadowBitcodeRegistry(const ShadowBitcodeRegistry &) = delete;

  std::mutex Mutex;
  std::unordered_map<std::string, llvm::StringRef> Modules;
};

} // namespace proteus

#endif
