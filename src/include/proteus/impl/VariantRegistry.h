// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROTEUS_VARIANT_REGISTRY_H
#define PROTEUS_VARIANT_REGISTRY_H

#include <cstdint>
#include <map>
#include <vector>

namespace proteus {
struct VariantSite {
  uint64_t Tag;
  uint64_t Type;
  unsigned Bits;
  unsigned Budget;
};
using VariantManifest = std::vector<VariantSite>;
using VariantSnapshot = std::map<uint64_t, std::vector<uint64_t>>;

// Each collector is invoked once, outside the registry lock. Returned storage
// belongs to this launch and can be moved/copied to an asynchronous task.
VariantSnapshot collectVariantSources(const VariantManifest &Manifest);
} // namespace proteus
#endif
