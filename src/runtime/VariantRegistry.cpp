// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "proteus/impl/VariantRegistry.h"
#include "proteus/Error.h"
#include "proteus/VariantSource.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace proteus {
namespace {
struct Source {
  uint64_t Type;
  unsigned Bits;
  detail::VariantCollector Collector;
  // Serialize mutable collector captures, even when launches overlap.
  std::recursive_mutex Mutex;
  Source(uint64_t Type, unsigned Bits, detail::VariantCollector Collector)
      : Type(Type), Bits(Bits), Collector(std::move(Collector)) {}
};
struct Registry {
  std::mutex Mutex;
  std::unordered_map<uint64_t, std::shared_ptr<Source>> Sources;
};
Registry &registry() {
  static Registry R;
  return R;
}
} // namespace

void detail::registerVariantSource(uint64_t Tag, uint64_t Type, unsigned Bits,
                                   VariantCollector Collector) {
  auto Entry = std::make_shared<Source>(Type, Bits, std::move(Collector));
  auto &R = registry();
  std::lock_guard<std::mutex> Lock(R.Mutex);
  auto It = R.Sources.find(Tag);
  if (It != R.Sources.end() &&
      (It->second->Type != Type || It->second->Bits != Bits))
    reportFatalError("Variant source type mismatch for tag " +
                     std::to_string(Tag));
  // Entry now owns the previous collector. Its captures are destroyed after
  // Lock is released, so their destructors may safely use the registry.
  R.Sources[Tag].swap(Entry);
}

void detail::unregisterVariantSource(uint64_t Tag) {
  std::shared_ptr<Source> Removed;
  auto &R = registry();
  std::lock_guard<std::mutex> Lock(R.Mutex);
  auto It = R.Sources.find(Tag);
  if (It != R.Sources.end()) {
    Removed = std::move(It->second);
    R.Sources.erase(It);
  }
}

VariantSnapshot collectVariantSources(const VariantManifest &Manifest) {
  std::map<uint64_t, std::shared_ptr<Source>> Sources;
  {
    auto &R = registry();
    std::lock_guard<std::mutex> Lock(R.Mutex);
    for (const auto &Site : Manifest) {
      auto It = R.Sources.find(Site.Tag);
      if (It == R.Sources.end())
        continue;
      if (It->second->Type != Site.Type || It->second->Bits != Site.Bits)
        reportFatalError("Variant source/device type mismatch for tag " +
                         std::to_string(Site.Tag));
      Sources.emplace(Site.Tag, It->second);
    }
  }
  VariantSnapshot Snapshot;
  for (const auto &[Tag, Entry] : Sources) {
    std::lock_guard<std::recursive_mutex> Lock(Entry->Mutex);
    auto Values = Entry->Collector();
    std::sort(Values.begin(), Values.end());
    Values.erase(std::unique(Values.begin(), Values.end()), Values.end());
    // Keep only effective sets. Missing, empty, and over-budget collections
    // share the generic cache entry. Never truncate an exhaustive set.
    bool Effective = false;
    for (const auto &Site : Manifest)
      Effective |=
          Site.Tag == Tag && !Values.empty() && Values.size() <= Site.Budget;
    if (Effective)
      Snapshot.emplace(Tag, std::move(Values));
  }
  return Snapshot;
}
} // namespace proteus
