#ifndef PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H
#define PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H

#include "proteus/Init.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace proteus {

struct JITPassPluginInsertion {
  std::string Pipeline;
  JITPassPluginPosition Position;

  bool operator==(const JITPassPluginInsertion &Other) const {
    return Pipeline == Other.Pipeline && Position == Other.Position;
  }
};

struct JITPassPluginConfig {
  std::string Path;
  std::optional<JITPassPluginInsertion> Insertion;
  std::string Fingerprint;
};

void registerJITPassPluginImpl(const std::string &PluginPath,
                               std::optional<JITPassPluginInsertion> Insertion);
void clearJITPassPluginsImpl();
std::vector<JITPassPluginConfig> getJITPassPluginConfigs();
// Changes whenever the registered plugins change.
uint64_t getJITPassPluginGeneration();

} // namespace proteus

#endif
