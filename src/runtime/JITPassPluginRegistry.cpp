#include "proteus/impl/JITPassPluginRegistry.h"

#include "proteus/Error.h"
#include "proteus/impl/Hashing.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <utility>

namespace proteus {
namespace {

class JITPassPluginRegistry {
public:
  static JITPassPluginRegistry &instance() {
    static JITPassPluginRegistry Registry;
    return Registry;
  }

  void registerPlugin(const std::string &PluginPath,
                      std::optional<JITPassPluginInsertion> Insertion) {
    if (PluginPath.empty())
      reportFatalError("JIT pass plugin path must be non-empty");
    if (Insertion && Insertion->Pipeline.empty())
      reportFatalError("JIT pass plugin pipeline must be non-empty");

    JITPassPluginConfig Config{
        normalizePath(PluginPath), std::move(Insertion), {}};
    Config.Fingerprint = computeFingerprint(Config);

    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = std::find_if(Plugins.begin(), Plugins.end(),
                           [&](const JITPassPluginConfig &Entry) {
                             return Entry.Path == Config.Path &&
                                    Entry.Insertion == Config.Insertion;
                           });
    Generation.fetch_add(1, std::memory_order_release);
    if (It != Plugins.end()) {
      It->Fingerprint = std::move(Config.Fingerprint);
      return;
    }

    Plugins.push_back(std::move(Config));
    HasPlugins.store(true, std::memory_order_release);
  }

  void clear() {
    std::lock_guard<std::mutex> Lock(Mutex);
    Plugins.clear();
    HasPlugins.store(false, std::memory_order_release);
    Generation.fetch_add(1, std::memory_order_release);
  }

  uint64_t generation() const {
    return Generation.load(std::memory_order_acquire);
  }

  std::vector<JITPassPluginConfig> snapshot() {
    // Fast path: avoid taking the lock on the common case where no plugins are
    // registered, which happens on every JIT kernel launch.
    if (!HasPlugins.load(std::memory_order_acquire))
      return {};

    std::lock_guard<std::mutex> Lock(Mutex);
    return Plugins;
  }

private:
  static std::string normalizePath(const std::string &PluginPath) {
    llvm::SmallString<256> RealPath;
    if (!llvm::sys::fs::real_path(PluginPath, RealPath))
      return std::string(RealPath.str());

    return PluginPath;
  }

  static std::string computeFingerprint(const JITPassPluginConfig &Config) {
    auto BufOrErr = llvm::MemoryBuffer::getFile(Config.Path);
    if (!BufOrErr) {
      std::string Fingerprint = Config.Path + "|";
      if (Config.Insertion)
        Fingerprint += Config.Insertion->Pipeline;
      return Fingerprint;
    }

    return hashValue(BufOrErr.get()->getBuffer()).toString();
  }

  std::mutex Mutex;
  std::atomic<bool> HasPlugins{false};
  std::atomic<uint64_t> Generation{0};
  std::vector<JITPassPluginConfig> Plugins;
};

} // namespace

void registerJITPassPluginImpl(
    const std::string &PluginPath,
    std::optional<JITPassPluginInsertion> Insertion) {
  JITPassPluginRegistry::instance().registerPlugin(PluginPath,
                                                   std::move(Insertion));
}

void clearJITPassPluginsImpl() { JITPassPluginRegistry::instance().clear(); }

std::vector<JITPassPluginConfig> getJITPassPluginConfigs() {
  return JITPassPluginRegistry::instance().snapshot();
}

uint64_t getJITPassPluginGeneration() {
  return JITPassPluginRegistry::instance().generation();
}

} // namespace proteus
