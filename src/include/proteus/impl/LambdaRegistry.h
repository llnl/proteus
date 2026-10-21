#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Logger.h"

#include <atomic>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <mutex>
#include <optional>

namespace proteus {

using namespace llvm;

// The LambdaRegistry stores the unique lambda type symbol and the values of Jit
// member variables in a map for retrieval by the Jit engines.
class LambdaRegistry {
public:
  static LambdaRegistry &instance() {
    static LambdaRegistry Singleton;
    return Singleton;
  }

  LambdaRegistry(const LambdaRegistry &) = delete;
  LambdaRegistry &operator=(const LambdaRegistry &) = delete;
  LambdaRegistry(LambdaRegistry &&) = delete;
  LambdaRegistry &operator=(LambdaRegistry &&) = delete;

  using JitVariantMap = DenseMap<int32_t, RuntimeConstant>;
  using JitVariantVec = SmallVector<JitVariantMap, 4>;
  struct DeviceLaunchInfo {
    SmallVector<uint64_t, 4> LambdaCalleeInfo;
    LambdaCallsiteRuntimeConstantsMap CallsiteRuntimeConstants;
  };

  std::optional<JitVariantMap> getHostJitVariant(uint64_t FunctorID) const {
    const auto &CurrentVariants = currentHostJitVariables();
    if (CurrentVariants.empty())
      return std::nullopt;

    auto It = CurrentVariants.find(FunctorID);
    if (It == CurrentVariants.end() || It->second.empty())
      return std::nullopt;

    return It->second;
  }

  void appendHostJitVariable(uint64_t FunctorID, const RuntimeConstant &RC) {
    pendingHostJitVariables()[FunctorID][RC.Pos] = RC;
  }

  void commitHostJitVariables(uint64_t FunctorID) {
    auto &PendingVariants = pendingHostJitVariables();
    auto PendingIt = PendingVariants.find(FunctorID);
    if (PendingIt == PendingVariants.end())
      return;
    if (PendingIt->second.empty())
      return;

    // Swap and clear instead of moving so neither map frees and reallocates its
    // buckets on every register_lambda call.
    auto &Current = currentHostJitVariables()[FunctorID];
    std::swap(Current, PendingIt->second);
    PendingIt->second.clear();
  }

  void eraseHostJitVariables(uint64_t FunctorID) {
    currentHostJitVariables().erase(FunctorID);
  }

  bool emptyHost() const { return currentHostJitVariables().empty(); }

  void beginDeviceLaunch() {
    auto &PendingLaunchInfo = pendingDeviceLaunchInfo();
    auto &CurrentLaunchInfo = currentDeviceLaunchInfo();
    PendingLaunchInfo.LambdaCalleeInfo.clear();
    PendingLaunchInfo.CallsiteRuntimeConstants.clear();
    CurrentLaunchInfo.LambdaCalleeInfo.clear();
    CurrentLaunchInfo.CallsiteRuntimeConstants.clear();
  }

  void appendDeviceCallsiteRuntimeConstant(uint64_t LambdaID,
                                           uint32_t CallsiteIndex,
                                           const RuntimeConstant &RC) {
    auto &PendingLaunchInfo = pendingDeviceLaunchInfo();
    auto &RCVec = PendingLaunchInfo.CallsiteRuntimeConstants[CallsiteIndex];
    RCVec.push_back(RC);
    if (llvm::find(PendingLaunchInfo.LambdaCalleeInfo, LambdaID) ==
        PendingLaunchInfo.LambdaCalleeInfo.end()) {
      PendingLaunchInfo.LambdaCalleeInfo.push_back(LambdaID);
    }
  }

  void finalizeDeviceLaunch() {
    auto &PendingLaunchInfo = pendingDeviceLaunchInfo();
    auto &CurrentLaunchInfo = currentDeviceLaunchInfo();
    for (auto &KV : PendingLaunchInfo.CallsiteRuntimeConstants) {
      llvm::sort(KV.second,
                 [](const RuntimeConstant &L, const RuntimeConstant &R) {
                   return L.Pos < R.Pos;
                 });
    }
    std::swap(CurrentLaunchInfo, PendingLaunchInfo);
    PendingLaunchInfo.LambdaCalleeInfo.clear();
    PendingLaunchInfo.CallsiteRuntimeConstants.clear();
  }

  // Valid until clearDeviceLaunchInfo or the next launch.
  const DeviceLaunchInfo &peekDeviceLaunchInfo() const {
    return currentDeviceLaunchInfo();
  }

  void clearDeviceLaunchInfo() {
    auto &CurrentLaunchInfo = currentDeviceLaunchInfo();
    CurrentLaunchInfo.LambdaCalleeInfo.clear();
    CurrentLaunchInfo.CallsiteRuntimeConstants.clear();
  }

  void populateLambdaRegistrationCodeCache(void *Kernel,
                                           void *RegistrationFunc) {
    std::lock_guard<std::mutex> Lock(DeviceRegistrationMutex);
    KernelToLambdaRegistration[Kernel] = RegistrationFunc;
    HasDeviceRegistrations.store(true, std::memory_order_release);
  }

  void invokeRegisterLambdaConstants(void *Kernel, void **Args) {
    void *RegistrationFunc = nullptr;
    if (HasDeviceRegistrations.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> Lock(DeviceRegistrationMutex);
      auto It = KernelToLambdaRegistration.find(Kernel);
      if (It != KernelToLambdaRegistration.end())
        RegistrationFunc = It->second;
    }
    if (!RegistrationFunc) {
      beginDeviceLaunch();
      return;
    }
    auto RegisterFunc = reinterpret_cast<void (*)(void **)>(RegistrationFunc);
    RegisterFunc(Args);
  }

private:
  explicit LambdaRegistry() = default;

  static DenseMap<uint64_t, JitVariantMap> &pendingHostJitVariables() {
    static thread_local DenseMap<uint64_t, JitVariantMap> PendingHostJitVars;
    return PendingHostJitVars;
  }

  static DenseMap<uint64_t, JitVariantMap> &currentHostJitVariables() {
    static thread_local DenseMap<uint64_t, JitVariantMap> CurrentHostJitVars;
    return CurrentHostJitVars;
  }

  static DeviceLaunchInfo &pendingDeviceLaunchInfo() {
    static thread_local DeviceLaunchInfo PendingLaunchInfo;
    return PendingLaunchInfo;
  }

  static DeviceLaunchInfo &currentDeviceLaunchInfo() {
    static thread_local DeviceLaunchInfo CurrentLaunchInfo;
    return CurrentLaunchInfo;
  }

  mutable std::mutex DeviceRegistrationMutex;
  std::atomic<bool> HasDeviceRegistrations{false};
  DenseMap<void *, void *> KernelToLambdaRegistration;
};

} // namespace proteus

#endif
