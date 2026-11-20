#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Logger.h"

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

    // Member-wise, so storage is kept; std::swap would deallocate.
    void swap(DeviceLaunchInfo &Other) {
      LambdaCalleeInfo.swap(Other.LambdaCalleeInfo);
      CallsiteRuntimeConstants.swap(Other.CallsiteRuntimeConstants);
    }

    void clear() {
      LambdaCalleeInfo.clear();
      CallsiteRuntimeConstants.clear();
    }
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

    currentHostJitVariables()[FunctorID] = std::move(PendingIt->second);
    PendingVariants.erase(PendingIt);
  }

  void eraseHostJitVariables(uint64_t FunctorID) {
    currentHostJitVariables().erase(FunctorID);
  }

  bool emptyHost() const { return currentHostJitVariables().empty(); }

  void beginDeviceLaunch() {
    pendingDeviceLaunchInfo().clear();
    currentDeviceLaunchInfo().clear();
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
    CurrentLaunchInfo.swap(PendingLaunchInfo);
    PendingLaunchInfo.clear();
  }

  // Valid until clearDeviceLaunchInfo or the next launch.
  const DeviceLaunchInfo &peekDeviceLaunchInfo() const {
    return currentDeviceLaunchInfo();
  }

  void clearDeviceLaunchInfo() { currentDeviceLaunchInfo().clear(); }

  // Copies rather than moves, so the registry keeps its storage.
  void
  takeDeviceLaunchInfo(SmallVector<uint64_t> &LambdaCalleeInfo,
                       LambdaCallsiteRuntimeConstantsMap &LambdaJitValuesMap) {
    auto &CurrentLaunchInfo = currentDeviceLaunchInfo();
    LambdaCalleeInfo.assign(CurrentLaunchInfo.LambdaCalleeInfo.begin(),
                            CurrentLaunchInfo.LambdaCalleeInfo.end());
    LambdaJitValuesMap = CurrentLaunchInfo.CallsiteRuntimeConstants;
    CurrentLaunchInfo.clear();
  }

  void populateLambdaRegistrationCodeCache(void *Kernel,
                                           void *RegistrationFunc) {
    std::lock_guard<std::mutex> Lock(DeviceRegistrationMutex);
    KernelToLambdaRegistration[Kernel] = RegistrationFunc;
  }

  void invokeRegisterLambdaConstants(void *Kernel, void **Args) {
    void *RegistrationFunc = nullptr;
    {
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
  DenseMap<void *, void *> KernelToLambdaRegistration;
};

} // namespace proteus

#endif
