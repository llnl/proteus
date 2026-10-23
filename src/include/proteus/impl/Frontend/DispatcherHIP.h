#ifndef PROTEUS_FRONTEND_DISPATCHER_HIP_H
#define PROTEUS_FRONTEND_DISPATCHER_HIP_H

#if PROTEUS_ENABLE_HIP

#include "proteus/Error.h"
#include "proteus/impl/Frontend/DispatcherDevice.h"
#include "proteus/impl/Frontend/HIPToolchain.h"
#include "proteus/impl/JitEngineDeviceHIP.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

class DispatcherHIP : public DispatcherDevice<JitEngineDeviceHIP> {
public:
  static DispatcherHIP &instance() {
    static DispatcherHIP D{"DispatcherHIP", JitEngineDeviceHIP::instance()};
    return D;
  }

  DispatcherHIP(const std::string &Label, JitEngineDeviceHIP &Jit)
      : DispatcherDevice(Label, TargetModelType::HIP, Jit) {}

protected:
  // Link ROCm device libraries (ocml/ockl + oclc config) so HIPRTC can
  // resolve __ocml_* calls produced by math lowering.
  void linkDeviceLibraries(Module &M) override {
    TIMESCOPE(DispatcherHIP, linkDeviceLibraries);
    const auto &Toolchain = resolveHIPToolchain();

    auto LoadBitcode = [&](const llvm::SmallString<256> &Path) {
      auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
      if (!BufferOrErr || !BufferOrErr.get())
        reportFatalError("DispatchHIP: failed to read ROCm bitcode file: " +
                         Path.str().str() + " (" + Toolchain.Origin + ")");
      auto Parsed = llvm::parseBitcodeFile(
          BufferOrErr->get()->getMemBufferRef(), M.getContext());
      if (!Parsed)
        reportFatalError("DispatchHIP: failed to parse ROCm bitcode file: " +
                         Path.str().str() + " (" + Toolchain.Origin + ")");
      return std::move(Parsed.get());
    };

    auto AppendBitcodePath =
        [&](llvm::SmallVectorImpl<llvm::SmallString<256>> &Paths,
            llvm::StringRef Filename) {
          llvm::SmallString<256> Path{Toolchain.DeviceLibDir};
          llvm::sys::path::append(Path, Filename);
          Paths.push_back(std::move(Path));
        };

    auto Exists = [&](llvm::StringRef Filename) -> bool {
      llvm::SmallString<256> Path{Toolchain.DeviceLibDir};
      llvm::sys::path::append(Path, Filename);
      return llvm::sys::fs::exists(Path);
    };

    auto AppendExistingBitcodePath =
        [&](llvm::SmallVectorImpl<llvm::SmallString<256>> &Paths,
            llvm::StringRef Filename) {
          if (Exists(Filename))
            AppendBitcodePath(Paths, Filename);
        };

    auto PickFirstExisting =
        [&](std::initializer_list<llvm::StringRef> Candidates)
        -> llvm::StringRef {
      for (auto C : Candidates) {
        if (Exists(C))
          return C;
      }
      return {};
    };

    llvm::SmallVector<llvm::SmallString<256>, 8> LibsToLink;
    AppendBitcodePath(LibsToLink, "ocml.bc");
    AppendBitcodePath(LibsToLink, "ockl.bc");

    // ABI: prefer the newest available.
    if (auto Abi = PickFirstExisting({"oclc_abi_version_600.bc",
                                      "oclc_abi_version_500.bc",
                                      "oclc_abi_version_400.bc"});
        !Abi.empty()) {
      AppendBitcodePath(LibsToLink, Abi);
    } else {
      reportFatalError(
          std::string("DispatchHIP: missing oclc ABI bitcode under ") +
          Toolchain.DeviceLibDir + " (" + Toolchain.Origin +
          "; expected oclc_abi_version_{600,500,400}.bc)");
    }

    // ISA: derived from device arch like "gfx90a" -> "90a".
    const std::string DeviceArch = Jit.getDeviceArch().str();
    if (!llvm::StringRef{DeviceArch}.starts_with("gfx"))
      reportFatalError("DispatchHIP: unexpected HIP device arch: " +
                       DeviceArch);
    const llvm::StringRef IsaSuffix = llvm::StringRef{DeviceArch}.drop_front(3);
    const std::string IsaFile = ("oclc_isa_version_" + IsaSuffix + ".bc").str();
    if (!Exists(IsaFile))
      reportFatalError(std::string("DispatchHIP: missing ISA bitcode file ") +
                       IsaFile + " under " + Toolchain.DeviceLibDir + " (" +
                       Toolchain.Origin + "; DeviceArch=" + DeviceArch + ")");
    AppendBitcodePath(LibsToLink, IsaFile);

    // Math/FP mode defaults (safe defaults, can be revisited later).
    // These configuration libraries are optional across ROCm releases.
    // In particular, ROCm 10 does not ship the DAZ and correctly-rounded
    // sqrt variants, while clang simply omits them from its link command.
    AppendExistingBitcodePath(LibsToLink, "oclc_unsafe_math_off.bc");
    AppendExistingBitcodePath(LibsToLink, "oclc_finite_only_off.bc");
    AppendExistingBitcodePath(LibsToLink, "oclc_daz_opt_off.bc");
    AppendExistingBitcodePath(LibsToLink, "oclc_correctly_rounded_sqrt_on.bc");

    // Wavefront size selection: RDNA is typically wave32; CDNA/gfx9 wave64.
    const bool IsWave32 = llvm::StringRef{DeviceArch}.starts_with("gfx10") ||
                          llvm::StringRef{DeviceArch}.starts_with("gfx11") ||
                          llvm::StringRef{DeviceArch}.starts_with("gfx12");
    AppendExistingBitcodePath(LibsToLink, IsWave32
                                              ? "oclc_wavefrontsize64_off.bc"
                                              : "oclc_wavefrontsize64_on.bc");

    llvm::Linker Linker{M};
    for (const auto &Path : LibsToLink) {
      auto LibMod = LoadBitcode(Path);
      Linker.linkInModule(std::move(LibMod),
                          llvm::Linker::Flags::LinkOnlyNeeded);
    }
  }
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_H
