#include "proteus/impl/Frontend/HIPToolchain.h"

#include "proteus/Error.h"

#if PROTEUS_ENABLE_HIP

#include "proteus/impl/Config.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>

#include <cctype>
#include <mutex>
#include <optional>
#include <utility>

#ifndef PROTEUS_HIP_INSTALL_ROOT
#define PROTEUS_HIP_INSTALL_ROOT ""
#endif

namespace proteus {

namespace {

std::optional<std::string> getNonEmptyEnvVar(llvm::StringRef VarName) {
  std::string Name = VarName.str();
  auto Value = getEnvOrDefaultString(Name.c_str());
  if (!Value || Value->empty())
    return std::nullopt;
  return Value;
}

std::string canonicalizeExistingPath(llvm::StringRef Path) {
  llvm::SmallString<256> RealPath;
  if (!llvm::sys::fs::real_path(Path, RealPath))
    return RealPath.str().str();
  return Path.str();
}

std::string requireExistingDirectory(llvm::StringRef Path,
                                     llvm::StringRef Context) {
  if (!llvm::sys::fs::is_directory(Path))
    reportFatalError(Context.str() + ": expected directory, got " + Path.str());
  return canonicalizeExistingPath(Path);
}

std::string readTextFile(llvm::StringRef Path, llvm::StringRef Context) {
  auto BufferOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrErr || !BufferOrErr.get())
    reportFatalError(Context.str() + ": failed to read " + Path.str());
  return std::string(BufferOrErr.get()->getBuffer());
}

std::optional<std::pair<std::string, std::string>>
resolveRootFromEnv(llvm::StringRef VarName) {
  auto Value = getNonEmptyEnvVar(VarName);
  if (!Value)
    return std::nullopt;

  return std::pair<std::string, std::string>{
      requireExistingDirectory(*Value, VarName), VarName.str() + "=" + *Value};
}

std::optional<std::pair<std::string, std::string>> resolveRootFromBuild() {
  if (llvm::StringRef(PROTEUS_HIP_INSTALL_ROOT).empty())
    return std::nullopt;

  return std::pair<std::string, std::string>{
      requireExistingDirectory(PROTEUS_HIP_INSTALL_ROOT,
                               "Build-time ROCm installation root"),
      "build-time ROCm installation root " +
          std::string(PROTEUS_HIP_INSTALL_ROOT)};
}

std::string resolveDeviceLibDirFromRoot(llvm::StringRef Root) {
  llvm::SmallString<256> Candidate(Root);
  llvm::sys::path::append(Candidate, "amdgcn", "bitcode");
  return requireExistingDirectory(Candidate, "ROCm device library directory");
}

std::optional<std::string> extractVersionPrefix(llvm::StringRef Text) {
  std::size_t Cursor = 0;
  while (Cursor < Text.size() &&
         !std::isdigit(static_cast<unsigned char>(Text[Cursor]))) {
    ++Cursor;
  }

  if (Cursor == Text.size())
    return std::nullopt;

  std::size_t End = Cursor;
  while (End < Text.size() &&
         (std::isdigit(static_cast<unsigned char>(Text[End])) ||
          Text[End] == '.')) {
    ++End;
  }

  if (End == Cursor)
    return std::nullopt;
  return Text.substr(Cursor, End - Cursor).str();
}

std::optional<std::string> parseMacroValue(llvm::StringRef Text,
                                           llvm::StringRef MacroName) {
  while (!Text.empty()) {
    std::size_t LineEnd = Text.find_first_of("\r\n");
    llvm::StringRef Line = Text.substr(0, LineEnd);
    if (LineEnd == llvm::StringRef::npos) {
      Text = llvm::StringRef{};
    } else {
      Text = Text.substr(LineEnd);
      while (!Text.empty() && (Text.front() == '\n' || Text.front() == '\r'))
        Text = Text.drop_front();
    }

    Line = Line.trim();
    if (!Line.consume_front("#define"))
      continue;

    Line = Line.ltrim();
    if (!Line.consume_front(MacroName))
      continue;
    if (!Line.empty() &&
        !std::isspace(static_cast<unsigned char>(Line.front())))
      continue;

    Line = Line.ltrim();
    std::size_t End = 0;
    while (End < Line.size() &&
           std::isdigit(static_cast<unsigned char>(Line[End]))) {
      ++End;
    }
    if (End == 0)
      continue;
    return Line.substr(0, End).str();
  }

  return std::nullopt;
}

std::string readHIPVersionFromHeader(llvm::StringRef Path) {
  std::string Text = readTextFile(Path, "HIP version header");
  auto Major = parseMacroValue(Text, "HIP_VERSION_MAJOR");
  auto Minor = parseMacroValue(Text, "HIP_VERSION_MINOR");
  if (!Major || !Minor) {
    reportFatalError("HIP version header: missing HIP_VERSION_MAJOR or "
                     "HIP_VERSION_MINOR in " +
                     Path.str());
  }

  auto Patch = parseMacroValue(Text, "HIP_VERSION_PATCH");
  if (Patch)
    return *Major + "." + *Minor + "." + *Patch;
  return *Major + "." + *Minor;
}

std::string readHIPVersionFromRoot(llvm::StringRef Root) {
  // .info/version is the ROCm distribution version, which need not match
  // the HIP API/runtime version.  For example, ROCm 10 installations may
  // package HIP 7.15.  Validate against the same HIP version CMake detects.
  for (llvm::StringRef RelativePath :
       {"include/hip/hip_version.h",
        "include/hip/amd_detail/amd_hip_version.h"}) {
    llvm::SmallString<256> HeaderPath(Root);
    llvm::sys::path::append(HeaderPath, RelativePath);
    if (llvm::sys::fs::is_regular_file(HeaderPath))
      return readHIPVersionFromHeader(HeaderPath);
  }

  llvm::SmallString<256> VersionFile(Root);
  llvm::sys::path::append(VersionFile, ".info", "version");
  if (llvm::sys::fs::is_regular_file(VersionFile)) {
    std::string Text = readTextFile(VersionFile, "ROCm version file");
    if (auto Version = extractVersionPrefix(Text))
      return *Version;
    reportFatalError("ROCm version file: could not parse version from " +
                     VersionFile.str().str());
  }

  reportFatalError("Could not locate HIP/ROCm version metadata under " +
                   Root.str() +
                   ". Expected .info/version or a HIP version header.");
}

std::pair<std::string, std::string>
extractMajorMinorVersion(llvm::StringRef Version, llvm::StringRef Context) {
  auto ParseComponent = [&](std::size_t Start) -> std::optional<std::string> {
    if (Start >= Version.size())
      return std::nullopt;

    std::size_t End = Start;
    while (End < Version.size() &&
           std::isdigit(static_cast<unsigned char>(Version[End]))) {
      ++End;
    }
    if (End == Start)
      return std::nullopt;
    return Version.substr(Start, End - Start).str();
  };

  auto Major = ParseComponent(0);
  if (!Major)
    reportFatalError(Context.str() + ": could not parse major version from " +
                     Version.str());

  std::size_t MinorStart = Major->size();
  if (MinorStart >= Version.size() || Version[MinorStart] != '.')
    reportFatalError(Context.str() + ": could not parse minor version from " +
                     Version.str());
  ++MinorStart;

  auto Minor = ParseComponent(MinorStart);
  if (!Minor)
    reportFatalError(Context.str() + ": could not parse minor version from " +
                     Version.str());

  return {*Major, *Minor};
}

void validateHIPVersion(llvm::StringRef Root, llvm::StringRef Origin,
                        llvm::StringRef RuntimeVersion) {
  constexpr llvm::StringRef BuildHIPVersion = PROTEUS_HIP_VERSION;
  auto [BuildMajor, BuildMinor] =
      extractMajorMinorVersion(BuildHIPVersion, "Build HIP version");
  auto [RuntimeMajor, RuntimeMinor] =
      extractMajorMinorVersion(RuntimeVersion, "Runtime HIP version");
  if (BuildMajor != RuntimeMajor || BuildMinor != RuntimeMinor) {
    reportFatalError(
        "HIP version mismatch: build used " + BuildHIPVersion.str() + ", but " +
        Origin.str() + " resolved to HIP version " + RuntimeVersion.str() +
        " at " + Root.str() + ". Expected matching major and minor versions.");
  }
}

ResolvedHIPToolchain resolveHIPToolchainImpl() {
  std::optional<std::pair<std::string, std::string>> Root;
  for (llvm::StringRef Var : {"PROTEUS_ROCM_PATH", "ROCM_PATH"}) {
    Root = resolveRootFromEnv(Var);
    if (Root)
      break;
  }
  if (!Root)
    Root = resolveRootFromBuild();

  if (!Root) {
    reportFatalError("Failed to resolve the ROCm installation root. Set "
                     "PROTEUS_ROCM_PATH or ROCM_PATH.");
  }

  std::string RuntimeVersion = readHIPVersionFromRoot(Root->first);
  validateHIPVersion(Root->first, Root->second, RuntimeVersion);

  return {Root->first, resolveDeviceLibDirFromRoot(Root->first),
          std::move(RuntimeVersion), Root->second};
}

} // namespace

const ResolvedHIPToolchain &resolveHIPToolchain() {
  static std::once_flag CacheOnce;
  static std::optional<ResolvedHIPToolchain> Cache;

  std::call_once(CacheOnce, []() { Cache = resolveHIPToolchainImpl(); });
  return *Cache;
}

} // namespace proteus

#endif

#if !PROTEUS_ENABLE_HIP

namespace proteus {

const ResolvedHIPToolchain &resolveHIPToolchain() {
  reportFatalError("HIP toolchain requested in a non-HIP build");
  static const ResolvedHIPToolchain Unreachable{};
  return Unreachable;
}

} // namespace proteus

#endif
