#ifndef PROTEUS_CORE_LLVM_H
#define PROTEUS_CORE_LLVM_H

static_assert(__cplusplus >= 201703L,
              "This header requires C++17 or later due to LLVM.");

#include "proteus/Error.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/JITPassPluginRegistry.h"
#include "proteus/impl/Logger.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#if __has_include(<llvm/Plugins/PassPlugin.h>)
#include <llvm/Plugins/PassPlugin.h>
#elif __has_include(<llvm/Passes/PassPlugin.h>)
#include <llvm/Passes/PassPlugin.h>
#else
#error "Cannot find LLVM PassPlugin.h"
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>

#if LLVM_VERSION_MAJOR >= 18
#include <llvm/TargetParser/SubtargetFeature.h>
// This convoluted logic below is because AMD ROCm 5.7.1 identifies as LLVM 17
// but includes the header SubtargetFeature.h to a different directory than
// upstream LLVM 17. We basically detect if it's the HIP version and include it
// from the expected MC directory, otherwise from TargetParser.
#elif LLVM_VERSION_MAJOR == 17
#if defined(__HIP_PLATFORM_HCC__) || defined(HIP_VERSION_MAJOR)
#include <llvm/MC/SubtargetFeature.h>
#else
#include <llvm/TargetParser/SubtargetFeature.h>
#endif
#else
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#error "Unsupported LLVM version " STRINGIFY(LLVM_VERSION_MAJOR)
#endif
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace proteus {
using namespace llvm;

namespace detail {

inline Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel = 3) {
  Triple TT(M.getTargetTriple());
  auto CGOptLevel = CodeGenOpt::getLevel(OptLevel);
  if (CGOptLevel == std::nullopt)
    reportFatalError("Invalid opt level");

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TT);

  std::optional<Reloc::Model> RelocModel;
  if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel = M.getCodeModel();

  // Use default target options.
  // TODO: Customize based on AOT compilation flags or by creating a
  // constructor that sets target options based on the triple.
  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      M.getTargetTriple(), Arch, Features.getString(), Options, RelocModel,
      CodeModel, CGOptLevel.value()));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

inline std::string getDefaultOptimizationPipeline(char OptLevel) {
  switch (OptLevel) {
  case '0':
    return "default<O0>";
  case '1':
    return "default<O1>";
  case '2':
    return "default<O2>";
  case '3':
    return "default<O3>";
  case 's':
#if LLVM_VERSION_MAJOR > 22
    return "default<O2>";
#else
    return "default<Os>";
#endif
  case 'z':
#if LLVM_VERSION_MAJOR > 22
    return "default<O2>";
#else
    return "default<Oz>";
#endif
  default:
    reportFatalError(std::string("Unsupported optimization level ") + OptLevel);
  }
  return "";
}

inline std::string composeOptimizationPassPipeline(
    std::optional<std::string> PassPipeline, char OptLevel,
    const std::vector<JITPassPluginConfig> &Plugins) {
  std::string Pipeline;
  for (const auto &Plugin : Plugins) {
    if (!Plugin.Insertion ||
        Plugin.Insertion->Position != JITPassPluginPosition::Prepend)
      continue;
    if (!Pipeline.empty())
      Pipeline += ",";
    Pipeline += Plugin.Insertion->Pipeline;
  }

  if (!Pipeline.empty())
    Pipeline += ",";
  Pipeline += PassPipeline ? std::move(*PassPipeline)
                           : getDefaultOptimizationPipeline(OptLevel);

  for (const auto &Plugin : Plugins) {
    if (!Plugin.Insertion ||
        Plugin.Insertion->Position != JITPassPluginPosition::Append)
      continue;
    Pipeline += ",";
    Pipeline += Plugin.Insertion->Pipeline;
  }

  return Pipeline;
}

// LLVM 23 removed the textual Os/Oz pipeline names.  Keep accepting these
// names in PROTEUS_OPT_PIPELINE for compatibility with existing callers, but
// translate them only at the point where LLVM parses the pipeline.  The
// caller-facing trace can therefore continue to report the requested
// pipeline.
inline std::string normalizeOptimizationPassPipeline(StringRef PassPipeline) {
#if LLVM_VERSION_MAJOR > 22
  std::string Normalized = PassPipeline.str();
  for (auto [Old, New] :
       {std::pair<StringRef, StringRef>{"default<Os>", "default<O2>"},
        {"default<Oz>", "default<O2>"}}) {
    size_t Pos = 0;
    while ((Pos = Normalized.find(Old.str(), Pos)) != std::string::npos) {
      Normalized.replace(Pos, Old.size(), New.str());
      Pos += New.size();
    }
  }
  return Normalized;
#else
  return PassPipeline.str();
#endif
}

inline void applySizeOptimizationAttribute(Module &M,
                                           Attribute::AttrKind AttrKind) {
  for (Function &F : M)
    F.addFnAttr(AttrKind);
}

inline void applySizeOptimizationAttribute(Module &M, char OptLevel) {
  if (OptLevel == 'z')
    applySizeOptimizationAttribute(M, Attribute::MinSize);
  else if (OptLevel == 's')
    applySizeOptimizationAttribute(M, Attribute::OptimizeForSize);
}

inline void applySizeOptimizationAttribute(Module &M, StringRef PassPipeline) {
  if (PassPipeline.contains("default<Oz>"))
    applySizeOptimizationAttribute(M, Attribute::MinSize);
  else if (PassPipeline.contains("default<Os>"))
    applySizeOptimizationAttribute(M, Attribute::OptimizeForSize);
}

inline bool
hasJITPassPluginInsertion(const std::vector<JITPassPluginConfig> &Plugins) {
  return std::any_of(Plugins.begin(), Plugins.end(),
                     [](const JITPassPluginConfig &Plugin) {
                       return Plugin.Insertion.has_value();
                     });
}

inline std::vector<std::string>
getUniqueJITPassPluginPaths(const std::vector<JITPassPluginConfig> &Plugins) {
  std::vector<std::string> Paths;
  Paths.reserve(Plugins.size());
  for (const auto &Plugin : Plugins) {
    if (std::find(Paths.begin(), Paths.end(), Plugin.Path) == Paths.end())
      Paths.push_back(Plugin.Path);
  }
  return Paths;
}

inline std::vector<PassPlugin>
loadJITPassPlugins(const std::vector<JITPassPluginConfig> &Plugins) {
  std::vector<PassPlugin> LoadedPlugins;
  const auto PluginPaths = getUniqueJITPassPluginPaths(Plugins);
  LoadedPlugins.reserve(PluginPaths.size());
  for (const auto &PluginPath : PluginPaths) {
    auto LoadedPlugin = PassPlugin::Load(PluginPath);
    if (!LoadedPlugin)
      reportFatalError("Failed to load JIT pass plugin '" + PluginPath +
                       "': " + toString(LoadedPlugin.takeError()));
    LoadedPlugins.push_back(std::move(*LoadedPlugin));
  }
  return LoadedPlugins;
}

inline void runOptimizationPassPipeline(
    Module &M, StringRef Arch, const std::string &PassPipeline,
    unsigned CodegenOptLevel,
    const std::vector<JITPassPluginConfig> &Plugins = {}) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, CodegenOptLevel);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  auto LoadedPlugins = loadJITPassPlugins(Plugins);
  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  for (const auto &Plugin : LoadedPlugins)
    Plugin.registerPassBuilderCallbacks(PB);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager Passes;
#if LLVM_VERSION_MAJOR > 22
  // Preserve the pre-LLVM 23 Os/Oz semantics while normalizing their removed
  // textual pipelines to O2 below.
  applySizeOptimizationAttribute(M, PassPipeline);
#endif
  const std::string NormalizedPipeline =
      normalizeOptimizationPassPipeline(PassPipeline);
  if (auto E = PB.parsePassPipeline(Passes, NormalizedPipeline))
    reportFatalError("Error: " + toString(std::move(E)));

  Passes.run(M, MAM);
}

inline void runOptimizationPassPipeline(
    Module &M, StringRef Arch, char OptLevel = '3',
    unsigned CodegenOptLevel = 3,
    const std::vector<JITPassPluginConfig> &Plugins = {}) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, CodegenOptLevel);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  auto LoadedPlugins = loadJITPassPlugins(Plugins);
  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  for (const auto &Plugin : LoadedPlugins)
    Plugin.registerPassBuilderCallbacks(PB);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  OptimizationLevel OptSetting;
#if LLVM_VERSION_MAJOR > 22
  // Preserve the pre-LLVM 23 Os/Oz semantics with O2 plus function attrs.
  applySizeOptimizationAttribute(M, OptLevel);
#endif
  switch (OptLevel) {
  case '0':
    OptSetting = OptimizationLevel::O0;
    break;
  case '1':
    OptSetting = OptimizationLevel::O1;
    break;
  case '2':
    OptSetting = OptimizationLevel::O2;
    break;
  case '3':
    OptSetting = OptimizationLevel::O3;
    break;
  case 's':
#if LLVM_VERSION_MAJOR > 22
    // LLVM 23 uses O2 plus the optsize function attribute.
    OptSetting = OptimizationLevel::O2;
#else
    OptSetting = OptimizationLevel::Os;
#endif
    break;
  case 'z':
#if LLVM_VERSION_MAJOR > 22
    // LLVM 23 uses O2 plus the minsize function attribute.
    OptSetting = OptimizationLevel::O2;
#else
    OptSetting = OptimizationLevel::Oz;
#endif
    break;
  default:
    reportFatalError(std::string("Unsupported optimization level ") + OptLevel);
  };

  ModulePassManager Passes = PB.buildPerModuleDefaultPipeline(OptSetting);
  Passes.run(M, MAM);
}

} // namespace detail

struct InitLLVMTargets {
  InitLLVMTargets() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
  }
};

struct OptimizationPipelineConfig {
  std::optional<std::string> PassPipeline;
  char OptLevel;
  unsigned CodegenOptLevel;

  explicit OptimizationPipelineConfig(const CodeGenerationConfig &CGConfig)
      : OptLevel(CGConfig.optLevel()),
        CodegenOptLevel(CGConfig.codeGenOptLevel()) {
    if (auto Pipeline = CGConfig.optPipeline())
      PassPipeline = Pipeline.value();
  }

  OptimizationPipelineConfig(std::optional<std::string> PassPipeline,
                             char OptLevel, unsigned CodegenOptLevel)
      : PassPipeline(std::move(PassPipeline)), OptLevel(OptLevel),
        CodegenOptLevel(CodegenOptLevel) {}
};

inline void optimizeIR(Module &M, StringRef Arch,
                       const OptimizationPipelineConfig &OptConfig) {
  TIMESCOPE("proteus::optimizeIR");
  Timer T(Config::get().ProteusEnableTimers);

  const auto Plugins = getJITPassPluginConfigs();
  const bool UseTextualPipeline =
      OptConfig.PassPipeline || detail::hasJITPassPluginInsertion(Plugins);
  const std::string FinalPipeline =
      UseTextualPipeline
          ? detail::composeOptimizationPassPipeline(OptConfig.PassPipeline,
                                                    OptConfig.OptLevel, Plugins)
          : std::string();

  if (UseTextualPipeline) {
    auto TraceOut = [](const std::string &PassPipeline) {
      SmallString<128> S;
      raw_svector_ostream OS(S);
      OS << "[CustomPipeline] " << PassPipeline << "\n";
      return S;
    };

    if (Config::get().traceSpecializations())
      Logger::trace(TraceOut(FinalPipeline));

    detail::runOptimizationPassPipeline(M, Arch, FinalPipeline,
                                        OptConfig.CodegenOptLevel, Plugins);
  } else {
    detail::runOptimizationPassPipeline(M, Arch, OptConfig.OptLevel,
                                        OptConfig.CodegenOptLevel, Plugins);
  }

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "optimizeIR optlevel "
                       << (UseTextualPipeline
                               ? StringRef(FinalPipeline)
                               : StringRef(&OptConfig.OptLevel, 1))
                       << " codegenopt " << OptConfig.CodegenOptLevel << " "
                       << T.elapsed() << " ms\n");
}

inline std::unique_ptr<Module>
linkModules(LLVMContext &Ctx,
            SmallVector<std::unique_ptr<Module>> LinkedModules) {
  TIMESCOPE("proteus::linkModules");
  if (LinkedModules.empty())
    reportFatalError("Expected jit module");

  auto LinkedModule = std::make_unique<llvm::Module>("JitModule", Ctx);
  Linker IRLinker(*LinkedModule);
  // Link in all the proteus-enabled extracted modules.
  for (auto &LinkedM : LinkedModules) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM)))
      reportFatalError("Linking failed");
  }

  return LinkedModule;
}

inline void pruneDanglingNVVMAnnotations(Module &M) {
  NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
  if (!Annotations)
    return;

  SmallVector<MDNode *> LiveEntries;
  for (MDNode *Entry : Annotations->operands()) {
    // Global DCE nulls out the entry of a function it removes and stripping
    // debug info rewrites such a node to an empty tuple, !{}.
    // LLVM's nvvm.annotations which LLVM's later UpgradeNVVMAnnotations
    // pass reads unconditionally, so we need to prune out such entries.
    // Otherwise, UpgradeNVVMAnnotations runs out of bounds!
    if (!Entry || Entry->getNumOperands() == 0)
      continue;

    if (!mdconst::dyn_extract_or_null<GlobalValue>(Entry->getOperand(0)))
      continue;

    LiveEntries.push_back(Entry);
  }

  if (LiveEntries.size() == Annotations->getNumOperands())
    return;

  Annotations->clearOperands();
  for (MDNode *Entry : LiveEntries)
    Annotations->addOperand(Entry);
}

inline void runCleanupPassPipeline(Module &M) {
  TIMESCOPE("proteus::runCleanupPassPipeline");
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager Passes;
  Passes.addPass(GlobalDCEPass());
  // Passes.addPass(StripDeadDebugInfoPass());
  Passes.addPass(StripDeadPrototypesPass());

  Passes.run(M, MAM);

  pruneDanglingNVVMAnnotations(M);

  // Keep line-table debug info so that recorded device modules retain the
  // kernel's source file and line information.
  stripNonLineTableDebugInfo(M);
}

inline void findFunctionsWithU64Metadata(
    llvm::Module &M, llvm::StringRef Key,
    llvm::SmallVectorImpl<std::pair<llvm::Function *, std::uint64_t>> &Out) {
  llvm::SmallDenseMap<llvm::Function *, std::uint64_t, 32> Seen;
  for (llvm::Function &F : M) {
    llvm::MDNode *Node = F.getMetadata(Key);
    if (!Node || Node->getNumOperands() < 1)
      continue;
    auto *CAM = llvm::dyn_cast<llvm::ConstantAsMetadata>(Node->getOperand(0));
    auto *CI =
        CAM ? llvm::dyn_cast<llvm::ConstantInt>(CAM->getValue()) : nullptr;
    if (!CI)
      continue;
    std::uint64_t Id = CI->getZExtValue();
    if (Seen.try_emplace(&F, Id).second)
      Out.emplace_back(&F, Id);
  }
}

inline void
findFunctionsWithU64Metadata(llvm::Module &M, llvm::StringRef Key,
                             llvm::SmallVectorImpl<std::uint64_t> &Out) {
  llvm::SmallDenseMap<llvm::Function *, std::uint64_t, 32> Seen;
  for (llvm::Function &F : M) {
    llvm::MDNode *Node = F.getMetadata(Key);
    if (!Node || Node->getNumOperands() < 1)
      continue;
    auto *CAM = llvm::dyn_cast<llvm::ConstantAsMetadata>(Node->getOperand(0));
    auto *CI =
        CAM ? llvm::dyn_cast<llvm::ConstantInt>(CAM->getValue()) : nullptr;
    if (!CI)
      continue;
    std::uint64_t Id = CI->getZExtValue();
    if (Seen.try_emplace(&F, Id).second)
      Out.emplace_back(Id);
  }
}

inline bool hasU64Metadata(Function *F, StringRef Key) {
  return F->getMetadata(Key) != nullptr;
}

inline std::optional<uint64_t> getU64Metadata(Function *F, StringRef Key) {
  if (!F)
    return std::nullopt;

  auto *Node = F->getMetadata(Key);
  if (!Node || Node->getNumOperands() < 1)
    return std::nullopt;

  auto *CAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(0));
  auto *CI = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
  if (!CI)
    return std::nullopt;

  return CI->getZExtValue();
}

inline void pruneIR(Module &M, bool UnsetExternallyInitialized = true) {
  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);

  // Remove llvm.compiler.used
  if (auto *CompilerUsed = M.getGlobalVariable("llvm.compiler.used"))
    M.eraseGlobalVariable(CompilerUsed);

  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  SmallVector<GlobalVariable *> GlobalsToErase;
  for (auto &GV : M.globals()) {
    auto Name = GV.getName();
    if (Name.starts_with("__clang_gpu_used_external") ||
        Name.starts_with("_jit_bitcode") || Name.starts_with("__hip_cuid")) {
      GlobalsToErase.push_back(&GV);
      removeFromUsedLists(M, [&GV](Constant *C) {
        if (auto *Global = dyn_cast<GlobalVariable>(C))
          return Global == &GV;
        return false;
      });
    }
  }
  for (auto *GV : GlobalsToErase) {
    M.eraseGlobalVariable(GV);
  }

  // Remove externaly_initialized attributes.
  if (UnsetExternallyInitialized)
    for (auto &GV : M.globals())
      if (GV.isExternallyInitialized())
        GV.setExternallyInitialized(false);
}

inline void internalize(Module &M, StringRef PreserveFunctionName) {
  auto *F = M.getFunction(PreserveFunctionName);
  // Internalize others besides the kernel function.
  internalizeModule(M, [&F](const GlobalValue &GV) {
    // Do not internalize the kernel function.
    if (&GV == F)
      return true;

    // Internalize everything else.
    return false;
  });
}

inline std::optional<uint64_t> getFunctionU64Metadata(Function &F,
                                                      StringRef Key) {
  MDNode *Node = F.getMetadata(Key);
  if (!Node || Node->getNumOperands() < 1)
    return std::nullopt;

  auto *CAM = dyn_cast<ConstantAsMetadata>(Node->getOperand(0));
  auto *CI = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
  if (!CI)
    return std::nullopt;

  return CI->getZExtValue();
}

} // namespace proteus

#endif
