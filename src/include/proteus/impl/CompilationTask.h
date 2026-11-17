#ifndef PROTEUS_COMPILATION_TASK_H
#define PROTEUS_COMPILATION_TASK_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/CoreLLVMDevice.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Utils.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

namespace proteus {

using namespace llvm;

class CompilationTask {
private:
  Dispatcher *Dispatch;
  MemoryBufferRef Bitcode;
  HashT HashValue;
  KernelName Name;
  dim3 BlockDim;
  dim3 GridDim;
  SmallVector<RuntimeConstant> RCVec;
  VariantSnapshot Variants;
  SmallVector<uint64_t> LambdaCalleeInfo;
  LambdaCallsiteRuntimeConstantsMap LambdaCallsiteRuntimeConstants;
  std::unordered_map<std::string, GlobalVarInfo> VarNameToGlobalInfo;
  const CodeGenerationConfig *CGConfig;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  int MinBlocksPerSM;
  bool SpecializeArgs;
  bool SpecializeDims;
  bool SpecializeDimsRange;
  bool SpecializeLaunchBounds;

  std::unique_ptr<Module> cloneKernelModule(LLVMContext &Ctx) {
    TIMESCOPE(CompilationTask, cloneKernelModule);
    auto ClonedModule = parseBitcodeFile(Bitcode, Ctx);
    if (auto E = ClonedModule.takeError()) {
      reportFatalError("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::move(*ClonedModule);
  }

  void dumpOptimizedIR(Module &M) {
    if (Config::get().traceIRDump()) {
      llvm::outs() << "LLVM IR module post optimization " << M << "\n";
    }
    if (DumpIR) {
      const auto CreateDumpDirectory = []() {
        const std::string DumpDirectory = ".proteus-dump";
        std::filesystem::create_directory(DumpDirectory);
        return DumpDirectory;
      };

      static const std::string DumpDirectory = CreateDumpDirectory();

      saveToFile(DumpDirectory + "/device-jit-" + HashValue.toString() + ".ll",
                 M);
    }
  }

public:
  CompilationTask(
      Dispatcher &Dispatch, MemoryBufferRef Bitcode, HashT HashValue,
      KernelName Name, dim3 BlockDim, dim3 GridDim,
      const SmallVector<RuntimeConstant> &RCVec,
      const VariantSnapshot &Variants,
      const SmallVector<uint64_t> &LambdaCalleeInfo,
      const LambdaCallsiteRuntimeConstantsMap &LambdaCallsiteRuntimeConstants,
      const std::unordered_map<std::string, GlobalVarInfo> &VarNameToGlobalInfo,
      const CodeGenerationConfig &CGConfig, bool DumpIR,
      bool RelinkGlobalsByCopy)
      : Dispatch(&Dispatch), Bitcode(Bitcode), HashValue(HashValue),
        Name(std::move(Name)), BlockDim(BlockDim), GridDim(GridDim),
        RCVec(RCVec), Variants(Variants), LambdaCalleeInfo(LambdaCalleeInfo),
        LambdaCallsiteRuntimeConstants(LambdaCallsiteRuntimeConstants),
        VarNameToGlobalInfo(VarNameToGlobalInfo), CGConfig(&CGConfig),
        DumpIR(DumpIR), RelinkGlobalsByCopy(RelinkGlobalsByCopy),
        MinBlocksPerSM(
            CGConfig.minBlocksPerSM(BlockDim.x * BlockDim.y * BlockDim.z)),
        SpecializeArgs(CGConfig.specializeArgs()),
        SpecializeDims(CGConfig.specializeDims()),
        SpecializeDimsRange(CGConfig.specializeDimsRange()),
        SpecializeLaunchBounds(CGConfig.specializeLaunchBounds()) {
    if (Config::get().traceSpecializations()) {
      llvm::SmallString<128> S;
      llvm::raw_svector_ostream OS(S);
      OS << "[KernelConfig] ID:" << this->Name.base() << " ";
      CGConfig.dump(OS);
      OS << "\n";
      Logger::trace(OS.str());
    }
  }

  // Delete copy operations.
  CompilationTask(const CompilationTask &) = delete;
  CompilationTask &operator=(const CompilationTask &) = delete;

  // Use default move operations.
  CompilationTask(CompilationTask &&) noexcept = default;
  CompilationTask &operator=(CompilationTask &&) noexcept = default;

  HashT getHashValue() const { return HashValue; }

  std::unique_ptr<MemoryBuffer> compile() {
    TIMESCOPE(CompilationTask, compile);
    struct TimerRAII {
      std::chrono::high_resolution_clock::time_point Start, End;
      HashT HashValue;
      TimerRAII(HashT HashValue) : HashValue(HashValue) {
        if (Config::get().ProteusDebugOutput) {
          Start = std::chrono::high_resolution_clock::now();
        }
      }

      ~TimerRAII() {
        if (Config::get().ProteusDebugOutput) {
          auto End = std::chrono::high_resolution_clock::now();
          auto Duration = End - Start;
          auto Milliseconds =
              std::chrono::duration_cast<std::chrono::milliseconds>(Duration)
                  .count();
          Logger::logs("proteus")
              << "Compiled HashValue " << HashValue.toString() << " for "
              << Milliseconds << "ms\n";
        }
      }
    } Timer{HashValue};

    LLVMContext Ctx;
    std::unique_ptr<Module> M = cloneKernelModule(Ctx);

    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".input.ll", *M));

    proteus::specializeIR(*M, Name.base(), Name.suffix(), BlockDim, GridDim,
                          RCVec, Variants, LambdaCalleeInfo,
                          LambdaCallsiteRuntimeConstants, SpecializeArgs,
                          SpecializeDims, SpecializeDimsRange,
                          SpecializeLaunchBounds, MinBlocksPerSM);

    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".specialized.ll", *M));

    replaceGlobalVariablesWithPointers(*M, VarNameToGlobalInfo);

    Dispatch->optimizeModule(*M, *CGConfig);
    dumpOptimizedIR(*M);

    auto ObjBuf = Dispatch->codegenModule(*M, *CGConfig);
    if (!RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(),
                                   VarNameToGlobalInfo);

    return ObjBuf;
  }
};

} // namespace proteus

#endif
