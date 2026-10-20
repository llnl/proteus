//===-- JitEngineHost.cpp -- JIT Engine for CPU using ORC ---===//
//
// Part of Proteus Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JitEngine interface for dynamic compilation and optimization
// of CPU code.
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/JitEngineHost.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Frontend/DispatcherHost.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/TransformArgumentSpecialization.h"
#include "proteus/impl/TransformLambdaSpecialization.h"
#include <optional>
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
#include "proteus/impl/CompilerInterfaceDevice.h"
#endif

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Object/SymbolSize.h>
#include <llvm/TargetParser/Host.h>

#include <cstring>
#include <memory>

using namespace proteus;
using namespace llvm;
using namespace llvm::orc;

namespace {

DenseMap<int, proteus::RuntimeConstant>
collectCurrentLambdaJitValues(std::optional<uint64_t> FunctorID) {
  DenseMap<int, proteus::RuntimeConstant> EmptyMap{};

  if (!FunctorID)
    return EmptyMap;

  auto VariantOpt = LambdaRegistry::instance().getHostJitVariant(*FunctorID);
  if (!VariantOpt || VariantOpt->empty())
    return EmptyMap;

  return *VariantOpt;
}

} // namespace

JitEngineHost &JitEngineHost::instance() {
  static JitEngineHost Jit;
  return Jit;
}

void JitEngineHost::addStaticLibrarySymbols() {
  // Create a SymbolMap for static symbols.
  SymbolMap SymbolMap;

#if PROTEUS_ENABLE_CUDA
  // NOTE: For CUDA codes we link the CUDA runtime statically to access device
  // global variables. So, if the host JIT module uses CUDA functions, we
  // need to resolve them statically in the JIT module's linker.

  // Insert cudaLaunchKernel if available, using the Proteus CUDA runtime
  // builtin initialization to avoid dependencies on the CUDA runtime library.
  if (__proteus_cudaLaunchKernel_ptr) {
    SymbolMap[LLJITPtr->mangleAndIntern("cudaLaunchKernel")] =
        orc::ExecutorSymbolDef(orc::ExecutorAddr{reinterpret_cast<uintptr_t>(
                                   __proteus_cudaLaunchKernel_ptr)},
                               JITSymbolFlags::Exported);
  }

#endif

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
  // Add __proteus_launch_kernel as a static symbol.
  SymbolMap[LLJITPtr->mangleAndIntern("__proteus_launch_kernel")] =
      orc::ExecutorSymbolDef(orc::ExecutorAddr{reinterpret_cast<uintptr_t>(
                                 __proteus_launch_kernel)},
                             JITSymbolFlags::Exported);

#endif
  // Register the symbol in the main JIT dynamic library.
  cantFail(LLJITPtr->getMainJITDylib().define(absoluteSymbols(SymbolMap)));
}

void JitEngineHost::dumpSymbolInfo(
    const object::ObjectFile &LoadedObj,
    const RuntimeDyld::LoadedObjectInfo &ObjInfo) {
  // Dump information about symbols.
  auto Pid = sys::Process::getProcessId();
  std::error_code EC;
  raw_fd_ostream OFD("/tmp/perf-" + std::to_string(Pid) + ".map", EC,
                     sys::fs::OF_Append);
  if (EC)
    reportFatalError("Cannot open perf map file");
  for (auto SymSizePair : object::computeSymbolSizes(LoadedObj)) {
    auto Sym = SymSizePair.first;
    auto Size = SymSizePair.second;
    auto SymName = Sym.getName();
    // Skip any unnamed symbols.
    if (!SymName || SymName->empty())
      continue;
    // The relative address of the symbol inside its section.
    auto SymAddr = Sym.getAddress();
    if (!SymAddr)
      continue;
    // The address the functions was loaded at.
    auto LoadedSymAddress = *SymAddr;
    auto SymbolSection = Sym.getSection();
    if (SymbolSection) {
      // Compute the load address of the symbol by adding the section load
      // address.
      LoadedSymAddress += ObjInfo.getSectionLoadAddress(*SymbolSection.get());
    }
    PROTEUS_DBG(Logger::logs("proteus")
                << format("Address range: [%12p, %12p]", LoadedSymAddress,
                          LoadedSymAddress + Size)
                << "\tSymbol: " << *SymName << "\n");

    if (Size > 0)
      OFD << format("%lx %x)", LoadedSymAddress, Size) << " " << *SymName
          << "\n";
  }

  OFD.close();
}

void JitEngineHost::notifyLoaded(MaterializationResponsibility & /*R*/,
                                 const object::ObjectFile &Obj,
                                 const RuntimeDyld::LoadedObjectInfo &LOI) {
  dumpSymbolInfo(Obj, LOI);
}

JitEngineHost::~JitEngineHost() = default;

DispatcherHost &JitEngineHost::getDispatcher() {
  if (!Dispatch)
    reportFatalError("Dispatcher has not been created by the engine");
  return *Dispatch;
}

void JitEngineHost::specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                                 ArrayRef<RuntimeConstant> RCArray) {
  TIMESCOPE(JitEngineHost, specializeIR);
  Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

  // Replace argument uses with runtime constants.
  // TODO: change NumRuntimeConstants to size_t at interface.
  MDNode *Node = F->getMetadata("jit_arg_nos");
  assert(Node && "Expected metata for jit argument positions");
  PROTEUS_DBG(Logger::logs("proteus") << "Metadata jit for F " << F->getName()
                                      << " = " << *Node << "\n");

  // Replace argument uses with runtime constants.
  SmallVector<int32_t> ArgPos;
  for (unsigned int I = 0; I < Node->getNumOperands(); ++I) {
    ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(Node->getOperand(I));
    ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
    int ArgNo = ConstInt->getZExtValue();
    ArgPos.push_back(ArgNo);
  }
  TransformArgumentSpecialization::transform(M, *F, RCArray);

  auto FunctorID = getFunctionU64Metadata(*F, "proteus.registered_lambda");
  auto LambdaJitValuesMap = collectCurrentLambdaJitValues(FunctorID);

  if (!LambdaJitValuesMap.empty()) {
    TransformLambdaSpecialization::transformHostFunction(M, *F,
                                                         LambdaJitValuesMap);
    LambdaRegistry::instance().eraseHostJitVariables(*FunctorID);
  }

  F->setName(FnName + Suffix);

  if (Config::get().ProteusDebugOutput) {
    if (verifyModule(M, &errs()))
      reportFatalError("Broken module found, JIT compilation aborted!");
    else
      Logger::logs("proteus") << "Module verified!\n";
  }
}

void *JitEngineHost::compileAndLink(StringRef FnName, char *IR, int IRSize,
                                    void **Args,
                                    ArrayRef<RuntimeConstantInfo *> RCInfoArray,
                                    uint64_t *FunctorIDPtr) {
  TIMESCOPE(JitEngineHost, compileAndLink);

  StringRef StrIR(IR, IRSize);

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(Args, RCInfoArray);
  std::optional<uint64_t> FunctorID =
      FunctorIDPtr ? std::optional<uint64_t>{*FunctorIDPtr} : std::nullopt;
  DenseMap<int, proteus::RuntimeConstant> LambdaJitValuesMap =
      collectCurrentLambdaJitValues(FunctorID);

  const auto &CGConfig = Config::get().getCGConfig();
  HashT HashValue = hash(StrIR, FnName, RCVec, LambdaJitValuesMap,
                         hashCodeGenConfig(CGConfig),
                         hashRuntimeSpecializationConfig(CGConfig));
  if (Config::get().ProteusDebugOutput) {
    Logger::logs("proteus")
        << "Hashing: " << " FnName " << FnName << " RCVec [ ";
    for (const auto &RC : RCVec)
      Logger::logs("proteus") << RC.Value.Int64Val << ",";
    Logger::logs("proteus") << " ] LambdaMap [ ";
    for (auto &KV : LambdaJitValuesMap)
      Logger::logs("proteus")
          << KV.first << ":" << KV.second.Value.Int32Val << ",";
    Logger::logs("proteus") << " ] -> Hash " << HashValue.getValue() << "\n";
  }

  KernelName Name{FnName.str(), HashValue};
  std::string Suffix = Name.suffix();
  std::string MangledFnName = Name.mangled();
  DispatcherHost &Dispatch = getDispatcher();

  // Lookup the function pointer in the code cache.
  void *JitFnPtr = Dispatch.lookupFunction(Name.base(), HashValue);
  if (JitFnPtr) {
    if (FunctorID)
      LambdaRegistry::instance().eraseHostJitVariables(*FunctorID);
    return JitFnPtr;
  }

  std::unique_ptr<CompiledLibrary> Library =
      Dispatch.lookupCompiledLibrary(HashValue);
  if (!Library) {
    Timer T(Config::get().ProteusEnableTimers);
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Diag;
    auto M = parseIR(MemoryBufferRef(StrIR, "JitModule"), Diag, *Ctx);
    if (!M)
      reportFatalError("Error parsing IR: " + Diag.getMessage());

    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus") << "Parse IR " << FnName << " "
                                                 << T.elapsed() << " ms\n");

    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".input.ll", *M));
    // Specialize the module using runtime values.
    specializeIR(*M, FnName, Suffix, RCVec);
    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".specialized.ll", *M));

    Library = std::make_unique<CompiledLibrary>(
        Dispatch.compile(std::move(Ctx), std::move(M), HashValue, CGConfig));
  }

  JitFnPtr = Dispatch.insertFunction(Name, HashValue, *Library);

  PROTEUS_DBG(Logger::logs("proteus")
              << "===\n"
              << *LLJITPtr->getExecutionSession().getSymbolStringPool()
              << "===\n");

  PROTEUS_DBG(Logger::logs("proteus")
              << "=== JIT compile: " << FnName << " Mangled " << MangledFnName
              << " RC HashValue " << HashValue.toString() << " Addr "
              << JitFnPtr << "\n");
  if (FunctorID)
    LambdaRegistry::instance().eraseHostJitVariables(*FunctorID);
  return JitFnPtr;
}

std::unique_ptr<MemoryBuffer>
JitEngineHost::compileOnly(Module &M, const CodeGenerationConfig &CGConfig) {
  TIMESCOPE(JitEngineHost, compileOnly);
  // Create the target machine using JITTargetMachineBuilder to match ORC JIT
  // loading.
  auto ExpectedTM =
      JITTargetMachineBuilder::detectHost()->createTargetMachine();
  if (auto E = ExpectedTM.takeError())
    reportFatalError("Expected target machine: " + toString(std::move(E)));
  std::unique_ptr<TargetMachine> TM = std::move(*ExpectedTM);

  // Set up the output stream.
  SmallVector<char, 0> ObjBuffer;
  raw_svector_ostream ObjStream(ObjBuffer);

  // Set up the pass manager.
  legacy::PassManager PM;
  // Add optimization passes.
  optimizeIR(M, sys::getHostCPUName(), OptimizationPipelineConfig(CGConfig));

  // Add the target passes to emit object code.
  if (TM->addPassesToEmitFile(PM, ObjStream, nullptr,
                              CodeGenFileType::ObjectFile)) {
    reportFatalError("Target machine cannot emit object file");
  }

  // Run the passes.
  PM.run(M);

  return MemoryBuffer::getMemBufferCopy(
      StringRef(ObjBuffer.data(), ObjBuffer.size()));
}

void JitEngineHost::loadCompiledLibrary(CompiledLibrary &Library) {
  TIMESCOPE(JitEngineHost, loadCompiledLibrary);
  // Create an isolated JITDyLib context and load the compiled library for
  // linking and symbol retrieval.
  auto &ES = LLJITPtr->getExecutionSession();
  auto ExpectedJitDyLib = ES.createJITDylib(
      "JitDyLib_" + std::to_string(reinterpret_cast<uintptr_t>(&Library)));
  if (auto E = ExpectedJitDyLib.takeError()) {
    reportFatalError("Error creating library jit dylib: " +
                     toString(std::move(E)));
  }

  JITDylib &CreatedDylib = *ExpectedJitDyLib;
  Library.JitDyLib = &CreatedDylib;

  if (Library.isSharedObject()) {
    // Load the dynamic library through a generator using the dynamic library
    // file.
    auto Gen = ExitOnErr(llvm::orc::DynamicLibrarySearchGenerator::Load(
        Library.DynLibPath.c_str(),
        LLJITPtr->getDataLayout().getGlobalPrefix()));
    Library.JitDyLib->addGenerator(std::move(Gen));
  } else {
    // Resolve symbols from main/process before materialization through the main
    // JIT dynamic library. It is simpler and avoids duplication than adding
    // symbols to the code library JIT dylib every time.
    Library.JitDyLib->addToLinkOrder(LLJITPtr->getMainJITDylib(),
                                     JITDylibLookupFlags::MatchAllSymbols);

    // Add the object from the compiled library.
    if (auto E = LLJITPtr->addObjectFile(*Library.JitDyLib,
                                         std::move(Library.ObjectModule)))
      reportFatalError("Error loading object file: " + toString(std::move(E)));
  }
}

void *JitEngineHost::getFunctionAddress(StringRef FnName,
                                        CompiledLibrary &Library) {
  TIMESCOPE(JitEngineHost, getFunctionAddress);
  // Lookup the function address corresponding the dynamic library context of
  // the compiled library.
  assert(Library.JitDyLib && "Expected non-null JIT dylib");
  auto EntryAddr = ExitOnErr(LLJITPtr->lookup(*Library.JitDyLib, FnName));

  void *JitFnPtr = (void *)EntryAddr.getValue();
  assert(JitFnPtr && "Expected non-null JIT function pointer");

  return JitFnPtr;
}

JitEngineHost::JitEngineHost() {
  ExitOnErr.setBanner("JIT: ");
  // Create the LLJIT instance.
  // TODO: Fix support for debugging jitted code. This appears to be
  // the correct interface (see orcv2 examples) but it does not work.
  // By dumpSymbolInfo() the debug sections are not populated. Why?
  LLJITPtr = ExitOnErr(
      LLJITBuilder()
#if LLVM_VERSION_MAJOR >= 21
#if LLVM_VERSION_MAJOR > 22
          .setObjectLinkingLayerCreator(
              [&](ExecutionSession &ES,
                  jitlink::JITLinkMemoryManager &) {
                auto GetMemMgr = [](const MemoryBuffer &) {
                  return std::make_unique<SectionMemoryManager>();
                };
                auto ObjLinkingLayer =
                    std::make_unique<RTDyldObjectLinkingLayer>(ES, GetMemMgr);

                // Register the event listener.
                ObjLinkingLayer->registerJITEventListener(
                    *JITEventListener::createGDBRegistrationListener());

                // Make sure the debug info sections aren't stripped.
                ObjLinkingLayer->setProcessAllSections(true);

                if (Config::get().ProteusDebugOutput) {
                  ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
                }
                return std::unique_ptr<ObjectLayer>(std::move(ObjLinkingLayer));
              })
#else
          .setObjectLinkingLayerCreator([&](ExecutionSession &ES) {
            auto GetMemMgr = [](const MemoryBuffer &) {
              return std::make_unique<SectionMemoryManager>();
            };
            auto ObjLinkingLayer =
                std::make_unique<RTDyldObjectLinkingLayer>(ES, GetMemMgr);

            // Register the event listener.
            ObjLinkingLayer->registerJITEventListener(
                *JITEventListener::createGDBRegistrationListener());

            // Make sure the debug info sections aren't stripped.
            ObjLinkingLayer->setProcessAllSections(true);

            if (Config::get().ProteusDebugOutput) {
              ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
            }
            return ObjLinkingLayer;
          })
#endif
#else
          .setObjectLinkingLayerCreator(
              [&](ExecutionSession &ES, const Triple &) {
                auto GetMemMgr = []() {
                  return std::make_unique<SectionMemoryManager>();
                };
                auto ObjLinkingLayer =
                    std::make_unique<RTDyldObjectLinkingLayer>(ES, GetMemMgr);

                // Register the event listener.
                ObjLinkingLayer->registerJITEventListener(
                    *JITEventListener::createGDBRegistrationListener());

                // Make sure the debug info sections aren't stripped.
                ObjLinkingLayer->setProcessAllSections(true);

                if (Config::get().ProteusDebugOutput) {
                  ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
                }
                return ObjLinkingLayer;
              })
#endif
          .create());
  // Use the main JIT dynamic library to add a generator for host process
  // symbols.
  orc::MangleAndInterner Mangle(LLJITPtr->getExecutionSession(),
                                LLJITPtr->getDataLayout());
  LLJITPtr->getMainJITDylib().addGenerator(
      ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          LLJITPtr->getDataLayout().getGlobalPrefix(),
          [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
            return Name != MainName;
          })));

  // Add static library functions to the main JIT dynamic library.
  addStaticLibrarySymbols();

  // Create the dispatcher eagerly so cache statistics are always reported,
  // even when no function is ever JIT-compiled.
  Dispatch = std::make_unique<DispatcherHost>("JitEngineHost", *this);
}
