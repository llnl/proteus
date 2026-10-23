#include "AnnotationHandler.h"

#include "Helpers.h"
#include "Types.h"

#include "proteus/Error.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

namespace helpers {
#if defined(_WIN32)
#include <tlhelp32.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

static uint32_t getParentProcessId() {
#if defined(_WIN32)
  DWORD pid = GetCurrentProcessId();
  uint32_t ppid = 0;
  HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (h == INVALID_HANDLE_VALUE)
    return 0;
  PROCESSENTRY32 pe;
  pe.dwSize = sizeof(pe);
  if (Process32First(h, &pe)) {
    do {
      if (pe.th32ProcessID == pid) {
        ppid = pe.th32ParentProcessID;
        break;
      }
    } while (Process32Next(h, &pe));
  }
  CloseHandle(h);
  return ppid;
#else
  return static_cast<uint32_t>(getppid());
#endif
}

[[maybe_unused]] static uint32_t getProcessId() {
#if defined(_WIN32)
  return static_cast<uint32_t>(GetCurrentProcessId());
#else
  return static_cast<uint32_t>(getpid());
#endif
}
} // namespace helpers

using namespace llvm;
using namespace llvm::PatternMatch;

// LLVM 17 misses the m_GEP pattern match method, so we define it here guarded
// on the LLVM version.
#if LLVM_VERSION_MAJOR == 17
/// Matches instructions with Opcode and any number of operands
template <unsigned Opcode, typename... OperandTypes> struct AnyOps_match {
  std::tuple<OperandTypes...> Operands;

  AnyOps_match(const OperandTypes &...Ops) : Operands(Ops...) {}

  // Operand matching works by recursively calling match_operands, matching the
  // operands left to right. The first version is called for each operand but
  // the last, for which the second version is called. The second version of
  // match_operands is also used to match each individual operand.
  template <int Idx, int Last>
  std::enable_if_t<Idx != Last, bool> match_operands(const Instruction *I) {
    return match_operands<Idx, Idx>(I) && match_operands<Idx + 1, Last>(I);
  }

  template <int Idx, int Last>
  std::enable_if_t<Idx == Last, bool> match_operands(const Instruction *I) {
    return std::get<Idx>(Operands).match(I->getOperand(Idx));
  }

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<Instruction>(V);
      return I->getNumOperands() == sizeof...(OperandTypes) &&
             match_operands<0, sizeof...(OperandTypes) - 1>(I);
    }
    return false;
  }
};
/// Matches GetElementPtrInst.
template <typename... OperandTypes>
inline auto m_GEP(const OperandTypes &...Ops) {
  return AnyOps_match<Instruction::GetElementPtr, OperandTypes...>(Ops...);
}
#endif

// Strip intermittent casts.
inline static Value *stripAllCasts(Value *V) {
  while (auto *C = dyn_cast<CastInst>(V))
    V = C->getOperand(0);

  return V;
};

static Argument *tryFindArgFromLoad(Value *V) {
  auto *LI = dyn_cast<LoadInst>(V);
  if (!LI)
    return nullptr;

  StoreInst *SingleStore = nullptr;
  Value *LoadSource = LI->getPointerOperand();
  for (auto *U : LoadSource->users()) {
    auto *SI = dyn_cast<StoreInst>(U);
    if (!SI || SI->getPointerOperand() != LoadSource)
      continue;

    if (SingleStore != nullptr)
      reportFatalError("Expected single store");

    SingleStore = SI;
  }

  Value *Val = stripAllCasts(SingleStore->getValueOperand());
  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg;

  return tryFindArgFromLoad(Val);
}

/// If V ultimately came from a store of an argument into an alloca or
/// addrspacecast, return that argument, otherwise return nullptr.
static Argument *getOriginatingArgument(Value *V,
                                        const Instruction *AnnotCall) {

  // Pattern rules to identify an originating argument.
  if (AnnotCall->getParent() != &AnnotCall->getFunction()->getEntryBlock())
    reportFatalError(
        "Expected instrumentation annotation to be in the entry basic block");

  const BasicBlock &EntryBB = AnnotCall->getFunction()->getEntryBlock();
  // AMDGPU IR may pass by-value objects to jit_object through an addrspacecast
  // of the private alloca. Normalize both destinations before matching the
  // memcpy that initialized that alloca.
  V = stripAllCasts(V);
  for (auto &I : EntryBB) {
    // Iterate up to the instrumentation function call.
    if (&I == AnnotCall)
      break;

    if (auto *Memcpy = dyn_cast<MemCpyInst>(&I);
        Memcpy && stripAllCasts(Memcpy->getRawDest()) == V) {
      if (auto *Arg = dyn_cast<Argument>(stripAllCasts(Memcpy->getRawSource())))
        return Arg;
    }
  }

  if (auto *Arg = dyn_cast<Argument>(V)) {
    return Arg;
  }

  // Find the argument by walking a load of a stack slot, which is the typical
  // O0 code generation. An alternative would be to run mem2reg on a clone of
  // the function to avoid affecting the original module.
  return tryFindArgFromLoad(V);
}

static SmallPtrSet<Argument *, 16>
tryGatherCoercedArguments(Value *Obj, const Instruction *AnnotCall) {
  // The annotation receives the generic (addrspace 0) pointer.  For objects
  // passed by value Clang materializes the object in the alloca address space
  // and inserts an addrspacecast before calling jit_object.  Strip that cast
  // before looking for the stack slot; otherwise this routine aborts even
  // though the underlying value is an alloca.
  Obj = stripAllCasts(Obj);
  AllocaInst *AI = dyn_cast<AllocaInst>(Obj);
  if (!AI)
    reportFatalError("Expected object in local alloca");

  // Pattern rules to identify arguments for coerced objects.
  if (AnnotCall->getParent() != &AnnotCall->getFunction()->getEntryBlock())
    reportFatalError(
        "Expected instrumentation annotation to be in the entry basic block");

  const BasicBlock &EntryBB = AnnotCall->getFunction()->getEntryBlock();

  // We keep track of coerced arguments in a set because there are cases where
  // the same function argument is used multiple times to initialize an object,
  // for example if it is a static array to extract values from.
  SmallPtrSet<Argument *, 16> CoercedArgs;

  auto MatchStore = [](const Instruction &I, Value *Dst) -> Argument * {
    Value *V = nullptr;
    ConstantInt *OffsetCI = nullptr;
    ConstantInt *Idx0 = nullptr;
    ConstantInt *Idx1 = nullptr;

    if (match(&I, m_Store(m_Value(V),
                          m_GEP(m_Specific(Dst), m_ConstantInt(OffsetCI))))) {
      Argument *Arg = dyn_cast<Argument>(V);
      if (Arg)
        return Arg;

      Value *Src = nullptr;

      if (match(stripAllCasts(V), m_ExtractValue(m_Value(Src)))) {
        Arg = dyn_cast<Argument>(Src);
        if (Arg)
          return Arg;
      }

      reportFatalError("Expected argument");
    }

    if (match(&I, m_Store(m_Value(V), m_Specific(Dst)))) {
      Argument *Arg = dyn_cast<Argument>(V);
      if (Arg)
        return Arg;

      Value *Src = nullptr;

      if (match(V, m_ExtractValue(m_Value(Src)))) {
        Arg = dyn_cast<Argument>(Src);
        if (Arg)
          return Arg;
      }

      reportFatalError("Expected argument");
    }

    if (match(&I,
              m_Store(m_Value(V), m_GEP(m_Specific(Dst), m_ConstantInt(Idx0),
                                        m_ConstantInt(Idx1))))) {
      Argument *Arg = dyn_cast<Argument>(V);
      if (Arg)
        return Arg;

      Value *Src = nullptr;
      if (match(V, m_ExtractValue(m_Value(Src)))) {
        Arg = dyn_cast<Argument>(Src);
        if (Arg)
          return Arg;
      }

      reportFatalError("Expected argument");
    }

    return nullptr;
  };

  for (auto &I : EntryBB) {
    // Iterate up to the instrumentation function call.
    if (&I == AnnotCall)
      break;

    if (auto *Arg = MatchStore(I, AI)) {
      CoercedArgs.insert(Arg);
    } else if (auto *Memcpy = dyn_cast<MemCpyInst>(&I);
               Memcpy && stripAllCasts(Memcpy->getRawDest()) == AI) {
      if (!CoercedArgs.empty())
        reportFatalError("Expected empty coerced arguments");

      Value *Src = Memcpy->getRawSource();

      // The source of the copy is the by-value kernel argument in current
      // LLVM/Clang.  Older IR used stores into another temporary, which is
      // why the store walk below is still retained.
      if (auto *Arg = dyn_cast<Argument>(stripAllCasts(Src)))
        CoercedArgs.insert(Arg);

      // Restart detection for the Src operand.
      const Instruction &E = I;
      for (auto &I : EntryBB) {
        if (&I == &E)
          break;

        if (auto *Arg = MatchStore(I, Src))
          CoercedArgs.insert(Arg);
      }

      break;
    }
  }

  return CoercedArgs;
}

static RuntimeConstantInfo
createVectorRuntimeConstantInfo(RuntimeConstantType RCType,
                                const Argument *Arg) {
  VectorType *VecTy = dyn_cast<VectorType>(Arg->getType());
  if (!VecTy)
    reportFatalError("Expected vector type");

  ElementCount EC = VecTy->getElementCount();
  int32_t NumElts = EC.getKnownMinValue();
  if (EC.isScalable())
    reportFatalError("Unsupported scalable vector type");

  Type *EltTy = VecTy->getElementType();
  RuntimeConstantType EltRCType = convertTypeToRuntimeConstantType(EltTy);
  RuntimeConstantInfo RCI{RCType, static_cast<int32_t>(Arg->getArgNo()),
                          NumElts, EltRCType};
  return RCI;
}

static bool isLambdaFunction(const Function &F) {
  std::string DemangledName = demangle(F.getName().str());
  return StringRef{DemangledName}.contains("'lambda") &&
         StringRef{DemangledName}.contains(")::operator()");
}

static bool isLambdaFunctorWrapperOperator(const Function &F) {
  std::string DemangledName = demangle(F.getName().str());
  return StringRef{DemangledName}.contains("LambdaFunctorWrapper") &&
         StringRef{DemangledName}.contains("::operator()");
}

AnnotationHandler::AnnotationHandler(Module &M) : M(M), Types(M) {}

void AnnotationHandler::parseAnnotations(
    MapVector<Function *, JitFunctionInfo> &JitFunctionInfoMap,
    const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
    bool ForceJitAnnotateAll) {
  auto AttachMetadata = [](Function &F) {
    // Add metadata to JIT compiled functions so that they are identifiable
    // by HIP LTO when assembling JIT modules from the linked LTO module.
    MDString *MDStr = MDString::get(F.getContext(), "proteus.jit");
    MDNode *MD = MDNode::get(F.getContext(), MDStr);
    F.setMetadata("proteus.jit", MD);
  };
  // Forcing annotations overrides any user annotations and makes all kernels
  // jit annotated without any runtime constant arguments.
  if (ForceJitAnnotateAll) {
    // Remove any pre-existing global annotations.
    removeJitGlobalAnnotations();
    removeProteusLambdaAnnotations();

    // Create jit annotations for all kernel functions.
    for (auto &F : M.getFunctionList()) {
      if (F.isDeclaration())
        continue;
      if (!(isDeviceKernel(&F) || StubToKernelMap.contains(&F)))
        continue;

      // Communicate back in the JitFunctionInfoMap.
      JitFunctionInfoMap.insert({&F, JitFunctionInfo{{}, ""}});
      AttachMetadata(F);
    }

    return;
  }

  // Parse any proteus::jit_arg, proteus::jit_array, proteus::jit_object
  // annotations.
  SmallPtrSet<Function *, 32> JitArgAnnotations;
  SmallPtrSet<Function *, 32> JitArrayAnnotations;
  SmallPtrSet<Function *, 32> JitObjectAnnotations;
  for (auto &F : M.getFunctionList()) {
    std::string DemangledName = demangle(F.getName().str());
    if (StringRef{DemangledName}.contains("proteus::jit_arg"))
      JitArgAnnotations.insert(&F);
    if (StringRef{DemangledName}.contains("proteus::jit_array"))
      JitArrayAnnotations.insert(&F);
    if (StringRef{DemangledName}.contains("proteus::jit_object")) {
      JitObjectAnnotations.insert(&F);
    }
  }

  MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> RCInfoMap;
  if (!JitArgAnnotations.empty())
    parseJitArgAnnotations(JitArgAnnotations, RCInfoMap);

  if (!JitArrayAnnotations.empty())
    parseJitArrayAnnotations(JitArrayAnnotations, RCInfoMap);

  if (!JitObjectAnnotations.empty())
    parseJitObjectAnnotations(JitObjectAnnotations, RCInfoMap);

  // Lastly, parse global annotations and remove them after parsing.
  parseJitGlobalAnnotations(StubToKernelMap, RCInfoMap);
  removeJitGlobalAnnotations();
  removeProteusLambdaAnnotations();

  // If this is device compilation the pass emits a JSON file that stores this
  // information for the host compilation pass to parse for instrumentation.
  // Otherwise, it is host compilation and attempts to parse the JSON manifest
  // file.  The JSON file is uniquely named using the TU unique file ID.
  if (isDeviceCompilation(M)) {
    if (!RCInfoMap.empty())
      createDeviceManifestFile(RCInfoMap);
  } else {
    parseManifestFileAnnotations(StubToKernelMap, RCInfoMap);
  }

  // Populate the JitFunctionInfoMap using the runtime constant info map, which
  // gathers all annotations parsed and from the manifest file.
  for (auto &[F, RCInfos] : RCInfoMap) {
    // Sort the runtime constant infos by argument position for determinism.
    auto RCInfoVec = RCInfos.takeVector();
    std::sort(RCInfoVec.begin(), RCInfoVec.end());

    JitFunctionInfoMap.insert(
        {F, JitFunctionInfo{SmallSetVector<RuntimeConstantInfo, 16>{
                                RCInfoVec.begin(), RCInfoVec.end()},
                            ""}});
    AttachMetadata(*F);
  }
}

void AnnotationHandler::parseManifestFileAnnotations(
    const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Parse the JSON manifest from device compilation, if there exists one, and
  // update JitFunctionInfoMap.
  SmallString<64> UniqueFilename = getUniqueManifestFilename();
  // If there is no manifest file, return early.
  if (!sys::fs::exists(UniqueFilename))
    return;

  auto ErrorOrManifestBuf = MemoryBuffer::getFile(UniqueFilename);
  if (!ErrorOrManifestBuf)
    reportFatalError("Error reading json manifest file " + UniqueFilename);

  std::unique_ptr<MemoryBuffer> ManifestBuf = std::move(*ErrorOrManifestBuf);
  auto ExpectedJsonValue = json::parse(ManifestBuf->getBuffer());
  if (auto E = ExpectedJsonValue.takeError())
    reportFatalError("Failed to parse json: " + toString(std::move(E)));

  json::Value ManifestValue = *ExpectedJsonValue;
  json::Object *Manifest = ManifestValue.getAsObject();
  if (!Manifest)
    reportFatalError("Failed to parse json: manifest object");

  json::Array *KernelArray = Manifest->getArray("manifest");
  if (!KernelArray)
    reportFatalError("Failed to parse json: kernel array");

  for (auto &Entry : *KernelArray) {
    json::Object *KernelObject = Entry.getAsObject();
    if (!KernelObject)
      reportFatalError("Failed parsing json: kernel object");

    auto OptionalKernelSym = KernelObject->getString("symbol");
    if (!OptionalKernelSym)
      reportFatalError("Failed parsing json: function symbol");

    StringRef KernelSym = *OptionalKernelSym;

    // Find the device stub function searching the StubToKernelMap.
    Function *F = nullptr;
    for (auto [Stub, KernelSymGV] : StubToKernelMap) {
      ConstantDataArray *CDA =
          dyn_cast<ConstantDataArray>(KernelSymGV->getInitializer());
      if (!CDA)
        reportFatalError("Expected ConstantDataArray");
      if (!CDA->isString())
        reportFatalError("Expected string constant storing the kernel symbol");

      // Get the value as a CString to avoid including an extra null
      // terminator character that spuriously fails the following comparison.
      StringRef MappedKernelSym = CDA->getAsCString();
      if (MappedKernelSym == KernelSym) {
        F = dyn_cast<Function>(Stub);
        if (!F)
          reportFatalError("Expected stub function");
        break;
      }
    }

    if (!F) {
      DEBUG(Logger::logs("proteus-pass")
            << "Warning: missing device stub Function for kernel sym "
            << KernelSym << " in manifest file: "
            << UniqueFilename + ", for source file: " << M.getSourceFileName()
            << ", kernel may be unused" << "\n");
      continue;
    }

    json::Array *JsonRCInfoArray = KernelObject->getArray("rc");
    if (!JsonRCInfoArray)
      reportFatalError("Failed parsing json: jit args");
    // Update the RCInfoMap for the stub function proxying the kernel.
    auto &RCInfos = RCInfoMap[F];
    for (auto J : *JsonRCInfoArray) {
      auto CreateRuntimeConstantInfo = [](json::Object *JsonRCDict) {
        json::Value *ArgV = JsonRCDict->get("arg");
        if (!ArgV)
          reportFatalError("Expected arg in runtime constant info");
        auto OptArgNo = ArgV->getAsInteger();
        int32_t ArgNo = *OptArgNo;

        json::Value *TypeV = JsonRCDict->get("type");
        if (!TypeV)
          reportFatalError("Expected type in runtime constant info");
        auto OptRCType = TypeV->getAsInteger();
        RuntimeConstantType RCType =
            static_cast<RuntimeConstantType>(*OptRCType);

        if (RCType == RuntimeConstantType::ARRAY) {
          json::Value *EltTypeV = JsonRCDict->get("elt-type");
          auto OptEltType = EltTypeV->getAsInteger();
          if (!OptEltType)
            reportFatalError("Expected elt-type in runtime constant info");
          RuntimeConstantType EltType =
              static_cast<RuntimeConstantType>(*OptEltType);

          json::Value *NumEltsV = JsonRCDict->get("num-elts");
          json::Value *NumEltsPosV = JsonRCDict->get("num-elts-pos");
          json::Value *NumEltsTypeV = JsonRCDict->get("num-elts-type");

          // Check for semantic errors.
          if (NumEltsV && NumEltsPosV)
            reportFatalError(
                "Setting both NumElts and NumEltsPos is incorrect");

          if (!NumEltsV && !NumEltsPosV)
            reportFatalError("Either NumElts or NumEltsPos must be set");

          if (NumEltsPosV && !NumEltsTypeV)
            reportFatalError("NumEltsType must be set if NumEltsPos is set");

          if (!NumEltsPosV && NumEltsTypeV)
            reportFatalError("NumEltsPos must be set if NumEltsType is set");

          if (NumEltsV) {
            int32_t NumElts = NumEltsV->getAsInteger().value();
            return RuntimeConstantInfo{RCType, ArgNo, NumElts, EltType};
          }

          int32_t NumEltsPos = NumEltsPosV->getAsInteger().value();
          RuntimeConstantType NumEltsType = static_cast<RuntimeConstantType>(
              NumEltsTypeV->getAsInteger().value());
          return RuntimeConstantInfo{RCType, ArgNo, EltType, NumEltsType,
                                     NumEltsPos};
        }

        if (RCType == RuntimeConstantType::OBJECT) {
          json::Value *SizeV = JsonRCDict->get("size");
          if (!SizeV)
            reportFatalError("Expected key 'size' type in runtime "
                             "constant info for object");
          auto OptSize = SizeV->getAsInteger();
          if (!OptSize)
            reportFatalError("Expected key 'size' type is integer");
          int32_t Size = *OptSize;

          json::Value *PassByValueV = JsonRCDict->get("pass-by-value");
          if (!PassByValueV)
            reportFatalError("Expected key 'pass-by-value' type in runtime "
                             "constant info for object");
          auto OptPassByValue = PassByValueV->getAsBoolean();
          if (!OptPassByValue)
            reportFatalError("Expected key 'pass-by-value' is boolean");
          bool PassByValue = *OptPassByValue;

          return RuntimeConstantInfo{RCType, ArgNo, Size, PassByValue};
        }

        return RuntimeConstantInfo{RCType, ArgNo};
      };

      RuntimeConstantInfo RCI = CreateRuntimeConstantInfo(J.getAsObject());
      if (!RCInfos.insert(RCI))
        reportFatalError("Duplicate JIT annotation for argument (0-index): " +
                         std::to_string(RCI.ArgInfo.Pos));
    }
  }

  std::remove(UniqueFilename.c_str());
}

SmallString<64> AnnotationHandler::getUniqueManifestFilename() {
  auto TmpPath = std::filesystem::temp_directory_path();

  // Use the parent process id and the unique file id to uniquely identify the
  // manifest file for a specific compilation, assuming the parent process is
  // the same for both device and host compilation.
  return {TmpPath.string(),
          "/",
          "proteus-device-manifest-",
          std::to_string(helpers::getParentProcessId()),
          "-",
          getUniqueFileID(M),
          ".json"};
}

void AnnotationHandler::appendToGlobalAnnotations(
    SmallVector<Constant *> &NewAnnotations) {
  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  SmallVector<Constant *> CurrentAnnotations;
  // If there is an llvm.global.annotations global variable we get the info
  // and append, otherwise we need to create it.
  if (GlobalAnnotations) {
    if (Constant *Init = GlobalAnnotations->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentAnnotations.reserve(N + 1);
      for (unsigned I = 0; I != N; ++I) {
        CurrentAnnotations.push_back(cast<Constant>(Init->getOperand(I)));
      }
    }
    GlobalAnnotations->eraseFromParent();
  }

  CurrentAnnotations.insert(CurrentAnnotations.end(), NewAnnotations.begin(),
                            NewAnnotations.end());

  ArrayType *AT =
      ArrayType::get(Types.GlobalAnnotationEltTy, CurrentAnnotations.size());
  Constant *Init = ConstantArray::get(AT, CurrentAnnotations);
  auto *AnnotationsGV = new GlobalVariable(M, Init->getType(), false,
                                           GlobalValue::AppendingLinkage, Init,
                                           "llvm.global.annotations");
  AnnotationsGV->setSection("llvm.metadata");
}

Constant *AnnotationHandler::createJitAnnotation(
    Function *F, const SmallSetVector<RuntimeConstantInfo, 16> &RCInfos) {
  // llvm.global.annotations entry format:
  //  ptr: (addrspace 1) Function pointer
  //  ptr: (addrspace 4) Annotations string
  //  ptr: (addrspace 4) Source file string
  //  i32: Line number,
  //  ptr: (addrspace 1) Arguments pointer
  IRBuilder<> IRB{M.getContext()};
  constexpr size_t NumElts = 5;
  Constant *AnnotationVals[NumElts];

  if (isDeviceCompilation(M)) {
    constexpr unsigned GlobalAddressSpace = 1;
    constexpr unsigned ConstantAddressSpace = 4;
    AnnotationVals[0] = cast<Constant>(
        IRB.CreateAddrSpaceCast(F, IRB.getPtrTy(GlobalAddressSpace)));
    AnnotationVals[1] =
        IRB.CreateGlobalString("jit", ".str", ConstantAddressSpace, &M);
    AnnotationVals[2] = IRB.CreateGlobalString(M.getSourceFileName(), "",
                                               ConstantAddressSpace, &M);
  } else {
    AnnotationVals[0] = F;
    AnnotationVals[1] = IRB.CreateGlobalString("jit", ".str", 0, &M);
    AnnotationVals[2] =
        IRB.CreateGlobalString(M.getSourceFileName(), "", 0, &M);
  }
  // We don't know the line number, hence we store 0.
  AnnotationVals[3] = IRB.getInt32(0);

  // If there aren't any runtime constant arguments set annotation value to
  // the null value and return early.
  if (RCInfos.empty()) {
    AnnotationVals[4] =
        Constant::getNullValue(Types.GlobalAnnotationEltTy->getElementType(4));

    Constant *NewAnnotation = ConstantStruct::get(
        Types.GlobalAnnotationEltTy, ArrayRef{AnnotationVals, NumElts});

    return NewAnnotation;
  }

  // Create the JIT argument information in long form to include the argument
  // number and possible size. Each RCInfo will create a global variable
  // string with key-value pairs to store the information.
  SmallVector<Type *> RCInfoTypes{RCInfos.size(), Types.PtrTy};
  StructType *RCInfoStructTy = StructType::get(M.getContext(), RCInfoTypes);
  SmallVector<Constant *> RCInfoConsts;
  for (const auto &RCInfo : RCInfos) {
    json::Object JsonRCDict;
    JsonRCDict["arg"] = RCInfo.ArgInfo.Pos + 1;
    JsonRCDict["type"] = static_cast<int32_t>(RCInfo.ArgInfo.Type);

    if (RCInfo.OptArrInfo) {
      json::Object JsonArrInfo;
      JsonArrInfo["elt-type"] =
          static_cast<int32_t>(RCInfo.OptArrInfo->EltType);

      if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
        JsonArrInfo["num-elts-type"] =
            static_cast<int32_t>(RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
        JsonArrInfo["num-elts-pos"] = RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos;
      } else {
        JsonArrInfo["num-elts"] = RCInfo.OptArrInfo->NumElts;
      }

      JsonRCDict["arrinfo"] = std::move(JsonArrInfo);
    }

    if (RCInfo.OptObjInfo) {
      json::Object JsonObjInfo;
      JsonObjInfo["size"] = RCInfo.OptObjInfo->Size;
      JsonObjInfo["pass-by-value"] = RCInfo.OptObjInfo->PassByValue;

      JsonRCDict["objinfo"] = std::move(JsonObjInfo);
    }

    std::string RCInfoStr;
    raw_string_ostream OS{RCInfoStr};
    OS << json::Value(std::move(JsonRCDict));

    if (isDeviceCompilation(M)) {
      constexpr unsigned ConstAddressSpace = 4;
      Constant *StrConst =
          IRB.CreateGlobalString(RCInfoStr, ".str", ConstAddressSpace, &M);
      RCInfoConsts.push_back(ConstantExpr::getPointerBitCastOrAddrSpaceCast(
          StrConst, Types.PtrTy));
    } else
      RCInfoConsts.push_back(IRB.CreateGlobalString(RCInfoStr, ".str", 0, &M));
  }
  Constant *RCInfoInit = ConstantStruct::get(RCInfoStructTy, RCInfoConsts);

  GlobalVariable *RCInfoGV = nullptr;
  if (isDeviceCompilation(M)) {
    constexpr unsigned GlobalAddressSpace = 1;
    RCInfoGV = new GlobalVariable(
        M, RCInfoInit->getType(), true, GlobalValue::PrivateLinkage, RCInfoInit,
        ".args", nullptr, llvm::GlobalValue::NotThreadLocal,
        GlobalAddressSpace);
  } else {
    RCInfoGV =
        new GlobalVariable(M, RCInfoInit->getType(), true,
                           GlobalValue::PrivateLinkage, RCInfoInit, ".args");
  }
  AnnotationVals[4] = RCInfoGV;

  Constant *NewAnnotation = ConstantStruct::get(
      Types.GlobalAnnotationEltTy, ArrayRef{AnnotationVals, NumElts});

  return NewAnnotation;
}

void AnnotationHandler::createDeviceManifestFile(
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Emit JSON file manifest which contains the kernel symbol and
  // JIT-annotated arguments.
  SmallString<64> UniqueFilename = getUniqueManifestFilename();

  int FD = -1;
  std::error_code EC = sys::fs::openFileForWrite(UniqueFilename, FD);
  if (EC)
    reportFatalError("Error creating device manifest file: " + UniqueFilename +
                     ", error: " + EC.message());
  raw_fd_ostream OS(FD, /*shouldClose=*/true);
  if (OS.has_error())
    reportFatalError("Error opening device manifest file: " + UniqueFilename +
                     ", error: " + OS.error().message());

  json::Object ManifestInfo;
  json::Array KernelArray;

  for (auto &[F, RCInfos] : RCInfoMap) {
    // Skip non-kernel device functions that were JIT annotated (e.g., a lambda)
    // as their argument handling is independent of the manifest.
    if (!isDeviceKernel(F))
      continue;

    json::Object KernelObject;
    KernelObject["symbol"] = F->getName();

    json::Array JsonRCInfoArray;
    for (auto &RCInfo : RCInfos) {
      json::Object JsonRCDict;
      JsonRCDict["arg"] = RCInfo.ArgInfo.Pos;
      JsonRCDict["type"] = static_cast<int32_t>(RCInfo.ArgInfo.Type);

      if (RCInfo.OptArrInfo) {
        JsonRCDict["elt-type"] =
            static_cast<int32_t>(RCInfo.OptArrInfo->EltType);

        if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
          JsonRCDict["num-elts-type"] =
              static_cast<int32_t>(RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
          JsonRCDict["num-elts-pos"] = RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos;
        } else {
          JsonRCDict["num-elts"] = RCInfo.OptArrInfo->NumElts;
        }
      }

      if (RCInfo.OptObjInfo) {
        JsonRCDict["size"] = static_cast<int32_t>(RCInfo.OptObjInfo->Size);
        JsonRCDict["pass-by-value"] = RCInfo.OptObjInfo->PassByValue;
      }

      JsonRCInfoArray.push_back(std::move(JsonRCDict));
    }

    KernelObject["rc"] = std::move(JsonRCInfoArray);

    KernelArray.push_back(std::move(KernelObject));
  }

  ManifestInfo["manifest"] = std::move(KernelArray);

  OS << formatv("{0:2}", json::Value(std::move(ManifestInfo)));
  OS.close();
}

void AnnotationHandler::parseJitArgAnnotations(
    SmallPtrSetImpl<Function *> &JitArgAnnotations,
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Iterate over all proteus::jit_arg annotations and store the information
  // in RCInfoMap.
  SmallVector<CallBase *> CallsToDelete;
  for (Function *AnnotationF : JitArgAnnotations) {
    for (User *Usr : AnnotationF->users()) {
      CallBase *CB = dyn_cast<CallBase>(Usr);
      if (!CB)
        continue;

      if (CB->getNumUses() != 0)
        reportFatalError("Expected zero uses of the annotation function");

      CallsToDelete.push_back(CB);

      Function *JitFunction = CB->getFunction();
      assert(CB->arg_size() == 1 && "Expected single argument");
      auto *V = CB->getArgOperand(0);

      Argument *Arg = getOriginatingArgument(V, CB);
      if (!Arg)
        reportFatalError(
            "Expected non-null argument. Possible cause: proteus::jit_arg "
            "argument is not an argument of the enclosing function.");

      auto &ConstantArgs = RCInfoMap[JitFunction];
      RuntimeConstantType RCType =
          convertTypeToRuntimeConstantType(Arg->getType());
      int32_t ArgNo = Arg->getArgNo();
      RuntimeConstantInfo RCI{
          RCType,
          ArgNo,
      };
      if (!ConstantArgs.insert(RCI))
        reportFatalError("Duplicate argument number found: " +
                         std::to_string(Arg->getArgNo()));
    }
  }

  // Remove JitArg annotations and their calls as they are not needed anymore.
  // Also removing them avoids errors with LTO and O0 compilation: LTO will
  // find those annotations and attempt to create a manifest which will cause
  // an error since it is not backed by a source filename.
  for (CallBase *CB : CallsToDelete)
    CB->eraseFromParent();

  for (Function *AnnotationF : JitArgAnnotations)
    AnnotationF->eraseFromParent();

  if (verifyModule(M, &errs()))
    reportFatalError("Broken JIT module found, compilation aborted!");
}

void AnnotationHandler::parseJitArrayAnnotations(
    SmallPtrSetImpl<Function *> &JitArrayAnnotations,
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Iterate over all proteus::jit_array annotations and store the information
  // in RCInfoMap.
  SmallVector<CallBase *> CallsToDelete;
  for (Function *AnnotationF : JitArrayAnnotations) {
    for (User *Usr : AnnotationF->users()) {
      CallBase *CB = dyn_cast<CallBase>(Usr);
      if (!CB)
        continue;

      if (CB->getNumUses() != 0)
        reportFatalError("Expected zero uses of the annotation function");

      CallsToDelete.push_back(CB);

      Function *JitFunction = CB->getFunction();
      assert(CB->arg_size() == 3 &&
             "Expected three arguments (value, size, element)");
      auto *V = CB->getArgOperand(0);
      auto *Size = CB->getArgOperand(1);
      auto *Elt = CB->getArgOperand(2);

      Argument *Arg = getOriginatingArgument(V, CB);
      if (!Arg)
        reportFatalError(
            "Expected non-null argument. Possible cause: proteus::jit_array "
            "argument is not an argument of the enclosing function.");

      auto &ConstantArgs = RCInfoMap[JitFunction];
      RuntimeConstantType RCType = RuntimeConstantType::ARRAY;
      int32_t ArgNo = Arg->getArgNo();

      int32_t NumElts = 0;
      RuntimeConstantType EltType =
          convertTypeToRuntimeConstantType(Elt->getType());
      auto *ConstInt = dyn_cast<ConstantInt>(Size);
      if (ConstInt) {
        NumElts = ConstInt->getZExtValue();

        RuntimeConstantInfo RCI{RCType, ArgNo, NumElts, EltType};
        if (!ConstantArgs.insert(RCI))
          reportFatalError("Duplicate argument number found: " +
                           std::to_string(Arg->getArgNo()));
      } else {
        // Find NumElts corresponding to a runtime constant argument.
        Argument *NumEltsArg = getOriginatingArgument(Size, CB);
        if (!NumEltsArg)
          reportFatalError("Expected non-null argument for size");

        RuntimeConstantType NumEltsType =
            convertTypeToRuntimeConstantType(NumEltsArg->getType());
        int32_t NumEltsPos = NumEltsArg->getArgNo();

        RuntimeConstantInfo RCI{RCType, ArgNo, EltType, NumEltsType,
                                NumEltsPos};
        if (!ConstantArgs.insert(RCI))
          reportFatalError("Duplicate argument number found: " +
                           std::to_string(Arg->getArgNo()));
      }
    }
  }

  // Remove JitArray annotations and their calls as they are not needed
  // anymore. Also removing them avoids errors with LTO and O0 compilation:
  // LTO will find those annotations and attempt to create a manifest which
  // will cause an error since it is not backed by a source filename.
  for (CallBase *CB : CallsToDelete)
    CB->eraseFromParent();

  for (Function *AnnotationF : JitArrayAnnotations)
    AnnotationF->eraseFromParent();

  if (verifyModule(M, &errs()))
    reportFatalError("Broken JIT module found, compilation aborted!");
}

void AnnotationHandler::parseJitObjectAnnotations(
    SmallPtrSetImpl<Function *> &JitObjectAnnotations,
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  // Iterate over all proteus::jit_object annotations and store the
  // information in the JitArgs map.
  SmallVector<CallBase *> CallsToDelete;
  for (Function *AnnotationF : JitObjectAnnotations) {
    for (User *Usr : AnnotationF->users()) {
      CallBase *CB = dyn_cast<CallBase>(Usr);
      if (!CB)
        continue;

      if (CB->getNumUses() != 0)
        reportFatalError("Expected zero uses of the annotation function");

      CallsToDelete.push_back(CB);

      Function *JitFunction = CB->getFunction();
      assert(CB->arg_size() == 2 &&
             "Expected three arguments (value, size, element)");
      auto *Ptr = CB->getArgOperand(0);
      auto *SizeV = CB->getArgOperand(1);

      Argument *Arg = getOriginatingArgument(Ptr, CB);
      if (Arg) {
        // Found originating pointer argument, so create runtime constant info
        // for the object.
        int32_t ArgNo = Arg->getArgNo();

        ConstantInt *SizeCI = dyn_cast<ConstantInt>(SizeV);
        if (!SizeCI)
          reportFatalError("Expected constant for the size argument of "
                           "proteus::jit_object");
        int32_t Size = SizeCI->getZExtValue();

        // If the object argument is passed byref or byval then the argument
        // is a pointer to the object, otherwise it is a pointer to a pointer
        // of the object.
        bool PassByValue =
            (Arg->hasByRefAttr() || Arg->hasByValAttr()) ? true : false;

        RuntimeConstantType RCType = RuntimeConstantType::OBJECT;
        RuntimeConstantInfo RCI{RCType, ArgNo, Size, PassByValue};

        auto &ConstantArgs = RCInfoMap[JitFunction];
        if (!ConstantArgs.insert(RCI))
          reportFatalError("Duplicate argument number found: " +
                           std::to_string(ArgNo));
      } else {
        // Detect coercion and gather scalar arguments to create runtime
        // constants infos for those.
        SmallPtrSet<Argument *, 16> CoercedArgs =
            tryGatherCoercedArguments(Ptr, CB);

        if (CoercedArgs.empty())
          reportFatalError(
              "Expected non-null argument. Possible cause: "
              "proteus::jit_object "
              "argument is not an argument of the enclosing function and "
              "coercion could not be detected.");

        // Create new runtime constants for scalars in coerced arguments.
        auto &ConstantArgs = RCInfoMap[JitFunction];
        for (Argument *Arg : CoercedArgs) {
          RuntimeConstantType RCType =
              convertTypeToRuntimeConstantType(Arg->getType());

          auto CreateRuntimeConstantInfo = [&RCType, &Arg]() {
            if (isScalarRuntimeConstantType(RCType)) {
              RuntimeConstantInfo RCI{RCType,
                                      static_cast<int32_t>(Arg->getArgNo())};
              return RCI;
            }

            if (RCType == RuntimeConstantType::STATIC_ARRAY) {
              int32_t NumElts = Arg->getType()->getArrayNumElements();
              Type *EltTy = Arg->getType()->getArrayElementType();
              RuntimeConstantType EltRCType =
                  convertTypeToRuntimeConstantType(EltTy);
              RuntimeConstantInfo RCI{RCType,
                                      static_cast<int32_t>(Arg->getArgNo()),
                                      NumElts, EltRCType};
              return RCI;
            }

            if (RCType == RuntimeConstantType::VECTOR) {
              RuntimeConstantInfo RCI =
                  createVectorRuntimeConstantInfo(RCType, Arg);
              return RCI;
            }

            reportFatalError("Expected scalar or static array type for "
                             "argument " +
                             std::to_string(Arg->getArgNo()));
          };

          RuntimeConstantInfo RCI = CreateRuntimeConstantInfo();
          if (!ConstantArgs.insert(RCI))
            reportFatalError("Duplicate argument number found: " +
                             std::to_string(Arg->getArgNo()));
        }
      }
    }
  }

  // Remove annotations and their calls as they are not needed anymore.  Also
  // removing them avoids errors with LTO and O0 compilation: LTO will find
  // those annotations and attempt to create a manifest which will cause an
  // error since it is not backed by a source filename.
  for (CallBase *CB : CallsToDelete)
    CB->eraseFromParent();

  for (Function *AnnotationF : JitObjectAnnotations)
    AnnotationF->eraseFromParent();

  if (verifyModule(M, &errs()))
    reportFatalError("Broken JIT module found, compilation aborted!");
}

static RuntimeConstantInfo parseAttributeJsonRuntimeConstantArray(
    RuntimeConstantType RCType, int32_t ArgNo, json::Object *JsonArrInfo) {
  RuntimeConstantType EltType;
  std::optional<int32_t> OptNumElts = std::nullopt;
  std::optional<int32_t> OptNumEltsPos = std::nullopt;
  std::optional<RuntimeConstantType> OptNumEltsType = std::nullopt;

  for (auto &[K, V] : *JsonArrInfo) {
    if (K == "elt-type") {
      auto OptEltType = V.getAsInteger();
      if (!OptEltType)
        reportFatalError(
            "Expected integer value for array info key 'elt-type'");
      EltType = static_cast<RuntimeConstantType>(*OptEltType);
    } else if (K == "num-elts") {
      OptNumElts = V.getAsInteger();
      if (!OptNumElts)
        reportFatalError(
            "Expected integer value for array info key 'num-elts'");
    } else if (K == "num-elts-pos") {
      OptNumEltsPos = V.getAsInteger();
      if (!OptNumEltsPos)
        reportFatalError(
            "Expected integer value for array info key 'num-elts-pos'");
    } else if (K == "num-elts-type") {
      auto OptV = V.getAsInteger();
      if (!OptV)
        reportFatalError(
            "Expected integer value for array info key 'num-elts-type'");
      OptNumEltsType = static_cast<RuntimeConstantType>(*OptV);
    }
  }

  // Semantic checks for num-elts and num-elts-pos keys.
  if (OptNumElts && OptNumEltsPos)
    reportFatalError("Expected either of mutually exclusive keys 'num-elts' "
                     "and 'num-elts-pos' to be set, but they are both set "
                     "for array info of runtime constant");

  if (!OptNumElts && !OptNumEltsPos)
    reportFatalError("Expected either NumElts or NumEltsPos set for array "
                     "info of runtime constant");

  if (OptNumEltsPos && !OptNumEltsType)
    reportFatalError("Expected NumEltsType is set when NumEltsPos is set for "
                     "array info of runtime constant");

  if (!OptNumEltsPos && OptNumEltsType)
    reportFatalError("Expected NumEltsPos is set when NumEltrType is set "
                     "for array info of runtime constant");

  if (OptNumElts) {
    return RuntimeConstantInfo{RCType, ArgNo, *OptNumElts, EltType};
  }

  return RuntimeConstantInfo{RCType, ArgNo, EltType, *OptNumEltsType,
                             *OptNumEltsPos};
}

static RuntimeConstantInfo parseAttributeJsonRuntimeConstantObject(
    RuntimeConstantType RCType, int32_t ArgNo, json::Object *JsonObjInfo) {
  int32_t Size;
  bool PassByValue;
  SmallVector<int32_t> ArgNums;
  SmallVector<int32_t> Offsets;

  for (auto &[K, V] : *JsonObjInfo) {
    if (K == "size") {
      auto OptSize = V.getAsInteger();
      if (!OptSize)
        reportFatalError("Expected integer value for key 'size'");
      Size = *OptSize;

      continue;
    }

    if (K == "pass-by-value") {
      auto OptPassByValue = V.getAsBoolean();
      if (!OptPassByValue)
        reportFatalError("Expected integer value for key 'size'");
      PassByValue = *OptPassByValue;

      continue;
    }

    reportFatalError("Unsupported key " + K.str());
  }

  return RuntimeConstantInfo{RCType, ArgNo, Size, PassByValue};
}

static llvm::StringRef getGlobalString(const llvm::GlobalVariable &GV) {
  if (!GV.hasInitializer())
    return {};

  const llvm::Constant *Init = GV.getInitializer();

  if (auto *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(Init)) {
    if (CDS->isCString())
      return CDS->getAsCString(); // drops trailing '\0'
    if (CDS->isString())
      return CDS->getAsString(); // may include '\0'
  }

  if (auto *CDA = llvm::dyn_cast<llvm::ConstantDataArray>(Init)) {
    if (CDA->isCString())
      return CDA->getAsCString();
    if (CDA->isString())
      return CDA->getAsString();
  }

  return {};
}

void AnnotationHandler::parseJitGlobalAnnotations(
    const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
    MapVector<Function *, SmallSetVector<RuntimeConstantInfo, 16>> &RCInfoMap) {
  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (!GlobalAnnotations)
    return;

  if (GlobalAnnotations->getInitializer()->isNullValue())
    return;

  auto *Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
  DEBUG(Logger::logs("proteus-pass") << "Annotation Array " << *Array << "\n");
  for (unsigned int I = 0; I < Array->getNumOperands(); I++) {
    auto *Entry = cast<ConstantStruct>(Array->getOperand(I));
    DEBUG(Logger::logs("proteus-pass") << "Entry " << *Entry << "\n");
    if (Entry->isNullValue()) {
      DEBUG(Logger::logs("proteus-pass") << "Skipping null entry...\n");
      continue;
    }

    auto *Fn = dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());
    auto *AnnotName = dyn_cast<GlobalVariable>(Entry->getOperand(1));
    auto AnnotStr = getGlobalString(*AnnotName);

    assert(Fn && "Expected function in entry operands");

    if (isDeviceCompilation(M) && AnnotStr.contains("jit")) {
      // Check the annotated function is a kernel function or a device
      // lambda for device compilation.
      if (!isDeviceKernel(Fn) && !isLambdaFunction(*Fn) &&
          !isLambdaFunctorWrapperOperator(*Fn))
        reportFatalError(
            std::string{} + __FILE__ + ":" + std::to_string(__LINE__) +
            " => Expected the annotated Fn " + Fn->getName() + " (" +
            demangle(Fn->getName().str()) +
            ") to be a kernel function or device lambda function!");
    } else {
      // Skip device kernel host stub functions for host compilation, to be
      // handled by device compilation and manifest generation.
      if (isDeviceKernelHostStub(StubToKernelMap, *Fn)) {
        DEBUG(Logger::logs("proteus-pass")
              << "Skip annotation for device kernel host stub function: "
              << Fn->getName()
              << " , this is handled separately through device compilation and "
                 "manifest generation.\n");
        continue;
      }
    }

    DEBUG(Logger::logs("proteus-pass")
          << "JIT Function " << Fn->getName() << "\n");

    auto *Annotation =
        cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

    DEBUG(Logger::logs("proteus-pass")
          << "Annotation " << Annotation->getAsCString() << "\n");

    // Needs CString for comparison to strip extract null byte character.
    if (Annotation->getAsCString().compare("jit"))
      continue;

    // Get or create the JitFunctionInfo for Fn.
    auto &RCInfos = RCInfoMap[Fn];

    if (Entry->getOperand(4)->isNullValue()) {
      RCInfos = {};
      continue;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n");
    DEBUG(Logger::logs("proteus-pass")
          << "Type AnnotArgs "
          << *Entry->getOperand(4)->getOperand(0)->getType() << "\n");
    auto *AnnotArgs = cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

    SmallSetVector<RuntimeConstantInfo, 16> ParsedRuntimeConstantInfo;
    auto InsertRuntimeConstantInfo = [&ParsedRuntimeConstantInfo](
                                         const RuntimeConstantInfo &RCI) {
      if (!ParsedRuntimeConstantInfo.insert(RCI))
        reportFatalError("Duplicate JIT annotation for argument (0-index): " +
                         std::to_string(RCI.ArgInfo.Pos));
    };

    for (unsigned int J = 0; J < AnnotArgs->getNumOperands(); ++J) {
      Constant *C = AnnotArgs->getOperand(J)->stripPointerCasts();
      // We parse either numeric arguments in short form or the key-value
      // pairs in long form.
      if (isa<ConstantInt>(C)) {
        auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(J));
        // We subtract 1 to convert to 0-start index.
        size_t ArgNo = Index->getValue().getZExtValue() - 1;
        if (ArgNo >= Fn->arg_size())
          reportFatalError("Expected ArgNo (0-index) " + std::to_string(ArgNo) +
                           " < arg_size " + std::to_string(Fn->arg_size()) +
                           " for function " + Fn->getName());

        RuntimeConstantType RCType =
            convertTypeToRuntimeConstantType(Fn->getArg(ArgNo)->getType());
        if (!isScalarRuntimeConstantType(RCType))
          reportFatalError(
              "Expected scalar type for attribute annotation, found " +
              toString(RCType));

        RuntimeConstantInfo RCI{RCType, static_cast<int32_t>(ArgNo)};
        InsertRuntimeConstantInfo(RCI);
      } else {
        auto *CDA = dyn_cast<ConstantDataArray>(C->getOperand(0));
        if (!CDA)
          reportFatalError("Expected constant data array string");

        StringRef RCDictStr = CDA->getAsCString();
        auto ExpectedRCDict = json::parse(RCDictStr);
        if (auto E = ExpectedRCDict.takeError())
          reportFatalError("Failed to parse runtime constant info json dict");

        json::Value RCDictV = *ExpectedRCDict;
        json::Object *RCDict = RCDictV.getAsObject();

        // Parse 'arg' and 'type' keys common across all runtime constant info
        // types.
        auto OptArgNo = RCDict->getInteger("arg");
        if (!OptArgNo)
          reportFatalError(
              "Expected 'arg' key in runtime constant info json dict");
        // We subtract 1 to convert to 0-start index.
        int32_t ArgNo = *OptArgNo - 1;

        auto OptRCType = RCDict->getInteger("type");
        if (!OptRCType) {
          // Auto-detect type for a scalar argument.
          Type *ArgType = Fn->getArg(ArgNo)->getType();
          if (ArgType->isPointerTy())
            reportFatalError(
                "Cannot auto-detect runtime type from argument "
                "LLVM type because it is a pointer. Expected 'type' key in "
                "runtime constant info json dict");
          OptRCType = convertTypeToRuntimeConstantType(ArgType);
        }
        RuntimeConstantType RCType =
            static_cast<RuntimeConstantType>(*OptRCType);

        if (RCType == RuntimeConstantType::OBJECT) {
          json::Object *JsonObjInfo = RCDict->getObject("objinfo");
          if (!JsonObjInfo)
            reportFatalError(
                "Expected 'objinfo' key for runtime constant object");
          RuntimeConstantInfo RCI = parseAttributeJsonRuntimeConstantObject(
              RCType, ArgNo, JsonObjInfo);
          InsertRuntimeConstantInfo(RCI);
        } else if (RCType == RuntimeConstantType::ARRAY ||
                   RCType == RuntimeConstantType::STATIC_ARRAY) {
          json::Object *JsonArrInfo = RCDict->getObject("arrinfo");
          if (!JsonArrInfo)
            reportFatalError(
                "Expected 'objinfo' key for runtime constant object");
          RuntimeConstantInfo RCI = parseAttributeJsonRuntimeConstantArray(
              RCType, ArgNo, JsonArrInfo);
          InsertRuntimeConstantInfo(RCI);
        } else if (RCType == RuntimeConstantType::VECTOR) {
          RuntimeConstantInfo RCI =
              createVectorRuntimeConstantInfo(RCType, Fn->getArg(ArgNo));
          InsertRuntimeConstantInfo(RCI);
        } else if (isScalarRuntimeConstantType(RCType)) {
          RuntimeConstantInfo RCI{RCType, ArgNo};
          InsertRuntimeConstantInfo(RCI);
        } else {
          reportFatalError("Unexpected runtime constant type: " +
                           std::to_string(static_cast<int32_t>(RCType)) +
                           " for function " + Fn->getName());
        }
      }
    }

    // Insert RC infos and sort JFI.ConstantArgs for determinism.
    for (auto &RCInfo : ParsedRuntimeConstantInfo) {
      if (!RCInfos.insert(RCInfo)) {
        reportFatalError("Duplicate entry found for arg no " +
                         std::to_string(RCInfo.ArgInfo.Pos));
      }
    }
  }
}

void AnnotationHandler::removeProteusLambdaAnnotations() {
  // Remove Proteus JIT global annotations after parsing.

  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (!GlobalAnnotations)
    return;

  Constant *Init = GlobalAnnotations->getInitializer();
  if (!Init)
    return;

  // Iterate over the annotations and keep only those that are not related to
  // proteus JIT.
  SmallVector<Constant *> KeepAnnotations;
  unsigned N = Init->getNumOperands();
  for (unsigned I = 0; I != N; ++I) {
    auto *Entry = dyn_cast<ConstantStruct>(Init->getOperand(I));
    if (!Entry) {
      DEBUG(Logger::logs("proteus-pass")
            << "Expected constant struct in global annotations, skipping null"
               "entry...\n");
      continue;
    }

    auto *Annotation =
        dyn_cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));
    if (!Annotation)
      reportFatalError("Expected constant data array as annotation string");

    StringRef AStr = Annotation->getAsCString();
    if (AStr.contains("proteus.register_lambda_impl") ||
        AStr.contains("proteus.register_call") ||
        AStr.contains("proteus.wrapper_call"))
      continue;

    KeepAnnotations.push_back(cast<Constant>(Init->getOperand(I)));
  }

  GlobalAnnotations->eraseFromParent();

  if (KeepAnnotations.empty())
    return;

  // Create new llvm.global.annotations global variable with the remaining
  // annotations.
  ArrayType *AT =
      ArrayType::get(Types.GlobalAnnotationEltTy, KeepAnnotations.size());
  Constant *NewInit = ConstantArray::get(AT, KeepAnnotations);
  auto *AnnotationsGV = new GlobalVariable(M, NewInit->getType(), false,
                                           GlobalValue::AppendingLinkage,
                                           NewInit, "llvm.global.annotations");
  AnnotationsGV->setSection("llvm.metadata");
}

void AnnotationHandler::removeJitGlobalAnnotations() {
  // Remove Proteus JIT global annotations after parsing.

  auto *GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (!GlobalAnnotations)
    return;

  Constant *Init = GlobalAnnotations->getInitializer();
  if (!Init)
    return;

  // Iterate over the annotations and keep only those that are not related to
  // proteus JIT.
  SmallVector<Constant *> KeepAnnotations;
  unsigned N = Init->getNumOperands();
  for (unsigned I = 0; I != N; ++I) {
    auto *Entry = dyn_cast<ConstantStruct>(Init->getOperand(I));
    if (!Entry) {
      DEBUG(Logger::logs("proteus-pass")
            << "Expected constant struct in global annotations, skipping null"
               "entry...\n");
      continue;
    }

    auto *Annotation =
        dyn_cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));
    if (!Annotation)
      reportFatalError("Expected constant data array as annotation string");

    StringRef AStr = Annotation->getAsCString();
    if (AStr == "jit")
      continue;

    KeepAnnotations.push_back(cast<Constant>(Init->getOperand(I)));
  }

  GlobalAnnotations->eraseFromParent();

  if (KeepAnnotations.empty())
    return;

  // Create new llvm.global.annotations global variable with the remaining
  // annotations.
  ArrayType *AT =
      ArrayType::get(Types.GlobalAnnotationEltTy, KeepAnnotations.size());
  Constant *NewInit = ConstantArray::get(AT, KeepAnnotations);
  auto *AnnotationsGV = new GlobalVariable(M, NewInit->getType(), false,
                                           GlobalValue::AppendingLinkage,
                                           NewInit, "llvm.global.annotations");
  AnnotationsGV->setSection("llvm.metadata");
}

} // namespace proteus
