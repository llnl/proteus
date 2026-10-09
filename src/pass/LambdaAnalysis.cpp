//===-- LambdaAnalysis.cpp -- Extact code/runtime info for Proteus JIT --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Instrument host lambda callsites to register runtime constants.  Record
//    and instrument the data layout for user-specified jit_variable calls.
//
//===----------------------------------------------------------------------===//
// clang-format off
#include "LambdaAnalysis.h"
#include "Helpers.h"
#include "Types.h"

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Logger.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
// clang-format on

using namespace llvm;
using namespace proteus;

//-----------------------------------------------------------------------------
// LambdaPass implementation
//-----------------------------------------------------------------------------
namespace {

struct LambdaJitVariableInfo {
  llvm::Value *Slot;
  llvm::Value *Offset;
  llvm::Type *Type;
};

class LambdaPassImpl {
public:
  LambdaPassImpl(Module &M) : Types(M) {}

  bool run(Module &M) {
    // Attach Metadata nodes corresponding to llvm.global.annotations to
    // individual function calls.
    addMetadataNodeWithNameFromLLVMGlobal(M, "proteus.wrapper_call");
    addMetadataNodeWithNameFromLLVMGlobal(M, "proteus.register_call");
    addMetadataNodeWithNameFromLLVMGlobal(M, "proteus.register_call_impl");

    // clang-format off
    // lambda pipeline: LambdaAnalysis sees early LLVM IR emitted at StartEPCallback.
    // proteus::register_lambda emits a LambdaFunctorWrapper template instantiation
    // for every lambda passed to register_lambda.  The goal of this analysis
    // is to determine which slots in a Functor/Lambda's storage will be folded into LLVM
    // IR at runtime and treated as constants. The analysis uses the following steps:
    // For the host module, we emit registration of runtime constants directly into the
    // functor's wrapper operator() method.
    //  (1) make a DenseSet containing anonymous lambda classes (class.anon type)
    //      as registered by __proteus_take_address inside __register_lambda_impl.
    //      Also track functor type --> corresponding class.anon type.  This correspondence
    //      is unique on a per-module level, and the exact class.anon names vary
    //      between host and device modules!  Type names are only stable within the module.
    //  (2) Find jit_variable calls and associate the calls offset with registered,
    //      class.anon types
    //  (3) Look at register_lambda calls, identify the call operators of the functors
    //      from the callsite
    //      (a) (HOST ONLY) Inject __proteus_register_lambda_runtime_constant into the functor operator()
    //          IR with the call operator functor ID as the key.  Use the analysis from (2) to
    //          determine which llvm::Value to pass to the call.
    // (4) (RUNTIME) (DEVICE) Look at all the call operators within a kernel, replace
    //     with runtime constants now propagated from the registry.
    // (5) (RUNTIME) (HOST) For host operator() calls, we trust the values registered
    //     within the LambdaRegistry by __proteus_register_lambda_runtime_constant, and
    //     cleanup those values after cache hit/specialization
    //
    // NOTE: We need this for both host and device modules, since the runtime
    // specialization paths expect the function metadata to be present after
    // cloning/extraction.
    // clang-format on

    DEBUG(Logger::logs("lambda-pass") << "=== Original Host Module\n"
                                      << M << "=== End Original Host Module\n");

    makeLambdaCallsUniquePerFunctorOperator(M);
    if (!isDeviceCompilation(M)) {
      DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          LambdaStorageTypeToJitIndices;
      DenseSet<StructType *> RegisteredLambdaStorageClasses;
      DenseMap<StructType *, StructType *> FunctorTypeToLambdaTypeMap;
      DenseMap<uint64_t, StructType *> FunctorIDToLambdaTypeMap;
      DenseMap<uint64_t, StructType *> FunctorIDToFunctorTypeMap;
      mapLambdaTypeToFunctorType(
          M, RegisteredLambdaStorageClasses, FunctorTypeToLambdaTypeMap,
          FunctorIDToLambdaTypeMap, FunctorIDToFunctorTypeMap);
      instrumentJitVariableStructIndex(M, RegisteredLambdaStorageClasses,
                                       LambdaStorageTypeToJitIndices);
      emitLambdaSchemaMetadata(M, LambdaStorageTypeToJitIndices,
                               FunctorIDToLambdaTypeMap);
      // Note: moving this outside the conditional will add host call to device
      // code, and break the semantics of the device runtime.  These
      // modifications can only occur to the Clang-generated host stub
      registerLambdaRuntimeConstants(M, LambdaStorageTypeToJitIndices,
                                     FunctorIDToLambdaTypeMap,
                                     FunctorIDToFunctorTypeMap);
    }

    DEBUG(Logger::logs("lambda-pass")
          << "=== Post Original Host Module\n"
          << M << "=== End Post Original Host Module\n");

    if (verifyModule(M, &errs()))
      reportFatalError("Broken original module found, compilation aborted!");

    return true;
  }

private:
  ProteusTypes Types;

  static void
  findAnnotatedFunctions(llvm::Module &M, llvm::StringRef Wanted,
                         llvm::SmallVectorImpl<llvm::Function *> &Out) {
    auto *GA = M.getGlobalVariable("llvm.global.annotations");
    if (!GA)
      return;

    auto *CA = llvm::dyn_cast<llvm::ConstantArray>(GA->getOperand(0));
    if (!CA)
      return;

    llvm::SmallDenseSet<llvm::Function *, 32> Seen;

    for (llvm::Value *Elt : CA->operands()) {
      auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Elt);
      if (!CS || CS->getNumOperands() < 5)
        continue;

      auto *F = llvm::dyn_cast<llvm::Function>(
          CS->getOperand(0)->stripPointerCasts());
      if (!F)
        continue;

      llvm::StringRef Ann;
      if (!llvm::getConstantStringInfo(CS->getOperand(1)->stripPointerCasts(),
                                       Ann))
        continue;
      if (Ann != Wanted)
        continue;

      if (!Seen.contains(F))
        Out.emplace_back(F);
      Seen.insert(F);
    }
  }

  static void findAnnotatedFunctions(
      llvm::Module &M, llvm::StringRef Wanted,
      llvm::SmallVectorImpl<std::pair<llvm::Function *, std::uint64_t>> &Out) {
    auto *GA = M.getGlobalVariable("llvm.global.annotations");
    if (!GA)
      return;

    auto *CA = llvm::dyn_cast<llvm::ConstantArray>(GA->getOperand(0));
    if (!CA)
      return;

    llvm::SmallDenseMap<llvm::Function *, std::uint64_t, 32> Seen;

    for (llvm::Value *Elt : CA->operands()) {
      auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Elt);
      if (!CS || CS->getNumOperands() < 5)
        continue;

      auto *F = llvm::dyn_cast<llvm::Function>(
          CS->getOperand(0)->stripPointerCasts());
      if (!F)
        continue;

      llvm::StringRef Ann;
      if (!llvm::getConstantStringInfo(CS->getOperand(1)->stripPointerCasts(),
                                       Ann))
        continue;
      if (Ann != Wanted)
        continue;

      // Parse the u64 payload (operand 4)
      std::uint64_t Id = 0;
      llvm::Value *ArgsPtr = CS->getOperand(4)->stripPointerCasts();
      if (llvm::isa<llvm::ConstantPointerNull>(ArgsPtr))
        continue;

      auto *ArgsGV = llvm::dyn_cast<llvm::GlobalVariable>(ArgsPtr);
      if (!ArgsGV)
        continue;

      auto *ArgsInit = ArgsGV->getInitializer();
      auto *ArgsCS = llvm::dyn_cast<llvm::ConstantStruct>(ArgsInit);
      if (!ArgsCS || ArgsCS->getNumOperands() < 1)
        continue;

      auto *CI = llvm::dyn_cast<llvm::ConstantInt>(ArgsCS->getOperand(0));
      if (!CI)
        continue;

      Id = CI->getZExtValue();

      if (Seen.try_emplace(F, Id).second)
        Out.emplace_back(F, Id);
    }
  }

  void setRegisteredLambdaMetadata(Function *F, uint64_t Id) {
    setU64FunctionMetadata(F, "proteus.registered_lambda", Id);
    if (!F)
      return;
    // We want tbe lambda clone to be inlined into the body of the operator()
    // ultimately. optnone requires noinline, e.g., at -O0.
    if (F->hasOptNone())
      return;
    F->removeFnAttr(Attribute::NoInline);
    F->addFnAttr(Attribute::AlwaysInline);
  }

  void setU64FunctionMetadata(Function *F, StringRef Key, uint64_t Id) {
    if (!F)
      return;

    if (auto *Existing = F->getMetadata(Key)) {
      if (Existing->getNumOperands() < 1)
        return;
      auto *CAM = dyn_cast<ConstantAsMetadata>(Existing->getOperand(0));
      auto *CI = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
      if (!CI)
        return;
      if (CI->getZExtValue() != Id)
        reportFatalError("Conflicting u64 metadata for " + Key.str() +
                         " on function " + F->getName().str());
      return;
    }

    LLVMContext &Ctx = F->getContext();
    auto *IdMD =
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Ctx), Id));
    F->setMetadata(Key, MDNode::get(Ctx, {IdMD}));
  }

  std::optional<uint64_t> getU64Metadata(Function *F, StringRef Key) {
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

  void addMetadataNodeWithNameFromLLVMGlobal(Module &M,
                                             StringRef MetadataName) {
    SmallVector<std::pair<Function *, uint64_t>, 16> WrapperCallFunctions;
    findAnnotatedFunctions(M, MetadataName, WrapperCallFunctions);
    for (auto &[F, Id] : WrapperCallFunctions)
      setU64FunctionMetadata(F, MetadataName, Id);
  }

  Function *cloneLambdaOperator(Function *LambdaOperator, uint64_t Suffix) {
    ValueToValueMapTy VMap;

    Function *NewFunc = Function::Create(
        LambdaOperator->getFunctionType(), LambdaOperator->getLinkage(),
        LambdaOperator->getName() + "_clone_" + std::to_string(Suffix),
        LambdaOperator->getParent());

    auto *NewArgIt = NewFunc->arg_begin();
    for (auto &OldArg : LambdaOperator->args()) {
      NewArgIt->setName(OldArg.getName());
      VMap[&OldArg] = &(*NewArgIt++);
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, LambdaOperator, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);

    return NewFunc;
  }

  // Go through all of the functor operator() methods emitted by the
  // proteus::register_lambda call.  Each operator()
  // has a unique ID, but multiple functor::operator() methods may call the same
  // lambda::operator() method.  Because each one needs to be specialized
  // separately at runtime, clone duplicate lambda::operator() methods.
  void makeLambdaCallsUniquePerFunctorOperator(Module &M) {
    SmallVector<std::pair<Function *, uint64_t>> FunctorOperatorMethods;
    DenseMap<Function *, SmallVector<std::pair<CallBase *, uint64_t>, 8>>
        LambdaCallToFunctor;
    // The preprocessor tags each operator() with a proteus.wrapper_call.  Use
    // the annotation to find the LambdaFunctorWrapper::operator() calls.
    findAnnotatedFunctions(M, "proteus.wrapper_call", FunctorOperatorMethods);

    for (auto &[FunctorOperator, FunctorId] : FunctorOperatorMethods) {
      // Proteus will prune llvm.global.annotations, so add a metadata node with
      // the preprocessor-emitted FunctorID and proteus.wrapper_call keyword.
      setU64FunctionMetadata(FunctorOperator, "proteus.wrapper_call",
                             FunctorId);
      for (auto &BB : *FunctorOperator) {
        for (auto &I : BB) {
          CallBase *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            continue;
          std::string CalledFunctionName = Callee->getName().str();
          std::string DemangledCalledFunctionName =
              demangle(CalledFunctionName);
          bool IsCallOp = DemangledCalledFunctionName.find("operator()") !=
                          std::string::npos;

          bool LooksLikeLambda =
              DemangledCalledFunctionName.find("{lambda") !=
                  std::string::npos || // common demangle form
              DemangledCalledFunctionName.find("::$_") !=
                  std::string::npos || // your main::$_0 form
              CalledFunctionName.find("$_") !=
                  std::string::npos || // survives even if demangle fails
              llvm::StringRef(CalledFunctionName)
                  .starts_with("_ZZ"); // local-scope Itanium

          if (!IsCallOp || !LooksLikeLambda)
            continue;

          LambdaCallToFunctor[Callee].emplace_back(CB, FunctorId);
        }
      }
    }

    DenseMap<Function *, uint64_t> FunctionsToAnnotate;
    for (auto [LambdaOperator, LambdaOperatorCBVec] : LambdaCallToFunctor) {
      DenseMap<uint64_t, SmallVector<CallBase *, 8>> CallsByFunctorId;
      for (auto &[LambdaOpCB, CallerId] : LambdaOperatorCBVec)
        CallsByFunctorId[CallerId].push_back(LambdaOpCB);

      if (CallsByFunctorId.empty())
        continue;

      uint64_t PrimaryFunctorId = std::numeric_limits<uint64_t>::max();
      // Ensure a deterministic ordering of assignment
      for (auto &KV : CallsByFunctorId)
        if (KV.first < PrimaryFunctorId)
          PrimaryFunctorId = KV.first;

      FunctionsToAnnotate.try_emplace(LambdaOperator, PrimaryFunctorId);

      for (auto &[CallerId, Calls] : CallsByFunctorId) {
        if (CallerId == PrimaryFunctorId)
          continue;

        Function *NewOperator = cloneLambdaOperator(LambdaOperator, CallerId);
        for (CallBase *LambdaOpCB : Calls)
          LambdaOpCB->setCalledFunction(NewOperator);

        FunctionsToAnnotate.try_emplace(NewOperator, CallerId);
      }
    }

    for (auto &KV : FunctionsToAnnotate) {
      Function *F = KV.first;
      uint64_t Id = KV.second;
      setRegisteredLambdaMetadata(F, Id);
    }
  }

  StructType *getStructAllocaFromCB(CallBase *CB) {
    auto *Alloca = dyn_cast<AllocaInst>(CB->getArgOperand(0));
    if (!Alloca)
      reportFatalError("Non alloca passed to CB\n");
    auto *AnonClassTy = dyn_cast<StructType>(Alloca->getAllocatedType());
    if (!AnonClassTy)
      reportFatalError(
          "Non struct type allocated and passed to __proteus_take_address");
    return AnonClassTy;
  }

  StructType *getRegisteredAnonClassFromCB(
      CallBase *CB, DenseSet<StructType *> &RegisteredLambdaStorageClasses) {
    Function *F = CB->getCalledFunction();
    constexpr const char *ErrorMsg =
        "Error in Proteus JitInterface API.  Please report this bug at "
        "https://github.com/LLNL/proteus.";
    if (!hasU64Metadata(F, "proteus.register_call_impl")) {
      return nullptr;
    }
    StructType *AnonLambdaType = nullptr;
    for (auto &BB : *F)
      for (auto &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || !CB->getCalledFunction()->getName().contains(
                       "__proteus_take_address"))
          continue;
        auto *AnonTy = getStructAllocaFromCB(CB);
        if (!AnonTy)
          reportFatalError(ErrorMsg);
        RegisteredLambdaStorageClasses.insert(AnonTy);
        AnonLambdaType = AnonTy;
        return AnonLambdaType;
      }
    if (!AnonLambdaType)
      reportFatalError(ErrorMsg);
    return AnonLambdaType;
  }

  /// Look at all functions with register_call metadata.  Find call to
  /// __proteus_take_address, identify functor type.  Find call to function
  /// with metadata proteus.register_call_impl, find call to
  /// __proteus_take_address in the nested implementation call, and map onto the
  /// functor type.
  void mapLambdaTypeToFunctorType(
      Module &M, DenseSet<StructType *> &RegisteredLambdaStorageClasses,
      DenseMap<StructType *, StructType *> &FunctorTypeToLambdaTypeMap,
      DenseMap<uint64_t, StructType *> &FunctorIDToLambdaTypeMap,
      DenseMap<uint64_t, StructType *> &FunctorIDToFunctorTypeMap) {
    SmallVector<Function *> RegisterFunctions;
    constexpr const char *ErrorMsg =
        "Error in Proteus JitInterface API.  Please report this bug at "
        "https://github.com/LLNL/proteus.";
    findAnnotatedFunctions(M, "proteus.register_call", RegisterFunctions);
    if (RegisterFunctions.empty())
      DEBUG(Logger::logs("lambda-pass") << "No register func CBs found\n");
    for (auto *RegisterFunc : RegisterFunctions) {
      DEBUG(Logger::logs("lambda-pass")
            << "Analyzing register_func " << *RegisterFunc << "\n");
      StructType *AnonClassType = nullptr;
      StructType *FunctorType = nullptr;
      std::optional<uint64_t> FunctorID;
      for (auto &BB : *RegisterFunc)
        for (auto &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          StructType *RegisteredTy =
              getRegisteredAnonClassFromCB(CB, RegisteredLambdaStorageClasses);
          if (RegisteredTy) {
            AnonClassType = RegisteredTy;
            FunctorID = getU64Metadata(CB->getCalledFunction(),
                                       "proteus.register_call_impl");
          }
          if (!CB->getCalledFunction()->getName().contains(
                  "__proteus_take_address"))
            continue;
          auto *FunctorTy = getStructAllocaFromCB(CB);
          if (!FunctorTy)
            reportFatalError(ErrorMsg);
          FunctorType = FunctorTy;
        }
      if (!FunctorType || !AnonClassType)
        reportFatalError(ErrorMsg);
      FunctorTypeToLambdaTypeMap[FunctorType] = AnonClassType;
      if (FunctorID) {
        FunctorIDToLambdaTypeMap[*FunctorID] = AnonClassType;
        FunctorIDToFunctorTypeMap[*FunctorID] = FunctorType;
      }
    }
  }

  FunctionCallee getProteusFinalizeCall(Module &M) {
    FunctionType *FnTy = FunctionType::get(Types.VoidTy, {Types.Int64Ty},
                                           /*isVarArg=*/false);
    return M.getOrInsertFunction("__proteus_finalize_register", FnTy);
  }

  FunctionCallee getJitRegisterLambdaRuntimeConstant(Module &M) {
    FunctionType *FnTy =
        FunctionType::get(Types.VoidTy,
                          {Types.Int32Ty, Types.Int32Ty, Types.Int32Ty,
                           Types.PtrTy, Types.Int64Ty},
                          /*isVarArg=*/false);
    return M.getOrInsertFunction("__proteus_register_lambda_runtime_constant",
                                 FnTy);
  }

  std::optional<RuntimeConstantType> getRCTypeForLLVMType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return RuntimeConstantType::BOOL;
    if (Ty->isIntegerTy(8))
      return RuntimeConstantType::INT8;
    if (Ty->isIntegerTy(32))
      return RuntimeConstantType::INT32;
    if (Ty->isIntegerTy(64))
      return RuntimeConstantType::INT64;
    if (Ty->isFloatTy())
      return RuntimeConstantType::FLOAT;
    if (Ty->isDoubleTy())
      return RuntimeConstantType::DOUBLE;
    if (Ty->isFP128Ty() || Ty->isPPC_FP128Ty() || Ty->isX86_FP80Ty())
      return RuntimeConstantType::LONG_DOUBLE;
    if (Ty->isPointerTy())
      return RuntimeConstantType::PTR;
    return std::nullopt;
  }

  void emitLambdaSchemaMetadata(
      Module &M,
      DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          &LambdaStorageTypeToJitIndices,
      DenseMap<uint64_t, StructType *> &FunctorIDToLambdaTypeMap) {
    if (LambdaStorageTypeToJitIndices.empty())
      return;

    NamedMDNode *SchemaMD =
        M.getOrInsertNamedMetadata(LambdaSchemaMetadataName);
    for (const auto &KV : FunctorIDToLambdaTypeMap) {
      uint64_t FunctorID = KV.first;
      auto *LambdaStorageType = KV.second;
      auto It = LambdaStorageTypeToJitIndices.find(LambdaStorageType);
      if (It == LambdaStorageTypeToJitIndices.end())
        continue;

      for (const auto &JitVarInfo : It->second) {
        auto *SlotC = dyn_cast<ConstantInt>(JitVarInfo.Slot);
        auto *OffsetC = dyn_cast<ConstantInt>(JitVarInfo.Offset);
        if (!SlotC || !OffsetC)
          reportFatalError("Failed to emit lambda schema metadata");
        auto RCType = getRCTypeForLLVMType(
            LambdaStorageType->getElementType(SlotC->getZExtValue()));
        if (!RCType)
          reportFatalError("Failed to emit lambda schema metadata");

        LLVMContext &Ctx = M.getContext();
        Metadata *Ops[] = {
            ConstantAsMetadata::get(
                ConstantInt::get(Type::getInt64Ty(Ctx), FunctorID)),
            ConstantAsMetadata::get(
                ConstantInt::get(Type::getInt32Ty(Ctx),
                                 static_cast<uint32_t>(SlotC->getZExtValue()))),
            ConstantAsMetadata::get(ConstantInt::get(
                Type::getInt32Ty(Ctx),
                static_cast<uint32_t>(OffsetC->getZExtValue()))),
            ConstantAsMetadata::get(ConstantInt::get(
                Type::getInt32Ty(Ctx), static_cast<int32_t>(RCType.value())))};
        SchemaMD->addOperand(MDNode::get(Ctx, Ops));
      }
    }
  }

  /// instrumentJitVariableStructIndex modifies calls to proteus::jit_variable
  /// by injecting the corresponding jit variable offset into the call.
  /// todo(bowen): add assert on jit_variable having an alloca within the same
  /// procedure
  void instrumentJitVariableStructIndex(
      Module &M, DenseSet<StructType *> &RegisteredLambdaStorageClasses,
      DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          &JitIndices) {
    DEBUG(Logger::logs("lambda-pass")
          << "finding jit variables users..." << "\n");

    SmallVector<Function *, 16> JitFunctions;

    // todo(bowen) do this with metadata instead of name demangling
    for (auto &F : M.getFunctionList()) {
      std::string DemangledName = demangle(F.getName().str());
      if (StringRef{DemangledName}.contains("proteus::jit_variable")) {
        JitFunctions.push_back(&F);
      }
    }

    for (auto &F : JitFunctions) {
      FunctionType *FTy = F->getFunctionType();
      auto *LoadType = FTy->getParamType(0);
      if (!LoadType)
        reportFatalError("jit function return type null??\n");
      SmallVector<Value *> WorkList;
      DenseSet<Value *> Discovered;
      for (auto *Usr : F->users()) {
        CallBase *CB = dyn_cast<CallBase>(Usr);
        if (!CB)
          continue;
        for (auto *Usr : CB->users())
          WorkList.push_back(Usr);
      }
      // These two integers need to be equivalent--we track down each callsite
      // to an offset in the struct.
      uint16_t NumJitVars = WorkList.size();
      uint16_t NumSlotsFound = 0;

      while (!WorkList.empty()) {
        Value *CurVal = WorkList.back();
        WorkList.pop_back();
        if (Discovered.contains(CurVal))
          continue;
        DEBUG(Logger::logs("lambda-pass")
              << "VISITING JIT VARIABLE OPERAND " << *CurVal << "\n");
        Discovered.insert(CurVal);

        if (StoreInst *Store = dyn_cast<StoreInst>(CurVal))
          WorkList.push_back(Store->getPointerOperand());
        else if (AllocaInst *Alloca = dyn_cast<AllocaInst>(CurVal)) {
          Constant *Slot = ConstantInt::get(Types.Int32Ty, 0);
          auto *PossibleLamType =
              dyn_cast<StructType>(Alloca->getAllocatedType());
          if (!PossibleLamType ||
              !RegisteredLambdaStorageClasses.contains(PossibleLamType))
            continue;
          ++NumSlotsFound;
          JitIndices[PossibleLamType].emplace_back(
              LambdaJitVariableInfo{Slot, Slot, LoadType});
          if (!LoadType)
            reportFatalError("Load type is null\n");
        } else if (CastInst *Cast = dyn_cast<CastInst>(CurVal)) {
          for (auto *Usr : Cast->users())
            WorkList.push_back(Usr);
        } else if (GetElementPtrInst *GEP =
                       dyn_cast<GetElementPtrInst>(CurVal)) {
          StructType *PossibleLamType =
              dyn_cast<StructType>(GEP->getSourceElementType());
          if (!PossibleLamType ||
              !RegisteredLambdaStorageClasses.contains(PossibleLamType)) {
            reportFatalError("jit_variable called on unregistered lambda");
            continue;
          }
          auto *Slot = GEP->getOperand(GEP->getNumOperands() - 1);
          const StructLayout *SL =
              M.getDataLayout().getStructLayout(PossibleLamType);
          ConstantInt *SlotC = dyn_cast<ConstantInt>(Slot);
          if (!SlotC)
            reportFatalError("Expected constant slot");

          auto Offset = SL->getElementOffset(SlotC->getZExtValue());
          Constant *OffsetCI = ConstantInt::get(Types.Int32Ty, Offset);

          JitIndices[PossibleLamType].emplace_back(
              LambdaJitVariableInfo{Slot, OffsetCI, LoadType});
          ++NumSlotsFound;
        }
      }

      if (NumJitVars != NumSlotsFound) {
        DEBUG(Logger::logs("lambda-pass").flush());
        reportFatalError(
            "Analysis found jit_variable callsite outside lambda capture list");
      }
    }
  }

  /// Register runtime constants for host lambdas directly at the callsite by
  /// reading off of the closure pointer using offsets deduced from the
  /// jit_variable analysis earlier.
  void registerLambdaRuntimeConstants(
      Module &M,
      DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          &LambdaStorageTypeToJitIndices,
      DenseMap<uint64_t, StructType *> &FunctorIDToLambdaTypeMap,
      DenseMap<uint64_t, StructType *> &FunctorIDToFunctorTypeMap) {
    constexpr const char *ErrorMsg =
        "Error in LambdaAnalysis pass caused by unexpected Clang LLVM IR "
        "within LambdaFunctorWrapper.   Please report this bug at "
        "https://github.com/LLNL/proteus.\n";
    if (LambdaStorageTypeToJitIndices.empty())
      return;
    DEBUG(Logger::logs("lambda-pass")
          << "registering lambda functions" << "\n");
    SmallVector<std::pair<Function *, uint64_t>, 16> FunctorOperators;
    // At the start of the LLVM pipline, the lambda functor operator will
    // allocate a typed struct corresponding to its storage class.  It will also
    // call the lambda.
    findAnnotatedFunctions(M, "proteus.wrapper_call", FunctorOperators);
    for (auto [Function, ID] : FunctorOperators) {
      CallBase *LambdaCall = nullptr;
      for (auto &BB : *Function) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (CB && hasU64Metadata(CB->getCalledFunction(),
                                   "proteus.registered_lambda"))
            LambdaCall = CB;
        }
      }
      auto FunctorTypeIt = FunctorIDToFunctorTypeMap.find(ID);
      auto LambdaTypeIt = FunctorIDToLambdaTypeMap.find(ID);
      if (!LambdaCall || FunctorTypeIt == FunctorIDToFunctorTypeMap.end() ||
          LambdaTypeIt == FunctorIDToLambdaTypeMap.end() ||
          Function->arg_empty())
        reportFatalError(ErrorMsg);
      StructType *FunctorStructType = FunctorTypeIt->second;
      StructType *AnonLambdaStorageType = LambdaTypeIt->second;

      // Insert __proteus_register_lambda_runtime_constant calls for each lambda
      // type participating in this kernel launch.  The insertion point is right
      // before lambda launch--since we know we have a valid closure pointer at
      // this point.
      constexpr const char *InsertionErrorMsg =
          "Error in LambdaAnalysis pass caused by unexpected jit_variable "
          "analysis result.   Please report this bug at "
          "https://github.com/LLNL/proteus.\n";
      IRBuilder<> Builder(LambdaCall);
      Value *LambdaStoragePtr =
          Builder.CreateStructGEP(FunctorStructType, Function->getArg(0), 0);
      auto JitVarFn = getJitRegisterLambdaRuntimeConstant(M);
      auto FinalizeFn = getProteusFinalizeCall(M);
      auto JitVarIt = LambdaStorageTypeToJitIndices.find(AnonLambdaStorageType);
      if (JitVarIt == LambdaStorageTypeToJitIndices.end())
        continue;
      for (auto JitVarInfo : JitVarIt->getSecond()) {
        if (!JitVarInfo.Type)
          reportFatalError(InsertionErrorMsg);
        ConstantInt *SlotC = dyn_cast<ConstantInt>(JitVarInfo.Slot);
        if (!SlotC)
          reportFatalError(InsertionErrorMsg);
        const auto Idx = SlotC->getZExtValue();
        Type *FieldType = AnonLambdaStorageType->getElementType(Idx);
        auto ProteusRCType = getRCTypeForLLVMType(FieldType);
        if (!ProteusRCType)
          reportFatalError(InsertionErrorMsg);

        Value *FieldPtr = Builder.CreateStructGEP(AnonLambdaStorageType,
                                                  LambdaStoragePtr, Idx);
        Builder.CreateCall(
            JitVarFn,
            {Builder.getInt32(static_cast<int32_t>(ProteusRCType.value())),
             Builder.getInt32(Idx), JitVarInfo.Offset, FieldPtr,
             Builder.getInt64(ID)});
      }
      // Insert the finalize call
      Builder.CreateCall(FinalizeFn, {Builder.getInt64(ID)});
    }
  }
};

} // namespace

bool proteus::runLambdaAnalysis(Module &M) {
  LambdaPassImpl PPI{M};
  return PPI.run(M);
}
