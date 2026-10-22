//===-- TransformArgumentSpecialization.h -- Specialize arguments --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_ARGUMENT_SPECIALIZATION_H
#define PROTEUS_TRANSFORM_ARGUMENT_SPECIALIZATION_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/Debug.h>

namespace proteus {

using namespace llvm;

class TransformArgumentSpecialization {
private:
  template <typename T>
  static ArrayRef<T> createArrayRef(const RuntimeConstant &RC) {
    T *TypedPtr = reinterpret_cast<T *>(RC.ArrInfo.Blob.get());
    if (RC.ArrInfo.NumElts <= 0)
      reportFatalError("Invalid number of elements in array: " +
                       std::to_string(RC.ArrInfo.NumElts));

    return ArrayRef<T>(TypedPtr, RC.ArrInfo.NumElts);
  }

  static Constant *createConstantDataArray(Module &M,
                                           const RuntimeConstant &RC) {
    //  Dispatch based on element type.
    switch (RC.ArrInfo.EltType) {
    case RuntimeConstantType::BOOL:
      return ConstantDataArray::get(M.getContext(), createArrayRef<bool>(RC));
    case RuntimeConstantType::INT8:
      return ConstantDataArray::get(M.getContext(), createArrayRef<int8_t>(RC));
    case RuntimeConstantType::INT32:
      return ConstantDataArray::get(M.getContext(),
                                    createArrayRef<int32_t>(RC));
    case RuntimeConstantType::INT64:
      return ConstantDataArray::get(M.getContext(),
                                    createArrayRef<int64_t>(RC));
    case RuntimeConstantType::FLOAT:
      return ConstantDataArray::get(M.getContext(), createArrayRef<float>(RC));
    case RuntimeConstantType::DOUBLE:
      return ConstantDataArray::get(M.getContext(), createArrayRef<double>(RC));
    default:
      reportFatalError("Unsupported array element type: " +
                       toString(RC.ArrInfo.EltType));
    }
  }

  static Constant *createConstantDataVector(Module &M,
                                            const RuntimeConstant &RC) {
    //  Dispatch based on element type.
    switch (RC.ArrInfo.EltType) {
    case RuntimeConstantType::INT8:
      return ConstantDataVector::get(M.getContext(),
                                     createArrayRef<uint8_t>(RC));
    case RuntimeConstantType::INT32:
      return ConstantDataVector::get(M.getContext(),
                                     createArrayRef<uint32_t>(RC));
    case RuntimeConstantType::INT64:
      return ConstantDataVector::get(M.getContext(),
                                     createArrayRef<uint64_t>(RC));
    case RuntimeConstantType::FLOAT:
      return ConstantDataVector::get(M.getContext(), createArrayRef<float>(RC));
    case RuntimeConstantType::DOUBLE:
      return ConstantDataVector::get(M.getContext(),
                                     createArrayRef<double>(RC));
    default:
      reportFatalError("Unsupported vector element type: " +
                       toString(RC.ArrInfo.EltType));
    }
  }

public:
  static void transform(Module &M, Function &F,
                        ArrayRef<RuntimeConstant> RCArray) {
    auto &Ctx = M.getContext();

    // Replace argument uses with runtime constants.
    for (const auto &RC : RCArray) {
      int ArgNo = RC.Pos;
      Argument *Arg = F.getArg(ArgNo);
      Type *ArgType = Arg->getType();
      Constant *C = nullptr;

      switch (RC.Type) {
      case RuntimeConstantType::BOOL: {
        C = ConstantInt::get(ArgType, RC.Value.BoolVal);
        break;
      }
      case RuntimeConstantType::INT8: {
        C = ConstantInt::get(ArgType, RC.Value.Int8Val);
        break;
      }
      case RuntimeConstantType::INT32: {
        C = ConstantInt::get(ArgType, RC.Value.Int32Val);
        break;
      }
      case RuntimeConstantType::INT64: {
        C = ConstantInt::get(ArgType, RC.Value.Int64Val);
        break;
      }
      case RuntimeConstantType::FLOAT: {
        // Logger::logs("proteus") << "RC is Float\n";
        C = ConstantFP::get(ArgType, RC.Value.FloatVal);
        break;
      }
      case RuntimeConstantType::DOUBLE: {
        C = ConstantFP::get(ArgType, RC.Value.DoubleVal);
        break;
      }
      case RuntimeConstantType::LONG_DOUBLE: {
        // NOTE: long double on device should correspond to plain double.
        // XXX: CUDA with a long double SILENTLY fails to create a working
        // kernel in AOT compilation, with or without JIT.
        C = ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
        break;
      }
      case RuntimeConstantType::PTR: {
        auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
        C = ConstantExpr::getIntToPtr(IntC, ArgType);
        break;
      }
      case RuntimeConstantType::ARRAY: {
        Constant *CDA = createConstantDataArray(M, RC);
        // Create a global variable to hold the array.
        GlobalVariable *GV = new GlobalVariable(
            M, CDA->getType(), true, GlobalValue::PrivateLinkage, CDA);

        // Cast to the expected pointer type.
        C = ConstantExpr::getBitCast(GV, ArgType);
        break;
      }
      case RuntimeConstantType::STATIC_ARRAY: {
        C = createConstantDataArray(M, RC);
        break;
      }
      case RuntimeConstantType::VECTOR: {
        C = createConstantDataVector(M, RC);
        break;
      }
      case RuntimeConstantType::OBJECT: {
        Constant *CDA = ConstantDataArray::getRaw(
            StringRef{reinterpret_cast<const char *>(RC.ObjInfo.Blob.get()),
                      static_cast<size_t>(RC.ObjInfo.Size)},
            RC.ObjInfo.Size, Type::getInt8Ty(M.getContext()));
        // Create a global variable to hold the array.
        GlobalVariable *GV = new GlobalVariable(
            M, CDA->getType(), true, GlobalValue::PrivateLinkage, CDA);
        // Set alignment (16 bytes) to safely load int/float/double.
        GV->setAlignment(Align(16));

        // Cast to the expected pointer type.
        C = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, ArgType);
        break;
      }
      default: {
        std::string TypeString;
        raw_string_ostream TypeOstream(TypeString);
        ArgType->print(TypeOstream);
        reportFatalError("JIT Incompatible type in runtime constant: " +
                         TypeOstream.str());
      }
      }

      auto TraceOut = [](Function &F, int ArgNo, Constant *C) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[ArgSpec] Replaced Function " << F.getName() << " ArgNo "
           << ArgNo << " with value " << *C->stripPointerCasts() << "\n";

        return S;
      };

      PROTEUS_DBG(Logger::logs("proteus") << TraceOut(F, ArgNo, C));
      if (Config::get().traceSpecializations())
        Logger::trace(TraceOut(F, ArgNo, C));
      Arg->replaceAllUsesWith(C);
    }
  }
};

} // namespace proteus

#endif
