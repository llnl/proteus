// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_inst_visitor_call_context.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct LambdaPair {
  F First;
  F Second;
};

// Both calls instantiate this same function. The analysis must keep the
// formal argument's offset and returned pointer separate for each call site.
__device__ __attribute__((noinline, optnone)) static void *
returnFieldAlias(void *Field) {
  return Field;
}

template <typename Pair>
__device__ __attribute__((noinline, optnone)) static Pair *
returnPairAlias(Pair *PairValue) {
  return static_cast<Pair *>(returnFieldAlias(PairValue));
}

// LLVM shape:
//   %second.field = getelementptr %pair, ptr %storage, i32 0, i32 1
//   %second.alias = call ptr @returnFieldAlias(ptr %second.field)
//   %base.alias   = call ptr @returnFieldAlias(ptr %storage)
//   call void @llvm.memcpy(ptr %base.alias, ptr %replacement, sizeof(F))
//   %second.from.base = getelementptr %pair, ptr %base.alias, i32 0, i32 1
//   %selected = phi ptr [ %second.alias, ... ], [ %second.from.base, ... ]
//   call void %selected()
//
// Storage and Storage.First have the same address. The first and second
// returnFieldAlias calls therefore obtain aliases to the two fields while
// sharing one callee Argument and ReturnInst in LLVM IR. Relative to the
// tracked second field, the calls enter that argument at offsets sizeof(F) and
// zero. The first field is overwritten through the returned base alias, while
// the second field is invoked through that same call site's return. Mixing the
// field-alias context into the base-alias context loses the field offset.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeSecondAfterFirstOverwrite(F *First, F *Replacement, F *Second,
                                bool UseDirectAlias) {
  LambdaPair<F> Storage{*First, *Second};
  auto *SecondAlias = reinterpret_cast<F *>(returnFieldAlias(&Storage.Second));
  auto *BaseAlias = returnPairAlias(&Storage);
  __builtin_memcpy(&BaseAlias->First, Replacement, sizeof(F));
  F *Selected = UseDirectAlias ? SecondAlias : &BaseAlias->Second;
  (*Selected)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCallContext(F First, F Replacement, F Second, bool UseDirectAlias) {
  invokeSecondAfterFirstOverwrite(&First, &Replacement, &Second,
                                  UseDirectAlias);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("call-context %d\n", X);
      });
}

int main() {
  auto First = makeBody(811);
  auto Replacement = makeBody(821);
  auto Second = makeBody(823);
  kernelCallContext<<<1, 1>>>(First, Replacement, Second, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 823
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 {{811|821}}
// CHECK: call-context 823
// clang-format on
