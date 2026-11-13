// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_mixed_field_overwrite.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// This file creates a substantial challenge for the clobbering-aware LambdaInstUseVisitor analysis.
// We create a register_lambda type functor, then write to one slot with a store and other with a
// callbase
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteThirdCapture(F *Body, int Replacement) {
  reinterpret_cast<volatile int *>(Body)[2] = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyThenOverwriteCaptures(const F *Original, int StoreReplacement,
                          int CallReplacement) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 3 * sizeof(int));

  // The final closure combines three provenances: the whole-closure copy, a
  // direct StoreInst, and a write hidden behind a non-intrinsic CallBase.
  F Mixed = *Original;
  reinterpret_cast<volatile int *>(&Mixed)[1] = StoreReplacement;
  overwriteThirdCapture(&Mixed, CallReplacement);
  Mixed();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelMixedFieldOverwrite(F Body, int StoreReplacement, int CallReplacement) {
  copyThenOverwriteCaptures(&Body, StoreReplacement, CallReplacement);
}

static auto makeBody(int First, int Second, int Third) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second),
       Third = proteus::jit_variable(Third)] __host__ __device__ {
        printf("mixed field overwrite %d %d %d\n", First, Second, Third);
      });
}

int main() {
  auto Body = makeBody(941, 947, 949);
  kernelMixedFieldOverwrite<<<1, 1>>>(Body, 953, 967);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// The backend may eventually prove each field independently. Until then, it
// must reject the mixed-provenance closure rather than specialize either
// overwritten capture from the stale whole-closure initializer.
// CHECK: [KernelConfig] ID:{{.*}}kernelMixedFieldOverwrite
// CHECK-NOT: [LambdaSpec]
// CHECK: mixed field overwrite 941 953 967
