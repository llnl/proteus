// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_source_history.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_source_history.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// Resolving Target's copy must also resolve Source's bytes immediately before
// that copy. Source's pointer identity alone gives its stale kernel argument.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSourceWrite(F *Target, F *Source, const F *Replacement) {
  __builtin_memcpy(Source, Replacement, sizeof(F));
  __builtin_memcpy(Target, Source, sizeof(F));
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSource(F Initial, F Source, F Replacement) {
  static_assert(std::is_trivially_copyable_v<F>);
  invokeAfterSourceWrite(&Initial, &Source, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("source history %d\n", X);
      });
}

int main() {
  kernelSource<<<1, 1>>>(makeBody(201), makeBody(203), makeBody(209));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// A correct specialization or a conservative decline is acceptable.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 201
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 203
// RESULT: source history 209
// clang-format on
