// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_caller_history.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_caller_history.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// The invocation frame has no writes, but its caller replaces the complete
// closure. Follow memory history in each frame before proceeding outward.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
innerInvoke(F *Target) {
  (*Target)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
outerWrite(F *Target, const F *Replacement) {
  __builtin_memcpy(Target, Replacement, sizeof(F));
  innerInvoke(Target);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCaller(F Initial, F Replacement) {
  static_assert(std::is_trivially_copyable_v<F>);
  outerWrite(&Initial, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("caller history %d\n", X);
      });
}

int main() {
  kernelCaller<<<1, 1>>>(makeBody(301), makeBody(309));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// A correct specialization or a conservative decline is acceptable.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 301
// RESULT: caller history 309
// clang-format on
