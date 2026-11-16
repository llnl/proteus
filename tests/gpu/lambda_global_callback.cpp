// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_global_callback.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_global_callback.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// A registered wrapper calls a helper with no arguments. That helper can
// still overwrite another closure through globals, so inspecting only the
// wrapper's pointer arguments cannot prove the call leaves Copy unchanged.
template <typename F> __device__ F *GlobalTarget;
template <typename F> __device__ const F *GlobalReplacement;

template <typename F>
__device__ __attribute__((noinline, optnone)) static void writeGlobal() {
  __builtin_memcpy(GlobalTarget<F>, GlobalReplacement<F>, sizeof(F));
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("global callback %d\n", X);
      });
}

template <typename F> static auto makeTrigger(int Value) {
  return proteus::register_lambda(
      [V = proteus::jit_variable(Value)] __host__ __device__ {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        if (V)
          writeGlobal<F>();
#endif
      });
}

template <typename F, typename T>
__global__ __attribute__((annotate("jit"))) static void
kernelGlobal(F Initial, F Replacement, T Trigger) {
  static_assert(std::is_trivially_copyable_v<F>);
  F Copy = Initial;
  GlobalTarget<F> = &Copy;
  GlobalReplacement<F> = &Replacement;
  Trigger();
  Copy();
}

int main() {
  auto Initial = makeBody(401);
  kernelGlobal<<<1, 1>>>(Initial, makeBody(409),
                         makeTrigger<decltype(Initial)>(1));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// A correct specialization or a conservative decline is acceptable.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 401
// RESULT: global callback 409
// clang-format on
