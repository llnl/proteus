// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_clobber_query_range.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct LambdaEnvelope {
  std::uint64_t Unrelated;
  F Body;
};

// Copy the complete aggregate, then overwrite bytes preceding the registered
// lambda. The later write must not hide the initializer of Body.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterUnrelatedFieldOverwrite(F *Body) {
  LambdaEnvelope<F> Source{0x1111111111111111ULL, *Body};
  LambdaEnvelope<F> Copy = Source;
  Copy.Unrelated = 0x2222222222222222ULL;
  Copy.Body();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelUnrelatedPrefixOverwrite(F Body) {
  invokeAfterUnrelatedFieldOverwrite(&Body);
}

int main() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(619)] __host__ __device__ {
        printf("unrelated prefix overwrite %d\n", X);
      });

  kernelUnrelatedPrefixOverwrite<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: [LambdaSpec] Replacing slot 0 with i32 619
// CHECK: unrelated prefix overwrite 619
