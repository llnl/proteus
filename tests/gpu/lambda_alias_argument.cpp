// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_alias_argument.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_alias_argument.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// The two formals alias at runtime. A use walk rooted at Target alone cannot
// conclude that its pointee is unchanged because it never visits Alias's copy.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughAlias(F *Target, F *Alias, const F *Replacement) {
  __builtin_memcpy(Alias, Replacement, sizeof(F));
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAlias(F Initial, F Replacement, F Other) {
  static_assert(std::is_trivially_copyable_v<F>);
  invokeThroughAlias(&Initial, &Initial,
                     &Replacement); // CB1 target is modified
  invokeThroughAlias(&Replacement, &Initial,
                     &Other); // CB2 target is unmodified
  // In CB1, Target and Alias connect through the same caller object. In CB2,
  // they connect to different objects. The memcpy is a clobber of Target in
  // CB1, but not CB2. Combining contexts conservatively must retain that write.
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("alias argument %d\n", X);
      });
}

int main() {
  kernelAlias<<<1, 1>>>(makeBody(101), makeBody(109), makeBody(117));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// A correct specialization or a conservative decline is acceptable.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 101
// RESULT: alias argument 109
// RESULT-NEXT: alias argument 109
// clang-format on
