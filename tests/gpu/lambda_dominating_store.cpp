// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_dominating_store.%ext | %FILECHECK: %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_dominating_store.%ext | %FILECHECK: %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename Lambda>
__global__ void kernel_store(Lambda Body, Lambda Body2, Lambda Body3,
                             bool UseSecond) {
  helper(&Body, &Body2, &Body3, UseSecond);
}

template <typename F> __device__ void invoke(F *Fn) { (*Fn)(); }

template <typename F>
__device__ void helper(F *Opt1, F *Opt2, F *Opt3, bool UseSecond) {
  F *volatile Slot = nullptr;

  // A dominance-aware analysis should follow only the store that reaches the
  // invoke below, and ignore the later overwrite.
  if (UseSecond)
    Slot = Opt2;
  else
    Slot = Opt1;

  invoke(&Slot);

  // This store does not dominate the invoke and should not influence
  // provenance, but the current visitAllocaInst walks all alloca users.
  Slot = Opt3;
}

auto getTestLambda(int V) {
  return proteus::register_lambda(
      [=, X = proteus::jit_variable(V)] __host__ __device__ {
        printf("case phi %d\n", X);
      });
}

static void store_case(int V1, int V2, bool UseSecond) {
  kernel_store<<<1, 1>>>(getTestLambda(V1), getTestLambda(V2),
                         getTestLambda(44), UseSecond);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int V1 = 55;
  int V2 = 77;
  // dominance-aware resolution should be 55
  store_case(V1, V2, false);
  // dominance-aware resolution should be 77
  store_case(V1, V2, true);

  // repeat to exercise caching once specialization works
  store_case(V1, V2, false);
  store_case(V1, V2, true);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 55
// CHECK: case phi 55
// CHECK: case phi 77
// CHECK: case phi 55
// CHECK: case phi 77
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 2 accesses 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK: [proteus][JitEngineDevice] ObjectCacheChain rank 0 with 1 level(s):
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// clang-format on
