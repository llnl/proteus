// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_def.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_def.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int A = 42;
  auto Lambda =
      [=, A = proteus::jit_variable(A)] __device__()
          __attribute__((annotate("jit"))) { printf("Lambda A %d\n", A); };
  run(Lambda);
  run(Lambda);
  run(Lambda);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK-COUNT-3: Lambda A 42
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 2 accesses 3
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
