// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_spec_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_spec_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;

  auto ZeroLambda = [=, C = proteus::jit_variable(Zero)]()
                        __attribute__((annotate("jit"))) { printInt(C); };

  auto OneLambda = [=, C = proteus::jit_variable(One)]()
                       __attribute__((annotate("jit"))) { printInt(C); };

  auto TwoLambda = [=, C = proteus::jit_variable(Two)]()
                       __attribute__((annotate("jit"))) { printInt(C); };

  run(ZeroLambda);
  run(OneLambda);
  run(TwoLambda);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
