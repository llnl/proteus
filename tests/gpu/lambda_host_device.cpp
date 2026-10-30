// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_host_device.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_host_device.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename T> void launcher(T &&LB) {
  auto RegisteredFunctor = proteus::register_lambda(LB);
  kernel<<<1, 1>>>(RegisteredFunctor);
  gpuErrCheck(gpuDeviceSynchronize());
  RegisteredFunctor();
}

int main() {
  int A = 42;
  launcher([=, A = proteus::jit_variable(A)] __host__ __device__()
               __attribute__((annotate("jit"))) { printf("Lambda %d\n", A); });
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK: Lambda 42
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK: Lambda 42
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 1 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
