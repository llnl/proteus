// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/shared_array.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/shared_array.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// Third run runs with proteus disabled to test fallback shared memory allocation.
// RUN: PROTEUS_DISABLE=1 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/shared_array.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename Lambda> __global__ void kernel(Lambda &&Body) {
  printf("Kernel\n");
  Body();
}

template <typename Lambda> void launcher(Lambda &&Body) {
  kernel<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int Dims = 3;
  launcher(proteus::register_lambda(
      [=, Dims = proteus::jit_variable(Dims)] __host__ __device__() {
        double *Array = proteus::shared_array<double, 10>(Dims);
        Array[0] = 1.0;
        Array[1] = 2.0;
        Array[2] = 3.0;
        printf("Lambda Array[0] %lf Array[1] %lf Array[2] %lf\n", Array[0],
               Array[1], Array[2]);
      }));

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 3
// CHECK-FIRST: [SharedArray] Replace CB double* proteus::shared_array<double, 10ul, 0>(unsigned long, unsigned long) with @.proteus.shared = internal addrspace(3) global [24 x i8] undef, align 16
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel
// CHECK: Lambda Array[0] 1.000000 Array[1] 2.000000 Array[2] 3.000000
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-SECOND: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
