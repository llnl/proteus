// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_multiple.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_multiple.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T> __global__ void kernel(T LB) { LB(); }

__global__ __attribute__((annotate("jit"))) void kernelSimple() {
  printf("Kernel simple\n");
}

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

void lambdaCaller(int V) {
  run(proteus::register_lambda(
      [=, V = proteus::jit_variable(V)] __device__() { printf("V %d\n", V); }));
}

int main() {
  // We expect that lambdas will specialize and NOT hit the cache since its
  // kernel invocation is templated on the unique lambda type.  The
  // non-templated kernelSimple should hit the cache as it is independent of the
  // lambda (type and JIT variables).
  lambdaCaller(1);
  kernelSimple<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  lambdaCaller(2);

  lambdaCaller(1);
  kernelSimple<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  lambdaCaller(2);
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: V 1
// CHECK: Kernel simple
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: V 2
// CHECK: V 1
// CHECK: Kernel simple
// CHECK: V 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 3 accesses 6
// CHECK-COUNT-3: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
