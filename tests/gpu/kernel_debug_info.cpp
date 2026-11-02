// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="ir-dump" %build/kernel_debug_info.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ __attribute__((always_inline)) double scale(double X, double A) {
  return A * X;
}

__global__ __attribute__((annotate("jit", 2))) void kernel(double *X,
                                                           double A) {
  X[threadIdx.x] = scale(X[threadIdx.x], A);
}

int main() {
  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double)));
  X[0] = 1.0;
  kernel<<<1, 1>>>(X, 2.0);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("X %f\n", X[0]);
  gpuErrCheck(gpuFree(X));

  return 0;
}

// The kernel is compiled with full debug info. The JIT module keeps source
// locations, including those of the inlined scale(), only as line directives.
// clang-format off
// CHECK-NOT: emissionKind: {{LineTablesOnly|FullDebug}}
// CHECK: !DICompileUnit({{.*}}emissionKind: DebugDirectivesOnly
// CHECK-NOT: emissionKind: {{LineTablesOnly|FullDebug}}
// CHECK: X 2.000000
