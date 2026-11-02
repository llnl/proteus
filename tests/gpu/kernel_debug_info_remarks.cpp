// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="ir-dump" %build/kernel_debug_info_remarks.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit", 2))) void kernel(double *X,
                                                           double A) {
  X[threadIdx.x] *= A;
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

// The kernel is compiled with optimization remarks and no -g, so clang gives
// it a NoDebug compile unit that carries locations only for the remarks.
// clang-format off
// CHECK-NOT: emissionKind: {{DebugDirectivesOnly|LineTablesOnly|FullDebug}}
// CHECK: !DICompileUnit({{.*}}emissionKind: NoDebug
// CHECK-NOT: emissionKind: {{DebugDirectivesOnly|LineTablesOnly|FullDebug}}
// CHECK: X 2.000000
