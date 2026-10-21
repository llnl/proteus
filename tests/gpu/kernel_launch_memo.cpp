// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/kernel_launch_memo.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit", 1))) void kernel(int V, int *Out) {
  *Out = V;
}

template <typename T>
__global__ __attribute__((annotate("jit"))) void lambdaKernel(T LB) {
  LB();
}

template <typename T> void launchLambda(T &&LB) {
  lambdaKernel<<<1, 1>>>(proteus::register_lambda(LB));
}

int main() {
  int *Out;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));

  // Cycle through more distinct values than recent launches are memoized, so
  // warm launches must keep telling specializations apart.
  int Mismatches = 0;
  for (int I = 0; I < 36; ++I) {
    int V = I % 6;
    kernel<<<1, 1>>>(V, Out);
    gpuErrCheck(gpuDeviceSynchronize());
    Mismatches += (*Out != V);
  }

  for (int I = 0; I < 36; ++I) {
    int C = I % 6;
    launchLambda([=, C = proteus::jit_variable(C)] __device__
                 __attribute__((annotate("jit"))) () { *Out = C + 100; });
    gpuErrCheck(gpuDeviceSynchronize());
    Mismatches += (*Out != C + 100);
  }

  printf("Mismatches %d\n", Mismatches);
  gpuErrCheck(gpuFree(Out));
  return 0;
}

// clang-format off
// CHECK-COUNT-6: [ArgSpec] Replaced Function _Z6kerneliPi ArgNo 0 with value i32
// CHECK-NOT: [ArgSpec] Replaced Function _Z6kerneliPi
// CHECK-COUNT-6: [LambdaSpec] Replacing slot 0 with i32
// CHECK-NOT: [LambdaSpec] Replacing slot
// CHECK: Mismatches 0
