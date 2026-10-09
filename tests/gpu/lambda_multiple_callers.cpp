// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_multiple_callers.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// There is one wrapper call in this function but two caller contexts carrying
// different kernel arguments. Without call-context-sensitive provenance,
// selecting either caller would depend on Value::users() traversal order.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeShared(F *Body) {
  (*Body)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelMultipleCallers(F First, F Second) {
  invokeShared(&First);
  invokeShared(&Second);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("multiple callers %d\n", X);
      });
}

int main() {
  auto First = makeBody(919);
  auto Second = makeBody(929);
  kernelMultipleCallers<<<1, 1>>>(First, Second);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelMultipleCallers
// CHECK-NOT: [LambdaSpec]
// CHECK: multiple callers 919
// CHECK: multiple callers 929
// clang-format on
