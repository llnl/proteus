// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_arithmetic_argument.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_arithmetic_argument.%ext | %FILECHECK %s --check-prefixes=SPEC,RESULT
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ __attribute__((noinline, optnone)) static void
unrelatedArithmetic(int Value, int *Output) {
  *Output = Value * 3 + 1;
}

// Acceptance criterion for the conservative call-write proof: both pointers
// enter this frame as arguments. Mapping the arithmetic helper's destination
// back to Output does not establish that it is disjoint from Body. Decline
// specialization without assuming that distinct formals denote distinct objects.
// Contrast lambda_unrelated_arithmetic.cpp, where Output is a fresh local in
// this frame and the test requires specialization. optnone preserves the calls
// and pointer spills so optimization cannot supply the missing context for us.
template <typename F>
__device__ __attribute__((noinline, optnone)) static int
invokeAfterArithmetic(F *Body, int *Output) {
  unrelatedArithmetic(7, Output);
  return (*Body)() + *Output;
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArithmeticArgument(F Body, int *Output, int *Result) {
  *Result = invokeAfterArithmetic(&Body, Output);
}

int main() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(53)] __host__ __device__ { return X + 0; });
  int *DeviceValues = nullptr;
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&DeviceValues), 2 * sizeof(int)));
  // Runtime storage is disjoint. The test requires a decline because the
  // initial proof is local to invokeAfterArithmetic, not because this launch
  // actually overwrites the closure.
  kernelArithmeticArgument<<<1, 1>>>(Body, DeviceValues, DeviceValues + 1);
  gpuErrCheck(gpuDeviceSynchronize());
  int Values[2] = {};
  gpuErrCheck(gpuMemcpy(Values, DeviceValues, sizeof(Values),
                       gpuMemcpyDeviceToHost));
  gpuErrCheck(gpuFree(DeviceValues));
  printf("arithmetic argument output %d result %d\n", Values[0], Values[1]);
  return 0;
}

// clang-format off
// SPEC-NOT: [LambdaSpec]
// RESULT: arithmetic argument output 22 result 75
// SPEC-NOT: [LambdaSpec]
// clang-format on
