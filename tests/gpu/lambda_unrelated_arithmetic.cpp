// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_unrelated_arithmetic.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unrelated_arithmetic.%ext | %FILECHECK %s --check-prefixes=SPEC,RESULT
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// Write through a pointer that is separate from the closure. optnone keeps
// the call and the pointer argument's local spill visible in the device IR.
__device__ __attribute__((noinline, optnone)) static void
unrelatedArithmetic(int Value, int *Output) {
  *Output = Value * 3 + 1;
}

__device__ static int Counter;

// Exercise the general isolation proof with user code and synchronization,
// independently of any device printf implementation.
__device__ __attribute__((noinline, optnone)) static void unrelatedGlobalWrite() {
  atomicAdd(&Counter, 1);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPrivateClosure(F Body, int *Result) {
  unrelatedGlobalWrite();
  *Result = Body();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static int
invokeAfterArithmetic(F *Body, int Input) {
  int Noise = 0;
  unrelatedArithmetic(Input, &Noise);
  return (*Body)() + Noise;
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelUnrelatedArithmetic(F Body, int Input, int *Result) {
  *Result = invokeAfterArithmetic(&Body, Input);
}

int main() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(41)] __host__ __device__ { return X + 0; });
  int *DeviceResult = nullptr;
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&DeviceResult), sizeof(int)));
  kernelUnrelatedArithmetic<<<1, 1>>>(Body, 7, DeviceResult);
  gpuErrCheck(gpuDeviceSynchronize());
  int Result = 0;
  gpuErrCheck(gpuMemcpy(&Result, DeviceResult, sizeof(Result),
                        gpuMemcpyDeviceToHost));
  gpuErrCheck(gpuFree(DeviceResult));
  // Printing happens only on the host, after the device computation.
  printf("arithmetic helper result %d\n", Result);
  fflush(stdout);
  auto PrivateBody = proteus::register_lambda(
      [X = proteus::jit_variable(59)] __host__ __device__ { return X + 0; });
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&DeviceResult), sizeof(int)));
  kernelPrivateClosure<<<1, 1>>>(PrivateBody, DeviceResult);
  gpuErrCheck(gpuDeviceSynchronize());
  gpuErrCheck(gpuMemcpy(&Result, DeviceResult, sizeof(Result),
                        gpuMemcpyDeviceToHost));
  gpuErrCheck(gpuFree(DeviceResult));
  printf("private closure result %d\n", Result);
  return 0;
}

// clang-format off
// SPEC: [LambdaSpec] Replacing slot 0 with i32 41
// RESULT: arithmetic helper result 63
// SPEC: [LambdaSpec] Replacing slot 0 with i32 59
// RESULT: private closure result 59
// clang-format on
