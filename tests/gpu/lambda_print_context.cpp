// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_print_context.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_print_context.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> __device__ F *UtilityDestination;
template <typename F> __device__ const F *UtilitySource;

// This models a print-like utility writing through hidden state. The explicit
// copy is our model's write; it is not a claim that printf itself overwrites a
// closure. The print-only control below uses the actual builtin without it.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
printWithHiddenWrite() {
  __builtin_memcpy(UtilityDestination<F>, UtilitySource<F>, sizeof(F));
  printf("Context utility\n");
}

template <bool HiddenWrite, typename F>
__device__ __attribute__((noinline, optnone)) static int
invokeAfterPrint(F *Target) {
  if constexpr (HiddenWrite)
    printWithHiddenWrite<F>();
  else
    printf("Context utility\n");
  return (*Target)();
}

template <bool HiddenWrite, typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPrintContext(F Initial, F Replacement, F Other, int *Result) {
  if constexpr (HiddenWrite) {
    UtilityDestination<F> = &Initial;
    UtilitySource<F> = &Replacement;
  }
  // In the hidden-write case, the first invocation's target is overwritten.
  Result[0] = invokeAfterPrint<HiddenWrite>(&Initial);
  if constexpr (HiddenWrite)
    UtilitySource<F> = &Other;
  // The same helper now writes Initial while invoking Replacement.
  Result[1] = invokeAfterPrint<HiddenWrite>(&Replacement);
}

template <bool HiddenWrite> static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ { return X + 0; });
}

template <bool HiddenWrite> static void run(int *DeviceResult) {
  constexpr int Initial = HiddenWrite ? 1421 : 1401;
  kernelPrintContext<HiddenWrite><<<1, 1>>>(
      makeBody<HiddenWrite>(Initial), makeBody<HiddenWrite>(Initial + 8),
      makeBody<HiddenWrite>(Initial + 16), DeviceResult);
  gpuErrCheck(gpuDeviceSynchronize());
  int Result[2];
  gpuErrCheck(
      gpuMemcpy(Result, DeviceResult, sizeof(Result), gpuMemcpyDeviceToHost));
  printf("print context %d: %d %d\n", HiddenWrite, Result[0], Result[1]);
  fflush(stdout);
}

int main() {
  int *DeviceResult = nullptr;
  gpuErrCheck(
      gpuMalloc(reinterpret_cast<void **>(&DeviceResult), 2 * sizeof(int)));
  run<false>(DeviceResult);
  run<true>(DeviceResult);
  gpuErrCheck(gpuFree(DeviceResult));
}

// Correct specialization or conservative rejection is acceptable. These
// checks exercise runtime correctness, not a new specialization guarantee.
// clang-format off
// RESULT: print context 0: 1401 1409
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1421
// RESULT: print context 1: 1429 1429
// clang-format on
