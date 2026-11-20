// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_hidden_call_writes.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_hidden_call_writes.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct WritePlan {
  F *Destination;
  const F *Source;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyLeaf(F *Destination, const F *Source) {
  __builtin_memcpy(Destination, Source, sizeof(F));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyMiddle(F *Destination, const F *Source) {
  copyLeaf(Destination, Source);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
executePlan(const WritePlan<F> *Plan) {
  copyMiddle(Plan->Destination, Plan->Source);
}

// Body and Plan are separate arguments. The allocation containing Plan is
// disjoint from Body, but the pointer stored inside Plan aliases Body.
template <typename F>
__device__ __attribute__((noinline, optnone)) static int
invokePlan(F *Body, const WritePlan<F> *Plan) {
  executePlan(Plan);
  return (*Body)();
}

template <typename Action>
__device__ __attribute__((noinline, optnone)) static void
dispatchAction(const Action *ActionPtr) {
  (*ActionPtr)();
}

// The callable's captured pointer can reach Body even though the call's
// explicit argument points at a different callable object.
template <typename F, typename Action>
__device__ __attribute__((noinline, optnone)) static int
invokeAction(F *Body, const Action *ActionPtr) {
  dispatchAction(ActionPtr);
  return (*Body)();
}

template <typename F> __device__ F *PublishedDestination;
template <typename F> __device__ const F *PublishedSource;

template <typename F>
__device__ __attribute__((noinline, optnone)) static void writePublished() {
  copyMiddle(PublishedDestination<F>, PublishedSource<F>);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
publishAndWrite(const WritePlan<F> *Plan) {
  PublishedDestination<F> = Plan->Destination;
  PublishedSource<F> = Plan->Source;
  writePublished<F>();
}

// Publication happens inside the selected call, before its nested call with
// no arguments. A private-allocation proof must account for this publication.
template <typename F>
__device__ __attribute__((noinline, optnone)) static int
invokePublished(F *Body, const WritePlan<F> *Plan) {
  publishAndWrite(Plan);
  return (*Body)();
}

// Here substitution through both copy helpers reaches a fresh local object
// in this frame. The call summary can prove that Body is unchanged.
template <typename F>
__device__ __attribute__((noinline, optnone)) static int
invokeWithScratch(F *Body, const F *Replacement) {
  F Scratch = *Replacement;
  copyMiddle(&Scratch, Replacement);
  return (*Body)();
}

template <int Case, typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelHiddenWrites(F Initial, F Replacement, F Noise, int *Result) {
  if constexpr (Case == 0) {
    WritePlan<F> Plan{&Initial, &Replacement};
    Result[0] = invokePlan(&Initial, &Plan);
  } else if constexpr (Case == 1) {
    auto Action = [Destination = &Initial, Source = &Replacement] __device__ {
      copyMiddle(Destination, Source);
    };
    Result[0] = invokeAction(&Initial, &Action);
  } else if constexpr (Case == 2) {
    WritePlan<F> Plan{&Initial, &Replacement};
    Result[0] = invokePublished(&Initial, &Plan);
  } else if constexpr (Case == 3) {
    // The very same helper is harmless at its first call and overwrites the
    // closure at its second. A function-wide cached no-write answer is unsafe.
    copyMiddle(&Noise, &Replacement);
    Result[0] = Initial();
    copyMiddle(&Initial, &Replacement);
    Result[1] = Initial();
  } else {
    Result[0] = invokeWithScratch(&Initial, &Replacement);
  }
}

template <int Case> static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ { return X + 0; });
}

template <int Case> static void run(int *DeviceResult) {
  constexpr int Initial = 1201 + 20 * Case;
  kernelHiddenWrites<Case><<<1, 1>>>(makeBody<Case>(Initial),
                                     makeBody<Case>(Initial + 8),
                                     makeBody<Case>(Initial + 4), DeviceResult);
  gpuErrCheck(gpuDeviceSynchronize());
  int Result[2] = {};
  gpuErrCheck(gpuMemcpy(Result, DeviceResult, (Case == 3 ? 2 : 1) * sizeof(int),
                        gpuMemcpyDeviceToHost));
  if constexpr (Case == 3)
    printf("hidden write %d: %d %d\n", Case, Result[0], Result[1]);
  else
    printf("hidden write %d: %d\n", Case, Result[0]);
  fflush(stdout);
}

int main() {
  int *DeviceResult = nullptr;
  gpuErrCheck(
      gpuMalloc(reinterpret_cast<void **>(&DeviceResult), 2 * sizeof(int)));
  run<0>(DeviceResult);
  run<1>(DeviceResult);
  run<2>(DeviceResult);
  run<3>(DeviceResult);
  run<4>(DeviceResult);
  gpuErrCheck(gpuFree(DeviceResult));
}

// Harmful cases permit a correct replacement or a conservative decline.
// Case 3 permits rejection: replacement recovery through a helper with two
// callers is currently unsupported. Its two results must still differ; reusing
// a harmless-call answer for the overwriting call would violate that check.
// Case 4 requires useful specialization across proven disjoint writes.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1201
// RESULT: hidden write 0: 1209
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1221
// RESULT: hidden write 1: 1229
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1241
// RESULT: hidden write 2: 1249
// RESULT: hidden write 3: 1261 1269
// SPEC: [LambdaSpec] Replacing slot 0 with i32 1281
// RESULT: hidden write 4: 1281
// clang-format on
