// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_pointer_history.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_pointer_history.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct Context {
  F *Body;
};
template <typename F> __device__ F *PublishedBody;
template <typename F> __device__ const F *PublishedFinal;

template <typename F>
__device__ __attribute__((noinline, optnone)) static void writePublished() {
  __builtin_memcpy(PublishedBody<F>, PublishedFinal<F>, sizeof(F));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeEscapedAlias(F *Target, F *Alias, const F *Final) {
  PublishedBody<F> = Alias;
  PublishedFinal<F> = Final;
  writePublished<F>();
  (*Target)();
}

// Recovering the field's pointer does not discharge writes to its pointee.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeField(Context<F> *C, const F *Replacement) {
  F *Body = C->Body;
  __builtin_memcpy(Body, Replacement, sizeof(F));
  (*Body)();
}

// A pointer supplier also changes the supplied closure's bytes. Projecting
// Replacement into the caller must retain the supplier's memory effects.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
supply(F **Slot, F *Replacement, const F *Final) {
  *Slot = Replacement;
  __builtin_memcpy(Replacement, Final, sizeof(F));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeSupplier(F *Initial, F *Replacement, const F *Final) {
  F *Slot = Initial;
  supply(&Slot, Replacement, Final);
  (*Slot)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeCopiedPointer(F *Initial, const F *Final) {
  F *Source = Initial;
  F *Slot = nullptr;
  __builtin_memcpy(&Slot, &Source, sizeof(Slot));
  F *Body = Slot;
  __builtin_memcpy(Body, Final, sizeof(F));
  (*Body)();
}

template <int Case, typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelHistory(F Initial, F Replacement, F Final) {
  if constexpr (Case == 0) {
    Context<F> C{&Initial};
    invokeField(&C, &Final);
  } else if constexpr (Case == 1) {
    invokeSupplier(&Initial, &Replacement, &Final);
  } else if constexpr (Case == 2) {
    F Copy = Initial;
    F *volatile Slot = &Copy;
    F *Body = Slot;
    __builtin_memcpy(Body, &Final, sizeof(F));
    (*Body)();
  } else if constexpr (Case == 3) {
    invokeCopiedPointer(&Initial, &Final);
  } else {
    invokeEscapedAlias(&Initial, &Initial, &Final);
  }
}

template <int Case> static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("pointer history %d %d\n", Case, X);
      });
}

int main() {
  kernelHistory<0>
      <<<1, 1>>>(makeBody<0>(1001), makeBody<0>(1003), makeBody<0>(1009));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelHistory<1>
      <<<1, 1>>>(makeBody<1>(1013), makeBody<1>(1019), makeBody<1>(1021));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelHistory<2>
      <<<1, 1>>>(makeBody<2>(1031), makeBody<2>(1033), makeBody<2>(1039));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelHistory<3>
      <<<1, 1>>>(makeBody<3>(1051), makeBody<3>(1061), makeBody<3>(1063));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelHistory<4>
      <<<1, 1>>>(makeBody<4>(1079), makeBody<4>(1081), makeBody<4>(1087));
  gpuErrCheck(gpuDeviceSynchronize());
}

// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1001
// RESULT: pointer history 0 1009
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1013
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1019
// RESULT: pointer history 1 1021
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1031
// RESULT: pointer history 2 1039
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1051
// RESULT: pointer history 3 1063
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1079
// RESULT: pointer history 4 1087
// clang-format on
