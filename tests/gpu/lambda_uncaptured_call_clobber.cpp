// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_uncaptured_call_clobber.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// This call is a MemoryDef because its body contains an opaque memory clobber.
__device__ __attribute__((noinline, optnone)) static void
opaqueUnrelatedWrite() {
  asm volatile("" : : : "memory");
}

__device__ static void *EscapedSlot;

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
escapePointerSlot(F **Slot) {
  EscapedSlot = Slot;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossUncapturedCall(F *Initial, F *Replacement) {
  F *Slot = Initial;
  Slot = Replacement;
  opaqueUnrelatedWrite();
  (*Slot)();
  // Escaping the slot later causes BasicAA to conservatively consider the
  // earlier opaque call a possible clobber. Capture information alone cannot
  // prove that an arbitrary call is read-only, so specialization must stop.
  escapePointerSlot(&Slot);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelUncapturedCallClobber(F Initial, F Replacement) {
  invokeAcrossUncapturedCall(&Initial, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("uncaptured call clobber %d\n", X);
      });
}

int main() {
  auto Initial = makeBody(907);
  auto Replacement = makeBody(911);
  kernelUncapturedCallClobber<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelUncapturedCallClobber
// CHECK-NOT: [LambdaSpec]
// CHECK: uncaptured call clobber 911
// clang-format on
