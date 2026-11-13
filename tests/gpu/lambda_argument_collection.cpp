// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_argument_collection.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((pure, noinline)) static int
readCapture(const F *Body) {
  return *reinterpret_cast<const int *>(Body);
}

// The scan must finish successfully without a writer: the read-only call
// precedes the invocation, while the memset cannot reach it. The unrelated
// printf also must not turn an empty argument-use collection into a decline.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeBeforeWrite(F *Target) {
  printf("before write %d\n", readCapture(Target));
  (*Target)();
  __builtin_memset(Target, 0, sizeof(F));
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentNoPriorWriter(F Body) {
  invokeBeforeWrite(&Body);
}

// An unsupported use is not evidence that the argument is unchanged, even
// when no interpretable write candidate was collected.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterOpaqueUse(F *Target) {
  asm volatile("" : : "r"(Target) : "memory");
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentOpaqueUse(F Body) {
  invokeAfterOpaqueUse(&Body);
}

// The forward walk reaches a select with an untracked incoming pointer. It
// cannot follow that alias to its memcpy, so the unfinished alias must block
// a NoTrackedWrite result rather than expose Initial's stale capture.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSelectedWrite(F *Target, F *Other, const F *Replacement, bool Flag) {
  F *Selected = Flag ? Target : Other;
  __builtin_memcpy(Selected, Replacement, sizeof(F));
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentUnresolvedAlias(F Initial, F Other, F Replacement, bool Flag) {
  invokeAfterSelectedWrite(&Initial, &Other, &Replacement, Flag);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
writeThroughSlot(F **Slot, const F *Replacement) {
  __builtin_memcpy(*Slot, Replacement, sizeof(F));
}

// Following proven local reloads alone is insufficient when the pointer
// slot's address reaches a helper. The argument's pointee can change even
// though the invocation still loads the original Target pointer.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSlotEscape(F *Target, const F *Replacement) {
  F *Slot = Target;
  writeThroughSlot(&Slot, Replacement);
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentSlotEscape(F Initial, F Replacement) {
  invokeAfterSlotEscape(&Initial, &Replacement);
}

// Each case has a distinct wrapper type so a decline in a negative case
// cannot disable specialization of the positive case.
template <int Case> static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("argument collection %d %d\n", Case, X);
      });
}

int main() {
  kernelArgumentNoPriorWriter<<<1, 1>>>(makeBody<0>(907));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelArgumentOpaqueUse<<<1, 1>>>(makeBody<1>(919));
  gpuErrCheck(gpuDeviceSynchronize());
  kernelArgumentUnresolvedAlias<<<1, 1>>>(makeBody<2>(923), makeBody<2>(927),
                                          makeBody<2>(929), true);
  gpuErrCheck(gpuDeviceSynchronize());
  kernelArgumentSlotEscape<<<1, 1>>>(makeBody<3>(937), makeBody<3>(941));
  gpuErrCheck(gpuDeviceSynchronize());
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentNoPriorWriter
// CHECK: [LambdaSpec] Replacing slot 0 with i32 907
// CHECK: before write 907
// CHECK: argument collection 0 907
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentOpaqueUse
// CHECK-NOT: [LambdaSpec]
// CHECK: argument collection 1 919
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentUnresolvedAlias
// CHECK-NOT: [LambdaSpec]
// CHECK: argument collection 2 929
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentSlotEscape
// CHECK-NOT: [LambdaSpec]
// CHECK: argument collection 3 941
// clang-format on
