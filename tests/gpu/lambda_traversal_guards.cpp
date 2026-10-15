// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_traversal_guards.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// Pins the traversal guards that keep the kernel-arg analysis conservative
// (and terminating) on adversarial call graphs. Each shape must DECLINE --
// and, for the recursive shape, must decline rather than hang the compile.
// Each case uses a distinct registered closure type so one decline cannot
// mask another's.

// --- Case 1: one helper entered with conflicting offsets ---------------------
// writeSlot is entered twice while resolving one closure, with the tracked
// pointer at different remaining offsets (slot 0 vs slot 1). ValueRangeMap
// facts for the callee's formal are not call-context scoped, so the analysis
// must reject the conflicting re-entry rather than silently retarget the
// first entry's facts.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
writeSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterConflictingEntries(F *Initial, F *First, F *Second) {
  F *volatile SlotA = Initial;
  F *volatile SlotB = Initial;
  writeSlot(&SlotA, First);
  writeSlot(&SlotB, Second);
  (*SlotA)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelConflictingEntries(F Initial, F First, F Second) {
  invokeAfterConflictingEntries(&Initial, &First, &Second);
}

// --- Case 2: recursive pointer-returning helper ------------------------------
// Each traversal entry into a pointer-returning helper creates a fresh
// return continuation, and visited-tracking is continuation-scoped, so
// without the recursion guard this shape re-enqueues work without bound:
// the required outcome is a conservative decline, not a hung compile.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *
passThroughRecursive(F *Ptr, int Depth) {
  if (Depth <= 0)
    return Ptr;
  return passThroughRecursive(Ptr, Depth - 1);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughRecursion(F *Body, int Depth) {
  F *Resolved = passThroughRecursive(Body, Depth);
  (*Resolved)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelRecursiveHelper(F Body, int Depth) {
  invokeThroughRecursion(&Body, Depth);
}

// --- Case 3: helper whose address escapes ------------------------------------
// forwardInvoke has exactly one direct callsite, but its address is also
// taken, so an unseen indirect callsite could supply the lambda argument;
// single-caller reasoning in the backward argument walk must reject it.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void forwardInvoke(F *Body) {
  (*Body)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughEscapedHelper(F *Body, bool Direct) {
  void (*volatile Escaped)(F *) = &forwardInvoke<F>;
  if (Direct)
    forwardInvoke(Body);
  else
    Escaped(Body);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelEscapedHelper(F Body, bool Direct) {
  invokeThroughEscapedHelper(&Body, Direct);
}

// --- Case 4: pointer-returning callee with multiple returns ------------------
// Both returns yield pointers derived from the same argument, but the
// backward walk analyzes exactly one value-returning ReturnInst; a callee
// with several must be rejected rather than trusting whichever return the
// analysis happens to find first.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *
selectSlot(F *A, F *B, bool UseB) {
  if (UseB)
    return B;
  return A;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughMultiReturn(F *A, F *B, bool UseB) {
  F *Resolved = selectSlot(A, B, UseB);
  (*Resolved)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelMultiReturn(F A, F B, bool UseB) {
  invokeThroughMultiReturn(&A, &B, UseB);
}

static auto makeConflictBody(int V) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(V)] __host__ __device__ {
        printf("conflicting entries %d\n", X);
      });
}

static auto makeRecursionBody(int V) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(V)] __host__ __device__ {
        printf("recursive helper %d\n", X);
      });
}

static auto makeEscapeBody(int V) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(V)] __host__ __device__ {
        printf("escaped helper %d\n", X);
      });
}

static auto makeMultiReturnBody(int V) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(V)] __host__ __device__ {
        printf("multi return %d\n", X);
      });
}

int main() {
  auto ConflictInitial = makeConflictBody(401);
  auto ConflictFirst = makeConflictBody(409);
  auto ConflictSecond = makeConflictBody(419);
  kernelConflictingEntries<<<1, 1>>>(ConflictInitial, ConflictFirst,
                                     ConflictSecond);
  gpuErrCheck(gpuDeviceSynchronize());

  auto RecursionBody = makeRecursionBody(421);
  kernelRecursiveHelper<<<1, 1>>>(RecursionBody, 3);
  gpuErrCheck(gpuDeviceSynchronize());

  auto EscapeBody = makeEscapeBody(431);
  kernelEscapedHelper<<<1, 1>>>(EscapeBody, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto MultiA = makeMultiReturnBody(433);
  auto MultiB = makeMultiReturnBody(439);
  kernelMultiReturn<<<1, 1>>>(MultiA, MultiB, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelConflictingEntries
// CHECK-NOT: [LambdaSpec]
// CHECK: conflicting entries 409
// CHECK: [KernelConfig] ID:{{.*}}kernelRecursiveHelper
// CHECK-NOT: [LambdaSpec]
// CHECK: recursive helper 421
// CHECK: [KernelConfig] ID:{{.*}}kernelEscapedHelper
// CHECK-NOT: [LambdaSpec]
// CHECK: escaped helper 431
// CHECK: [KernelConfig] ID:{{.*}}kernelMultiReturn
// CHECK-NOT: [LambdaSpec]
// CHECK: multi return 439
// clang-format on
