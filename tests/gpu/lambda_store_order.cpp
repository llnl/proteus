// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_store_order.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThenOverwrite(F Initial, F Later) {
  F *volatile Slot = &Initial;
  (*Slot)();
  Slot = &Later;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteThenInvoke(F Initial, F Replacement) {
  F *volatile Slot = &Initial;
  Slot = &Replacement;
  (*Slot)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
branchThenInvoke(F First, F Second, bool UseSecond) {
  F *volatile Slot;
  if (UseSecond)
    Slot = &Second;
  else
    Slot = &First;
  (*Slot)();
}

template <typename F>
__global__ static void kernelStoreOrder(F Initial, F Other) {
  invokeThenOverwrite(Initial, Other);
}

template <typename F>
__global__ static void kernelLatestStore(F Initial, F Replacement) {
  overwriteThenInvoke(Initial, Replacement);
}

template <typename F>
__global__ static void kernelBranchStore(F First, F Second, bool UseSecond) {
  branchThenInvoke(First, Second, UseSecond);
}

template <typename F> __global__ static void kernelMutatedCapture(F Body) {
  // Force a second write to the captured integer itself.  Specializing from
  // the parameter initializer instead of this nearest write miscompiles the
  // call even though no pointer spill is involved.
  __atomic_store_n(reinterpret_cast<int *>(&Body), 181, __ATOMIC_RELAXED);
  Body();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
replaceCaptureThenInvoke(F Body, int Replacement) {
  *reinterpret_cast<volatile int *>(&Body) = Replacement;
  Body();
}

template <typename F>
__global__ static void kernelCaptureFromArg(F Body, int Replacement) {
  replaceCaptureThenInvoke(Body, Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("store order %d\n", X);
      });
}

// Keep the intentionally ambiguous case on a distinct closure type so its
// conservative analysis failure does not suppress the positive cases above.
static auto makeAmbiguousBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("branch store %d\n", X);
      });
}

static auto makeMutatedBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("mutated capture %d\n", X);
      });
}

static auto makeReplacedBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("capture from arg %d\n", X);
      });
}

int main() {
  auto Before = makeBody(127);
  auto After = makeBody(131);
  kernelStoreOrder<<<1, 1>>>(Before, After);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Initial = makeBody(137);
  auto Replacement = makeBody(139);
  kernelLatestStore<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto BranchFirst = makeAmbiguousBody(149);
  auto BranchSecond = makeAmbiguousBody(151);
  kernelBranchStore<<<1, 1>>>(BranchFirst, BranchSecond, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Mutated = makeMutatedBody(179);
  kernelMutatedCapture<<<1, 1>>>(Mutated);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Replaced = makeReplacedBody(191);
  kernelCaptureFromArg<<<1, 1>>>(Replaced, 193);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 127
// CHECK: store order 127
// CHECK: [LambdaSpec] Replacing slot 0 with i32 139
// CHECK: store order 139
// CHECK: [KernelConfig] ID:{{.*}}kernelBranchStore
// CHECK-NOT: [LambdaSpec]
// CHECK: branch store 151
// CHECK: [KernelConfig] ID:{{.*}}kernelMutatedCapture
// CHECK-NOT: [LambdaSpec]
// CHECK: mutated capture 181
// CHECK: [LambdaSpec] Replacing slot 0 with i32 193
// CHECK: capture from arg 193
// clang-format on
