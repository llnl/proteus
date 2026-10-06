// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_call_shapes.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *
chooseBody(F *First, F *Second, bool UseSecond) {
  if (UseSecond)
    return Second;
  return First;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *
chooseSameBody(F *Body, bool FirstPath) {
  if (FirstPath)
    return Body;
  return Body;
}

// Different return paths carry different kernel arguments.  Picking the first
// ReturnInst (or first reaching store to a lowered return slot) is unsound.
template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelMultipleReturns(F First, F Second, bool UseSecond) {
  (*chooseBody(&First, &Second, UseSecond))();
}

// Distinct CFG paths and SSA loads still have one semantic pointer source.
template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSameReturns(F Body, bool FirstPath) {
  (*chooseSameBody(&Body, FirstPath))();
}

template <typename F>
__device__ __attribute__((noinline)) static F *identityBody(F *Body) {
  return Body;
}

// With optimization, this ternary is a pointer select rather than a branch
// spill.  The arms are distinct call results with the same provenance.
template <typename F>
__device__ __attribute__((always_inline)) static F *
selectSameBody(F *Body, bool AlternatePath) {
  auto *First = identityBody(Body);
  auto *Second = identityBody(Body);
  return AlternatePath ? Second : First;
}

// Different select arms must not be collapsed to the first operand.
template <typename F>
__device__ __attribute__((always_inline)) static F *
selectDifferentBody(F *First, F *Second, bool UseSecond) {
  return UseSecond ? Second : First;
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSelectSame(F Body, bool AlternatePath) {
  (*selectSameBody(&Body, AlternatePath))();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSelectDifferent(F First, F Second, bool UseSecond) {
  (*selectDifferentBody(&First, &Second, UseSecond))();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeIndirect(F *Body) {
  using Forwarder = F *(*)(F *);
  volatile Forwarder Forward = &identityBody<F>;
  (*Forward(Body))();
}

// getCalledFunction() is null for an indirect call.  The provenance analysis
// must decline this shape instead of dereferencing the null callee.
template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelIndirectCall(F Body) {
  invokeIndirect(&Body);
}

static auto makeReturnBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("multiple returns %d\n", X);
      });
}

static auto makeIndirectBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("indirect call %d\n", X);
      });
}

static auto makeSameReturnBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("same returns %d\n", X);
      });
}

static auto makeSelectSameBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("select same %d\n", X);
      });
}

int main() {
  auto First = makeReturnBody(157);
  auto Second = makeReturnBody(163);
  kernelMultipleReturns<<<1, 1>>>(First, Second, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Same = makeSameReturnBody(173);
  kernelSameReturns<<<1, 1>>>(Same, false);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Indirect = makeIndirectBody(167);
  kernelIndirectCall<<<1, 1>>>(Indirect);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SelectSame = makeSelectSameBody(179);
  kernelSelectSame<<<1, 1>>>(SelectSame, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SelectFirst = makeReturnBody(181);
  auto SelectSecond = makeReturnBody(191);
  kernelSelectDifferent<<<1, 1>>>(SelectFirst, SelectSecond, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelMultipleReturns
// CHECK-NOT: [LambdaSpec]
// CHECK: multiple returns 163
// CHECK: [KernelConfig] ID:{{.*}}kernelSameReturns
// CHECK-NOT: [LambdaSpec]
// CHECK: same returns 173
// CHECK: [KernelConfig] ID:{{.*}}kernelIndirectCall
// CHECK-NOT: [LambdaSpec]
// CHECK: indirect call 167
// CHECK: [KernelConfig] ID:{{.*}}kernelSelectSame
// CHECK-NOT: [LambdaSpec]
// CHECK: select same 179
// CHECK: [KernelConfig] ID:{{.*}}kernelSelectDifferent
// CHECK-NOT: [LambdaSpec]
// CHECK: multiple returns 191
// clang-format on
