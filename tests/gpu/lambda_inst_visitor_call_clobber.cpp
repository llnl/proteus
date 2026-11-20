// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_inst_visitor_call_clobber.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

static PROTEUS_HOST_DEVICE auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("inst visitor call clobber %d\n", X);
      });
}

// Use a distinct lambda type for the analysis-failure case. Lambda analysis is
// performed for all callsites of a registered-lambda operator together, so a
// failure for this operator must not disable specialization of makeBody's
// operator as well.
static PROTEUS_HOST_DEVICE auto makeUntraceableBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("untraceable inst visitor call clobber %d\n", X);
      });
}

static PROTEUS_HOST_DEVICE auto makeMemcpyBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("inst visitor memcpy clobber %d\n", X);
      });
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwritePointerSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
dontOverwritePointerSlot(F *volatile *, F *) {}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwritePointerSlotNonKernelPtr(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

// Keep these helpers unoptimized so their device IR contains a pointer alloca,
// an initializer store, a call that may overwrite that alloca, and the final
// load used to invoke the registered lambda.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughClobberedAlloca(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  overwritePointerSlot(&Slot, Replacement);
  (*Slot)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughAmbiguousFunc(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  dontOverwritePointerSlot(&Slot, Replacement);
  (*Slot)();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughNonKernelFunc(F *Initial) {
  alignas(F) unsigned char ReplacementStorage[sizeof(F)] = {};
  F *Replacement = reinterpret_cast<F *>(ReplacementStorage);
  *reinterpret_cast<int *>(Replacement) = 120;
  F *volatile Slot = Initial;
  overwritePointerSlotNonKernelPtr(&Slot, Replacement);
  (*Slot)();
}

// MemorySSA identifies the second pre-load memcpy as the reaching clobber.
// PointerClobberResolver cannot summarize the intrinsic and falls back to
// LambdaInstUseVisitor. The visitor must neither select the first memcpy nor
// cross the load boundary and select the final, post-load memcpy.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughOrderedMemcpys(F *Initial, F *FirstReplacement,
                            F *FinalReplacement, F *PostLoadReplacement) {
  F *Slot = Initial;
  __builtin_memcpy(&Slot, &FirstReplacement, sizeof(Slot));
  __builtin_memcpy(&Slot, &FinalReplacement, sizeof(Slot));
  F *Selected = Slot;
  __builtin_memcpy(&Slot, &PostLoadReplacement, sizeof(Slot));
  (*Selected)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelInstVisitorCallClobber(F Initial, F Replacement) {
  invokeThroughClobberedAlloca(&Initial, &Replacement);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelInstVisitorCallDoesntClobber(F Initial, F Replacement) {
  invokeThroughAmbiguousFunc(&Initial, &Replacement);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAnalysisShouldFail(F Initial) {
  invokeThroughNonKernelFunc(&Initial);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelInstVisitorOrderedMemcpys(F Initial, F FirstReplacement,
                                F FinalReplacement, F PostLoadReplacement) {
  invokeThroughOrderedMemcpys(&Initial, &FirstReplacement, &FinalReplacement,
                              &PostLoadReplacement);
}

int main() {
  auto Initial = makeBody(397);
  auto Replacement = makeBody(401);

  kernelInstVisitorCallClobber<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());

  kernelInstVisitorCallDoesntClobber<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto UntraceableInitial = makeUntraceableBody(419);
  kernelAnalysisShouldFail<<<1, 1>>>(UntraceableInitial);
  gpuErrCheck(gpuDeviceSynchronize());

  auto MemcpyInitial = makeMemcpyBody(421);
  auto MemcpyFirst = makeMemcpyBody(431);
  auto MemcpyFinal = makeMemcpyBody(433);
  auto MemcpyPostLoad = makeMemcpyBody(439);
  kernelInstVisitorOrderedMemcpys<<<1, 1>>>(MemcpyInitial, MemcpyFirst,
                                            MemcpyFinal, MemcpyPostLoad);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 401
// CHECK: inst visitor call clobber 401
// CHECK: [LambdaSpec] Replacing slot 0 with i32 397
// CHECK: inst visitor call clobber 397
// CHECK: [KernelConfig] ID:{{.*}}kernelAnalysisShouldFail
// CHECK-NOT: [LambdaSpec]
// CHECK: untraceable inst visitor call clobber 120
// CHECK: [LambdaSpec] Replacing slot 0 with i32 433
// CHECK: inst visitor memcpy clobber 433
// clang-format on
