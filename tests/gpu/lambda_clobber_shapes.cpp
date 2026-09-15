// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_clobber_shapes.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterPredecessorOverwrite(F *Initial, F *Replacement, bool Execute) {
  F *volatile Slot = Initial;
  if (Execute)
    Slot = Replacement;
  else
    return;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPredecessorOverwrite(F Initial, F Replacement, bool Execute) {
  invokeAfterPredecessorOverwrite(&Initial, &Replacement, Execute);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSameBranchOverwrite(F *Initial, F *Replacement, bool LeftPath) {
  F *volatile Slot = Initial;
  if (LeftPath)
    Slot = Replacement;
  else
    Slot = Replacement;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSameBranchOverwrite(F Initial, F Replacement, bool LeftPath) {
  invokeAfterSameBranchOverwrite(&Initial, &Replacement, LeftPath);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterDifferentBranchOverwrite(F *First, F *Second, bool UseSecond) {
  F *volatile Slot;
  if (UseSecond)
    Slot = Second;
  else
    Slot = First;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDifferentBranchOverwrite(F First, F Second, bool UseSecond) {
  invokeAfterDifferentBranchOverwrite(&First, &Second, UseSecond);
}

template <typename F> struct PointerPair {
  F *Invoked;
  F *Unrelated;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterNonoverlapOverwrite(F *Initial, F *Replacement, F *NoiseInitial,
                               F *NoiseReplacement) {
  PointerPair<F> Pair{Initial, NoiseInitial};
  Pair.Invoked = Replacement;
  Pair.Unrelated = NoiseReplacement;
  (*Pair.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelNonoverlapOverwrite(F Initial, F Replacement, F NoiseInitial,
                          F NoiseReplacement) {
  invokeAfterNonoverlapOverwrite(&Initial, &Replacement, &NoiseInitial,
                                 &NoiseReplacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwritePointerInCall(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterCallOverwrite(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  overwritePointerInCall(&Slot, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCallOverwrite(F Initial, F Replacement) {
  invokeAfterCallOverwrite(&Initial, &Replacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSameLoopOverwrite(F *Initial, F *Replacement, int Iterations) {
  F *volatile Slot = Initial;
  Slot = Replacement;
  for (int I = 0; I < Iterations; ++I)
    Slot = Replacement;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSameLoopOverwrite(F Initial, F Replacement, int Iterations) {
  invokeAfterSameLoopOverwrite(&Initial, &Replacement, Iterations);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterAmbiguousLoopOverwrite(F *Initial, F *Replacement, int Iterations) {
  F *volatile Slot = Initial;
  for (int I = 0; I < Iterations; ++I)
    Slot = Replacement;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAmbiguousLoopOverwrite(F Initial, F Replacement, int Iterations) {
  invokeAfterAmbiguousLoopOverwrite(&Initial, &Replacement, Iterations);
}

static auto makePredecessorBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("predecessor clobber %d\n", X);
      });
}

static auto makeSameBranchBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("same branch clobber %d\n", X);
      });
}

static auto makeDifferentBranchBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("different branch clobber %d\n", X);
      });
}

static auto makeNonoverlapBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("nonoverlap clobber %d\n", X);
      });
}

static auto makeCallBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("call clobber %d\n", X);
      });
}

static auto makeSameLoopBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("same loop clobber %d\n", X);
      });
}

static auto makeAmbiguousLoopBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("ambiguous loop clobber %d\n", X);
      });
}

int main() {
  auto PredecessorInitial = makePredecessorBody(229);
  auto PredecessorReplacement = makePredecessorBody(233);
  kernelPredecessorOverwrite<<<1, 1>>>(PredecessorInitial,
                                       PredecessorReplacement, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SameBranchInitial = makeSameBranchBody(235);
  auto SameBranchReplacement = makeSameBranchBody(239);
  kernelSameBranchOverwrite<<<1, 1>>>(SameBranchInitial, SameBranchReplacement,
                                      true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto DifferentBranchFirst = makeDifferentBranchBody(241);
  auto DifferentBranchSecond = makeDifferentBranchBody(251);
  kernelDifferentBranchOverwrite<<<1, 1>>>(DifferentBranchFirst,
                                           DifferentBranchSecond, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto NonoverlapInitial = makeNonoverlapBody(257);
  auto NonoverlapReplacement = makeNonoverlapBody(263);
  auto NoiseInitial = makeNonoverlapBody(269);
  auto NoiseReplacement = makeNonoverlapBody(271);
  kernelNonoverlapOverwrite<<<1, 1>>>(NonoverlapInitial, NonoverlapReplacement,
                                      NoiseInitial, NoiseReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto CallInitial = makeCallBody(273);
  auto CallReplacement = makeCallBody(277);
  kernelCallOverwrite<<<1, 1>>>(CallInitial, CallReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SameLoopInitial = makeSameLoopBody(281);
  auto SameLoopReplacement = makeSameLoopBody(283);
  kernelSameLoopOverwrite<<<1, 1>>>(SameLoopInitial, SameLoopReplacement, 2);
  gpuErrCheck(gpuDeviceSynchronize());

  auto AmbiguousLoopInitial = makeAmbiguousLoopBody(293);
  auto AmbiguousLoopReplacement = makeAmbiguousLoopBody(307);
  kernelAmbiguousLoopOverwrite<<<1, 1>>>(AmbiguousLoopInitial,
                                         AmbiguousLoopReplacement, 2);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 233
// CHECK: predecessor clobber 233
// CHECK: [LambdaSpec] Replacing slot 0 with i32 239
// CHECK: same branch clobber 239
// CHECK: [KernelConfig] ID:{{.*}}kernelDifferentBranchOverwrite
// CHECK-NOT: [LambdaSpec]
// CHECK: different branch clobber 251
// CHECK: [LambdaSpec] Replacing slot 0 with i32 263
// CHECK: nonoverlap clobber 263
// CHECK: [LambdaSpec] Replacing slot 0 with i32 277
// CHECK: call clobber 277
// CHECK: [LambdaSpec] Replacing slot 0 with i32 283
// CHECK: same loop clobber 283
// CHECK: [KernelConfig] ID:{{.*}}kernelAmbiguousLoopOverwrite
// CHECK-NOT: [LambdaSpec]
// CHECK: ambiguous loop clobber 307
// clang-format on
