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

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwritePointerTransitively(F *volatile *Slot, F *Replacement) {
  overwritePointerInCall(Slot, Replacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterNestedCallOverwrite(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  overwritePointerTransitively(&Slot, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelNestedCallOverwrite(F Initial, F Replacement) {
  invokeAfterNestedCallOverwrite(&Initial, &Replacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSamePointerOnBothPaths(F *volatile *Slot, F *Replacement,
                                bool LeftPath) {
  if (LeftPath)
    overwritePointerInCall(Slot, Replacement);
  else
    overwritePointerTransitively(Slot, Replacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterSameCallBranchOverwrite(F *Initial, F *Replacement, bool LeftPath) {
  F *volatile Slot = Initial;
  overwriteSamePointerOnBothPaths(&Slot, Replacement, LeftPath);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSameCallBranchOverwrite(F Initial, F Replacement, bool LeftPath) {
  invokeAfterSameCallBranchOverwrite(&Initial, &Replacement, LeftPath);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteDifferentPointersOnPaths(F *volatile *Slot, F *First, F *Second,
                                  bool UseSecond) {
  if (UseSecond)
    overwritePointerInCall(Slot, Second);
  else
    overwritePointerTransitively(Slot, First);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterDifferentCallBranchOverwrite(F *Initial, F *First, F *Second,
                                        bool UseSecond) {
  F *volatile Slot = Initial;
  overwriteDifferentPointersOnPaths(&Slot, First, Second, UseSecond);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDifferentCallBranchOverwrite(F Initial, F First, F Second,
                                   bool UseSecond) {
  invokeAfterDifferentCallBranchOverwrite(&Initial, &First, &Second, UseSecond);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterLocalSupersedesCall(F *Initial, F *CallReplacement,
                               F *FinalReplacement) {
  F *volatile Slot = Initial;
  overwritePointerInCall(&Slot, CallReplacement);
  Slot = FinalReplacement;
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelLocalSupersedesCall(F Initial, F CallReplacement, F FinalReplacement) {
  invokeAfterLocalSupersedesCall(&Initial, &CallReplacement, &FinalReplacement);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossUnrelatedCallOverwrite(F *Initial, F *Replacement, F *NoiseInitial,
                                   F *NoiseReplacement) {
  F *volatile Slot = Initial;
  F *volatile NoiseSlot = NoiseInitial;
  Slot = Replacement;
  overwritePointerInCall(&NoiseSlot, NoiseReplacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelUnrelatedCallOverwrite(F Initial, F Replacement, F NoiseInitial,
                             F NoiseReplacement) {
  invokeAcrossUnrelatedCallOverwrite(&Initial, &Replacement, &NoiseInitial,
                                     &NoiseReplacement);
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

static auto makeNestedCallBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("nested call clobber %d\n", X);
      });
}

static auto makeSameCallBranchBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("same call branch clobber %d\n", X);
      });
}

static auto makeDifferentCallBranchBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("different call branch clobber %d\n", X);
      });
}

static auto makeLocalAfterCallBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("local after call clobber %d\n", X);
      });
}

static auto makeUnrelatedCallBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("unrelated call clobber %d\n", X);
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

  auto NestedCallInitial = makeNestedCallBody(311);
  auto NestedCallReplacement = makeNestedCallBody(313);
  kernelNestedCallOverwrite<<<1, 1>>>(NestedCallInitial, NestedCallReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SameCallBranchInitial = makeSameCallBranchBody(317);
  auto SameCallBranchReplacement = makeSameCallBranchBody(331);
  kernelSameCallBranchOverwrite<<<1, 1>>>(SameCallBranchInitial,
                                          SameCallBranchReplacement, false);
  gpuErrCheck(gpuDeviceSynchronize());

  auto DifferentCallBranchInitial = makeDifferentCallBranchBody(337);
  auto DifferentCallBranchFirst = makeDifferentCallBranchBody(347);
  auto DifferentCallBranchSecond = makeDifferentCallBranchBody(349);
  kernelDifferentCallBranchOverwrite<<<1, 1>>>(DifferentCallBranchInitial,
                                               DifferentCallBranchFirst,
                                               DifferentCallBranchSecond, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto LocalAfterCallInitial = makeLocalAfterCallBody(353);
  auto LocalAfterCallIntermediate = makeLocalAfterCallBody(359);
  auto LocalAfterCallFinal = makeLocalAfterCallBody(367);
  kernelLocalSupersedesCall<<<1, 1>>>(
      LocalAfterCallInitial, LocalAfterCallIntermediate, LocalAfterCallFinal);
  gpuErrCheck(gpuDeviceSynchronize());

  auto UnrelatedCallInitial = makeUnrelatedCallBody(373);
  auto UnrelatedCallReplacement = makeUnrelatedCallBody(379);
  auto UnrelatedCallNoiseInitial = makeUnrelatedCallBody(383);
  auto UnrelatedCallNoiseReplacement = makeUnrelatedCallBody(389);
  kernelUnrelatedCallOverwrite<<<1, 1>>>(
      UnrelatedCallInitial, UnrelatedCallReplacement, UnrelatedCallNoiseInitial,
      UnrelatedCallNoiseReplacement);
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
// CHECK: [LambdaSpec] Replacing slot 0 with i32 313
// CHECK: nested call clobber 313
// CHECK: [LambdaSpec] Replacing slot 0 with i32 331
// CHECK: same call branch clobber 331
// CHECK: [KernelConfig] ID:{{.*}}kernelDifferentCallBranchOverwrite
// CHECK-NOT: [LambdaSpec]
// CHECK: different call branch clobber 349
// CHECK: [LambdaSpec] Replacing slot 0 with i32 367
// CHECK: local after call clobber 367
// CHECK: [LambdaSpec] Replacing slot 0 with i32 379
// CHECK: unrelated call clobber 379
// clang-format on
