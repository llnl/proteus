// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_callee_join_no_phi.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// A helper overwrites the closure with a full-extent memcpy and then branches
// on an unrelated value before returning. The arms hold only register
// operations (a call to a memory(none) function), so MemorySSA places no
// MemoryPhi at the return block. MemorySSAClobberOracle::getStateBefore walks
// back from the helper's ReturnInst, finds no MemoryDef in that block, no
// MemoryPhi, and more than one predecessor, and reports live-on-entry; the
// reconciliation then declines with "did not identify a clobbering
// instruction" although the memcpy is the unique reaching definition. The
// straight-line twin pins the shape that already specializes.
__device__ __attribute__((const, noinline)) static int pureScale(int V) {
  return V * 3 + 1;
}

// Deliberately not optnone: the arms must stay register-only.
template <typename F>
__device__ __attribute__((noinline)) static int
fillStraight(F *Dst, const F *Src, int Flag) {
  __builtin_memcpy(Dst, Src, sizeof(F));
  return pureScale(Flag);
}

template <typename F>
__device__ __attribute__((noinline)) static int
fillDiamond(F *Dst, const F *Src, int Flag) {
  __builtin_memcpy(Dst, Src, sizeof(F));
  int V = 0;
  if (Flag)
    V = pureScale(Flag);
  return V;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterStraightFill(const F *Initial, const F *Replacement, int Flag) {
  F Copy = *Initial;
  int R = fillStraight(&Copy, Replacement, Flag);
  Copy();
  if (R < 0)
    printf("unreachable %d\n", R);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterDiamondFill(const F *Initial, const F *Replacement, int Flag) {
  F Copy = *Initial;
  int R = fillDiamond(&Copy, Replacement, Flag);
  Copy();
  if (R < 0)
    printf("unreachable %d\n", R);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelStraightLineFill(F Initial, F Replacement, int Flag) {
  invokeAfterStraightFill(&Initial, &Replacement, Flag);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDiamondFill(F Initial, F Replacement, int Flag) {
  invokeAfterDiamondFill(&Initial, &Replacement, Flag);
}

// Distinct closure types so one analysis outcome cannot mask the other's.
static auto makeStraightBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("straight-line fill %d\n", X);
      });
}

static auto makeDiamondBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("diamond fill %d\n", X);
      });
}

int main() {
  auto StraightInitial = makeStraightBody(701);
  auto StraightReplacement = makeStraightBody(709);
  kernelStraightLineFill<<<1, 1>>>(StraightInitial, StraightReplacement, 1);
  gpuErrCheck(gpuDeviceSynchronize());

  auto DiamondInitial = makeDiamondBody(719);
  auto DiamondReplacement = makeDiamondBody(727);
  kernelDiamondFill<<<1, 1>>>(DiamondInitial, DiamondReplacement, 1);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelStraightLineFill
// CHECK: [LambdaSpec] Replacing slot 0 with i32 709
// CHECK: straight-line fill 709
// CHECK: [KernelConfig] ID:{{.*}}kernelDiamondFill
// CHECK: [LambdaSpec] Replacing slot 0 with i32 727
// CHECK: diamond fill 727
// clang-format on
