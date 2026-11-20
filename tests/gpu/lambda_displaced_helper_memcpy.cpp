// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_displaced_helper_memcpy.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// The closure lives at a nonzero offset inside an envelope, and a helper
// receives a pointer to that field and overwrites it with a full-extent memcpy.
// LambdaInstUseVisitor::visitMemIntrinsic derives its offset correction from
// the helper frame's base pointer (the formal, displacement 0) instead of the
// visitor root (the envelope, displacement 4), so the resolved source offset
// is off by the envelope header. With a 16-byte source object the wrong offset
// stays in bounds and the analysis folds Body.Y as slot 0 and Spare.X as slot
// 1. Specialization must use Replacement.Body's values or decline.
template <typename F> struct Envelope {
  int Hdr;
  F Body;
};

template <typename F> struct Pair {
  F Body;
  F Spare;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyInto(F *Dst, const F *Src) {
  __builtin_memcpy(Dst, Src, sizeof(F));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeEnvelopeBody(const F *Initial, const Pair<F> *Replacement) {
  Envelope<F> Env{7, *Initial};
  copyInto(&Env.Body, &Replacement->Body);
  Env.Body();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDisplacedHelperMemcpy(F Initial, Pair<F> Replacement) {
  invokeEnvelopeBody(&Initial, &Replacement);
}

static auto makeBody(int A, int B) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(A),
       Y = proteus::jit_variable(B)] __host__ __device__ {
        printf("displaced helper memcpy %d %d\n", X, Y);
      });
}

int main() {
  auto Initial = makeBody(599, 598);
  Pair<decltype(Initial)> Replacement{makeBody(601, 607), makeBody(613, 617)};
  kernelDisplacedHelperMemcpy<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelDisplacedHelperMemcpy
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 607
// CHECK: displaced helper memcpy 601 607
// clang-format on
