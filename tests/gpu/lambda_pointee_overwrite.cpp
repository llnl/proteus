// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_pointee_overwrite.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// A helper receives the closure by pointer, overwrites the pointee as a whole
// object (legal under the frontend contract: the copy covers the entire
// wrapper), then invokes it directly on the pointer argument. No StoreInst
// writes through the argument itself, so the direct-argument clobber guard in
// LambdaArgVisitor::visitArgument never queries MemorySSA; the walk crosses to
// the kernel and folds Initial's capture while the device executes
// Replacement's. Specialization must either use Replacement's value or
// decline; it must never fold Initial's.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwritePointeeThenInvoke(F *Target, const F *Replacement) {
  __builtin_memcpy(Target, Replacement, sizeof(F));
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPointeeOverwrite(F Initial, F Replacement) {
  overwritePointeeThenInvoke(&Initial, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("pointee overwrite %d\n", X);
      });
}

int main() {
  auto Initial = makeBody(503);
  auto Replacement = makeBody(509);
  kernelPointeeOverwrite<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelPointeeOverwrite
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 503
// CHECK: pointee overwrite 509
// clang-format on
