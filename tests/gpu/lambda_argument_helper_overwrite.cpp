// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_argument_helper_overwrite.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// The write is a call in the invocation frame and a memcpy in the writer's
// frame. Resolving its source requires following the argument's uses through
// that call before tracing the argument back to the kernel.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
writeClosure(F *Dst, const F *Src) {
  __builtin_memcpy(Dst, Src, sizeof(F));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterWriter(F *Target, const F *Replacement) {
  writeClosure(Target, Replacement);
  (*Target)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentHelperOverwrite(F Initial, F Replacement) {
  invokeAfterWriter(&Initial, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("argument helper overwrite %d\n", X);
      });
}

int main() {
  auto Initial = makeBody(801);
  auto Replacement = makeBody(809);
  kernelArgumentHelperOverwrite<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentHelperOverwrite
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 801
// CHECK: [LambdaSpec] Replacing slot 0 with i32 809
// CHECK: argument helper overwrite 809
// clang-format on
