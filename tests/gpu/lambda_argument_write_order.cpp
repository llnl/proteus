// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_argument_write_order.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// The argument has three writer uses. Only the second copy reaches the lambda
// call: the first is overwritten, and the third occurs after the call. A use
// walk must collect the writes; MemorySSA must select the one at the boundary.
template <typename F>
__device__ __attribute__((noinline)) static void
invokeBetweenWrites(F *__restrict__ Target, const F *__restrict__ First,
                    const F *__restrict__ Selected,
                    const F *__restrict__ After) {
  __builtin_memcpy(Target, First, sizeof(F));
  int FirstObserved = *reinterpret_cast<volatile const int *>(Target);
  __builtin_memcpy(Target, Selected, sizeof(F));
  (*Target)();
  __builtin_memcpy(Target, After, sizeof(F));
  if (FirstObserved == -1)
    printf("unreachable %d\n", FirstObserved);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelArgumentWriteOrder(F Initial, F First, F Selected, F After) {
  invokeBetweenWrites(&Initial, &First, &Selected, &After);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("argument write order %d\n", X);
      });
}

int main() {
  auto Initial = makeBody(811);
  auto First = makeBody(821);
  auto Selected = makeBody(829);
  auto After = makeBody(839);
  kernelArgumentWriteOrder<<<1, 1>>>(Initial, First, Selected, After);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelArgumentWriteOrder
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 811
// CHECK: [LambdaSpec] Replacing slot 0 with i32 829
// CHECK: argument write order 829
// clang-format on
