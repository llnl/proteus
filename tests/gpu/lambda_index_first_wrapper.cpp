// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_index_first_wrapper.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__host__ __device__ __attribute__((noinline)) static void invokeInner(F Inner) {
  Inner();
}

template <typename F>
__host__ __device__ __attribute__((noinline)) static void
invokeWithIndexFirstSlot(F Body, int I) {
  int LocalIndex = I;
  int *IndexPtr = &LocalIndex;

  auto Inner = [IndexPtr, Body] __host__ __device__() { Body(*IndexPtr); };
  invokeInner(Inner);
}

template <typename F> __global__ static void kernel(F Wrapped) {
  if (threadIdx.x == 0)
    Wrapped(5);
}

static auto makeWrapped(int X, int *Out) {
  auto Registered = proteus::register_lambda(
      [X = proteus::jit_variable(X), Out] __host__ __device__(int I) {
        *Out = X + I;
        printf("index-first wrapper %d %d -> %d\n", X, I, *Out);
      });

  return [Registered] __host__ __device__(int I) {
    invokeWithIndexFirstSlot(Registered, I);
  };
}

int main() {
  int *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));
  *Out = -1;

  kernel<<<1, 1>>>(makeWrapped(37, Out));
  gpuErrCheck(gpuDeviceSynchronize());
  printf("host observed %d\n", *Out);

  if (*Out != 42)
    printf("mismatch expected 42 got %d\n", *Out);

  gpuErrCheck(gpuFree(Out));
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 37
// CHECK: index-first wrapper 37 5 -> 42
// CHECK: host observed 42
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// clang-format on
