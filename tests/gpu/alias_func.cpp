// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
// 1) Forward‐declare the real device function
extern "C" __device__ __attribute__((used)) void foo(int V);

// 2) Create the alias (must refer to the mangled/assembler name).  LLVM 23's
// AMDGPU frontend no longer emits source-level device aliases, so use an
// equivalent forwarding function with that compiler.
#if __clang_major__ >= 23
extern "C" __device__ __attribute__((used, noinline)) void foo_alias(int V) {
  foo(V);
}
#else
extern "C" __device__ __attribute__((used)) void foo_alias(int V)
    __attribute__((alias("foo")));
#endif

// 3) Define the aliasee
extern "C" __device__ __attribute__((used)) void foo(int V) {
  printf("foo %d\n", V);
}
#endif

// A trivial kernel that forces emission of both symbols
__global__ __attribute__((annotate("jit"))) void kernel() {
#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
  printf("Kernel\n");
  foo_alias(42);
#endif
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: Kernel
// CHECK: foo 42
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
