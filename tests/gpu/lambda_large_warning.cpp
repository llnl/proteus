// clang-format off
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_large_warning.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_large_warning.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// This test is meant to test as a regression when Proteus is built with -Werror -Wall against the amdclang message
// warning: local memory global used by non-kernel function.  This warning was caused by storing LambdaFunctorWrapper
// information containing pointers to functions referencing shared memory inside llvm.global.annotations.
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include "proteus/JitInterface.h"
template <typename T> struct Functor {

  __attribute__((noinline)) __host__ __device__ void operator()(T &&Lam,
                                                                const int I) {
    Lam(I);
  }
};

template <typename T> __global__ void kernel1d(int N, T LB) {
  const int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N) {
    LB(I);
  }
}

template <typename T> void launch(int N, T &&LB) {
  auto RegisteredFunctor = proteus::register_lambda(LB);
  kernel1d<<<1, 32>>>(N, RegisteredFunctor);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int *X = nullptr;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(int)));
  X[0] = -1;
  int A = 4;

  launch(32, [=, A = proteus::jit_variable(A)] __device__
         __attribute__((annotate("jit"))) (int I) {
           // The warning is triggered by LDS/shared-memory globals defined in
           // the registered lambda body being used through the non-kernel
           // Proteus wrapper.
           __shared__ int Smem[32];
           Smem[threadIdx.x] = I;
           __syncthreads();
           if (threadIdx.x == 0) {
             X[0] = Smem[0] + A;
           }
         });

  std::cout << "x[0] = " << X[0] << "\n";
  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 4
// CHECK: x[0] = 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}}
// NumExecs 1 NumHits 0 CHECK-FIRST: [proteus][JitEngineDevice] ObjectCacheChain
// rank 0 with 1 level(s): CHECK-FIRST: [proteus][JitEngineDevice] StorageCache
// rank 0 hits 0 accesses 1 CHECK-SECOND: [proteus][JitEngineDevice]
// StorageCache rank 0 hits 1 accesses 1
