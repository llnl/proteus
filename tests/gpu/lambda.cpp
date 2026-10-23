// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include "proteus/JitInterface.h"

template <typename T> __global__ void kernel(T LB) {
  std::size_t I = blockIdx.x + threadIdx.x;
  if (I == 0)
    LB();
}

template <typename T> __global__ void kernel(size_t N, T LB) {
  std::size_t I = blockDim.x * blockIdx.x + threadIdx.x;
  if (I < N)
    LB(I);
}

template <typename T> void registerRun(T &&LB) {
  kernel<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

template <typename T> void registerRun(int N, T &&LB) {
  constexpr int BlockSize = 256;
  const int NumBlocks = (N + BlockSize - 1) / BlockSize;
  kernel<<<NumBlocks, BlockSize>>>(N, proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

inline void launch(double C, double *X) {
  registerRun([=, C = proteus::jit_variable(C)] __device__
              __attribute__((annotate("jit"))) () { X[0] = C + 2; });
  std::cout << "x[0] = " << X[0] << "\n";
}

int main() {
  double A{3.14};
  double B{1.23};
  double C{4.56};
  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 2));

  registerRun([=, A = proteus::jit_variable(A)] __device__
              __attribute__((annotate("jit"))) () { X[0] = A; });

  std::cout << "x[0] = " << X[0] << "\n";

  registerRun(2, [=, C = proteus::jit_variable(C)] __device__
              __attribute__((annotate("jit"))) (int I) { X[I] = C; });

  std::cout << "x[0:1] = " << X[0] << "," << X[1] << "\n";

  run(proteus::register_lambda(
      [=, A = proteus::jit_variable(A), B = proteus::jit_variable(B)] __device__
      __attribute__((annotate("jit"))) () { X[0] = A + B; }));
  std::cout << "x[0] = " << X[0] << "\n";

  launch(C, X);
  gpuErrCheck(gpuFree(X));
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with double 3.140000e+00
// CHECK: x[0] = 3.14
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with double 4.560000e+00
// CHECK: x[0:1] = 4.56,4.56
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with double 1.230000e+00
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with double 3.140000e+00
// CHECK: x[0] = 4.37
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with double 4.560000e+00
// CHECK: x[0] = 6.56
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 4
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 4 accesses 4
