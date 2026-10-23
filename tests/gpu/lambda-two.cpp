// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda-two.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda-two.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "proteus/JitInterface.h"

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ static void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T, typename L> __global__ static void kernel(T LB, L LB2) {
  std::size_t I = blockIdx.x + threadIdx.x;
  if (I == 0) {
    LB(I);
    LB2(I);
  }
}

template <typename LambdaType> class Abstraction {
public:
  LambdaType Lambda;
  Abstraction(const LambdaType &Lam) : Lambda(Lam) {};
  __host__ __device__ auto operator()(size_t I) { return Lambda(I); }
};

template <typename T, typename L> static void registerRun(T &&LB, L &&LB2) {
  constexpr int BlockSize = 256;
  constexpr int N = 1000;
  const int NumBlocks = (N + BlockSize - 1) / BlockSize;
  auto F2 = proteus::register_lambda(LB2);
  auto F1 = proteus::register_lambda(LB);
  kernel<<<NumBlocks, BlockSize>>>(F1, F2);
  gpuErrCheck(gpuDeviceSynchronize());
}

static auto declareLambda(int Rc1, int Rc2) {
  return [=, Rc1 = proteus::jit_variable(Rc1), Rc2 = proteus::jit_variable(Rc2)]
      __attribute__((annotate("jit"))) (size_t) {
        printInt(Rc1);
        printInt(Rc2);
      };
}

inline static void launch(int C, int D) {
  registerRun([=, C = proteus::jit_variable(C)] __device__ __attribute__((
                  annotate("jit"))) (size_t) { printf("Integer = %d\n", C); },
              [=, D = proteus::jit_variable(D)] __device__ __attribute__((
                  annotate("jit"))) (size_t) { printf("Integer = %d\n", D); });
}

int main() {
  launch(1, 2);

  registerRun(declareLambda(42, 24), declareLambda(24, 42));

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 24
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 42
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 24
// CHECK: Integer = 42
// CHECK: Integer = 24
// CHECK: Integer = 24
// CHECK: Integer = 42
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
// clang-format on
