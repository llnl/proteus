// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory_forwarded.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <utility>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__host__ __device__ void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T> __global__ void kernel(T LB) { LB(); }

auto declareLambda(int rc1, int rc2) {
  return proteus::register_lambda(
      [=, C = proteus::jit_variable(rc1), C2 = proteus::jit_variable(rc2)]()
          __attribute__((annotate("jit"))) {
            printInt(C);
            printInt(C2);
          });
}

__attribute__((noinline)) gpuError_t launchIndirect(const void *Func, dim3 Grid,
                                                    dim3 Block, void **Args) {
  return gpuLaunchKernel(Func, Grid, Block, Args, 0, 0);
}

template <typename T> void run(T &&LB) {
  using LambdaT = std::decay_t<T>;
  LambdaT Body = std::forward<T>(LB);
  const void *Func = reinterpret_cast<const void *>(&kernel<LambdaT>);
  void *Args[] = {&Body};
  gpuErrCheck(launchIndirect(Func, dim3(1), dim3(1), Args));
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int Zero = 0;
  int One = 1;

  auto ZeroLambda = declareLambda(Zero, One);
  auto OneLambda = declareLambda(One, Zero);

  run(ZeroLambda);
  run(OneLambda);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK: Integer = 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// clang-format on
