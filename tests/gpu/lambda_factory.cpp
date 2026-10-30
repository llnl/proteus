// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>
#include <tuple>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__host__ __device__ void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename L1, typename L2, typename L3> struct Abstraction {
  L1 lambda_1;
  L2 lambda_2;
  L3 lambda_3;
  Abstraction(L1 &&l1, L2 &&l2, L3 &&l3)
      : lambda_1(l1), lambda_2(l2), lambda_3(l3) {};

  __host__ __device__ auto operator()() {
    lambda_1();
    lambda_2();
    lambda_3();
  }
};

// Macro can be invoked function style, either within a Function where the
// arguments passed to jit_variable are function parameters, or Constants as
// below
auto declareLambda(int rc1, int rc2) {
  return proteus::register_lambda([=, C = proteus::jit_variable(rc1), D = 5,
                                   C2 = proteus::jit_variable(rc2)]()
                                      __attribute__((annotate("jit"))) {
                                        printInt(C);
                                        printInt(C2);
                                        printInt(D);
                                      });
}

auto declareLambdaThree(int rc1, int rc2, int rc3) {
  return [=, C = proteus::jit_variable(rc1), D = 4,
          C2 = proteus::jit_variable(rc2), rc3 = proteus::jit_variable(rc3)]()
             __attribute__((annotate("jit"))) {
               printInt(C);
               printInt(C2);
               printInt(rc3);
               printInt(D);
             };
}

auto forwardLambda(int rc1, int, int rc3) {
  int loc = 1;
  return declareLambdaThree(rc1, loc, rc3);
}

template <typename T> void run(T &&LB) {
  using LambdaT = std::decay_t<T>;
  LambdaT Body = std::forward<T>(LB);
  auto func = reinterpret_cast<const void *>(&kernel<LambdaT>);
  void *args[] = {&Body};
  gpuErrCheck(gpuLaunchKernel(func, dim3(1), dim3(1), args, 0, 0));
  gpuErrCheck(gpuDeviceSynchronize());
}

void forDeclareLambda() {
  for (int i = 0; i < 5; ++i) {
    auto lam = declareLambda(i, 2);
    run(lam);
  }
}

#define REGISTER_LAMBDA(Lambda) return Functor<__COUNTER__>(Lambda);

int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;

  auto ZeroLambda = declareLambda(Zero, One);

  auto OneLambda = declareLambda(One, Two);

  auto TwoLambda = declareLambda(Two, Zero);

  auto BigLam = proteus::register_lambda(declareLambdaThree(Zero, One, Two));

  auto ForLam = proteus::register_lambda(forwardLambda(Zero, One, Two));

  Abstraction A(std::move(ZeroLambda), std::move(OneLambda),
                std::move(TwoLambda));

  run(A);
  run(BigLam);
  run(ForLam);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 5
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 5
// CHECK: Integer = 2
// CHECK: Integer = 0
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 3 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 1 accesses 3
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
// clang-format on
