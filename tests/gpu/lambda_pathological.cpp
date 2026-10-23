// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_pathological.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_pathological.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

// Pathological lambda-specialization cases for KernelArgVisitor.
// Several cases are expected to fail under the current analysis and are meant
// to motivate a redesign of the interprocedural visitor.
//
// Each case is set up so that, if lambda specialization works, it should print
// its own integer, either 11, 22, 33, 44, or 55.

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename Lambda> __global__ void kernel(Lambda Body) { Body(); }

template <typename L1, typename L2> __global__ void kernel_case1(L1 A, L2 B) {
  helper_multi_caller(&A);
  helper_multi_caller(&B);
}

template <typename Lambda> __global__ void kernel_case3(Lambda Body) {
  helper_slot(&Body, true);
}

template <typename Lambda> __global__ void kernel_case4(Lambda Body) {
  helper_uintptr(&Body);
}

template <typename Lambda> __global__ void kernel_case5(Lambda Body) {
  helper_box(&Body);
}

template <typename Lambda> void run(Lambda Body) {
  kernel<<<1, 1>>>(proteus::register_lambda(Body));
  gpuErrCheck(gpuDeviceSynchronize());
}

template <typename F> __device__ void invoke(F *Fn) { (*Fn)(); }

template <typename F> __device__ void invoke_by_value(F Fn) { Fn(); }

template <typename F> struct Box {
  F *Fn;
};

template <typename F> __device__ void helper_multi_caller(F *Fn) { invoke(Fn); }

template <typename F> __device__ void helper_slot(F *Fn, bool Cond) {
  F *Slot = nullptr;
  if (Cond)
    Slot = Fn;
  else
    Slot = reinterpret_cast<F *>(static_cast<uintptr_t>(0x1234));
  invoke(Slot);
}

template <typename F> __device__ void helper_uintptr(F *Fn) {
  uintptr_t Bits = reinterpret_cast<uintptr_t>(Fn);
  auto *Recovered = reinterpret_cast<F *>(Bits);
  invoke(Recovered);
}

template <typename F> __device__ void helper_box(F *Fn) {
  Box<F> B{Fn};
  invoke(B.Fn);
}

// Case 1: same helper, multiple caller contexts.
// Current visitArgument() walks all users of the callee and can merge
// callsites.
static void case1() {
  int A = 11;
  int B = 22;
  auto LA = [=, X = proteus::jit_variable(A)] __host__ __device__ {
    printf("case1 A %d\n", X);
  };
  auto LB = [=, X = proteus::jit_variable(B)] __host__ __device__ {
    printf("case1 B %d\n", X);
  };

  kernel_case1<<<1, 1>>>(proteus::register_lambda(LA),
                         proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

// Case 2: helper reused from distinct kernels.
// This stresses mixing multiple kernel parents through a shared device helper.
template <typename Lambda> __global__ void kernel_case2_a(Lambda Body) {
  helper_multi_caller(&Body);
}

template <typename Lambda> __global__ void kernel_case2_b(Lambda Body) {
  helper_multi_caller(&Body);
}

static void case2() {
  int A = 33;
  int B = 44;
  auto LA = [=, X = proteus::jit_variable(A)] __device__ {
    printf("case2 A %d\n", X);
  };
  auto LB = [=, X = proteus::jit_variable(B)] __device__ {
    printf("case2 B %d\n", X);
  };
  kernel_case2_a<<<1, 1>>>(proteus::register_lambda(LA));
  gpuErrCheck(gpuDeviceSynchronize());
  kernel_case2_b<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

// Case 3: multiple stores to the same temporary slot, only one dominating the
// actual wrapper call on the executed path.
#if PATHO_CASE == 3
static void case3() {
  int V = 55;
  auto Body = [=, X = proteus::jit_variable(V)] __device__ {
    printf("case3 %d\n", X);
  };
  kernel_case3<<<1, 1>>>(proteus::register_lambda(Body));
  gpuErrCheck(gpuDeviceSynchronize());
}
#endif

// Case 4: exact pointer round-trip via uintptr_t.
static void case4() {
  int V = 66;
  auto Body = [=, X = proteus::jit_variable(V)] __device__ {
    printf("case4 %d\n", X);
  };
  kernel_case4<<<1, 1>>>(proteus::register_lambda(Body));
  gpuErrCheck(gpuDeviceSynchronize());
}

// Case 5: forwarding through an aggregate local before the eventual call.
static void case5() {
  int V = 77;
  auto Body = [=, X = proteus::jit_variable(V)] __device__ {
    printf("case5 %d\n", X);
  };
  kernel_case5<<<1, 1>>>(proteus::register_lambda(Body));
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  case1();
  case2();
#if PATHO_CASE == 3
  case3();
#endif
  case4();
  case5();

  return 0;
}

// clang-format off
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 22
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 11
// CHECK: case1 A 11
// CHECK: case1 B 22
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 33
// CHECK: case2 A 33
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 44
// CHECK: case2 B 44
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 66
// CHECK: case4 66
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 77
// CHECK: case5 77
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 5
// CHECK-COUNT-5: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] ObjectCacheChain rank 0 with 1 level(s):
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 5
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 5 accesses 5
// clang-format on
