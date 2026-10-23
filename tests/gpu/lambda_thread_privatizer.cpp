// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_thread_privatizer.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

struct ParamPack {
  char Dummy = 0;
};

struct LaunchContext {
  int Bias;
};

template <typename T>
__host__ __device__ __attribute__((noinline)) T &&
forward_ref(std::remove_reference_t<T> &Value) {
  return static_cast<T &&>(Value);
}

template <typename F> struct Privatizer {
  F Fn;

  __host__ __device__
      __attribute__((noinline)) explicit Privatizer(const F &Fn_)
      : Fn(Fn_) {}

  __host__ __device__ __attribute__((noinline)) static Privatizer
  thread_privatize(const F &Fn_) {
    return Privatizer(Fn_);
  }

  __host__ __device__ __attribute__((noinline)) F *get_priv() { return &Fn; }
};

template <typename Params, typename F, typename Ctx>
__host__ __device__ __attribute__((noinline)) void invoke_body(Params &&, F &Fn,
                                                               Ctx &CtxValue) {
  Fn(forward_ref<Ctx &>(CtxValue));
}

template <typename F>
__global__ void kernel_thread_privatizer(F Fn, ParamPack Params) {
  auto Private = Privatizer<F>::thread_privatize(Fn);
  F *Priv = Private.get_priv();
  LaunchContext Ctx{10};
  invoke_body(Params, *Priv, Ctx);
}

auto makeLambda(int X, int Y, bool UseAdd, int *Out) {
  return proteus::register_lambda([X = proteus::jit_variable(X),
                                   Y = proteus::jit_variable(Y),
                                   UseAdd = proteus::jit_variable(UseAdd),
                                   Out] __host__ __device__(LaunchContext Ctx) {
    int Result = UseAdd ? (X + Y) : (X - Y);
    *Out = Result + Ctx.Bias;
    printf("thread privatizer %d %d %d -> %d\n", X, Y, int(UseAdd), *Out);
  });
}

static void runCase(int X, int Y, bool UseAdd, int Expected) {
  int *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));
  *Out = -1;

  ParamPack Params{};
  auto Fn = makeLambda(X, Y, UseAdd, Out);
  kernel_thread_privatizer<<<1, 1>>>(Fn, Params);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("host observed %d\n", *Out);

  if (*Out != Expected)
    printf("mismatch expected %d got %d\n", Expected, *Out);

  gpuErrCheck(gpuFree(Out));
}

int main() {
  runCase(7, 3, true, 20);
  runCase(7, 3, false, 14);
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 3
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 7
// CHECK: thread privatizer 7 3 1 -> 20
// CHECK: host observed 20
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i8 0
// CHECK: thread privatizer 7 3 0 -> 14
// CHECK: host observed 14
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// clang-format on
