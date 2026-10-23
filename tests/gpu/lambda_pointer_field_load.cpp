// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_pointer_field_load.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// This matches the RAJA/MFEM shape: the registered functor is reached through
// a pointer field of an aggregate passed into a device helper.  The helper also
// reads unrelated pointer fields first, so the IR contains several
//   %p = load ptr, ptr %aggregate_field
// instructions before loading the functor pointer.
template <typename F> struct PointerFieldContext {
  int *First;
  int *Second;
  int *Limit;
  F *Body;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughPointerField(PointerFieldContext<F> *Context, int Offset) {
  int Index = *Context->First * *Context->Second + Offset;
  if (Index < *Context->Limit)
    (*Context->Body)(Index);
}

template <typename F> __global__ static void kernelPointerFieldLoad(F Body) {
  int First = 2;
  int Second = 3;
  int Limit = 7;
  PointerFieldContext<F> Context{&First, &Second, &Limit, &Body};
  invokeThroughPointerField(&Context, 0);
}

int main() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(211)] __host__ __device__(int Index) {
        printf("pointer field load %d %d\n", X, Index);
      });
  kernelPointerFieldLoad<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 211
// CHECK: pointer field load 211 6
// clang-format on
