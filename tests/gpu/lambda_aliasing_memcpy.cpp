// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_aliasing_memcpy.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_aliasing_memcpy.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwrite(F **Slot, const F *Replacement) {
  __builtin_memcpy(*Slot, Replacement, sizeof(F));
}

// optnone preserves the pointer-sized memcpy rather than turning it into a
// pointer load. WriteSlot and ReadSlot alias, without an SSA edge between them.
template <typename F>
__global__ __attribute__((annotate("jit"), optnone)) static void
kernelAliasingMemcpy(F Initial, F Replacement, F **WriteSlot, F **ReadSlot,
                     int *Result) {
  F Copy = Initial;
  F *Relay;
  *WriteSlot = &Copy;
  __builtin_memcpy(&Relay, ReadSlot, sizeof(Relay));
  overwrite(&Relay, &Replacement);
  *Result = Copy();
}

// Both the closure and the pointer slot live in local device storage. The
// caller supplies the same slot through two formals, so discovering their
// connection requires following the caller bindings as well as the memcpy.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeLocalSlot(F Initial, F Replacement, F **WriteSlot, F **ReadSlot) {
  F Copy = Initial;
  F *Relay;
  *WriteSlot = &Copy;
  __builtin_memcpy(&Relay, ReadSlot, sizeof(Relay));
  overwrite(&Relay, &Replacement);
  printf("local slot memcpy %d\n", Copy());
}

template <typename F>
__global__ __attribute__((annotate("jit"), optnone)) static void
kernelLocalSlot(F Initial, F Replacement) {
  F *Slot;
  invokeLocalSlot(Initial, Replacement, &Slot, &Slot);
}

// Separate closure types keep the cases independent of multi-caller limits.
template <int Case> static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ { return X; });
}

int main() {
  auto Initial = makeBody<0>(1441);
  using F = decltype(Initial);
  static_assert(std::is_trivially_copyable_v<F>);
  F **Slot;
  int *Result;
  gpuErrCheck(gpuMallocManaged(&Slot, sizeof(*Slot)));
  gpuErrCheck(gpuMallocManaged(&Result, sizeof(*Result)));
  kernelAliasingMemcpy<<<1, 1>>>(Initial, makeBody<0>(1449), Slot, Slot,
                                 Result);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("aliasing memcpy %d\n", *Result);
  fflush(stdout);
  gpuErrCheck(gpuFree(Result));
  gpuErrCheck(gpuFree(Slot));

  // This case needs no managed allocations, external slots, or globals.
  kernelLocalSlot<<<1, 1>>>(makeBody<1>(1461), makeBody<1>(1469));
  gpuErrCheck(gpuDeviceSynchronize());
}

// Neither case may specialize from the closure value preceding the overwrite.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1441
// RESULT: aliasing memcpy 1449
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1461
// RESULT: local slot memcpy 1469
// clang-format on
