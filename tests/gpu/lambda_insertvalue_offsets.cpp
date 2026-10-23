// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_insertvalue_offsets.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_insertvalue_offsets.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// These wrappers are intentionally returned by value.  Once SROA has split
// the return value, LLVM represents their construction as insertvalue chains
// and their use as extractvalue/GEP chains.

template <typename F> struct ShiftedEnvelope {
  std::uint64_t Prefix;
  F Body;

  __host__ __device__ __attribute__((noinline)) void operator()() const {
    Body();
  }
};

template <typename F>
__host__ __device__ __attribute__((noinline)) static ShiftedEnvelope<F>
makeShiftedEnvelope(F Body, std::uint64_t Prefix) {
  return {Prefix, Body};
}

// The registered lambda starts at offset zero in the kernel argument, but at
// offset eight in the returned aggregate.  Crossing the matching insertvalue
// must therefore rebase the tracked offset from eight to zero.
template <typename F> __global__ static void kernelShift(F Body) {
  static_assert(offsetof(ShiftedEnvelope<F>, Body) == 8);
  auto Envelope = makeShiftedEnvelope(Body, 0x1111111111111111ULL);
  Envelope();
}

// Applying the same transformation twice catches visitors that repair one
// aggregate boundary but retain stale state at the next one.
template <typename F> __global__ static void kernelDoubleShift(F Body) {
  static_assert(offsetof(ShiftedEnvelope<F>, Body) == 8);
  auto First = makeShiftedEnvelope(Body, 0x2222222222222222ULL);
  auto Second = makeShiftedEnvelope(First.Body, 0x3333333333333333ULL);
  Second();
}

template <typename F> struct PaddedEnvelope {
  std::uint8_t Prefix;
  alignas(16) F Body;
  std::uint64_t Suffix;

  __host__ __device__ __attribute__((noinline)) void operator()() const {
    Body(static_cast<int>(Suffix));
  }
};

template <typename F>
__host__ __device__ __attribute__((noinline)) static PaddedEnvelope<F>
makePaddedEnvelope(F Body, std::uint64_t Suffix) {
  return {0x44, Body, Suffix};
}

// This combines a larger non-zero offset with ABI padding and an insert after
// Body.  The analysis has to skip the suffix insert and use the DataLayout
// offset of Body rather than assuming tightly packed fields.
template <typename F>
__global__ static void kernelPadded(F Body, std::uint64_t Suffix) {
  static_assert(offsetof(PaddedEnvelope<F>, Body) == 16);
  auto Envelope = makePaddedEnvelope(Body, Suffix);
  Envelope();
}

static void runShift() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(31)] __host__ __device__ {
        printf("single insertvalue rebase %d\n", X);
      });
  kernelShift<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runDoubleShift() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(47)] __host__ __device__ {
        printf("double insertvalue rebase %d\n", X);
      });
  kernelDoubleShift<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runPadded() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(59)] __host__ __device__(int Suffix) {
        printf("padded insertvalue rebase %d suffix %d\n", X, Suffix);
      });
  kernelPadded<<<1, 1>>>(Body, 0x5555555555555555ULL);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  runShift();
  runDoubleShift();
  runPadded();
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 31
// CHECK: single insertvalue rebase 31
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 47
// CHECK: double insertvalue rebase 47
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 59
// CHECK: padded insertvalue rebase 59 suffix 1431655765
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 3
// CHECK-COUNT-3: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
// clang-format on
