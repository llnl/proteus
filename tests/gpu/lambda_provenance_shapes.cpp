// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_provenance_shapes.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_provenance_shapes.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct SlotArray {
  std::uint64_t Prefix;
  F Bodies[2];
  std::uint64_t Suffix;

  __host__ __device__ __attribute__((noinline)) void invokeSecond() const {
    Bodies[1]();
  }
};

template <typename F>
__host__ __device__ __attribute__((noinline)) static SlotArray<F>
makeSlotArray(F Body, std::uint64_t Prefix, std::uint64_t Suffix) {
  return {Prefix, {Body, Body}, Suffix};
}

// Optimized IR inserts [2 x F] at byte 8, while the invoked F is inside that
// inserted aggregate at byte 12.  The outer insertion contains the tracked
// byte but does not begin at the tracked byte.
template <typename F>
__global__ static void kernelArrayMember(F Body, std::uint64_t Prefix,
                                         std::uint64_t Suffix) {
  static_assert(offsetof(SlotArray<F>, Bodies) == 8);
  static_assert(offsetof(SlotArray<F>, Bodies) + sizeof(F) == 12);
  auto Slots = makeSlotArray(Body, Prefix, Suffix);
  Slots.invokeSecond();
}

template <typename F> struct CopySource {
  std::uint64_t Prefix;
  F Body;
};

template <typename F> struct CopyDestination {
  std::uint64_t Prefix[2];
  F Body;
};

// Keep the memory transfer visible so LambdaInstUseVisitor must translate the
// tracked byte from destination offset 16 to source offset 8.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyBody(CopyDestination<F> *Destination, const CopySource<F> *Source) {
  __builtin_memcpy(&Destination->Body, &Source->Body, sizeof(F));
}

template <typename F> __global__ static void kernelOffsetMemcpy(F Body) {
  static_assert(offsetof(CopySource<F>, Body) == 8);
  static_assert(offsetof(CopyDestination<F>, Body) == 16);
  CopySource<F> Source{0x1111111111111111ULL, Body};
  CopyDestination<F> Destination{{0x2222222222222222ULL, 0x3333333333333333ULL},
                                 Body};
  copyBody(&Destination, &Source);
  Destination.Body();
}

template <typename F> struct PartialCopyValue {
  std::uint64_t Prefix[2];
  F Body;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyPrefixOnly(PartialCopyValue<F> *Destination,
               const PartialCopyValue<F> *Source) {
  __builtin_memcpy(Destination, Source, sizeof(std::uint64_t));
}

// The memcpy is a use of Destination, but it does not cover Body at byte 16.
// Treating every memcpy as a reaching definition incorrectly attributes the
// invocation to Source.Body.
template <typename F>
__global__ static void kernelPartialMemcpy(F DestinationBody, F SourceBody) {
  static_assert(offsetof(PartialCopyValue<F>, Body) == 16);
  PartialCopyValue<F> Destination{
      {0x1111111111111111ULL, 0x2222222222222222ULL}, DestinationBody};
  PartialCopyValue<F> Source{{0x3333333333333333ULL, 0x4444444444444444ULL},
                             SourceBody};
  copyPrefixOnly(&Destination, &Source);
  Destination.Body();
}

// Reaching memcpy through its source operand is a read, not a definition of
// Source.Body.  The source's initializer must remain its provenance.
template <typename F>
__global__ static void kernelMemcpySource(F SourceBody, F DestinationBody) {
  PartialCopyValue<F> Source{{0x5555555555555555ULL, 0x6666666666666666ULL},
                             SourceBody};
  PartialCopyValue<F> Destination{
      {0x7777777777777777ULL, 0x8888888888888888ULL}, DestinationBody};
  copyPrefixOnly(&Destination, &Source);
  Source.Body();
}

template <typename F> struct NestedBody {
  std::uint64_t Prefix;
  F Body;
};

template <typename F>
__device__ __attribute__((noinline)) static F *
getNestedBody(NestedBody<F> *Nested) {
  return &Nested->Body;
}

template <typename F>
__device__ __attribute__((noinline)) static F *
forwardNestedBody(NestedBody<F> *Nested) {
  return getNestedBody(Nested);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *
roundTripInteriorPointer(F *Body) {
  auto *Bytes = reinterpret_cast<char *>(Body);
  auto *Nested =
      reinterpret_cast<NestedBody<F> *>(Bytes - offsetof(NestedBody<F>, Body));
  return &Nested->Body;
}

// The pointer returned by each call is interior to the argument.  This checks
// that analyzeFunction composes the GEP offset across two call boundaries.
template <typename F>
__global__ static void kernelReturnedInteriorPointer(F Body) {
  static_assert(offsetof(NestedBody<F>, Body) == 8);
  NestedBody<F> Nested{0x4444444444444444ULL, Body};
  (*forwardNestedBody(&Nested))();
}

// The backwards walk sees +offsetof(Body), then the helper's negative GEP,
// then +offsetof(Body) at the caller.  Offsets must compose per instruction.
template <typename F> __global__ static void kernelNegativeGep(F Body) {
  NestedBody<F> Nested{0x9999999999999999ULL, Body};
  (*roundTripInteriorPointer(&Nested.Body))();
}

static void runArrayMember() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(71)] __host__ __device__ {
        printf("inserted array member %d\n", X);
      });
  kernelArrayMember<<<1, 1>>>(Body, 0x1111111111111111ULL,
                              0x2222222222222222ULL);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runOffsetMemcpy() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(83)] __host__ __device__ {
        printf("offset memcpy %d\n", X);
      });
  kernelOffsetMemcpy<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runReturnedInteriorPointer() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(97)] __host__ __device__ {
        printf("returned interior pointer %d\n", X);
      });
  kernelReturnedInteriorPointer<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

static auto makePartialCopyBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("partial memcpy %d\n", X);
      });
}

static void runPartialMemcpy() {
  auto DestinationBody = makePartialCopyBody(109);
  auto SourceBody = makePartialCopyBody(113);
  kernelPartialMemcpy<<<1, 1>>>(DestinationBody, SourceBody);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runMemcpySource() {
  auto SourceBody = makePartialCopyBody(127);
  auto DestinationBody = makePartialCopyBody(131);
  kernelMemcpySource<<<1, 1>>>(SourceBody, DestinationBody);
  gpuErrCheck(gpuDeviceSynchronize());
}

static void runNegativeGep() {
  auto Body = proteus::register_lambda(
      [X = proteus::jit_variable(137)] __host__ __device__ {
        printf("negative gep %d\n", X);
      });
  kernelNegativeGep<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  runArrayMember();
  runOffsetMemcpy();
  runReturnedInteriorPointer();
  runPartialMemcpy();
  runMemcpySource();
  runNegativeGep();
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 71
// CHECK: inserted array member 71
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 83
// CHECK: offset memcpy 83
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 97
// CHECK: returned interior pointer 97
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 109
// CHECK: partial memcpy 109
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 127
// CHECK: partial memcpy 127
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 137
// CHECK: negative gep 137
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 6
// CHECK-COUNT-6: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 6
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 6 accesses 6
// clang-format on
