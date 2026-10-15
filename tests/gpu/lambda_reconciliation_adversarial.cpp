// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_reconciliation_adversarial.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline)) static F returnByValue(const F *Source) {
  return *Source;
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterDuplicatingField(F *Source) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 2 * sizeof(std::uint32_t));

  // Returning the aggregate by value encourages Clang to materialize Copy as
  // parallel extractvalue/store operations instead of one memcpy.
  F Copy = returnByValue(Source);
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(&Copy);
  auto *Src = reinterpret_cast<const std::uint32_t *>(Source);
  Dst[0] = Src[0];
  Dst[1] = Src[0];
  Copy();
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterPermutingFields(F *Source) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 2 * sizeof(std::uint32_t));

  F Copy = returnByValue(Source);
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(&Copy);
  auto *Src = reinterpret_cast<const std::uint32_t *>(Source);
  Dst[0] = Src[1];
  Dst[1] = Src[0];
  Copy();
}

template <typename F>
__device__ __attribute__((noinline)) static void
overwriteSecondField(F *Body, const F *Donor) {
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(Body);
  auto *Src = reinterpret_cast<const std::uint32_t *>(Donor);
  Dst[1] = Src[1];
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterCallAndStore(F *Initial, F *StoreDonor, F *CallDonor) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 3 * sizeof(std::uint32_t));

  F Copy = *Initial;
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(&Copy);
  auto *Src = reinterpret_cast<const std::uint32_t *>(StoreDonor);
  Dst[1] = Src[1];
  overwriteSecondField(&Copy, CallDonor);
  Copy();
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterHeterogeneousStores(F *Initial, F *FirstDonor, F *SecondDonor) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 2 * sizeof(std::uint32_t));

  F Copy = returnByValue(Initial);
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(&Copy);
  auto *First = reinterpret_cast<const std::uint32_t *>(FirstDonor);
  auto *Second = reinterpret_cast<const std::uint32_t *>(SecondDonor);
  Dst[0] = First[0];
  Dst[1] = Second[1];
  Copy();
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterPartialByteStore(F *Source, std::uint8_t Replacement) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 2 * sizeof(std::uint32_t));

  F Copy = returnByValue(Source);
  // Character writes may legally modify an object's representation. A
  // byte-sized store must not be accepted as the provenance of the complete
  // first capture, nor may the earlier whole-object value remain authoritative.
  reinterpret_cast<volatile std::uint8_t *>(&Copy)[0] = Replacement;
  Copy();
}

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterConditionalStore(F *Initial, F *FirstDonor, F *SecondDonor,
                            bool UseSecond) {
  static_assert(std::is_trivially_copyable_v<F>);
  static_assert(sizeof(F) == 2 * sizeof(std::uint32_t));

  F Copy = returnByValue(Initial);
  auto *Dst = reinterpret_cast<volatile std::uint32_t *>(&Copy);
  auto *First = reinterpret_cast<const std::uint32_t *>(FirstDonor);
  auto *Second = reinterpret_cast<const std::uint32_t *>(SecondDonor);
  if (UseSecond)
    Dst[0] = Second[0];
  else
    Dst[0] = First[0];
  Copy();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDuplicateField(F Body) {
  invokeAfterDuplicatingField(&Body);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPermuteFields(F Body) {
  invokeAfterPermutingFields(&Body);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCallAndStore(F Initial, F StoreDonor, F CallDonor) {
  invokeAfterCallAndStore(&Initial, &StoreDonor, &CallDonor);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelHeterogeneousStores(F Initial, F FirstDonor, F SecondDonor) {
  invokeAfterHeterogeneousStores(&Initial, &FirstDonor, &SecondDonor);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPartialByteStore(F Body, std::uint8_t Replacement) {
  invokeAfterPartialByteStore(&Body, Replacement);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelConditionalStore(F Initial, F FirstDonor, F SecondDonor,
                       bool UseSecond) {
  invokeAfterConditionalStore(&Initial, &FirstDonor, &SecondDonor, UseSecond);
}

static auto makeDuplicatePair(int First, int Second) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second)] __host__ __device__ {
        printf("reconciled pair %d %d\n", First, Second);
      });
}

static auto makePermutedPair(int First, int Second) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second)] __host__ __device__ {
        printf("permuted pair %d %d\n", First, Second);
      });
}

static auto makeHeterogeneousPair(int First, int Second) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second)] __host__ __device__ {
        printf("heterogeneous pair %d %d\n", First, Second);
      });
}

static auto makeBytePair(std::uint32_t First, std::uint32_t Second) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second)] __host__ __device__ {
        printf("byte-overwritten pair %u %u\n", First, Second);
      });
}

static auto makeConditionalPair(int First, int Second) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second)] __host__ __device__ {
        printf("conditional pair %d %d\n", First, Second);
      });
}

static auto makeTriple(int First, int Second, int Third) {
  return proteus::register_lambda(
      [First = proteus::jit_variable(First),
       Second = proteus::jit_variable(Second),
       Third = proteus::jit_variable(Third)] __host__ __device__ {
        printf("reconciled triple %d %d %d\n", First, Second, Third);
      });
}

int main() {
  auto Duplicate = makeDuplicatePair(101, 103);
  kernelDuplicateField<<<1, 1>>>(Duplicate);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Permute = makePermutedPair(107, 109);
  kernelPermuteFields<<<1, 1>>>(Permute);
  gpuErrCheck(gpuDeviceSynchronize());

  auto Initial = makeTriple(113, 127, 131);
  auto StoreDonor = makeTriple(137, 139, 149);
  auto CallDonor = makeTriple(151, 157, 163);
  kernelCallAndStore<<<1, 1>>>(Initial, StoreDonor, CallDonor);
  gpuErrCheck(gpuDeviceSynchronize());

  auto HeterogeneousInitial = makeHeterogeneousPair(167, 173);
  auto HeterogeneousFirst = makeHeterogeneousPair(179, 181);
  auto HeterogeneousSecond = makeHeterogeneousPair(191, 193);
  kernelHeterogeneousStores<<<1, 1>>>(HeterogeneousInitial,
                                      HeterogeneousFirst,
                                      HeterogeneousSecond);
  gpuErrCheck(gpuDeviceSynchronize());

  auto BytePair = makeBytePair(0x11223344u, 0x55667788u);
  kernelPartialByteStore<<<1, 1>>>(BytePair, 0xaau);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ConditionalInitial = makeConditionalPair(197, 199);
  auto ConditionalFirst = makeConditionalPair(211, 223);
  auto ConditionalSecond = makeConditionalPair(227, 229);
  kernelConditionalStore<<<1, 1>>>(ConditionalInitial, ConditionalFirst,
                                   ConditionalSecond, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// A whole-functor source is not authoritative after fields are duplicated,
// permuted, or overwritten from multiple sources.
// CHECK: [KernelConfig] ID:{{.*}}kernelDuplicateField
// CHECK-NOT: [LambdaSpec]
// CHECK: reconciled pair 101 101
// CHECK: [KernelConfig] ID:{{.*}}kernelPermuteFields
// CHECK-NOT: [LambdaSpec]
// CHECK: permuted pair 109 107
// CHECK: [KernelConfig] ID:{{.*}}kernelCallAndStore
// CHECK-NOT: [LambdaSpec]
// CHECK: reconciled triple 113 157 131
// CHECK: [KernelConfig] ID:{{.*}}kernelHeterogeneousStores
// CHECK-NOT: [LambdaSpec]
// CHECK: heterogeneous pair 179 193
// CHECK: [KernelConfig] ID:{{.*}}kernelPartialByteStore
// CHECK-NOT: [LambdaSpec]
// CHECK: byte-overwritten pair 287454122 1432778632
// CHECK: [KernelConfig] ID:{{.*}}kernelConditionalStore
// CHECK-NOT: [LambdaSpec]
// CHECK: conditional pair 227 199
