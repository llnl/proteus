// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_size_test.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// Exercises the functor-size plumbing (proteus.functor_size).
//
// Case 1 (kernelEnclosedFunctor): a registered two-slot functor captured by
// value inside an unregistered enclosing lambda. On device the closure is
// assembled slot-wise with no single write covering the functor, and the
// enclosure's own trailing capture is stored directly past the functor's
// extent. Specialization must survive.
//
// Case 2 (kernelEnvelopeSuffix): the functor is the leading member of an
// envelope whose trailing field is overwritten after the whole-envelope copy.
// The functor's bytes are untouched, so specialization must survive -- but
// without optnone the zero-index GEPs fold away, the reconciliation query
// falls back to an unbounded location, and MemorySSA selects the suffix store
// (outside the functor's extent) as the clobber: a spurious decline.
// EXPECTED TO FAIL until the query location is bounded by the functor's size
// (gate 8): the fix is MemoryLocation(base, proteus.functor_size), which
// makes the out-of-range suffix store non-aliasing by construction.
template <typename F>
__device__ __attribute__((noinline)) static void invoke(F Lambda) {
  Lambda();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelEnclosedFunctor(F Enclosure) {
  invoke(Enclosure);
}

template <typename F> struct Envelope {
  F Body;
  std::uint64_t Suffix;
};

template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterSuffixStore(F *Body, std::uint64_t Patch) {
  Envelope<F> Source{*Body, 0x1111111111111111ULL};
  Envelope<F> Copy = Source;
  Copy.Suffix = Patch;
  Copy.Body();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelEnvelopeSuffix(F Body, std::uint64_t Patch) {
  invokeAfterSuffixStore(&Body, Patch);
}

static auto makeBody(int A, int B) {
  return proteus::register_lambda([X = proteus::jit_variable(A),
                                   Y = proteus::jit_variable(B)] __host__
                                                                __device__ {
    printf("enclosed functor %d %d\n", X, Y);
  });
}

// A distinct closure type for the envelope case so one analysis outcome
// cannot mask the other's.
static auto makeSuffixBody(int A, int B) {
  return proteus::register_lambda([X = proteus::jit_variable(A),
                                   Y = proteus::jit_variable(B)] __host__
                                                                __device__ {
    printf("envelope suffix functor %d %d\n", X, Y);
  });
}

int main() {
  auto Body = makeBody(631, 757);
  int A = 5;
  auto Enclosure = [Body, A]() {
    printf("Hello from Enclosure %d\n", A);
    Body();
  };
  kernelEnclosedFunctor<<<1, 1>>>(Enclosure);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SuffixBody = makeSuffixBody(881, 999);
  kernelEnvelopeSuffix<<<1, 1>>>(SuffixBody, 0x2222222222222222ULL);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelEnclosedFunctor
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 631
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 757
// CHECK: Hello from Enclosure 5
// CHECK: enclosed functor 631 757
// CHECK: [KernelConfig] ID:{{.*}}kernelEnvelopeSuffix
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 881
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 999
// CHECK: envelope suffix functor 881 999
// clang-format on
