// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_aggregate_store_reconciliation.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct LambdaSources {
  F First;
  F Second;
};

template <typename F> struct __attribute__((packed)) ShiftedPointerSlot {
  std::uint32_t Padding;
  F *Invoked;
};

// AtZero occupies bytes [0, sizeof(F *)); Shifted.Invoked starts at byte four.
// Stores to the two fields therefore have different destinations that overlap.
template <typename F> union OverlappingPointerSlots {
  F *AtZero;
  ShiftedPointerSlot<F> Shifted;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughOverlappingStores(F *First, F *Second) {
  LambdaSources<F> Sources{*First, *Second};
  OverlappingPointerSlots<F> Slots;

  // Both values have the same pointer base but different source offsets. That
  // is insufficient to prove that these stores reconstruct an aggregate: the
  // destination ranges overlap and have no consistent field mapping.
  Slots.Shifted.Invoked = &Sources.Second;
  Slots.AtZero = &Sources.First;

  // The second store overlaps only the low half of Shifted.Invoked, so it
  // cannot serve as provenance for the complete pointer loaded below. Keep
  // the load and call in device IR without executing the malformed pointer.
  if (threadIdx.x != 0)
    (*Slots.Shifted.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelOverlappingStoreReconciliation(F First, F Second) {
  invokeThroughOverlappingStores(&First, &Second);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("overlapping aggregate stores %d\n", X);
      });
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  auto First = makeBody(727);
  auto Second = makeBody(733);
  kernelOverlappingStoreReconciliation<<<1, 1>>>(First, Second);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("aggregate reconciliation completed\n");
  return 0;
}

// Overlapping destination ranges are not a proven aggregate reconstruction.
// The analysis must reject this shape even though both source pointers share a
// common base.
// CHECK: [KernelConfig] ID:{{.*}}kernelOverlappingStoreReconciliation
// CHECK-NOT: [LambdaSpec]
// CHECK: aggregate reconciliation completed
