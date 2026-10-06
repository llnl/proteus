// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_slot_selection_supported.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
observeSlot(F *volatile *, F *) {}

// A read-only call after the replacement must not hide the reaching write.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossTrailingReadOnlyCall(F *Initial, F *Replacement, F *Noise) {
  F *volatile Slot = Initial;
  overwriteSlot(&Slot, Replacement);
  observeSlot(&Slot, Noise);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelTrailingReadOnlyCall(F Initial, F Replacement, F Noise) {
  invokeAcrossTrailingReadOnlyCall(&Initial, &Replacement, &Noise);
}

// A write after invocation cannot affect the value consumed by that call.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeBeforeLaterCall(F *Initial, F *Later) {
  F *volatile Slot = Initial;
  (*Slot)();
  overwriteSlot(&Slot, Later);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCallAfterInvoke(F Initial, F Later) {
  invokeBeforeLaterCall(&Initial, &Later);
}

// Both select operands identify the same slot, so control flow does not make
// the selected lambda ambiguous.
template <typename F>
__device__ __attribute__((noinline)) static F *volatile *
selectSameSlot(F *volatile *First, F *volatile *Second, bool UseSecond) {
  return UseSecond ? Second : First;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughSelectedSlot(F *Initial, F *Replacement, bool UseSecond) {
  F *volatile Slot = Initial;
  F *volatile *Alias = selectSameSlot(&Slot, &Slot, UseSecond);
  overwriteSlot(Alias, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelSelectedSlot(F Initial, F Replacement, bool UseSecond) {
  invokeThroughSelectedSlot(&Initial, &Replacement, UseSecond);
}

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makeReadOnlyBody, "trailing read-only call")
MAKE_BODY(makePostCallBody, "call after invoke")
MAKE_BODY(makeSelectedBody, "selected slot")

int main() {
  auto ReadOnlyInitial = makeReadOnlyBody(461);
  auto ReadOnlyReplacement = makeReadOnlyBody(463);
  auto ReadOnlyNoise = makeReadOnlyBody(467);
  kernelTrailingReadOnlyCall<<<1, 1>>>(ReadOnlyInitial, ReadOnlyReplacement,
                                       ReadOnlyNoise);
  gpuErrCheck(gpuDeviceSynchronize());

  auto PostCallInitial = makePostCallBody(523);
  auto PostCallLater = makePostCallBody(541);
  kernelCallAfterInvoke<<<1, 1>>>(PostCallInitial, PostCallLater);
  gpuErrCheck(gpuDeviceSynchronize());

  auto SelectedInitial = makeSelectedBody(601);
  auto SelectedReplacement = makeSelectedBody(607);
  kernelSelectedSlot<<<1, 1>>>(SelectedInitial, SelectedReplacement, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 463
// CHECK: trailing read-only call 463
// CHECK: [LambdaSpec] Replacing slot 0 with i32 523
// CHECK: call after invoke 523
// CHECK: [LambdaSpec] Replacing slot 0 with i32 607
// CHECK: selected slot 607
// clang-format on
