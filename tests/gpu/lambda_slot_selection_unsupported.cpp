// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_slot_selection_unsupported.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct PointerSlots {
  F *Noise;
  F *Invoked;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnSlotOnce(F *volatile *Slot) {
  return Slot;
}

// Calls in both branches produce a MemoryPhi. Even though both arguments are
// the same in this test, the current production contract rejects MemoryPhi.
template <typename F>
__device__ __attribute__((noinline)) static F *volatile *
returnSlotThroughPhi(F *volatile *First, F *volatile *Second, bool UseSecond) {
  F *volatile *Result;
  if (UseSecond)
    Result = returnSlotOnce(Second);
  else
    Result = returnSlotOnce(First);
  return Result;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughPhiSlot(F *Initial, F *Replacement, bool UseSecond) {
  F *volatile Slot = Initial;
  F *volatile *Alias = returnSlotThroughPhi(&Slot, &Slot, UseSecond);
  overwriteSlot(Alias, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelPhiSlot(F Initial, F Replacement, bool UseSecond) {
  invokeThroughPhiSlot(&Initial, &Replacement, UseSecond);
}

// Different return paths expose different aggregate fields.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnRuntimeSelectedField(PointerSlots<F> *Slots, bool ReturnNoise) {
  if (ReturnNoise)
    return &Slots->Noise;
  return &Slots->Invoked;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughAmbiguousReturnedField(F *Initial, F *Replacement, F *Noise,
                                    bool ReturnNoise) {
  PointerSlots<F> Slots{Noise, Initial};
  F *volatile *Alias = returnRuntimeSelectedField(&Slots, ReturnNoise);
  overwriteSlot(Alias, Replacement);
  (*Slots.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAmbiguousReturnedField(F Initial, F Replacement, F Noise,
                             bool ReturnNoise) {
  invokeThroughAmbiguousReturnedField(&Initial, &Replacement, &Noise,
                                      ReturnNoise);
}

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makePhiBody, "phi slot")
MAKE_BODY(makeAmbiguousReturnedFieldBody, "ambiguous returned field")

int main() {
  auto PhiInitial = makePhiBody(619);
  auto PhiReplacement = makePhiBody(631);
  kernelPhiSlot<<<1, 1>>>(PhiInitial, PhiReplacement, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto AmbiguousFieldInitial = makeAmbiguousReturnedFieldBody(577);
  auto AmbiguousFieldReplacement = makeAmbiguousReturnedFieldBody(587);
  auto AmbiguousFieldNoise = makeAmbiguousReturnedFieldBody(593);
  kernelAmbiguousReturnedField<<<1, 1>>>(AmbiguousFieldInitial,
                                         AmbiguousFieldReplacement,
                                         AmbiguousFieldNoise, false);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelPhiSlot
// CHECK-NOT: [LambdaSpec]
// CHECK: phi slot 631
// CHECK: [KernelConfig] ID:{{.*}}kernelAmbiguousReturnedField
// CHECK-NOT: [LambdaSpec]
// CHECK: ambiguous returned field 587
// clang-format on
