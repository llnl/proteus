// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_inst_visitor_adversarial.%ext | %FILECHECK %s
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

// LLVM shape: the returned pointer is an SSA select whose two arms both name
// the tracked slot. LambdaInstUseVisitor must traverse the select result to
// discover the later overwrite.
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

// LLVM shape: a pointer-returning helper returns the result of another
// pointer-returning helper. Both call boundaries must preserve the slot's
// identity before the overwrite is collected.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnSlotOnce(F *volatile *Slot) {
  return Slot;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnSlotTwice(F *volatile *Slot) {
  return returnSlotOnce(Slot);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughNestedReturn(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  F *volatile *Alias = returnSlotTwice(&Slot);
  overwriteSlot(Alias, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelNestedReturn(F Initial, F Replacement) {
  invokeThroughNestedReturn(&Initial, &Replacement);
}

// LLVM shape: calls in both branches feed an SSA phi. The two phi arms still
// resolve to the same tracked slot, so the write through the returned pointer
// has one unambiguous target.
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

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makeSelectedBody, "selected slot")
MAKE_BODY(makeNestedReturnBody, "nested return")
MAKE_BODY(makePhiBody, "phi slot")

int main() {
  auto SelectedInitial = makeSelectedBody(601);
  auto SelectedReplacement = makeSelectedBody(607);
  kernelSelectedSlot<<<1, 1>>>(SelectedInitial, SelectedReplacement, true);
  gpuErrCheck(gpuDeviceSynchronize());

  auto NestedInitial = makeNestedReturnBody(613);
  auto NestedReplacement = makeNestedReturnBody(617);
  kernelNestedReturn<<<1, 1>>>(NestedInitial, NestedReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto PhiInitial = makePhiBody(619);
  auto PhiReplacement = makePhiBody(631);
  kernelPhiSlot<<<1, 1>>>(PhiInitial, PhiReplacement, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 607
// CHECK: selected slot 607
// CHECK: [LambdaSpec] Replacing slot 0 with i32 617
// CHECK: nested return 617
// CHECK: [LambdaSpec] Replacing slot 0 with i32 631
// CHECK: phi slot 631
// clang-format on
