// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_clobber_adversarial.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
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

// Two writing calls are both relevant uses. MemorySSA, rather than collection
// order, must select the second call.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterTwoCalls(F *Initial, F *First, F *Final) {
  F *volatile Slot = Initial;
  overwriteSlot(&Slot, First);
  overwriteSlot(&Slot, Final);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelTwoCallClobbers(F Initial, F First, F Final) {
  invokeAfterTwoCalls(&Initial, &First, &Final);
}

// The final MemoryDef is a call, but it does not modify Slot. The clobber walk
// must continue to the preceding writing call.
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

template <typename F> struct PointerSlots {
  F *Noise;
  F *Invoked;
};

// The call receives the enclosing aggregate while writing only its second
// pointer field. This exercises interprocedural target-offset translation.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteAggregateField(PointerSlots<F> *Slots, F *Replacement,
                        F *NoiseReplacement) {
  Slots->Invoked = Replacement;
  Slots->Noise = NoiseReplacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterAggregateFieldCall(F *Initial, F *Replacement, F *NoiseInitial,
                              F *NoiseReplacement) {
  PointerSlots<F> Slots{NoiseInitial, Initial};
  overwriteAggregateField(&Slots, Replacement, NoiseReplacement);
  (*Slots.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAggregateFieldCall(F Initial, F Replacement, F NoiseInitial,
                         F NoiseReplacement) {
  invokeAfterAggregateFieldCall(&Initial, &Replacement, &NoiseInitial,
                                &NoiseReplacement);
}

// Hide the field relationship behind byte arithmetic in the callee. Alias
// analysis still has enough information to identify the second field.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteByteOffsetField(void *Storage, F *Replacement) {
  auto *Bytes = reinterpret_cast<unsigned char *>(Storage);
  auto *Slot = reinterpret_cast<F *volatile *>(Bytes + sizeof(F *));
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterByteOffsetCall(F *Initial, F *Replacement, F *Noise) {
  PointerSlots<F> Slots{Noise, Initial};
  overwriteByteOffsetField<F>(&Slots, Replacement);
  (*Slots.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelByteOffsetCall(F Initial, F Replacement, F Noise) {
  invokeAfterByteOffsetCall(&Initial, &Replacement, &Noise);
}

// A write after the lambda invocation is a collected use, but cannot be the
// reaching clobber at the invocation boundary.
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

// Returning the slot address creates a forward-use edge through a CallBase
// result. The use collector must continue through that result before asking
// MemorySSA which write reaches the invocation.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnSlotAddress(F *volatile *Slot) {
  return Slot;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughReturnedAlias(F *Initial, F *Replacement) {
  F *volatile Slot = Initial;
  F *volatile *Alias = returnSlotAddress(&Slot);
  overwriteSlot(Alias, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelReturnedAlias(F Initial, F Replacement) {
  invokeThroughReturnedAlias(&Initial, &Replacement);
}

// As above, but the returned alias points to a nonzero-offset field. Both the
// use collector and clobber resolver must preserve that offset across the
// return edge.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnInvokedField(PointerSlots<F> *Slots) {
  return &Slots->Invoked;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughReturnedField(F *Initial, F *Replacement, F *Noise) {
  PointerSlots<F> Slots{Noise, Initial};
  F *volatile *Alias = returnInvokedField(&Slots);
  overwriteSlot(Alias, Replacement);
  (*Slots.Invoked)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelReturnedFieldAlias(F Initial, F Replacement, F Noise) {
  invokeThroughReturnedField(&Initial, &Replacement, &Noise);
}

// Different return paths expose different fields. Even though this execution
// returns Invoked, the runtime flag prevents the analysis from choosing one
// return offset statically.
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

MAKE_BODY(makeTwoCallBody, "two call clobbers")
MAKE_BODY(makeReadOnlyBody, "trailing read-only call")
MAKE_BODY(makeAggregateBody, "aggregate field call")
MAKE_BODY(makeByteOffsetBody, "byte offset call")
MAKE_BODY(makePostCallBody, "call after invoke")
MAKE_BODY(makeReturnedAliasBody, "returned alias")
MAKE_BODY(makeReturnedFieldBody, "returned field alias")
MAKE_BODY(makeAmbiguousReturnedFieldBody, "ambiguous returned field")

int main() {
  auto TwoCallInitial = makeTwoCallBody(443);
  auto TwoCallFirst = makeTwoCallBody(449);
  auto TwoCallFinal = makeTwoCallBody(457);
  kernelTwoCallClobbers<<<1, 1>>>(TwoCallInitial, TwoCallFirst, TwoCallFinal);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ReadOnlyInitial = makeReadOnlyBody(461);
  auto ReadOnlyReplacement = makeReadOnlyBody(463);
  auto ReadOnlyNoise = makeReadOnlyBody(467);
  kernelTrailingReadOnlyCall<<<1, 1>>>(ReadOnlyInitial, ReadOnlyReplacement,
                                       ReadOnlyNoise);
  gpuErrCheck(gpuDeviceSynchronize());

  auto AggregateInitial = makeAggregateBody(479);
  auto AggregateReplacement = makeAggregateBody(487);
  auto AggregateNoiseInitial = makeAggregateBody(491);
  auto AggregateNoiseReplacement = makeAggregateBody(499);
  kernelAggregateFieldCall<<<1, 1>>>(AggregateInitial, AggregateReplacement,
                                     AggregateNoiseInitial,
                                     AggregateNoiseReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ByteOffsetInitial = makeByteOffsetBody(503);
  auto ByteOffsetReplacement = makeByteOffsetBody(509);
  auto ByteOffsetNoise = makeByteOffsetBody(521);
  kernelByteOffsetCall<<<1, 1>>>(ByteOffsetInitial, ByteOffsetReplacement,
                                 ByteOffsetNoise);
  gpuErrCheck(gpuDeviceSynchronize());

  auto PostCallInitial = makePostCallBody(523);
  auto PostCallLater = makePostCallBody(541);
  kernelCallAfterInvoke<<<1, 1>>>(PostCallInitial, PostCallLater);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ReturnedAliasInitial = makeReturnedAliasBody(547);
  auto ReturnedAliasReplacement = makeReturnedAliasBody(557);
  kernelReturnedAlias<<<1, 1>>>(ReturnedAliasInitial, ReturnedAliasReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ReturnedFieldInitial = makeReturnedFieldBody(563);
  auto ReturnedFieldReplacement = makeReturnedFieldBody(569);
  auto ReturnedFieldNoise = makeReturnedFieldBody(571);
  kernelReturnedFieldAlias<<<1, 1>>>(
      ReturnedFieldInitial, ReturnedFieldReplacement, ReturnedFieldNoise);
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
// CHECK: [LambdaSpec] Replacing slot 0 with i32 457
// CHECK: two call clobbers 457
// CHECK: [LambdaSpec] Replacing slot 0 with i32 463
// CHECK: trailing read-only call 463
// CHECK: [LambdaSpec] Replacing slot 0 with i32 487
// CHECK: aggregate field call 487
// CHECK: [LambdaSpec] Replacing slot 0 with i32 509
// CHECK: byte offset call 509
// CHECK: [LambdaSpec] Replacing slot 0 with i32 523
// CHECK: call after invoke 523
// CHECK: [LambdaSpec] Replacing slot 0 with i32 557
// CHECK: returned alias 557
// CHECK: [LambdaSpec] Replacing slot 0 with i32 569
// CHECK: returned field alias 569
// CHECK: [KernelConfig] ID:{{.*}}kernelAmbiguousReturnedField
// CHECK-NOT: [LambdaSpec]
// CHECK: ambiguous returned field 587
// clang-format on
