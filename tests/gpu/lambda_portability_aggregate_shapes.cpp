// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_portability_aggregate_shapes.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct PointerSlots {
  F *Noise;
  F *Invoked;
};

template <typename F> struct LambdaPair {
  F First;
  F Second;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

// A portability helper receives an aggregate and replaces its pointer fields.
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

// Returning a slot address through a helper must preserve its provenance.
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

// The returned alias points to a nonzero-offset aggregate field.
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

// A pointer-returning portability wrapper may itself call another wrapper.
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

// The same helper is called for distinct aggregate fields. Its formal argument
// and return value must remain specific to each call site.
__device__ __attribute__((noinline, optnone)) static void *
returnFieldAlias(void *Field) {
  return Field;
}

template <typename Pair>
__device__ __attribute__((noinline, optnone)) static Pair *
returnPairAlias(Pair *PairValue) {
  return static_cast<Pair *>(returnFieldAlias(PairValue));
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeSecondAfterFirstOverwrite(F *First, F *Replacement, F *Second,
                                bool UseDirectAlias) {
  LambdaPair<F> Storage{*First, *Second};
  auto *SecondAlias = reinterpret_cast<F *>(returnFieldAlias(&Storage.Second));
  auto *BaseAlias = returnPairAlias(&Storage);
  __builtin_memcpy(&BaseAlias->First, Replacement, sizeof(F));
  F *Selected = UseDirectAlias ? SecondAlias : &BaseAlias->Second;
  (*Selected)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCallContext(F First, F Replacement, F Second, bool UseDirectAlias) {
  invokeSecondAfterFirstOverwrite(&First, &Replacement, &Second,
                                  UseDirectAlias);
}

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makeAggregateBody, "aggregate field call")
MAKE_BODY(makeReturnedAliasBody, "returned alias")
MAKE_BODY(makeReturnedFieldBody, "returned field alias")
MAKE_BODY(makeNestedReturnBody, "nested return")
MAKE_BODY(makeCallContextBody, "call-context")

int main() {
  auto AggregateInitial = makeAggregateBody(479);
  auto AggregateReplacement = makeAggregateBody(487);
  auto AggregateNoiseInitial = makeAggregateBody(491);
  auto AggregateNoiseReplacement = makeAggregateBody(499);
  kernelAggregateFieldCall<<<1, 1>>>(AggregateInitial, AggregateReplacement,
                                     AggregateNoiseInitial,
                                     AggregateNoiseReplacement);
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

  auto NestedInitial = makeNestedReturnBody(613);
  auto NestedReplacement = makeNestedReturnBody(617);
  kernelNestedReturn<<<1, 1>>>(NestedInitial, NestedReplacement);
  gpuErrCheck(gpuDeviceSynchronize());

  auto First = makeCallContextBody(811);
  auto Replacement = makeCallContextBody(821);
  auto Second = makeCallContextBody(823);
  kernelCallContext<<<1, 1>>>(First, Replacement, Second, true);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 487
// CHECK: aggregate field call 487
// CHECK: [LambdaSpec] Replacing slot 0 with i32 557
// CHECK: returned alias 557
// CHECK: [LambdaSpec] Replacing slot 0 with i32 569
// CHECK: returned field alias 569
// CHECK: [LambdaSpec] Replacing slot 0 with i32 617
// CHECK: nested return 617
// CHECK: [LambdaSpec] Replacing slot 0 with i32 823
// CHECK-NOT: [LambdaSpec] Replacing slot 0 with i32 {{811|821}}
// CHECK: call-context 823
// clang-format on
