// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unsupported_memory_shapes.%ext | %FILECHECK %s
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

// Two mutating calls require composing separate interprocedural writes.
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

// Type-erased byte arithmetic writes an individual closure slot.
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

// The opaque call is a MemoryDef. A later escape prevents AA from proving
// that it cannot modify the local slot.
__device__ __attribute__((noinline, optnone)) static void
opaqueUnrelatedWrite() {
  asm volatile("" : : : "memory");
}

__device__ static void *EscapedSlot;

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
escapePointerSlot(F **Slot) {
  EscapedSlot = Slot;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossUncapturedCall(F *Initial, F *Replacement) {
  F *Slot = Initial;
  Slot = Replacement;
  opaqueUnrelatedWrite();
  (*Slot)();
  escapePointerSlot(&Slot);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelUncapturedCallClobber(F Initial, F Replacement) {
  invokeAcrossUncapturedCall(&Initial, &Replacement);
}

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makeTwoCallBody, "two call clobbers")
MAKE_BODY(makeByteOffsetBody, "byte offset call")
MAKE_BODY(makeUncapturedBody, "uncaptured call clobber")

int main() {
  auto TwoCallInitial = makeTwoCallBody(443);
  auto TwoCallFirst = makeTwoCallBody(449);
  auto TwoCallFinal = makeTwoCallBody(457);
  kernelTwoCallClobbers<<<1, 1>>>(TwoCallInitial, TwoCallFirst, TwoCallFinal);
  gpuErrCheck(gpuDeviceSynchronize());

  auto ByteOffsetInitial = makeByteOffsetBody(503);
  auto ByteOffsetReplacement = makeByteOffsetBody(509);
  auto ByteOffsetNoise = makeByteOffsetBody(521);
  kernelByteOffsetCall<<<1, 1>>>(ByteOffsetInitial, ByteOffsetReplacement,
                                 ByteOffsetNoise);
  gpuErrCheck(gpuDeviceSynchronize());

  auto UncapturedInitial = makeUncapturedBody(907);
  auto UncapturedReplacement = makeUncapturedBody(911);
  kernelUncapturedCallClobber<<<1, 1>>>(UncapturedInitial,
                                        UncapturedReplacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelTwoCallClobbers
// CHECK-NOT: [LambdaSpec]
// CHECK: two call clobbers 457
// CHECK: [KernelConfig] ID:{{.*}}kernelByteOffsetCall
// CHECK-NOT: [LambdaSpec]
// CHECK: byte offset call 509
// CHECK: [KernelConfig] ID:{{.*}}kernelUncapturedCallClobber
// CHECK-NOT: [LambdaSpec]
// CHECK: uncaptured call clobber 911
// clang-format on
