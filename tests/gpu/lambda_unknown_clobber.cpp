// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unknown_clobber.%ext 0 | %FILECHECK --check-prefix=AGGREGATE %s
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unknown_clobber.%ext 1 | %FILECHECK --check-prefix=OPAQUE %s
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unknown_clobber.%ext 2 | %FILECHECK --check-prefix=INDIRECT %s
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unknown_clobber.%ext 3 | %FILECHECK --check-prefix=ATOMIC %s
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_unknown_clobber.%ext 4 | %FILECHECK --check-prefix=CMPXCHG %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct HiddenSlot {
  F **Slot;
  F *Replacement;
  void *Padding[4];
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSlotThroughAggregate(HiddenSlot<F> *Hidden) {
  *Hidden->Slot = Hidden->Replacement;
}

// The call can overwrite Slot, but the pointer reaches the callee as an
// field of a separate aggregate. Failure to map the MemorySSA-selected call to
// the slot through its formal argument must not make the analysis continue to
// Initial's stale store.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterAggregateHiddenClobber(F *Initial, F *Replacement) {
  F *Slot = Initial;
  HiddenSlot<F> Hidden{
      &Slot, Replacement, {nullptr, nullptr, nullptr, nullptr}};
  overwriteSlotThroughAggregate(&Hidden);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAggregateHiddenClobber(F Initial, F Replacement) {
  invokeAfterAggregateHiddenClobber(&Initial, &Replacement);
}

// A side-effecting inline-assembly call is opaque to the interprocedural
// resolver. It may modify memory, so the analysis must decline rather than
// recover the store preceding the call.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossOpaqueCall(F *Initial) {
  F *Slot = Initial;
  asm volatile("" : : "r"(&Slot) : "memory");
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelOpaqueCallClobber(F Initial) {
  invokeAcrossOpaqueCall(&Initial);
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteIndirectly(F **Slot, F *Replacement) {
  *Slot = Replacement;
}

// An indirect call has no Function body that the resolver can summarize. The
// potentially clobbering branch therefore makes the reaching value unknown.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAcrossIndirectCall(F *Initial, F *Replacement, bool DoOverwrite) {
  F *Slot = Initial;
  if (DoOverwrite) {
    using OverwriteFn = void (*)(F **, F *);
    OverwriteFn volatile Overwrite = &overwriteIndirectly<F>;
    Overwrite(&Slot, Replacement);
  }
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelIndirectCallClobber(F Initial, F Replacement, bool DoOverwrite) {
  invokeAcrossIndirectCall(&Initial, &Replacement, DoOverwrite);
}

// atomicrmw is a MemoryDef that the pointer provenance resolver deliberately
// does not interpret. It must block specialization instead of exposing the
// older Initial store.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterAtomicExchange(F *Initial, F *Replacement, bool DoExchange) {
  F *Slot = Initial;
  if (DoExchange)
    __atomic_exchange_n(&Slot, Replacement, __ATOMIC_RELAXED);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelAtomicClobber(F Initial, F Replacement, bool DoExchange) {
  invokeAfterAtomicExchange(&Initial, &Replacement, DoExchange);
}

// cmpxchg is another unsupported MemoryDef. As with atomicrmw, either branch
// through this operation makes the reaching pointer value unknown.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterCompareExchange(F *Initial, F *Replacement, bool DoExchange) {
  F *Slot = Initial;
  if (DoExchange) {
    F *Expected = Initial;
    __atomic_compare_exchange_n(&Slot, &Expected, Replacement, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelCompareExchangeClobber(F Initial, F Replacement, bool DoExchange) {
  invokeAfterCompareExchange(&Initial, &Replacement, DoExchange);
}

#define MAKE_BODY(Name, Message)                                               \
  static auto Name(int Value) {                                                \
    return proteus::register_lambda(                                           \
        [X = proteus::jit_variable(Value)] __host__ __device__ {               \
          printf(Message " %d\n", X);                                          \
        });                                                                    \
  }

MAKE_BODY(makeAggregateHiddenBody, "aggregate-hidden clobber")
MAKE_BODY(makeOpaqueCallBody, "opaque call clobber")
MAKE_BODY(makeIndirectCallBody, "indirect call clobber")
MAKE_BODY(makeAtomicBody, "atomic clobber")
MAKE_BODY(makeCompareExchangeBody, "cmpxchg clobber")

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  if (argv[1][0] == '0') {
    auto Initial = makeAggregateHiddenBody(701);
    auto Replacement = makeAggregateHiddenBody(709);
    kernelAggregateHiddenClobber<<<1, 1>>>(Initial, Replacement);
  } else if (argv[1][0] == '1') {
    auto Initial = makeOpaqueCallBody(719);
    kernelOpaqueCallClobber<<<1, 1>>>(Initial);
  } else if (argv[1][0] == '2') {
    auto Initial = makeIndirectCallBody(723);
    auto Replacement = makeIndirectCallBody(725);
    kernelIndirectCallClobber<<<1, 1>>>(Initial, Replacement, false);
  } else if (argv[1][0] == '3') {
    auto Initial = makeAtomicBody(727);
    auto Replacement = makeAtomicBody(733);
    kernelAtomicClobber<<<1, 1>>>(Initial, Replacement, false);
  } else {
    auto Initial = makeCompareExchangeBody(739);
    auto Replacement = makeCompareExchangeBody(743);
    kernelCompareExchangeClobber<<<1, 1>>>(Initial, Replacement, false);
  }
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// AGGREGATE: [KernelConfig] ID:{{.*}}kernelAggregateHiddenClobber
// AGGREGATE-NOT: [LambdaSpec]
// AGGREGATE: aggregate-hidden clobber 709
// OPAQUE: [KernelConfig] ID:{{.*}}kernelOpaqueCallClobber
// OPAQUE-NOT: [LambdaSpec]
// OPAQUE: opaque call clobber 719
// INDIRECT: [KernelConfig] ID:{{.*}}kernelIndirectCallClobber
// INDIRECT-NOT: [LambdaSpec]
// INDIRECT: indirect call clobber 723
// ATOMIC: [KernelConfig] ID:{{.*}}kernelAtomicClobber
// ATOMIC-NOT: [LambdaSpec]
// ATOMIC: atomic clobber 727
// CMPXCHG: [KernelConfig] ID:{{.*}}kernelCompareExchangeClobber
// CMPXCHG-NOT: [LambdaSpec]
// CMPXCHG: cmpxchg clobber 739
// clang-format on
