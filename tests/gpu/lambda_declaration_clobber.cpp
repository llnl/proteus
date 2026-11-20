// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_declaration_clobber.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

extern "C" __device__ void overwriteSlotFromDeclaration(void **Slot,
                                                        void *Replacement);

// The definition is in another translation unit, so the Proteus pass sees
// only a declaration at this call. It must treat the MemorySSA-selected call
// as unresolved instead of continuing to Initial's stale store.
template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeAfterDeclarationClobber(F *Initial, F *Replacement) {
  F *Slot = Initial;
  overwriteSlotFromDeclaration(reinterpret_cast<void **>(&Slot), Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDeclarationClobber(F Initial, F Replacement) {
  invokeAfterDeclarationClobber(&Initial, &Replacement);
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("declaration clobber %d\n", X);
      });
}

int main() {
  auto Initial = makeBody(751);
  auto Replacement = makeBody(757);
  kernelDeclarationClobber<<<1, 1>>>(Initial, Replacement);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelDeclarationClobber
// CHECK-NOT: [LambdaSpec]
// CHECK: declaration clobber 757
// clang-format on
