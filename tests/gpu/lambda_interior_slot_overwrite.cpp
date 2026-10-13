// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_interior_slot_overwrite.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

// A foreign write lands on the functor's second registered slot while the
// first slot is re-stored from the authoritative source. The analysis anchors
// its clobber query at the narrow slot-0 store, so the slot-1 overwrite is
// invisible to it: it neither covers the tracked byte (candidate collection
// skips its GEP) nor aliases the 4-byte anchor. Concluding that the whole
// functor is authoritative from Initial is wrong -- slot 1 executes with
// Donor's value -- so specialization must decline.
template <typename F>
__device__ __attribute__((noinline)) static void
invokeAfterInteriorSlotOverwrite(F *Initial, F *Donor) {
  F Copy = *Initial;
  auto *CopyWords = reinterpret_cast<std::uint32_t *>(&Copy);
  auto *InitialWords = reinterpret_cast<const std::uint32_t *>(Initial);
  auto *DonorWords = reinterpret_cast<const std::uint32_t *>(Donor);
  CopyWords[0] = InitialWords[0];
  CopyWords[1] = DonorWords[1];
  Copy();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelInteriorSlotOverwrite(F Initial, F Donor) {
  invokeAfterInteriorSlotOverwrite(&Initial, &Donor);
}

static auto makeInteriorBody(int A, int B) {
  return proteus::register_lambda([X = proteus::jit_variable(A),
                                   Y = proteus::jit_variable(B)] __host__
                                                                __device__ {
    printf("interior slot overwrite %d %d\n", X, Y);
  });
}

int main() {
  auto Initial = makeInteriorBody(631, 757);
  auto Donor = makeInteriorBody(881, 999);
  kernelInteriorSlotOverwrite<<<1, 1>>>(Initial, Donor);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// The executed closure holds X from Initial but Y from Donor; no single
// kernel argument is authoritative for the functor's full extent, so the
// analysis must decline and the runtime must print Donor's Y.
// clang-format off
// CHECK: [KernelConfig] ID:{{.*}}kernelInteriorSlotOverwrite
// CHECK-NOT: [LambdaSpec]
// CHECK: interior slot overwrite 631 999
// clang-format on
