// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_memory_intrinsic_shapes.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct MemoryBox {
  std::uint64_t Prefix[2];
  F Body;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
clearPrefix(MemoryBox<F> *Box) {
  __builtin_memset(Box, 0, sizeof(std::uint64_t));
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelNonoverlapMemset(F Body) {
  MemoryBox<F> Box{{1, 2}, Body};
  clearPrefix(&Box);
  Box.Body();
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
copyDynamic(MemoryBox<F> *Destination, const MemoryBox<F> *Source,
            std::size_t Bytes) {
  __builtin_memcpy(Destination, Source, Bytes);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelDynamicMemcpy(F DestinationBody, F SourceBody, std::size_t Bytes) {
  MemoryBox<F> Destination{{3, 4}, DestinationBody};
  MemoryBox<F> Source{{5, 6}, SourceBody};
  copyDynamic(&Destination, &Source, Bytes);
  Destination.Body();
}

template <typename F> struct MoveSource {
  std::uint64_t Prefix;
  F Body;
};

template <typename F> struct MoveDestination {
  std::uint64_t Prefix[3];
  F Body;
};

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
moveBody(MoveDestination<F> *Destination, const MoveSource<F> *Source) {
  __builtin_memmove(&Destination->Body, &Source->Body, sizeof(F));
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelOffsetMemmove(F DestinationBody, F SourceBody) {
  MoveDestination<F> Destination{{7, 8, 9}, DestinationBody};
  MoveSource<F> Source{10, SourceBody};
  moveBody(&Destination, &Source);
  Destination.Body();
}

static auto makeMemsetBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("nonoverlap memset %d\n", X);
      });
}

static auto makeDynamicBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("dynamic memcpy %d\n", X);
      });
}

static auto makeMoveBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("offset memmove %d\n", X);
      });
}

int main() {
  kernelNonoverlapMemset<<<1, 1>>>(makeMemsetBody(197));
  gpuErrCheck(gpuDeviceSynchronize());

  auto DynamicDestination = makeDynamicBody(199);
  auto DynamicSource = makeDynamicBody(211);
  kernelDynamicMemcpy<<<1, 1>>>(DynamicDestination, DynamicSource,
                                sizeof(std::uint64_t));
  gpuErrCheck(gpuDeviceSynchronize());

  auto MoveDestinationBody = makeMoveBody(223);
  auto MoveSourceBody = makeMoveBody(227);
  kernelOffsetMemmove<<<1, 1>>>(MoveDestinationBody, MoveSourceBody);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 197
// CHECK: nonoverlap memset 197
// CHECK: [KernelConfig] ID:{{.*}}kernelDynamicMemcpy
// CHECK-NOT: [LambdaSpec]
// CHECK: dynamic memcpy 199
// The helper's source and destination formals are not noalias. When tracing
// the source, MemorySSA may therefore select the memmove as a possible write,
// and the analysis must not walk past it to expose a stale initializer.
// CHECK: [KernelConfig] ID:{{.*}}kernelOffsetMemmove
// CHECK-NOT: [LambdaSpec]
// CHECK: offset memmove 227
// clang-format on
