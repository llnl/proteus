// clang-format off
// RUN: PROTEUS_DISABLE=1 %build/lambda_readonly_plan.%ext | %FILECHECK %s --check-prefix=RESULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_readonly_plan.%ext | %FILECHECK %s --check-prefixes=RESULT,SPEC
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <new>
#include <type_traits>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F> struct WritePlan {
  F *Destination;
  const F *Source;
};

// Keep optimization enabled: Clang can infer readonly for Plan, although the
// pointer loaded from Plan->Destination is used to overwrite the closure.
template <typename F>
__device__ __attribute__((noinline)) static void
execute(const WritePlan<F> *Plan) {
  __builtin_memcpy(Plan->Destination, Plan->Source, sizeof(F));
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelReadonlyPlan(F &Body, const WritePlan<F> *__restrict Plan, int *Result) {
  // restrict separates the plan's own storage from Body, not the destination
  // pointer loaded from the plan. That destination still aliases Body.
  execute(Plan);
  *Result = Body();
}

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ { return X; });
}

int main() {
  auto Initial = makeBody(1401);
  auto Replacement = makeBody(1409);
  using F = decltype(Initial);
  static_assert(std::is_trivially_copyable_v<F>);
  F *Body, *Source;
  WritePlan<F> *Plan;
  int *Result;
  gpuErrCheck(gpuMallocManaged(&Body, sizeof(F)));
  gpuErrCheck(gpuMallocManaged(&Source, sizeof(F)));
  gpuErrCheck(gpuMallocManaged(&Plan, sizeof(*Plan)));
  gpuErrCheck(gpuMallocManaged(&Result, sizeof(*Result)));
  new (Body) F(Initial);
  new (Source) F(Replacement);
  // The container already holds the alias when the kernel begins. Its
  // initializing stores are absent from the device-side use-def graph.
  new (Plan) WritePlan<F>{Body, Source};
  kernelReadonlyPlan<<<1, 1>>>(*Body, Plan, Result);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("readonly plan %d\n", *Result);
  gpuErrCheck(gpuFree(Result));
  gpuErrCheck(gpuFree(Plan));
  gpuErrCheck(gpuFree(Source));
  gpuErrCheck(gpuFree(Body));
}

// A correct replacement specialization or a conservative decline is allowed.
// clang-format off
// SPEC-NOT: [LambdaSpec] Replacing slot 0 with i32 1401
// RESULT: readonly plan 1409
// clang-format on
