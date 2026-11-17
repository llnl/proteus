// clang-format off
// RUN: env PROTEUS_CACHE_DIR="%t.cache" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/collapse_cases.%ext | %FILECHECK %s
// RUN: env PROTEUS_CACHE_DIR="%t.cache" PROTEUS_TRACE_OUTPUT=cache-stats %build/collapse_cases.%ext | %FILECHECK %s --check-prefixes=RESULT,STORED
// RUN: env PROTEUS_CACHE_DIR="%t.async" PROTEUS_ASYNC_COMPILATION=1 PROTEUS_ASYNC_TEST_BLOCKING=1 %build/collapse_cases.%ext | %FILECHECK %s --check-prefix=RESULT
// clang-format on

#include "collapse_cases_source.h"
#include "gpu_common.h"
#include <cstdio>
#include <cstdlib>
#include <proteus/JitInterface.h>
#include <vector>

struct UnusedTag {
  using value_type = int;
};
struct Entry {
  Color color;
};
struct Group {
  Entry *entries;
};
struct Context {
  Group *groups;
};
struct IndexPair {
  int group, entry;
};
__device__ Context *device_state;

__global__ __attribute__((annotate("jit"))) void unused_kernel(int *Out) {
  *Out = proteus::jit_variants<UnusedTag, 2>(*Out);
}

// The tag is inside a callee reached through an index pair and device state.
__device__ __attribute__((noinline)) int lookup_color(IndexPair Loc, int X) {
  auto &D = device_state->groups[Loc.group].entries[Loc.entry];
  switch (proteus::jit_variants<ColorTag, 2>(D.color)) {
  case Color::Blue:
    return X + 11;
  case Color::Green:
    return X * 3;
  case Color::Red:
    return X - 17;
  case Color::Yellow:
    return X * X;
  case Color::Purple:
    return X / 2;
  case Color::Orange:
    return -X;
  default:
    return 999;
  }
}

__device__ __attribute__((noinline)) int forward_color(IndexPair Loc, int X) {
  return lookup_color(Loc, X);
}

template <class F>
__global__ __attribute__((annotate("jit"))) void apply_colors(F Fn) {
  Fn(threadIdx.x);
}

static int reference(Color C, int X) {
  switch (C) {
  case Color::Blue:
    return X + 11;
  case Color::Green:
    return X * 3;
  case Color::Red:
    return X - 17;
  case Color::Yellow:
    return X * X;
  case Color::Purple:
    return X / 2;
  case Color::Orange:
    return -X;
  default:
    return 999;
  }
}

int main() {
  constexpr int N = 64;
  Context *Ctx;
  IndexPair *Locations;
  int *Out;
  gpuErrCheck(gpuMallocManaged(&Ctx, sizeof(Context)));
  gpuErrCheck(gpuMallocManaged(&Ctx->groups, 2 * sizeof(Group)));
  for (int R = 0; R < 2; ++R)
    gpuErrCheck(gpuMallocManaged(&Ctx->groups[R].entries, 2 * sizeof(Entry)));
  gpuErrCheck(gpuMallocManaged(&Locations, N * sizeof(IndexPair)));
  gpuErrCheck(gpuMallocManaged(&Out, N * sizeof(int)));
  void *Global;
  gpuErrCheck(gpuGetSymbolAddress(&Global, device_state));
  gpuErrCheck(gpuMemcpy(Global, &Ctx, sizeof(Ctx), gpuMemcpyHostToDevice));
  for (int I = 0; I < N; ++I)
    Locations[I] = {I % 2, (I / 2) % 2};

  int Collections = 0;
  std::vector<Color> Allowed;
  auto Register = [&] { registerColorValues(Allowed, Collections); };
  Register();
  // Only reachable tags may be collected.
  proteus::register_variant_source<UnusedTag>([](auto) { std::abort(); });
  auto Fn = [=] __device__(int I) { Out[I] = forward_color(Locations[I], I); };
  auto Run = [&](const char *Name, std::vector<Color> Set, Color A, Color B) {
    Allowed = std::move(Set);
    for (int R = 0; R < 2; ++R)
      for (int D = 0; D < 2; ++D)
        Ctx->groups[R].entries[D].color = R ? B : A;
    apply_colors<<<1, N>>>(Fn);
    gpuErrCheck(gpuDeviceSynchronize());
    for (int I = 0; I < N; ++I) {
      int Expected = reference(I % 2 ? B : A, I);
      if (Out[I] != Expected) {
        std::fprintf(stderr, "%s lane %d: got %d expected %d\n", Name, I,
                     Out[I], Expected);
        std::abort();
      }
    }
    std::printf("PASS %s\n", Name);
  };
  using C = Color;
  Run("two colors", {C::Blue, C::Red}, C::Blue, C::Red);
  Run("reordered duplicates", {C::Red, C::Blue, C::Red}, C::Blue, C::Red);
  Run("new pair", {C::Green, C::Orange}, C::Green, C::Orange);
  Run("singleton", {C::Yellow}, C::Yellow, C::Yellow);
  Run("over budget", {C::Blue, C::Green, C::Red}, C::Blue, C::Red);
  Run("empty source", {}, C::Purple, C::Orange);
  Run("default", {C::Blue, static_cast<C>(99)}, C::Blue, static_cast<C>(99));
  proteus::unregister_variant_source<ColorTag>();
  Run("missing source", {}, C::Purple, C::Orange);
  Register();
  Run("replacement", {C::Red}, C::Red, C::Red);
#ifdef PROTEUS_TEST_AOT
  if (Collections != 0)
    std::abort();
#else
  if (Collections != 8)
    std::abort();
#endif
  proteus::unregister_variant_source<ColorTag>();
  proteus::unregister_variant_source<UnusedTag>();
  for (int R = 0; R < 2; ++R)
    gpuErrCheck(gpuFree(Ctx->groups[R].entries));
  gpuErrCheck(gpuFree(Ctx->groups));
  gpuErrCheck(gpuFree(Ctx));
  gpuErrCheck(gpuFree(Locations));
  gpuErrCheck(gpuFree(Out));
}

// CHECK: [CollapseCasesSpec] Switch cases 6 -> 2
// CHECK: [CollapseCasesSpec] {{.*}} Values 2 Budget 2 specialized
// CHECK: PASS two colors
// CHECK: PASS reordered duplicates
// CHECK: [CollapseCasesSpec] {{.*}} Values 2 Budget 2 specialized
// CHECK: PASS new pair
// CHECK: [CollapseCasesSpec] {{.*}} Values 1 Budget 2 specialized
// CHECK: PASS singleton
// CHECK: [CollapseCasesSpec] {{.*}} Values 0 Budget 2 generic
// CHECK: PASS over budget
// CHECK: PASS empty source
// CHECK: PASS default
// CHECK: PASS missing source
// CHECK: PASS replacement
// CHECK: MemoryCache rank 0 hits 3 accesses 9
// CHECK: StorageCache rank 0 hits 0 accesses 6
// RESULT: PASS two colors
// RESULT: PASS reordered duplicates
// RESULT: PASS new pair
// RESULT: PASS singleton
// RESULT: PASS over budget
// RESULT: PASS empty source
// RESULT: PASS default
// RESULT: PASS missing source
// RESULT: PASS replacement
// STORED: StorageCache rank 0 hits 6 accesses 6
