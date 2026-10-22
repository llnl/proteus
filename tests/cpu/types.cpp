// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/types | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST,%LLVM_FP80_CHECK_PREFIX
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/types | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdlib>

#include <iostream>

#include <proteus/JitInterface.h>

template <typename T> __attribute__((annotate("jit", 1))) void test(T Arg) {
  volatile T Local;
  Local = Arg;
  std::cout << "Arg " << Arg << "\n";
}

int main() {
  test(1);
  test(2l);
  test(3u);
  test(4ul);
  test(5ll);
  test(6ull);
  test(7.0f);
  test(8.0);
  test(9.0l);
  test(true);
  test('a');
  test((unsigned char)'a');
  int *Ptr = (int *)0x123;
  test(Ptr);

  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIiEvT_ ArgNo 0 with value i32 1
// CHECK: Arg 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIlEvT_ ArgNo 0 with value i64 2
// CHECK: Arg 2
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIjEvT_ ArgNo 0 with value i32 3
// CHECK: Arg 3
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testImEvT_ ArgNo 0 with value i64 4
// CHECK: Arg 4
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIxEvT_ ArgNo 0 with value i64 5
// CHECK: Arg 5
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIyEvT_ ArgNo 0 with value i64 6
// CHECK: Arg 6
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIfEvT_ ArgNo 0 with value float 7.000000e+00
// CHECK: Arg 7
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIdEvT_ ArgNo 0 with value double 8.000000e+00
// CHECK: Arg 8
// CHECK-LLVM22: [ArgSpec] Replaced Function _Z4testI{{e|g}}EvT_ ArgNo 0 with value {{x86_fp80 0xK40029000000000000000|ppc_fp128 0xM40220000000000000000000000000000|double 9.000000e\+00}}
// CHECK-LLVM23: [ArgSpec] Replaced Function _Z4testI{{e|g}}EvT_ ArgNo 0 with value {{x86_fp80 9\.000000e\+00|ppc_fp128 0xM40220000000000000000000000000000|double 9.000000e\+00}}
// CHECK: Arg 9
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIbEvT_ ArgNo 0 with value i1 true
// CHECK: Arg 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIcEvT_ ArgNo 0 with value i8 97
// CHECK: Arg a
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIhEvT_ ArgNo 0 with value i8 97
// CHECK: Arg a
// CHECK-FIRST: [ArgSpec] Replaced Function _Z4testIPiEvT_ ArgNo 0 with value ptr inttoptr (i64 291 to ptr)
// CHECK: Arg 0x123
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 13
// CHECK-COUNT-12: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 13
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 13 accesses 13
