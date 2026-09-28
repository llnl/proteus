// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/lambda_cast_capture | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

template <typename F> void run(F &&Func) { proteus::register_lambda(Func)(); }

int main() {
  int I = -5;
  unsigned U = 0x80000000u;
  float F = 1.5f;
  double D = 2.7;

  auto IntToLong =
      [=, X = (long)proteus::jit_variable(I)]()
          __attribute__((annotate("jit"))) { printf("IntToLong %ld\n", X); };
  run(IntToLong);

  auto UnsignedToULong = [=, X = (unsigned long)proteus::jit_variable(U)]()
                             __attribute__((annotate("jit"))) {
                               printf("UnsignedToULong %lu\n", X);
                             };
  run(UnsignedToULong);

  auto FloatToDouble =
      [=, X = (double)proteus::jit_variable(F)]()
          __attribute__((annotate("jit"))) { printf("FloatToDouble %f\n", X); };
  run(FloatToDouble);

  auto IntToDouble =
      [=, X = (double)proteus::jit_variable(I)]()
          __attribute__((annotate("jit"))) { printf("IntToDouble %f\n", X); };
  run(IntToDouble);

  auto DoubleToInt =
      [=, X = (int)proteus::jit_variable(D)]()
          __attribute__((annotate("jit"))) { printf("DoubleToInt %d\n", X); };
  run(DoubleToInt);

  return 0;
}

// CHECK: IntToLong -5
// CHECK: UnsignedToULong 2147483648
// CHECK: FloatToDouble 1.500000
// CHECK: IntToDouble -5.000000
// CHECK: DoubleToInt 2
