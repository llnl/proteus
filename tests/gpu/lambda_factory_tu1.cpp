// Shared-library TU 1 for cross-TU lambda factory testing.

#include "gpu_common.h"
#include "lambda_factory_header.h"

namespace {

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

} // namespace

extern "C" void run_lambda_factory_tu1() {
  int Ten = 10;
  int Eleven = 11;
  int Twelve = 12;

  auto LambdaA = declareLambda(Ten, Eleven);
  auto LambdaB = declareLambda(Twelve, Ten);

  run(LambdaA);
  run(LambdaB);
}
