// Shared-library TU 2 for cross-TU lambda factory testing.

#include "gpu_common.h"
#include "lambda_factory_header.h"

namespace {

template <typename T> __global__ void kernel(T LB) { LB(); }

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(proteus::register_lambda(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

} // namespace

extern "C" void run_lambda_factory_tu2() {
  int Twenty = 20;
  int TwentyOne = 21;
  int TwentyTwo = 22;

  auto LambdaA = declareLambda(Twenty, TwentyOne);
  auto LambdaB = declareLambda(TwentyTwo, Twenty);

  run(LambdaA);
  run(LambdaB);
}
