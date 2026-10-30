// NOLINTBEGIN
#include "gpu_common.h"

namespace MockRajaInterface {
template <typename T> struct Privatizer {
  using value_type = T;
  using reference_type = value_type &;
  value_type priv;

  __host__ __device__ Privatizer(const T &o) : priv{o} {}

  __host__ __device__ reference_type get_priv() { return priv; }

  __host__ __device__ auto operator()(size_t I) { return priv(I); }
};

template <typename T>
__host__ __device__ auto threadPrivatize(const T &item) -> Privatizer<T> {
  return Privatizer<T>{item};
}

template <typename Lambda>
__device__ __attribute__((always_inline)) inline void
globalWrapperBody(Lambda lam, size_t N) {
  auto privatizer = threadPrivatize(lam);
  // auto &body = privatizer.get_priv();
  std::size_t idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < N) {
    privatizer(idx);
  }
}

template <typename Lambda>
__global__ __attribute__((annotate("jit", 2))) void globalWrapper(Lambda lam,
                                                                  size_t N) {
  globalWrapperBody(lam, N);
}

template <typename Lambda>
__global__ void globalWrapperUnannotated(Lambda lam, size_t N) {
  globalWrapperBody(lam, N);
}

template <bool RegisteredLambda = false, typename Lambda, typename... Args,
          bool DeviceLaunch = true>
void forall(size_t N, Lambda &&lam, Args &&...) {
  const std::size_t GridSize = (((N) + (256) - 1) / (256));
  if constexpr (DeviceLaunch) {
#if PROTEUS_ENABLE_HIP
    if constexpr (RegisteredLambda)
      hipLaunchKernelGGL((globalWrapperUnannotated<Lambda>), dim3(GridSize),
                         dim3(256), 0, 0, lam, N);
    else
      hipLaunchKernelGGL((globalWrapper<Lambda>), dim3(GridSize), dim3(256), 0,
                         0, lam, N);
#elif PROTEUS_ENABLE_CUDA
    const void *func;
    if constexpr (RegisteredLambda)
      func = reinterpret_cast<const void *>(&globalWrapperUnannotated<Lambda>);
    else
      func = reinterpret_cast<const void *>(&globalWrapper<Lambda>);
    void *AArgs[] = {&lam, &N};
    cudaLaunchKernel(func, dim3(GridSize), dim3(256), AArgs, 0, 0);
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
  } else {
    if constexpr (RegisteredLambda)
      globalWrapperUnannotated(lam, N);
    else
      globalWrapper(lam, N);
  }
}
} // namespace MockRajaInterface
// NOLINTEND
