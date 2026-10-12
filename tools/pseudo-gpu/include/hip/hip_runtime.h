#ifndef PROTEUS_PSEUDO_GPU_HIP_RUNTIME_H
#define PROTEUS_PSEUDO_GPU_HIP_RUNTIME_H

// Minimal HIP API surface for "pseudo-GPU" compilation on machines without
// ROCm (e.g. macOS dev boxes).  This supports compile-time verification of
// ProteusPass analyses by driving the real HIP device pipeline:
//
//   clang++ -x hip --offload-arch=gfx90a --offload-device-only \
//       -nogpulib -nogpuinc -I tools/pseudo-gpu/include ... \
//       -fpass-plugin=libProteusPass.dylib -S -emit-llvm
//
// The pass runs on the amdgcn device module exactly as in a production HIP
// build; PROTEUS_DEBUG_OUTPUT=1 prints the analysis trace at compile time.
// Nothing here is intended for execution or linking.

#include <cstddef>
#include <cstdint>

// Target attribute macros normally supplied by HIP's clang wrapper headers
// (skipped by -nogpuinc).
#ifndef __global__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#endif

#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

struct dim3 {
  unsigned x, y, z;
  constexpr __host__ __device__ dim3(unsigned X = 1, unsigned Y = 1,
                                     unsigned Z = 1)
      : x(X), y(Y), z(Z) {}
};

typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorUnknown = 999,
} hipError_t;

typedef struct ihipStream_t *hipStream_t;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4,
} hipMemcpyKind;

extern "C" {
hipError_t hipLaunchKernel(const void *Func, dim3 Grid, dim3 Block,
                           void **Args, size_t SharedMem, hipStream_t Stream);
hipError_t hipDeviceSynchronize(void);
const char *hipGetErrorString(hipError_t Err);
hipError_t hipMalloc(void **Ptr, size_t Size);
hipError_t hipMallocManaged(void **Ptr, size_t Size, unsigned Flags);
hipError_t hipFree(void *Ptr);
hipError_t hipMemcpy(void *Dst, const void *Src, size_t Size,
                     hipMemcpyKind Kind);
hipError_t hipMemcpyFromSymbol(void *Dst, const void *Symbol, size_t Size,
                               size_t Offset, hipMemcpyKind Kind);
hipError_t hipGetSymbolAddress(void **DevPtr, const void *Symbol);
hipError_t hipStreamCreate(hipStream_t *Stream);
hipError_t hipStreamSynchronize(hipStream_t Stream);
hipError_t hipStreamDestroy(hipStream_t Stream);

// Triple-chevron <<<...>>> launch support expected by CodeGen.
hipError_t __hipPushCallConfiguration(dim3 Grid, dim3 Block,
                                      size_t SharedMem = 0,
                                      hipStream_t Stream = nullptr);
hipError_t __hipPopCallConfiguration(dim3 *Grid, dim3 *Block,
                                     size_t *SharedMem, hipStream_t *Stream);
} // extern "C"

template <typename T>
static inline hipError_t hipMallocManaged(T **Ptr, size_t Size,
                                          unsigned Flags = 1) {
  return hipMallocManaged(reinterpret_cast<void **>(Ptr), Size, Flags);
}

template <typename T>
static inline hipError_t hipMalloc(T **Ptr, size_t Size) {
  return hipMalloc(reinterpret_cast<void **>(Ptr), Size);
}

static inline hipError_t hipMemcpyFromSymbol(void *Dst, const void *Symbol,
                                             size_t Size) {
  return hipMemcpyFromSymbol(Dst, Symbol, Size, 0, hipMemcpyDeviceToHost);
}

#if defined(__HIP_DEVICE_COMPILE__)
// Declaration only; AMDGPU lowers printf during codegen, which the pseudo-GPU
// pipeline never reaches (-emit-llvm).
extern "C" __device__ int printf(const char *Fmt, ...);

// Work-item builtins give the same IR shapes as the real HIP headers.
#define threadIdx                                                              \
  (dim3{(unsigned)__builtin_amdgcn_workitem_id_x(),                            \
        (unsigned)__builtin_amdgcn_workitem_id_y(),                            \
        (unsigned)__builtin_amdgcn_workitem_id_z()})
#define blockIdx                                                               \
  (dim3{(unsigned)__builtin_amdgcn_workgroup_id_x(),                           \
        (unsigned)__builtin_amdgcn_workgroup_id_y(),                           \
        (unsigned)__builtin_amdgcn_workgroup_id_z()})
// Declaration-only accessors: resolve to extern device functions in IR, which
// is sufficient for pass-level analysis (never linked or executed).
extern "C" __device__ unsigned __pseudo_gpu_block_dim(int Axis);
extern "C" __device__ unsigned __pseudo_gpu_grid_dim(int Axis);
#define blockDim                                                               \
  (dim3{__pseudo_gpu_block_dim(0), __pseudo_gpu_block_dim(1),                  \
        __pseudo_gpu_block_dim(2)})
#define gridDim                                                                \
  (dim3{__pseudo_gpu_grid_dim(0), __pseudo_gpu_grid_dim(1),                    \
        __pseudo_gpu_grid_dim(2)})
#endif // __HIP_DEVICE_COMPILE__

#endif // PROTEUS_PSEUDO_GPU_HIP_RUNTIME_H
