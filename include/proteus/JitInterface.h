//===-- jit.h -- user interface to Proteus JIT library --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(readability-identifier-naming)

#ifndef PROTEUS_JIT_INTERFACE_H
#define PROTEUS_JIT_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Init.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

extern "C" __attribute__((used)) void
__proteus_register_lambda_runtime_constant(int32_t Type, int32_t Pos,
                                           int32_t Offset, const void *ValuePtr,
                                           uint64_t functor_id);

extern "C" void __proteus_take_address(void const *) noexcept;
extern "C" __attribute__((used)) void
__proteus_finalize_register(uint64_t Tag) noexcept;

namespace proteus {

#if defined(__CUDACC__) || defined(__HIP__)
#define PROTEUS_HOST_DEVICE __host__ __device__
#else
#define PROTEUS_HOST_DEVICE
#endif

template <typename T> __attribute__((noinline)) void jit_arg(T V) noexcept;
#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ void jit_arg(T V) noexcept;
#endif

template <typename T>
__attribute__((noinline)) void
jit_array(T V, [[maybe_unused]] size_t NumElts,
          [[maybe_unused]]
          typename std::remove_pointer<T>::type Velem = 0) noexcept;
#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ void
jit_array(T V, [[maybe_unused]] size_t NumElts,
          [[maybe_unused]]
          typename std::remove_pointer<T>::type Velem = 0) noexcept;
#endif

template <typename T>
__attribute__((noinline))
std::enable_if_t<std::is_trivially_copyable_v<std::remove_pointer_t<T>>, void>
jit_object(T *V, size_t Size = sizeof(std::remove_pointer_t<T>)) noexcept;

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ std::enable_if_t<
    std::is_trivially_copyable_v<std::remove_pointer_t<T>>, void>
jit_object(T *V, size_t Size = sizeof(T)) noexcept;
#endif

template <typename T>
__attribute__((noinline))
std::enable_if_t<!std::is_pointer_v<T> &&
                     std::is_trivially_copyable_v<std::remove_reference_t<T>>,
                 void>
jit_object(T &V, size_t Size = sizeof(std::remove_reference_t<T>)) noexcept;

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ std::enable_if_t<
    !std::is_pointer_v<T> &&
        std::is_trivially_copyable_v<std::remove_reference_t<T>>,
    void>
jit_object(T &V, size_t Size = sizeof(T)) noexcept;
#endif

namespace detail {
// todo: use LLVM hashing?
constexpr std::uint64_t fnv1a64(const char *s) {
  std::uint64_t h = 14695981039346656037ull;
  for (; *s; ++s) {
    h ^= (unsigned char)(*s);
    h *= 1099511628211ull;
  }
  return h;
}

// todo: make a test with lambda factory in a header, two separate cpp files
template <class Lambda> constexpr std::uint64_t functor_id() {
  return fnv1a64(__PRETTY_FUNCTION__); // includes Lambda + Ctr in the text
}

template <uint64_t FunctorID, typename Lambda> struct LambdaFunctorWrapper {
  using LambdaType = Lambda;
  static constexpr std::uint64_t functor_id = FunctorID;
  LambdaType lambda;

  // The second annotation payload is the closure's byte size, folded to a
  // constant by the compiler frontend at template instantiation -- host and
  // device compilations each record their own target's value. The kernel-arg
  // analysis uses it as the tracked entity's extent; unlike types in the IR,
  // it cannot be erased by optimization.
  template <typename... Args>
  PROTEUS_HOST_DEVICE __attribute__((annotate("jit")))
  __attribute__((annotate("proteus.wrapper_call", functor_id,
                          sizeof(LambdaType)))) decltype(auto)
  operator()(Args &&...args) const noexcept {
    return lambda(std::forward<Args>(args)...);
  }
};

template <typename... T> struct is_lambda_functor_wrapper : std::false_type {};

template <uint64_t FunctorID, typename Lambda>
struct is_lambda_functor_wrapper<LambdaFunctorWrapper<FunctorID, Lambda>>
    : std::true_type {};

template <std::uint64_t FunctorId, typename L>
PROTEUS_HOST_DEVICE inline auto tag_functor(L &&lambda) {
  return LambdaFunctorWrapper<FunctorId, std::decay_t<L>>{
      std::forward<L>(lambda)};
}

template <uint64_t ID, typename T>
[[nodiscard]] static __attribute__((noinline))
__attribute__((annotate("proteus.register_call_impl", ID))) auto
__register_lambda_impl(T &&t) noexcept {
  static_assert(!detail::is_lambda_functor_wrapper<std::decay_t<T>>::value);
  // Force LLVM to generate an AllocaInst of the underlying Clang--generated
  // anonymous class for T.  We remove this after recording the demangled
  // lambda name.
  using LambdaType = std::decay_t<T>;
  LambdaType local = t;
  __proteus_take_address(&local);
  auto result = tag_functor<ID>(std::forward<T>(t));
  return result;
}

} // namespace detail

template <class L>
[[nodiscard]] inline auto __attribute__((annotate("proteus.register_call")))
register_lambda(L &&lambda) noexcept {
  auto registered_lambda = ::proteus::detail::__register_lambda_impl<
      ::proteus::detail::functor_id<std::decay_t<L>>()>(
      std::forward<L>(lambda));
  __proteus_take_address(&registered_lambda);
  return registered_lambda;
}

template <typename T>
static __attribute__((noinline)) T jit_variable(T V) noexcept {
  return V;
}

#if defined(__CUDACC__) || defined(__HIP__)
// The function needs to be static for RDC compilation to resolve the static
// shared memory fallback.
template <typename T, size_t MAXN, int UniqueID = 0>
static __device__ __attribute__((noinline)) T *
shared_array([[maybe_unused]] size_t N,
             [[maybe_unused]] size_t ElemSize = sizeof(T)) {
  alignas(T) static __shared__ char shmem[sizeof(T) * MAXN];
  return reinterpret_cast<T *>(shmem);
}
#endif

} // namespace proteus

#endif

// NOLINTEND(readability-identifier-naming)
