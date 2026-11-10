// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROTEUS_VARIANT_SOURCE_H
#define PROTEUS_VARIANT_SOURCE_H

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace proteus {
namespace detail {
// Use the same spelling on host and device; never use a runtime address as ID.
template <class T> constexpr uint64_t variant_type_id() {
  const char *S = __PRETTY_FUNCTION__;
  uint64_t H = 14695981039346656037ull;
  for (; *S; ++S) {
    H ^= static_cast<unsigned char>(*S);
    H *= 1099511628211ull;
  }
  return H;
}

using VariantCollector = std::function<std::vector<uint64_t>()>;
void registerVariantSource(uint64_t Tag, uint64_t Type, unsigned Bits,
                           VariantCollector Collector);
void unregisterVariantSource(uint64_t Tag);

template <class T> constexpr unsigned variant_bits() {
  static_assert(std::is_integral_v<T> || std::is_enum_v<T>,
                "jit_variants requires an integer, Boolean, or enum");
  static_assert(sizeof(T) <= sizeof(uint64_t),
                "jit_variants supports values up to 64 bits");
  return std::is_same_v<T, bool> ? 1 : sizeof(T) * 8;
}

// The extraction pass preserves calls in JIT IR and lowers AOT calls to their
// first operand. optnone prevents identity folding before extraction. Without
// the plugin this definition still provides correct ordinary execution.
template <class T>
#if defined(__CUDACC__) || defined(__HIP__)
__host__ __device__
#endif
    __attribute__((noinline, optnone)) T
    __jit_variants_marker(T Value, uint64_t, uint64_t, unsigned,
                          unsigned) noexcept {
  return Value;
}
} // namespace detail

template <class Tag, class Collector>
void register_variant_source(Collector CollectorFn) {
  using T = typename Tag::value_type;
  constexpr unsigned Bits = detail::variant_bits<T>();
  detail::registerVariantSource(
      detail::variant_type_id<Tag>(), detail::variant_type_id<T>(), Bits,
      [CollectorFn = std::move(CollectorFn)]() mutable {
        std::vector<uint64_t> Values;
        CollectorFn([&](T Value) {
          uint64_t Raw = static_cast<uint64_t>(Value);
          if constexpr (Bits < 64)
            Raw &= (uint64_t{1} << Bits) - 1;
          Values.push_back(Raw);
        });
        return Values;
      });
}

template <class Tag> void unregister_variant_source() {
  detail::unregisterVariantSource(detail::variant_type_id<Tag>());
}

template <class Tag, unsigned MaxVariants>
#if defined(__CUDACC__) || defined(__HIP__)
__host__ __device__
#endif
    __attribute__((always_inline)) inline typename Tag::value_type
    jit_variants(typename Tag::value_type Value) noexcept {
  using T = typename Tag::value_type;
  constexpr uint64_t TagID = detail::variant_type_id<Tag>();
  constexpr uint64_t TypeID = detail::variant_type_id<T>();
  constexpr unsigned Bits = detail::variant_bits<T>();
  return detail::__jit_variants_marker(Value, TagID, TypeID, Bits, MaxVariants);
}
} // namespace proteus
#endif
