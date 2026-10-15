// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <cstdint>
#include <cstring>

#include <proteus/JitInterface.h>

// The IR analysis assumes closures move as whole objects (the witness model;
// docs/dev/design/lambda-closure-provenance.md). These are the type-erasure
// constructs that can break that assumption, diagnosed at the source level.

static auto makeBody(int A, int B) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(A), Y = proteus::jit_variable(B)] {
        return X + Y;
      });
}

using Registered = decltype(makeBody(0, 0));

template <typename F> struct Envelope {
  F Body;
  std::uint64_t Suffix;
};

// Interior-slot splice through a type-erased pointer: the shape of
// tests/gpu/lambda_interior_slot_overwrite.cpp.
void interiorSliceWrite(Registered *Initial, Registered *Donor) {
  Registered Copy = *Initial;
  // expected-error@+1 {{cannot be reinterpreted through a type-erased mutable pointer}}
  auto *Words = reinterpret_cast<std::uint32_t *>(&Copy);
  const auto *DonorWords =
      // expected-error@+1 {{cannot be reinterpreted through a type-erased mutable pointer}}
      reinterpret_cast<std::uint32_t *>(Donor);
  Words[1] = DonorWords[1];
  Copy();
}

// Wrong-length copy: legal C++, no casts, splices two sources into one
// functor. The IR analysis declines the shapes it can see; the frontend
// rejects them all.
void partialMemcpy(Registered *Initial, Registered *Donor) {
  Registered Copy = *Donor;
  // expected-error@+1 {{memory builtins cannot write into storage containing a registered lambda}}
  __builtin_memcpy(&Copy, Initial, 4);
  Copy();
}

// Builtin write through an envelope that holds the wrapper by value.
void envelopeMemset(Envelope<Registered> *E) {
  // expected-error@+1 {{memory builtins cannot write into storage containing a registered lambda}}
  __builtin_memset(E, 0, sizeof(*E));
}

// Reading through a type-erased const pointer does not license writes.
std::uint32_t constReadOk(const Registered *Body) {
  return *reinterpret_cast<const std::uint32_t *>(Body);
}

// Union storage reuse over a registered wrapper.
union WrapperPun {
  // expected-error@+1 {{cannot be a member of a union}}
  Registered A;
  std::uint64_t Raw;
};

int main() {
  auto Body = makeBody(1, 2);
  interiorSliceWrite(&Body, &Body);
  partialMemcpy(&Body, &Body);
  Envelope<Registered> E{makeBody(3, 4), 0};
  envelopeMemset(&E);
  (void)constReadOk(&Body);
  return 0;
}
