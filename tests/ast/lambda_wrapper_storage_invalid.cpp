// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <cstddef>
#include <cstdint>

#include <proteus/JitInterface.h>

// The IR analysis assumes closures move as whole objects (the witness model;
// docs/dev/design/lambda-closure-provenance.md). The checker diagnoses
// exactly the constructs that can break that assumption -- writes through
// type-erased pointers, partial-overlap memory builtins, unions -- while the
// whole-object and non-overlapping forms of the same constructs stay legal:
// the analysis suite itself uses them to exercise provenance resolution.

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

template <typename F> struct PrefixedEnvelope {
  std::uint64_t Prefix[2];
  F Body;
};

// Interior-slot splice: write through a type-erased pointer. The casts are
// recorded; the WRITE through them is the error.
void interiorSliceWrite(Registered *Initial, Registered *Donor) {
  Registered Copy = *Initial;
  auto *Words = reinterpret_cast<std::uint32_t *>(&Copy);
  const auto *DonorWords = reinterpret_cast<const std::uint32_t *>(Donor);
  // expected-error@+1 {{cannot be reinterpreted through a type-erased mutable pointer}}
  Words[1] = DonorWords[1];
  Copy();
}

// Write directly through the cast expression, no intermediate variable.
void directErasedWrite(Registered *Body) {
  // expected-error@+1 {{cannot be reinterpreted through a type-erased mutable pointer}}
  *reinterpret_cast<std::uint32_t *>(Body) = 7;
}

// Wrong-length copy: covers byte 0 of the closure but not its extent.
void partialMemcpy(Registered *Initial, Registered *Donor) {
  Registered Copy = *Donor;
  // expected-error@+1 {{memory builtins cannot write into storage containing a registered lambda}}
  __builtin_memcpy(&Copy, Initial, 4);
  Copy();
}

// Memset overlapping the wrapper: destroys registered-slot provenance even
// when it covers the whole object.
void envelopeMemset(Envelope<Registered> *E) {
  // expected-error@+1 {{memory builtins cannot write into storage containing a registered lambda}}
  __builtin_memset(E, 0, sizeof(*E));
}

// Dynamic length that may reach the wrapper: cannot be proven safe.
void dynamicLengthMemcpy(Registered *Dest, const Registered *Source,
                         unsigned Length) {
  // expected-error@+1 {{memory builtins cannot write into storage containing a registered lambda}}
  __builtin_memcpy(Dest, Source, Length);
}

// Union storage reuse over a registered wrapper.
union WrapperPun {
  // expected-error@+1 {{cannot be a member of a union}}
  Registered A;
  std::uint64_t Raw;
};

// ---- Legal forms of the same constructs (no diagnostics expected) ----------

// Whole-object copy of the closure through the builtin: the discipline
// itself, in builtin form (lambda_provenance_shapes exercises this shape).
void wholeClosureMemcpy(Registered *Dest, const Registered *Source) {
  __builtin_memcpy(Dest, Source, sizeof(Registered));
}

// Member-precise whole copy inside an envelope.
void memberMemcpy(PrefixedEnvelope<Registered> *Dest,
                  const PrefixedEnvelope<Registered> *Source) {
  __builtin_memcpy(&Dest->Body, &Source->Body, sizeof(Registered));
}

// Constant-length write that provably misses the wrapper (prefix only).
void prefixOnlyMemcpy(PrefixedEnvelope<Registered> *Dest,
                      const PrefixedEnvelope<Registered> *Source) {
  __builtin_memcpy(Dest, Source, sizeof(std::uint64_t));
}

// container_of-style address math: erasure for addressing, never written.
Registered *roundTripInteriorPointer(PrefixedEnvelope<Registered> *E) {
  auto *Bytes = reinterpret_cast<char *>(&E->Body);
  auto *Outer = reinterpret_cast<PrefixedEnvelope<Registered> *>(
      Bytes - offsetof(PrefixedEnvelope<Registered>, Body));
  return &Outer->Body;
}

// Reading through a type-erased const pointer.
std::uint32_t constReadOk(const Registered *Body) {
  return *reinterpret_cast<const std::uint32_t *>(Body);
}

int main() {
  auto Body = makeBody(1, 2);
  interiorSliceWrite(&Body, &Body);
  directErasedWrite(&Body);
  partialMemcpy(&Body, &Body);
  Envelope<Registered> E{makeBody(3, 4), 0};
  envelopeMemset(&E);
  dynamicLengthMemcpy(&Body, &Body, 4);
  wholeClosureMemcpy(&Body, &Body);
  PrefixedEnvelope<Registered> P{{0, 0}, makeBody(5, 6)};
  memberMemcpy(&P, &P);
  prefixOnlyMemcpy(&P, &P);
  (void)roundTripInteriorPointer(&P);
  (void)constReadOk(&Body);
  return 0;
}
