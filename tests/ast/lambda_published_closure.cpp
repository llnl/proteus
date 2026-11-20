// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <proteus/JitInterface.h>

static auto makeBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] { return X; });
}
using Registered = decltype(makeBody(0));

// Reject publication even though the closure itself has automatic or dynamic
// storage. A later invocation need not contain the publishing store in its IR.
template <typename F>
// clang-format off
// expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
// clang-format on
F *PublishedDestination;
template <typename F>
// clang-format off
// expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
// clang-format on
const F *PublishedSource;

void publish(Registered *Destination, const Registered *Source) {
  PublishedDestination<Registered> = Destination;
  PublishedSource<Registered> = Source;
}

void overwrite(const char *Message) {
  if (Message[0])
    __builtin_memcpy(PublishedDestination<Registered>,
                     PublishedSource<Registered>, sizeof(Registered));
}

struct Plan {
  Registered *Destination;
  const Registered *Source;
};
// clang-format off
// expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
// clang-format on
Plan GlobalPlan;
// clang-format off
// expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
// clang-format on
Registered *GlobalSlots[2];

void staticSlot(Registered *Body) {
  // clang-format off
  // expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
  // clang-format on
  static Registered *Slot = Body;
}

void *ErasedGlobal;
void publishErased(Registered *Body) {
  // clang-format off
  // expected-error@+2 {{a pointer to a registered lambda cannot be stored in global-lifetime storage}}
  // clang-format on
  ErasedGlobal = static_cast<void *>(Body);
}

// Local containers and pointer slots remain legal, including the local-slot
// aliasing-memcpy regression in the GPU suite.
void localStorage(Registered *Body) {
  Registered *Slot = Body;
  Plan LocalPlan{Body, Body};
  (void)Slot;
  (void)LocalPlan;
}
