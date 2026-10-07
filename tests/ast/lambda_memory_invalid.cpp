// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <cstddef>

#include <proteus/JitInterface.h>

template <typename F> struct PointerSlots {
  F *Noise;
  F *Invoked;
};

template <typename F> void overwriteSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F> void invokeAfterTwoCalls(F *Initial, F *First, F *Final) {
  F *volatile Slot = Initial;
  overwriteSlot(&Slot, First);
  // expected-error@+1 {{cannot be overwritten by multiple function calls}}
  overwriteSlot(&Slot, Final);
  (*Slot)();
}

template <typename F>
void overwriteByteOffsetField(void *Storage, F *Replacement) {
  auto *Bytes = reinterpret_cast<unsigned char *>(Storage);
  // expected-error@+1 {{cannot be accessed through type-erased byte offsets}}
  auto *Slot = reinterpret_cast<F *volatile *>(Bytes + sizeof(F *));
  *Slot = Replacement;
}

void *EscapedSlot;

template <typename F> void escapePointerSlot(F **Slot) { EscapedSlot = Slot; }

template <typename F> void invokeWithEscapedSlot(F *Initial, F *Replacement) {
  F *Slot = Initial;
  Slot = Replacement;
  (*Slot)();
  // expected-error@+1 {{cannot escape to global storage}}
  escapePointerSlot(&Slot);
}

template <typename F> F *volatile *returnSlotOnce(F *volatile *Slot) {
  return Slot;
}

template <typename F>
F *volatile *returnSlotThroughPhi(F *volatile *First, F *volatile *Second,
                                  bool UseSecond) {
  F *volatile *Result;
  if (UseSecond)
    Result = returnSlotOnce(Second);
  else
    Result = returnSlotOnce(First);
  // expected-error@+1 {{cannot be selected from multiple control-flow}}
  return Result;
}

template <typename F>
F *volatile *returnRuntimeSelectedField(PointerSlots<F> *Slots,
                                        bool ReturnNoise) {
  if (ReturnNoise)
    return &Slots->Noise;
  // expected-error@+1 {{cannot be selected from different aggregate fields}}
  return &Slots->Invoked;
}

auto makeRegisteredLambda() {
  return proteus::register_lambda([] {});
}

void instantiateUnsupportedShapes(bool SelectSecond) {
  auto Initial = makeRegisteredLambda();
  auto First = makeRegisteredLambda();
  auto Final = makeRegisteredLambda();

  invokeAfterTwoCalls(&Initial, &First, &Final);

  PointerSlots<decltype(Initial)> Slots{&First, &Initial};
  overwriteByteOffsetField<decltype(Initial)>(&Slots, &Final);

  invokeWithEscapedSlot(&Initial, &Final);

  auto *InitialSlot = &Initial;
  auto *SecondSlot = &Initial;
  auto *PhiSlot = returnSlotThroughPhi(&InitialSlot, &SecondSlot, SelectSecond);
  overwriteSlot(PhiSlot, &Final);

  auto *FieldSlot = returnRuntimeSelectedField(&Slots, SelectSecond);
  overwriteSlot(FieldSlot, &Final);
}
