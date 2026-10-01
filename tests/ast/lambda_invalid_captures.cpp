// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <proteus/JitInterface.h>

struct Accumulator {
  void operator+=(int &Value) { ++Value; }
};

struct IntRef {
  explicit IntRef(int &Ref) : Ref(Ref) {}
  int &Ref;
};

void mutate_through_member_operator(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        Accumulator Acc;
        // expected-error@+1 {{capture must remain read-only}}
        Acc += X;
        return X;
      });
  (void)Lambda;
}

void mutate_through_dereferenced_address(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        // expected-error@+1 {{capture must remain read-only}}
        *&X = 1;
        return X;
      });
  (void)Lambda;
}

void mutate_through_assigned_pointer(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        int *Alias;
        // expected-error@+1 {{capture must remain read-only}}
        Alias = &X;
        *Alias = 2;
        return X;
      });
  (void)Lambda;
}

void mutate_through_constructor_reference(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        // expected-error@+1 {{capture must remain read-only}}
        IntRef Ref(X);
        Ref.Ref = 3;
        return X;
      });
  (void)Lambda;
}
