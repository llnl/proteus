// clang-format off
// RUN: %clang -std=c++17 -fsyntax-only -Xclang -verify -I%proteus_include -fplugin=%plugin %s
// clang-format on

#include <proteus/JitInterface.h>

// Double registration also reaches the library's existing invariant guard.
// expected-error@* {{static assertion failed}}
// expected-note@* 2 {{in instantiation of function template specialization}}

struct Functor {
  void operator()() const {}
};

void invalid_register() {
  // expected-error@+1 {{requires a lambda closure object}}
  (void)proteus::register_lambda(Functor{});
}

void duplicate_register() {
  auto Registered = proteus::register_lambda([] {});
  // expected-error@+1 {{cannot be registered again}}
  (void)proteus::register_lambda(Registered);
}

void invalid_use(int Value) {
  // expected-error@+1 {{must be used directly as a lambda init-capture}}
  (void)proteus::jit_variable(Value);
}

void unregistered(int Value) {
  // expected-error@+1 {{must be registered with proteus::register_lambda}}
  auto Lambda = [X = proteus::jit_variable(Value)] { return X; };
  (void)Lambda;
}

void unsupported_type(short Value) {
  // expected-error@+2 {{does not support capture type 'short'}}
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value)] { return X; });
  (void)Lambda;
}

void mutate_capture(int Value) {
  // expected-error@+2 {{capture must remain read-only}}
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value)]() mutable { return ++X; });
  (void)Lambda;
}

void mutate_by_reference(int &Value) { ++Value; }

void escape_capture(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        // expected-error@+1 {{capture must remain read-only}}
        mutate_by_reference(X);
        // expected-error@+1 {{capture must remain read-only}}
        int *Alias = &X;
        return *Alias;
      });
  (void)Lambda;
}
