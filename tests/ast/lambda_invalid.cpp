// clang-format off
// RUN: %not %clang -std=c++17 -fsyntax-only -I%proteus_include -fplugin=%plugin %s 2>&1 | %FILECHECK %s

#include <proteus/JitInterface.h>

struct Functor {
  void operator()() const {}
};

void invalid_register() {
  (void)proteus::register_lambda(Functor{});
  // CHECK-DAG: error: proteus::register_lambda requires a lambda closure object
}

void duplicate_register() {
  auto Registered = proteus::register_lambda([] {});
  (void)proteus::register_lambda(Registered);
  // CHECK-DAG: error: a lambda returned by proteus::register_lambda cannot be registered again
}

void invalid_use(int Value) {
  (void)proteus::jit_variable(Value);
  // CHECK-DAG: error: proteus::jit_variable must be used directly as a lambda init-capture initializer
}

void unregistered(int Value) {
  auto Lambda = [X = proteus::jit_variable(Value)] { return X; };
  // CHECK-DAG: error: lambda containing proteus::jit_variable must be registered with proteus::register_lambda
  (void)Lambda;
}
// clang-format on

void unsupported_type(short Value) {
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value)] { return X; });
  // CHECK-DAG: error: proteus::jit_variable does not support capture type
  // 'short'
  (void)Lambda;
}

void mutate_capture(int Value) {
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value)]() mutable { return ++X; });
  // CHECK-DAG: error: proteus::jit_variable capture must remain read-only
  (void)Lambda;
}

void mutate_by_reference(int &Value) { ++Value; }

void escape_capture(int Value) {
  auto Lambda =
      proteus::register_lambda([X = proteus::jit_variable(Value)]() mutable {
        mutate_by_reference(X);
        int *Alias = &X;
        return *Alias;
      });
  (void)Lambda;
}
