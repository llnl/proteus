// RUN: %clang -std=c++17 -fsyntax-only -I%proteus_include -fplugin=%plugin %s

#include <proteus/JitInterface.h>

template <typename F> void register_indirectly(F &&Fn) {
  proteus::register_lambda(Fn)();
}

auto make_lambda(int Value) {
  return [X = proteus::jit_variable(Value)] { return X; };
}

template <int Offset> void direct_template(int Value) {
  auto Lambda = proteus::register_lambda(
      [=, X = proteus::jit_variable(Value)] { return X + Offset; });
  (void)Lambda;
}

void valid(int Value, int *Pointer) {
  register_indirectly(
      [X = proteus::jit_variable(Value), P = proteus::jit_variable(Pointer)] {
        *P = X;
        return *P;
      });
  auto Lambda = proteus::register_lambda(make_lambda(Value));
  (void)Lambda;
  direct_template<1>(Value);
}
