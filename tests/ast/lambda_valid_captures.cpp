// RUN: %clang -std=c++17 -fsyntax-only -I%proteus_include -fplugin=%plugin %s

#include <proteus/JitInterface.h>

void brace_init_capture(int Value) {
  auto Lambda =
      proteus::register_lambda([X{proteus::jit_variable(Value)}] { return X; });
  (void)Lambda;
}

void long_double_capture(long double Value) {
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value)] { return X; });
  (void)Lambda;
}

// Calling a functor passes the functor itself as argument 0, so arguments and
// parameters must not be paired by index.
void const_ref_argument_to_functor(int Value) {
  int Out = 0;
  auto Copy = [](const int &From, int &To) { To = From; };
  auto Lambda = proteus::register_lambda(
      [X = proteus::jit_variable(Value), &Out, Copy] { Copy(X, Out); });
  (void)Lambda;
}
