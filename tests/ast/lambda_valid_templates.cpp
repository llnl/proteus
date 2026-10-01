// RUN: %clang -std=c++17 -fsyntax-only -I%proteus_include -fplugin=%plugin %s

#include <proteus/JitInterface.h>

// RAJA-style wrapper that registers the lambda it is given.
template <typename Body> void forall(int N, Body &&B) {
  auto Registered = proteus::register_lambda(B);
  for (int I = 0; I < N; ++I)
    Registered(I);
}

// The capture type does not depend on T, so the uninstantiated pattern is
// checked even though only the instantiation reaches register_lambda.
template <typename T> void gather(T *Out, const T *In, int N, int Stride) {
  forall(N,
         [=, S = proteus::jit_variable(Stride)](int I) { Out[I] = In[I * S]; });
}

template <typename T> struct Scaler {
  void scale(T *Data, int N, int Factor) {
    forall(N, [=, F = proteus::jit_variable(Factor)](int I) { Data[I] *= F; });
  }
};

template <typename T> void never_instantiated(T *Out, int Value) {
  auto Lambda = proteus::register_lambda(
      [=, X = proteus::jit_variable(Value)] { Out[0] = X; });
  Lambda();
}

void valid(double *Out, const double *In, int N) {
  gather(Out, In, N, 4);
  Scaler<double>().scale(Out, N, 2);
}
