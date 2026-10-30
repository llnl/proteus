# Code Annotations

This page covers Proteus's annotation-based interface for runtime
specialization.
It includes both attribute-based annotations and the C++ annotation helpers for
specializing scalars, arrays, objects, and captured lambda state.

This interface requires compiling the user application with Clang so the
Proteus pass can observe the annotations and instrument the annotated regions.

If you want to construct JIT code directly instead of annotating existing code,
see the [C++ frontend API](cpp-frontend.md) or the [DSL API](dsl.md).

## Kernels / Functions

The interface for host functions and device kernel functions is the same.
Proteus relies on the `annotate` attribute, which allows metadata to be
attached to any function, whether it runs on the CPU or as a GPU kernel.

Here is a DAXPY (double-precision scaled vector addition) example:

```cpp title="Host"
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, size_t N, double *X, double *Y) {
  for (size_t I = 0; I < N; ++I)
    Y[I] = A * X[I] + Y[I];
}
```

```cuda title="Device"
__global__
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, size_t N, double *X, double *Y) {
  int Tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int Stride = blockDim.x * gridDim.x;
  for (int I = Tid; I < N; I += Stride)
    X[I] = A * X[I] + Y[I];
}
```

Here, the `annotate("jit", 1, 2)` attribute marks the function for JIT
compilation.
The comma-separated list after `jit` specifies which arguments
(1-indexed) Proteus should specialize: in this case, the scaling factor `A` and
the vector size `N`.

!!! note
    Function arguments are **1-indexed**.

When specializing, Proteus folds the designated arguments into constants.
This allows the downstream compiler optimization pipeline (for example at `-O3`)
to apply stronger transformations.
Folding runtime values into constants enhances analysis precision and unlocks
optimizations such as constant propagation, control-flow simplification, and
loop unrolling.

## Lambdas

Portability libraries such as RAJA and Kokkos make heavy use of lambdas, so
captured lambda state is an important specialization target.

Lambdas are slightly more complex than plain functions because they capture
variables in closures.
In practice, the greatest opportunities for specialization are often in the
captured variables rather than the argument list.

Proteus therefore requires lambda functions to be **registered** with the
runtime using `proteus::register_lambda`.
To mark captured variables for specialization, they must be explicitly wrapped
with `proteus::jit_variable`.

Here is a simple example:

```cpp title="Host"
auto DaxpyLambda = proteus::register_lambda(
    [=, A = proteus::jit_variable(A), N = proteus::jit_variable(N)]() {
      for (size_t I = 0; I < N; ++I)
        Y[I] = A * X[I] + Y[I];
    });
DaxpyLambda();
```

```cuda title="Device"
template <typename Lambda>
__global__
void kernel(Lambda &&Func) {
  Func();
}

auto DaxpyLambda = proteus::register_lambda(
    [=, A = proteus::jit_variable(A), N = proteus::jit_variable(N)] __device__() {
      for (size_t I = 0; I < N; ++I)
        Y[I] = A * X[I] + Y[I];
    });
kernel<<<blocks, threads>>>(DaxpyLambda);
```

Proteus automatically JIT-compiles a kernel that calls a registered device
lambda. Explicit kernel annotations remain available for specializing kernel
arguments or kernels without registered lambdas.

## C++ Annotations API

Beyond attribute-based annotations, Proteus also provides a C++ API for
specialization through annotation functions.
This API supports specialization of **scalars**, **arrays**, and **objects**.

When using these APIs, the contract is that annotated arrays and objects are
treated as **read-only** within the specialized code.
Specialized contents may be folded into constants, so these helpers are for
inputs that should not be mutated by the specialized region.

The C++ annotations API includes:

- `template<typename T> proteus::jit_arg(T Scalar)` for specializing a scalar
  value
- `template<typename T> proteus::jit_array(T *Arr, size_t NumElts)` for
  specializing array contents
- `template<typename T> proteus::jit_object(T *Obj or T &Obj)` for specializing
  a trivially copyable object (copyable via `std::memcpy`)

When you annotate a function or kernel with these APIs, Proteus will JIT-compile
it automatically.
There is no need to also use the `annotate` attribute.

Here is the same DAXPY example written with annotation functions:

```cpp title="Host"
void daxpy(double A, size_t N, double *X, double *Y) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
  for (std::size_t I = 0; I < N; ++I)
    Y[I] += X[I] * A;
}
```

```cuda title="Device"
__global__
void daxpy(double A, size_t N, double *X, double *Y) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
  int Tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int Stride = blockDim.x * gridDim.x;
  for (int I = Tid; I < N; I += Stride)
    X[I] = A * X[I] + Y[I];
}
```

`jit_array()` and `jit_object()` are used similarly:

```cpp title="Array Specialization"
void sum_prefix(const double *A, double *Out, size_t N) {
  proteus::jit_array(A, N);
  Out[0] = A[0];
  for (size_t I = 1; I < N; ++I)
    Out[I] = Out[I - 1] + A[I];
}
```

```cpp title="Object Specialization"
struct Params {
  int Tile;
  double Scale;
};

void saxpy(const Params &P, double *X, double *Y, size_t N) {
  proteus::jit_object(P);
  for (size_t I = 0; I < N; ++I)
    Y[I] += X[I] * P.Scale;
}
```
