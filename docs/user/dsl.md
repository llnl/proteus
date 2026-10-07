# DSL API

!!! warning "This is under development"
    The DSL interface is under active development and will evolve to include
    additional specialization features and a more streamlined syntax.
    However, the core concepts described here are expected to remain consistent,
    even as the implementation improves.

The DSL API is an alternative C++ interface for generating, managing, and
executing IR modules at runtime via JIT compilation.
It supports host and GPU targets and is intended for cases where you want to
construct JIT code programmatically instead of annotating existing source or
compiling source strings.

Unlike the annotation interface, this path does not require compiling the
application with Clang. The application can be built with any compatible
compiler because the JIT unit is constructed through runtime library APIs.

If you want source-string compilation, see the
[C++ frontend API](cpp-frontend.md).
If you want annotation-driven integration into existing code, see
[Code Annotations](annotations.md).

## Overview

Compared to code annotations, the DSL API offers more direct control over how
the JIT unit is built.
It introduces a mini-language for expressing computation in a C++-like style
while still feeding the same Proteus runtime specialization and dispatch
pipeline.

## Target and Backend Selection

`JitModule` takes a target string and an optional backend string:

- targets: `"host"`, `"cuda"`, `"hip"`
- backends: `"llvm"` (default) or `"mlir"`

The MLIR backend is available only when Proteus is built with
`-DPROTEUS_ENABLE_MLIR=ON`.

## Core Abstractions

The DSL introduces a small set of abstractions for programmatically generating
JIT code:

- **Types**: a subset of the C++ type system, including fundamental types such
  as `int`, `size_t`, `float`, `double`, and their pointer or reference forms
- **Modules**: analogous to translation units; modules encapsulate code and
  organize it into functions
- **Functions**: units of computation within modules, containing variables and
  statements
- **Variables**: typed entities supporting operations such as standard
  arithmetic (`+`, `-`, `*`, `/`) and indexed access (`[]`) for pointers
- **Statements**: control flow and actions within functions, including `If`,
  `For`, `Call` / `CallBuiltin`, and `Ret`

## Building Functions and Kernels

Use `addFunction<Sig>()` for normal functions or helper routines.
Use `addKernel<void(...)>()` for device entry points on CUDA or HIP targets.

Here is a concise example that demonstrates how to define a JIT module with the
DSL API:

```cpp
auto createJitKernel(size_t N) {
  auto J = std::make_unique<JitModule>("cuda");

  // Add a kernel with the signature: void add_vectors(double *A, double *B)
  // using vector size N as a runtime constant.
  auto KernelHandle = J->addKernel<void(double *, double *)>("add_vectors");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &I = F.declVar<size_t>("I");
    auto &Inc = F.declVar<size_t>("Inc");
    auto [A, B] = F.getArgs();
    auto &RunConstN = F.defRuntimeConst(N);

    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);

    Inc = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    F.beginFor(I, I, RunConstN, Inc);
    {
      A[I] = A[I] + B[I];
    }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}
```

This example shows how to build a CUDA kernel for vector addition.
The API provides constructs for declaring variables, accessing function
arguments, using built-in CUDA identifiers, and defining control flow in a
familiar C++-like style.

## Runtime Constants and Builtins

The vector size `N` is embedded as the runtime constant `RunConstN` with
`F.defRuntimeConst(N)`, which specializes the generated code to the actual
value seen at execution time.

GPU builtins such as `threadIdx.x`, `blockIdx.x`, `blockDim.x`, and
`gridDim.x` are accessed with `F.callBuiltin()`.
That is what lets the kernel compute a strided loop index while still being
constructed through the DSL rather than written as normal CUDA source.

## Compile and Execute Flow

Once the module is defined, it can be compiled and launched:

```cpp
// ... continuing from above ...

double *A = ...;
double *B = ...;
size_t N = ...;

auto [J, KernelHandle] = createJitKernel(N);
J->compile();

constexpr unsigned ThreadsPerBlock = 256;
unsigned NumBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

KernelHandle.launch(
    {NumBlocks, 1, 1}, {ThreadsPerBlock, 1, 1}, 0, nullptr, A, B);

cudaDeviceSynchronize();
```

Calling `J->compile()` finalizes the module and builds it into an in-memory
executable object.
The `KernelHandle` is then used to launch the kernel with grid and block
dimensions, plus optional launch parameters such as shared memory size and
stream.
Changing the target from `"cuda"` to `"hip"` lets the same construction pattern
target an AMD GPU.

## Examples

This example is included in the test suite:

- CPU: [add_vectors_runconst.cpp](https://github.com/LLNL/proteus/blob/main/tests/frontend/cpu/add_vectors_runconst.cpp)
- GPU: [add_vectors_runconst.cpp](https://github.com/LLNL/proteus/blob/main/tests/frontend/gpu/add_vectors_runconst.cpp)
