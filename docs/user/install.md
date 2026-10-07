# Installation

Proteus can be installed from source or via [Spack](https://github.com/spack/spack).
We recommend using the latest `main` branch, which is well-tested, stable, and
includes the most recent features.

Building Proteus installs two components:
- the **Proteus LLVM plugin pass** (`ProteusPass`), and
- the **Proteus runtime library** (`libproteus`).

The LLVM plugin pass is necessary **only if** you use the Code annotation
interface and requires compiling your application with Clang, besides linking with `libproteus`.
If you use the DSL, C++ frontend, or MLIR frontend APIs, you only need to link
your code with `libproteus`.
Integrate the components required by your chosen interface into your application
build system.
See the [Integration](integration.md) section for more details on integrating with your build system.

## Spack

We provide a Spack package recipe in `packaging/spack`.

Assuming you already have a Spack installation, you can add the Proteus repo and
install the package with:

```shell
git clone https://github.com/LLNL/proteus.git
spack repo add proteus/packaging/spack/spack_repo/proteus
spack install proteus
```

Useful variants include CUDA, ROCm, MPI, shared-library, test, and
implementation-header support. See `spack info proteus` for the full variant
list.

Some typical examples:

```shell
spack install proteus +cuda cuda_arch=90
spack install proteus +rocm amdgpu_target=gfx942
spack install proteus +mpi
```

## Building from source

Proteus uses `cmake` for building and requires an existing LLVM installation.
`LLVM_INSTALL_DIR` is mandatory and must point to the LLVM/Clang installation
prefix. CI currently covers LLVM 19/20/22 with CUDA 12.2 and AMD ROCm versions
6.4.3 (based on LLVM 19), 7.1.1 (based on LLVM 20), and 7.2.0 (based on LLVM 22).

The top-level `CMakeLists.txt` currently defines the following build options:

* `BUILD_SHARED`: build Proteus as a shared library (default is static).
* `ENABLE_TESTS`: build Proteus tests.
* `PROTEUS_ENABLE_MPI`: enable MPI support for shared caching.
* `PROTEUS_ENABLE_MLIR`: enable the MLIR backend.
* `PROTEUS_ENABLE_HIP`: enable HIP support.
* `PROTEUS_ENABLE_CUDA`: enable CUDA support.
* `ENABLE_DEVELOPER_COMPILER_FLAGS`: enable additional warning flags intended for development.
* `PROTEUS_INSTALL_IMPL_HEADERS`: install implementation headers in addition to the public API headers.

!!! info "Host, CUDA, and HIP support"
    Proteus always supports host JIT compilation.
    You can additionally enable CUDA or HIP JIT compilation by setting
    `PROTEUS_ENABLE_CUDA` or `PROTEUS_ENABLE_HIP` respectively.

!!! info "Backend requirements"
    A host-only build requires LLVM and Clang.
    A CUDA build additionally requires CUDA Toolkit 12 or newer.
    A HIP build additionally requires HIP 6.2 or newer and the ROCm device libraries.
    If you want to use the MLIR backend at runtime, build Proteus with
    `-DPROTEUS_ENABLE_MLIR=ON`.

A typical build looks like this:

```shell
git clone https://github.com/LLNL/proteus.git
cd proteus
mkdir -p build && cd build
cmake -DLLVM_INSTALL_DIR=<llvm-install-prefix> -DCMAKE_INSTALL_PREFIX=<install-path> <other options> ..
make -j install
```

For example, a CUDA build with Clang can be configured with:

```shell
cmake -DLLVM_INSTALL_DIR=/path/to/llvm -DPROTEUS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_C_COMPILER=/path/to/llvm/bin/clang -DCMAKE_CXX_COMPILER=/path/to/llvm/bin/clang++ -DCMAKE_CUDA_COMPILER=/path/to/llvm/bin/clang++ -DCMAKE_INSTALL_PREFIX=<install-path> ..
```

## Testing

We recommend enabling tests when deploying Proteus on a new machine and running
them once to verify the installation.

Testing requires:

* configuring with `-DENABLE_TESTS=ON`
* `lit` to be installed and discoverable, either in the LLVM installation or in the active Python environment
* LLVM `FileCheck`

Typical test workflow:

```shell
cmake -DLLVM_INSTALL_DIR=<llvm-install-prefix> -DENABLE_TESTS=ON <other options> ..
make -j
ctest --output-on-failure
```

If you encounter bugs or issues, please let us know via the
[Github issue tracker](https://github.com/LLNL/proteus/issues).
