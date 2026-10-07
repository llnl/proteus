[![docs (gh-pages)](https://github.com/LLNL/proteus/actions/workflows/gh-pages-docs.yml/badge.svg)](https://github.com/LLNL/proteus/actions/workflows/gh-pages-docs.yml)
[![Build and test](https://github.com/LLNL/proteus/actions/workflows/ci-build-test.yml/badge.svg)](https://github.com/LLNL/proteus/actions/workflows/ci-build-test.yml)
[![codecov](https://codecov.io/github/LLNL/proteus/graph/badge.svg?token=MEB0M2D0AC)](https://codecov.io/github/LLNL/proteus)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# <img src="docs/assets/proteus-logo.png" width="128" align="middle" /> Proteus

Proteus is a programmable runtime specialization and Just-In-Time (JIT) layer
built on LLVM. It embeds into existing C++ codebases and accelerates host, CUDA,
and HIP applications by using runtime context to specialize code and enable
optimizations beyond static compilation.

## Description
Standard ahead-of-time (AOT) compilation can only optimize a program with the
information available at build time. Proteus goes further by embedding
optimizing JIT compilation directly into C/C++ applications.

Runtime context, such as the actual values of variables during execution, lets
it **specialize** code on the fly and apply advanced compiler optimizations that
accelerate performance beyond what static compilation allows.

Several frontends are available, depending on how you want to describe JIT code:

| Interface | Input style | Best for | Specialization model | Requires Clang AOT? |
| --- | --- | --- | --- | --- |
| **Code annotations** | Existing C/C++/CUDA/HIP code | Incremental adoption in existing applications | Values, arrays, objects, and launch configuration | Yes |
| **C++ frontend API** | C++ source strings | Runtime-generated C++ and templates | Values, arrays, objects, and launch configuration | No |
| **LLVM IR frontend API** | LLVM IR text or bitcode | Reusing externally generated LLVM IR with Proteus caching and dispatch | Encoded in the provided LLVM IR | No |
| **MLIR frontend API** | MLIR source strings | Direct access to MLIR lowering | Encoded in the provided MLIR source | No |
| **Embedded DSL API** | Programmatic builders | Runtime code generation with high-level constructs | Values, arrays, and launch configuration | No |

These frontends can target host, CUDA, and HIP execution paths, with backend
support depending on how Proteus was configured:

| Interface | Host | CUDA | HIP | Notes |
| --- | --- | --- | --- | --- |
| **Code annotations** | Yes | Yes | Yes | Requires compiling with Clang and uses the Proteus LLVM pass |
| **C++ frontend API** | Yes | Yes | Yes | Uses Clang by default; CUDA paths can use NVCC |
| **LLVM IR frontend API** | Yes | Yes | Yes | Accepts LLVM IR text or bitcode and compiles it directly through LLVM |
| **MLIR frontend API** | Yes | Yes | Yes | Requires `PROTEUS_ENABLE_MLIR=ON` |
| **Embedded DSL API** | Yes | Yes | Yes | Uses the LLVM backend by default; MLIR backend requires `PROTEUS_ENABLE_MLIR=ON` |

CUDA, HIP, and MLIR support are available when Proteus is built with the
corresponding configuration options enabled.

Proteus includes both in-memory and persistent caching, ensuring that once code
has been compiled and optimized, the cost of recompilation is avoided.

Proteus consists of an LLVM pass and a runtime library that implements JIT
compilation and optimization using LLVM as a library.

* The **code annotation** interface requires compiling your application with Clang so the Proteus LLVM pass can parse annotations.
* The **DSL**, **C++ frontend**, **LLVM IR frontend**, and **MLIR frontend** APIs don’t depend on which AOT compiler you use.

In all cases, you link your application against the Proteus runtime library.
Details are provided [later](#integrating-with-your-build-system).

## Installation
Python API users can install `proteus-python` from wheels. C++ users should
install Proteus from source or via [spack](https://github.com/spack/spack).

### Spack
We provide a packaging recipe for Spack in the subdirectory `packaging/spack`.

Assuming you have a Spack installation and preferably using an isolated Spack
environment, you can add the spack repo by cloning Proteus and then install it
by running:
```bash
git clone https://github.com/LLNL/proteus.git
spack repo add proteus/packaging/spack/spack_repo/proteus
spack install proteus
```

We provide several variants to match different configurations, including CUDA,
ROCm, and MPI support.
A complete list of variants and their descriptions is available in the Spack
package file, or viewable through:
```bash
spack info proteus
```

Some typical examples:
```bash
# Install the latest version with CUDA support for sm_90 arch.
spack install proteus +cuda cuda_arch=90

# Install the latest version with ROCm support for gfx942 arch.
spack install proteus +rocm amdgpu_target=gfx942

# Install the latest version with MPI support.
spack install proteus +mpi
```

### Building from source
The project uses `cmake` and requires an LLVM installation.
CI tests currently cover LLVM 19, 20, 22 with CUDA versions 12.2, and AMD
ROCm versions 6.4.3 (based on LLVM 19), 7.1.1 (based on LLVM 20), 7.2.0 (based
on LLVM 22).

See the top-level `CMakeLists.txt` for the available build options.
A typical build looks like this:
```
mkdir -p build && cd build
cmake -DLLVM_INSTALL_DIR=<llvm_install_path> -DCMAKE_INSTALL_PREFIX=<install_path> ..
make install
```

The `scripts` directory contains setup scripts for building on different targets
(host-only, CUDA, ROCm) used on LLNL machines.
They also serve as good starting points to adapt for other environments.
Run them from the repository root:
```bash
source scripts/setup-<target>.sh
```
These scripts load environment modules (specific to LLNL systems) and create a
`build-<hostname>-<target>-<version>` directory with a
working configuration.

### Python wheels
Proteus now publishes a thin `proteus-python` shim package plus backend wheels.
The default install is shim-only from PyPI. Install an explicit backend from
the Proteus wheel index.

The shim package provides the Python import surface and backend discovery. The
native payload lives in backend-specific wheels published outside PyPI.

| Install | Backend | Target | Required compiler/toolchain |
| --- | --- | --- | --- |
| `pip install proteus-python` | shim only | Python API only | none |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-host-llvm22` | `proteus-python-backend-host-llvm22` | Host CPU | LLVM/Clang 22.x |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-cuda12-llvm22` | `proteus-python-backend-cuda12-llvm22` | Host CPU + NVIDIA CUDA GPU | CUDA 12.x plus LLVM/Clang 22.x |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-rocm72` | `proteus-python-backend-rocm72` | Host CPU + AMD ROCm GPU | ROCm 7.2.x |

Typical stable installs:

```bash
python -m pip install proteus-python
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-host-llvm22
```

```bash
python -m pip install proteus-python
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-cuda12-llvm22
```

```bash
python -m pip install proteus-python
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-rocm72
```

See [docs/dev/python-wheel.md](docs/dev/python-wheel.md) for packaging and
release details.

## Integrating with your build system

### CMake
To integrate Proteus with CMake, add the install prefix to `CMAKE_PREFIX_PATH`,
or pass it explicitly with
`-Dproteus_DIR=<install_path>/<libdir>/cmake/proteus` during configuration
where `<libdir>` is typically `lib` or `lib64`.
Then, in your project's `CMakeLists.txt` add:
```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(<target>)
```

If you only need the DSL, C++ frontend, LLVM IR frontend, or MLIR frontend APIs, you can link directly against
`proteusFrontend`.
In this case, you don’t need to compile your target with Clang:

```cmake
find_package(proteus CONFIG REQUIRED)

target_link_libraries(<target> ... proteusFrontend ...)
```

### Make
With `make`, annotation-based integration requires adding compilation and
linking flags, for example:
```bash
CXXFLAGS += -I<install_path>/include -fpass-plugin=<install_path>/<libdir>/libProteusPass.so

LDFLAGS += -L<install_path>/<libdir> -Wl,-rpath,<install_path>/<libdir> -lproteus $(llvm-config --libs) -lclang-cpp
```
If you don't use code annotations, you can omit the `-fpass-plugin` option,
since the LLVM pass is only needed for processing annotations.

## Using

Proteus's core optimization technique is **runtime constant folding**.
It replaces runtime values with constants during JIT compilation, which in turn
turbo-charges classical compiler optimizations such as loop unrolling,
control-flow simplification, and constant propagation.
Think of it as doing `constexpr`, but at runtime.

Values that can be folded include function or kernel arguments, kernel launch
dimensions, launch bounds, and other runtime variables.

Choose the interface that matches how your application wants to describe JIT
work:

| Interface | Detailed guide |
| --- | --- |
| Code annotations | [Code Annotations](https://software.llnl.gov/proteus/user/annotations/) |
| C++ frontend API | [C++ Frontend API](https://software.llnl.gov/proteus/user/cpp-frontend/) |
| LLVM IR frontend API | [LLVM IR Frontend API](https://software.llnl.gov/proteus/user/llvmir-frontend/) |
| MLIR frontend API | [MLIR Frontend API](https://software.llnl.gov/proteus/user/mlir-frontend/) |
| DSL API | [DSL API](https://software.llnl.gov/proteus/user/dsl/) |

Proteus generates a unique specialization for each distinct set of runtime
values and caches them in memory and on disk, so JIT overhead is minimized
within and across runs.

## Documentation

The [Proteus documentation](https://software.llnl.gov/proteus/) has more extensive information,
including a user's guide and developer manual.

## Contributing

We welcome contributions to Proteus in the form of pull requests targeting the
`main` branch of the repo, as well as questions, feature requests, or bug reports
via issues.

## Code of Conduct

Please note that Proteus has a [Code of Conduct](CODE_OF_CONDUCT.md).
By participating in the Proteus community, you agree to abide by its rules.

## Authors
Proteus was created by Giorgis Georgakoudis, georgakoudis1@llnl.gov.

Key contributors are:
- David Beckingsale, beckingsale1@llnl.gov
- Konstantinos Parasyris, parasyris1@llnl.gov
- John Bowen, bowen36@llnl.gov
- Zane Fink, fink12@llnl.gov
- Tal Ben Nun, bennun2@llnl.gov
- Thomas Stitt, stitt4@llnl.gov

## License

Proteus is distributed under the terms of the Apache License (Version 2.0) with
LLVM Exceptions.

All new contributions must be made under the Apache-2.0 with LLVM Exceptions license.

See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

LLNL-CODE-2000857
