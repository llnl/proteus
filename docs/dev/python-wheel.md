# Building the Python Wheels

Proteus now builds Python wheels as a package family rather than a single
monolithic wheel:

- `proteus-python`
  - pure-Python shim package
  - installs no native backend by default
- `proteus-python-backend-host-llvm22`
  - native host backend wheel for Linux and macOS
- `proteus-python-backend-cuda12-llvm22`
  - native CUDA backend wheel for Linux `x86_64`
- `proteus-python-backend-rocm72`
  - native ROCm backend wheel for Linux `x86_64`

The user-facing import surface remains:

```python
import proteus
```

At runtime, the shim discovers installed backend entry points and prefers the
highest-priority compatible backend:

1. explicit `PROTEUS_BACKEND_VARIANT`
2. explicit `PROTEUS_BACKEND_KIND`
3. runtime-compatible GPU backend
4. host fallback

## Packaging Model

The shim package is built from the repository root with `setuptools` and is the
only Python distribution published to PyPI/TestPyPI. Backend wheels are built
from dedicated subprojects, uploaded as GitHub Release assets, and exposed
through a static PEP 503 simple index on GitHub Pages:

- release channel: `https://software.llnl.gov/proteus/wheels/simple/`
- prerelease channel: `https://software.llnl.gov/proteus/wheels/test/`

The backend packages are built from dedicated subprojects:

- `packaging/python/backend-host`
- `packaging/python/backend-cuda`
- `packaging/python/backend-rocm`

Those backend projects use `scikit-build-core` and the repository CMake build,
but they install their native payload into backend-local packages instead of
directly under `proteus/`:

- `proteus_backend_host/`
- `proteus_backend_cuda/`
- `proteus_backend_rocm/`

Each backend package contains:

- backend-local `_proteus.*`
- backend-local `libproteus.*`
- a generated `manifest.json` describing the concrete built variant
- repaired vendored LLVM/Clang shared libraries when required by that backend
- backend registration metadata under the `proteus.backends` entry-point group

The distribution names encode the supported toolchain family for that wheel
line. Today that means:

- host backend: LLVM 22
- CUDA backend: CUDA 12 + LLVM 22
- ROCm backend: ROCm 7.2

The shim package exports:

- `proteus.active_backend`
- `proteus.active_backend_variant`
- `proteus.available_backends()`
- `proteus.available_backend_variants()`
- the selected backend's native API re-exported from `proteus`

## Build-Time Requirements

All backend wheel builds require:

- Python with `build`
- CMake
- `pybind11`
- `scikit-build-core`
- a usable LLVM/Clang installation discovered via `LLVM_INSTALL_DIR`

Platform-specific requirements:

- macOS host backend:
  - Homebrew LLVM 22
  - `delocate`
- Linux host backend:
  - the LLVM-enabled `manylinux_2_28` container
  - `auditwheel`
- Linux CUDA backend:
  - a local CUDA image derived from the LLVM-enabled `manylinux_2_28` container
  - CUDA Toolkit 12 with `libnvptxcompiler_static.a`
  - `auditwheel`
- Linux ROCm backend:
  - a local ROCm image derived from stock `manylinux_2_28`
  - ROCm HIP development toolchain
  - `auditwheel`

## Local Builds

### Shim Wheel

```bash
python3 -m venv /tmp/proteus-wheel-venv
/tmp/proteus-wheel-venv/bin/python -m pip install -U pip build setuptools setuptools-scm
/tmp/proteus-wheel-venv/bin/python -m build --wheel --outdir dist .
```

### Host Backend Wheel on macOS arm64

```bash
python3 -m venv /tmp/proteus-wheel-venv
/tmp/proteus-wheel-venv/bin/python -m pip install -U pip build scikit-build-core pybind11 delocate

brew install llvm@22

MACOSX_DEPLOYMENT_TARGET=14.0 \
LLVM_INSTALL_DIR="$(brew --prefix llvm@22)" \
  /tmp/proteus-wheel-venv/bin/python -m build --wheel \
  --outdir wheelhouse \
  packaging/python/backend-host
```

### Host Backend Wheel on Linux

```bash
bash packaging/python/image-scripts/build-manylinux-llvm-container.sh

python -m pip install -U pip build cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=ghcr.io/olympus-hpc/proteus-manylinux-llvm:22.1.3 \
  python -m cibuildwheel packaging/python/backend-host --output-dir wheelhouse
```

### CUDA 12 Backend Wheel on Linux

```bash
BASE_IMAGE=ghcr.io/olympus-hpc/proteus-manylinux-llvm:22.1.3 \
CUDA_VERSION=12.4.1 \
CUDA_MAJOR_MINOR=12-4 \
  bash packaging/python/image-scripts/build-manylinux-cuda-container.sh

python -m pip install -U pip build cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=proteus-manylinux-cuda-local:12.4.1 \
  LLVM_INSTALL_DIR=/opt/llvm-22.1.3 \
  CUDA_HOME=/usr/local/cuda \
  PATH=/opt/llvm-22.1.3/bin:/usr/local/cuda/bin:$PATH \
  python -m cibuildwheel packaging/python/backend-cuda --output-dir wheelhouse
```

### ROCm Backend Wheel on Linux

```bash
ROCM_VERSION=7.2.1 \
ROCM_EL_VERSION=8 \
  bash packaging/python/image-scripts/build-manylinux-rocm-container.sh

python -m pip install -U pip build cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=proteus-manylinux-rocm-local:7.2.1 \
  LLVM_INSTALL_DIR=/opt/rocm/llvm \
  ROCM_PATH=/opt/rocm \
  PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH \
  python -m cibuildwheel packaging/python/backend-rocm --output-dir wheelhouse
```

## Linux Container Images

The Linux wheel workflows use one maintained base image and build GPU-specific
images locally inside CI.

Host backend image inputs:

- `packaging/python/image-scripts/manylinux-llvm.Dockerfile`
- `packaging/python/image-scripts/build-llvm-manylinux.sh`
- `packaging/python/image-scripts/build-manylinux-llvm-container.sh`

CUDA backend image inputs:

- `packaging/python/image-scripts/manylinux-cuda.Dockerfile`
- `packaging/python/image-scripts/build-manylinux-cuda-container.sh`

ROCm backend image inputs:

- `packaging/python/image-scripts/manylinux-rocm.Dockerfile`
- `packaging/python/image-scripts/build-manylinux-rocm-container.sh`

Current CI image strategy:

- maintain `ghcr.io/olympus-hpc/proteus-manylinux-llvm:<llvm-version>`
- derive the CUDA image locally per CI run by adding the requested CUDA toolkit
- derive the ROCm image locally per CI run by installing the requested ROCm
  HIP development toolchain into stock `manylinux_2_28`

## Installed-Wheel Verification

Do not treat build-tree imports as sufficient. Install from built wheels into a
clean environment.

### Published Install Commands

Stable shim install from PyPI:

```bash
python -m pip install proteus-python
```

Stable backend installs from the GitHub Pages simple index:

```bash
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-host-llvm22
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-cuda12-llvm22
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ \
  proteus-python-backend-rocm72
```

Prerelease shim install from TestPyPI:

```bash
python -m pip install --pre --index-url https://test.pypi.org/simple/ proteus-python
```

Prerelease backend installs from the prerelease simple index:

```bash
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/test/ \
  proteus-python-backend-host-llvm22
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/test/ \
  proteus-python-backend-cuda12-llvm22
python -m pip install --index-url https://software.llnl.gov/proteus/wheels/test/ \
  proteus-python-backend-rocm72
```

### Default Host Install

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links dist \
  --find-links wheelhouse \
  dist/proteus_python-*.whl \
  proteus-python-backend-host-llvm22

export PROTEUS_CLANGXX_BIN=/path/to/clang++-22

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_selection.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wildcard_import.py
```

### Shim-Only Install

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install dist/proteus_python-*.whl
```

Validate that backend discovery reports no installed native backend:

```bash
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader_zero_backends.py
```

### CUDA Backend Install

Install the shim first, then the CUDA backend wheel from the local wheelhouse:

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links dist \
  --find-links wheelhouse \
  dist/proteus_python-*.whl \
  proteus-python-backend-cuda12-llvm22==<version>
```

On CPU-only machines, validate the host path:

```bash
export PROTEUS_CLANGXX_BIN=/path/to/clang++-22

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_selection.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wildcard_import.py
```

On Linux systems with an NVIDIA GPU and driver, additionally validate:

```bash
export CUDA_HOME=/usr/local/cuda

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_launch_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_pointer_validation.py
```

### ROCm Backend Install

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links dist \
  --find-links wheelhouse \
  dist/proteus_python-*.whl \
  proteus-python-backend-rocm72==<version>
```

Validate the installed-wheel path:

```bash
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_selection.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wildcard_import.py
```

## Runtime Contract

### Host Backend

- ships Proteus and the LLVM/Clang runtime libraries it links against
- does not ship a host C++ compiler toolchain
- still requires a host `clang++` whose major version matches the bundled
  LLVM/Clang runtime for `frontend="cpp", target="host"`
- for the current wheel line, use `clang++-22` or equivalent and set
  `PROTEUS_CLANGXX_BIN` when it is not the default `clang++` on `PATH`
- a mismatched `clang++` may fail while resolving builtin/system headers during
  the in-process frontend compile

| Install | Backend | Target | Required compiler/toolchain |
| --- | --- | --- | --- |
| `pip install proteus-python` | shim only | Python API only | none |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-host-llvm22` | `proteus-python-backend-host-llvm22` | Host CPU | LLVM/Clang 22.x |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-cuda12-llvm22` | `proteus-python-backend-cuda12-llvm22` | Host CPU + NVIDIA CUDA GPU | LLVM/Clang 22.x, plus CUDA 12.x |
| `pip install --index-url https://software.llnl.gov/proteus/wheels/simple/ proteus-python-backend-rocm72` | `proteus-python-backend-rocm72` | Host CPU + AMD ROCm GPU | ROCm 7.2.x |

### Backend Selection

- backend kinds are `host`, `cuda`, and `rocm`
- each installed backend wheel reports exactly one built variant id through
  `proteus.active_backend_variant`
- the current CI wheel line reports:
  - `host_llvm22`
  - `cuda12_llvm22`
  - `rocm72`
- use `PROTEUS_BACKEND_KIND` to force the backend kind when multiple backend
  wheels are installed
- use `PROTEUS_BACKEND_VARIANT` to require a specific installed variant id
- once a native backend is loaded, Proteus locks selection for the process and
  rejects attempts to switch to a different backend kind or variant

### CUDA Backend

- is a superset backend: host functionality remains available
- does not vendor `libcuda.so`, `libcudart`, or the CUDA Toolkit
- requires an installed NVIDIA driver for CUDA functionality
- requires a matching CUDA 12 toolkit root for runtime compilation
- for `frontend="cpp", target="cuda", compiler="nvcc"`, also requires an
  `nvcc` executable

CUDA toolkit resolution remains:

1. `PROTEUS_CUDA_HOME`
2. `CUDA_HOME`
3. `CUDA_PATH`

NVCC discovery for `compiler="nvcc"` remains:

1. `PROTEUS_NVCC_BIN`
2. `PATH`

For wheel-targeted CUDA builds on Linux, Proteus now resolves `libcuda.so.1`
with `dlopen()` at runtime instead of linking the wheel directly against the
CUDA driver shared library. This keeps the wheel compatible with manylinux
repair and lets host-only functionality work on machines without an NVIDIA
driver.

### ROCm Backend

- is a superset backend: host functionality remains available
- does not vendor the kernel driver
- requires a matching ROCm installation for HIP functionality

ROCm toolkit resolution remains:

1. `PROTEUS_ROCM_PATH`
2. `ROCM_PATH`
3. `/opt/rocm`

## CI Workflow

The wheel workflow is defined in `.github/workflows/ci-wheels.yml`.

It runs on:

- pull requests and pushes for wheel CI
- `workflow_dispatch` for manual runs
- GitHub releases for publishable builds

It produces six artifact groups:

- `wheels-shim`
- `sdist-shim`
- `wheels-host-linux`
- `wheels-host-macos`
- `wheels-cuda-linux`
- `wheels-rocm-linux`

Current target matrix:

- shim: pure Python
- host backend:
  - `macOS arm64`
  - `manylinux_2_28 x86_64`
- CUDA backend:
  - `manylinux_2_28 x86_64`
- ROCm backend:
  - `manylinux_2_28 x86_64`

The workflow builds all backend wheels with `cibuildwheel`, then performs
explicit installed-wheel validation using the newly built shim and backend
wheels from the local artifact directories.

On GitHub prereleases, the workflow publishes only the `proteus-python` shim
artifacts to TestPyPI. On full GitHub releases, it publishes only the
`proteus-python` shim artifacts to PyPI.

Backend wheels are uploaded as GitHub Release assets and exposed through the
GitHub Pages simple indexes:

- release channel: `https://software.llnl.gov/proteus/wheels/simple/`
- prerelease channel: `https://software.llnl.gov/proteus/wheels/test/`

The Pages publishing model is split by ownership:

- `.github/workflows/gh-pages-docs.yml`
  - updates only the `main` docs version on the machine-owned `gh-pages` branch
  - does not regenerate `/wheels/*`
- `.github/workflows/ci-wheels.yml`
  - on release or prerelease, publishes the released tag docs
  - on full releases, removes prior prerelease docs from `gh-pages`
  - updates the `latest` or `prerelease` alias for the new tag
  - regenerates `/wheels/simple/` and `/wheels/test/`
  - pushes normal commits to `gh-pages`

The remote `gh-pages` branch must exist before either Pages publishing workflow
can update docs. Create it manually once before enabling the workflows.

Trusted Publishing must be configured on both PyPI and TestPyPI for:

- `proteus-python`

Each publisher must trust:

- repository: `LLNL/proteus`
- workflow: `.github/workflows/ci-wheels.yml`
- environment: `testpypi` or `pypi`

Publishable releases must come from tags that resolve to public PEP 440
versions under `setuptools_scm`, for example:

- `v2026.03.1`
- `v2026.03.1rc1`
