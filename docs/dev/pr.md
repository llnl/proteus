# Creating a PR

This page describes the expected workflow for contributing changes to Proteus
through a pull request.
It is aimed at contributors who already have the repository checked out and want
clear guidance on how to get a change ready for review.

## Start Point

Proteus accepts pull requests against the `main` branch.
If you are planning a large refactor, a new interface surface, or a behavior
change that spans host and GPU paths, it is usually best to open an issue or
start a design discussion first so the implementation direction is aligned
before a large patch is prepared.

Before you start coding, it helps to identify which part of the system you are
changing:

- `src/pass` for annotation discovery and instrumentation
- `src/runtime` for specialization, dispatch, caching, and execution
- `src/runtime/Frontend` for the DSL, C++ frontend, and MLIR frontend implementations
- `include/proteus` for public headers and API surface
- `docs/` for user and developer documentation

For architecture context, read the [concepts guide](concepts.md).
For interface behavior, read the [user interface overview](../user/interface.md)
and the relevant user-guide subpage.

## Before Opening a PR

A good PR is usually narrow in scope and easy to validate.
Before opening one, try to make sure the branch does all of the following:

- solves one coherent problem or introduces one coherent feature
- includes tests for behavior changes when practical
- updates documentation when user-visible behavior, configuration, or public
  APIs change
- keeps unrelated cleanup out of the same branch

When you add a new docs page, remember to update `mkdocs.yml` so it appears in
site navigation.

## Build and Test Expectations

Proteus CI builds and tests pull requests on Linux across multiple LLVM
versions and both shared and static builds.
There is also a C++ linter workflow that checks formatting on changed lines.

You do not need to reproduce the full CI matrix locally, but you should run the
most relevant local checks you can for the area you touched.
A common local configuration is:

```bash
mkdir -p build
cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_TESTS=on \
  -DLLVM_INSTALL_DIR=$(llvm-config --prefix)
make -j
ctest --output-on-failure
```

A few practical notes:

- if you touch pass or runtime behavior, run the relevant `ctest` coverage for
  CPU, GPU, frontend, or integration tests as appropriate
- if you touch MLIR-specific code, test with `-DPROTEUS_ENABLE_MLIR=ON` when
  possible
- if you touch CUDA or HIP paths, say clearly in the PR description what target
  you tested and what you could not test locally
- if your change is documentation-only, say that explicitly so reviewers know a
  code-path validation gap is expected

The test tree lives under `tests/`, with coverage split across areas such as
`tests/cpu`, `tests/gpu`, `tests/frontend`, and `tests/integration`.

## Formatting and Code Hygiene

Proteus has repository formatting configuration in `.clang-format` and
`.clang-tidy`.
CI currently enforces `clang-format` checks on changed C++ lines, so it is worth
formatting touched code before you push.

For a quicker local feedback loop, the repository also includes a
`.pre-commit-config.yaml` with lightweight checks and a `clang-format` hook for
staged C and C++ files.
If you want to use it locally:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The local `clang-format` hook uses the `clang-format` executable available in
your `PATH` and applies the repository style from `.clang-format`.
Keep hook usage lightweight: formatting and hygiene checks are a good fit, while
full builds and test coverage should still be run manually or in CI.

In practice, try to keep a PR easy to read:

- prefer focused commits over one mixed change that combines refactors and
  behavior updates
- avoid drive-by renames or formatting-only churn in unrelated files
- keep public API changes and internal refactors clearly explained in the PR
  description
- update examples and docs when they would otherwise become misleading

## Writing the PR Description

A strong PR description saves reviewer time.
At minimum, include:

- what changed
- why the change is needed
- what you tested
- any target-specific limitations or untested paths
- any follow-up work that is intentionally left out

If the change affects specialization behavior, dispatch, caching, or compiler
integration, it helps to call that out directly so reviewers know which system
interactions to examine closely.

## Review Readiness

Before marking the PR ready for review, do a final pass for the questions a
reviewer is likely to ask:

- is the behavior change covered by tests or clearly justified if not
- are user-facing docs still accurate
- are host, CUDA, HIP, and MLIR implications addressed where relevant
- does the branch introduce configuration or build requirements that need to be
  documented
- is the diff scoped tightly enough to review efficiently

If a PR is intentionally partial, say so explicitly.
That is much easier to review than a branch that appears complete but leaves
important paths unexplained.

## Project Policies

Please keep in mind the project-level policies already called out in the main
repository documentation:

- pull requests should target `main`
- contributions are made under the Apache-2.0 with LLVM Exceptions license
- contributors are expected to follow the project's [Code of Conduct](https://github.com/LLNL/proteus/blob/main/CODE_OF_CONDUCT.md)

## Suggested Reading Path

For new contributors, a useful order is:

1. [concepts.md](concepts.md) for architecture
2. [interface.md](../user/interface.md) for the user-facing entry points
3. [config.md](../user/config.md) for runtime controls and cache-related knobs
4. [api.md](api.md) for symbol-level detail
