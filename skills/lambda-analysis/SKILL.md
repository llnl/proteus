---
name: lambda-analysis
description: Diagnose, extend, or test Proteus GPU lambda kernel-argument provenance analysis, especially LLVM IR pointer, memory, and control-flow shapes. Use for changes under KernelArgVisitor.h and KernelArgPtrUseVisitor.h; do not use for generic GPU kernel debugging.
---

# Lambda Analysis

Use this skill when changing the analysis that maps a registered lambda capture
back to a kernel argument and byte offset.

## Core model

The analysis is intentionally conservative: it should specialize only when it
can identify one kernel argument and one byte offset. A missed specialization
is preferable to specializing stale or ambiguous data.

- `src/pass/KernelArgVisitor.h` performs the backward provenance walk from a
  lambda wrapper call to a kernel argument. It tracks byte `Offset` through
  GEPs, aggregates, calls, loads, PHIs, selects, and memory transfers.
- `src/pass/KernelArgPtrUseVisitor.h` resolves a local pointer/address to its
  dominating write. It walks forward over uses because the main visitor has
  reached an allocation or pointer transform whose contents determine the
  provenance.
- `LambdaPtrUseAnalysis::Offset` is a correction consumed by callers that do
  `Offset -= Result.Offset`. For a transfer from destination base `D` to source
  base `S`, return `D - S`, not the source absolute offset.

Read both headers before changing either: the offset contract crosses their
boundary.

## Pointer loads: distinguish spills from program fields

A compiler pointer spill is a temporary local slot such as:

```llvm
%slot = alloca ptr
store ptr %value, ptr %slot
%reload = load ptr, ptr %slot
```

Use CFG-aware reaching-store recovery only for a load whose underlying storage
is an `alloca ptr`. `isPointerSpillLoad` encodes this distinction.

For a normal pointer field load, such as:

```llvm
%field = getelementptr ..., ptr %context, ..., i32 3
%body = load ptr, ptr %field
```

continue backward through `%field`; do not search for a local reaching store.
This is required by RAJA/MFEM-style context objects. The dedicated regression
is `tests/gpu/lambda_pointer_field_load.cpp`.

For actual spill slots, collect the closest pointer store along every incoming
CFG path. Continue only if all complete paths resolve recursively to the same
source pointer. Different sources, cycles, and incomplete coverage must not
be guessed.

## Ordering and memory semantics

`Value::users()` does not provide execution order. When resolving a local
capture, `getPreviousCoveringStore` scans backward in the same block from the
use boundary and selects the closest store covering the tracked byte. It must
not skip a possible intervening clobbering call.

Memory-transfer rules:

- A `memcpy`/`memmove` reached through its destination defines the destination
  bytes only if a constant length covers the tracked byte; trace to the source.
- Reaching it through its source is only a read.
- A dynamic-length transfer, a covering memset, or a dynamic-length memset is
  conservatively unsupported.
- A non-covering constant memset/transfer does not affect the tracked byte.

## Control flow and calls

- For PHIs and selects, analyze all incoming arms and accept only equal kernel
  function, argument index, byte offset, and changed runtime-constant layout.
- For indirect calls or declarations, fail cleanly and emit a debug reason;
  never dereference a null `getCalledFunction()`.
- Pointer-returning helper calls require composing offsets across caller and
  callee. Do not rely on the first return block when paths may disagree.
- Preserve the root lambda-call endpoint to prevent forward def-use traversal
  from cycling into the original wrapper invocation.

## Tests and IR-shape-first workflow

For every analysis change, use this agent workflow:

1. Identify the exact LLVM IR shape that fails or is currently unsupported.
   Inspect the pass trace and IR rather than inferring the shape from C++.
2. Add the smallest targeted HIP C++ regression that emits that shape. It must
   fail before the implementation change and assert the intended specialization
   or conservative decline with FileCheck.
3. Propose and make the narrowest change that handles that shape. Do not widen
   a rule from one instruction class to all pointer loads, stores, or calls
   without a matching provenance argument and test.
4. Build the new test, then run it with the relevant existing lambda tests and
   `lambda_mfem_style_launch` to catch interactions with established shapes.

Keep a positive case distinct from a conservative-decline case by using
different registered-lambda closure types; one failed analysis can otherwise
suppress specialization for a shared type.

Existing focused tests:

- `lambda_provenance_shapes.cpp`: nested insertvalue, GEP composition,
  interior returned pointers, and memcpy source/destination offsets.
- `lambda_store_order.cpp`: latest store, pointer spill ordering, branches,
  and capture overwrites.
- `lambda_call_shapes.cpp`: divergent/same return paths, indirect calls, and
  select-shaped routing.
- `lambda_memory_intrinsic_shapes.cpp`: memset, dynamic memcpy, and offset
  memmove.
- `lambda_pointer_field_load.cpp`: aggregate pointer field versus `alloca ptr`
  spill classification.
- `lambda_mfem_style_launch.cpp`: required MFEM-style regression guard.

Register a new GPU test in `tests/gpu/CMakeLists.txt`. Use FileCheck to assert
both observable output and whether `[LambdaSpec]` must be present or absent.

## Diagnostics and validation

Every newly introduced conservative failure must log a specific reason through
`DEBUG(Logger::logs("proteus-pass") << ...)`. Both visitors need a catch-all
`visitInstruction` diagnostic that includes the opcode and instruction.

Build and run only the affected tests first, then include MFEM:

```bash
cmake --build build-current-rocm-6.4.3 --target <targets> -j 16
ctest --test-dir build-current-rocm-6.4.3 \
  -R '^lambda_(...|mfem_style_launch)\.HIP$' --output-on-failure --timeout 30
```

GPU test execution needs the active allocation. When debugging a failure,
clear the test cache and use `PROTEUS_DEBUG_OUTPUT=1`; inspect the pass trace
around the first analysis warning and the corresponding LLVM IR, rather than
inferring provenance from source alone.
