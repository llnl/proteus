# Design decision: provenance horizon for slot-assembled closures

Status: DECIDED 2026-10-15 -- option A (see "Resolution" below). Revisit only
if soundness on IR that never passed the AST checker becomes a product
requirement. Owner: lambda kernel-arg analysis (PR #526 and successors).
Context anchors: `src/pass/KernelArgPtrUseVisitor.h` (forward use walk,
candidate reconciliation), `src/pass/KernelArgVisitor.h` (backward walk,
kernel-arg termination), `tests/gpu/lambda_mfem_style_launch.cpp`,
`tests/gpu/lambda_interior_slot_overwrite.cpp`, `tests/gpu/lambda_size_test.cpp`.

## The invariant and why it is a trilemma

Runtime specialization folds **every registered slot** of a closure from the
resolved location `(kernel argument K, byte offset O + slot offset)`. That is
correct iff, at the callsite, each slot byte's reaching definition traces back
to `K` at the offset-consistent position. The analysis, however, resolves
**one** use-def chain (the "witness": the chain defining the tracked byte),
because MemorySSA answers "which single access clobbers this location" and the
result type (`LambdaPtrUseAnalysis::DominatingWrite`) is one value.

A closure assembled slot-wise has N chains, one per slot. Inside the frame
performing assembly, a **legitimate parallel copy** (every slot stored from
the same source object) and a **foreign interior write** (one slot stored
from a different object) are the *same IR shape*: field stores whose value
operands have different SSA roots. The distinction exists only after
resolving each chain backward — in the limit, all the way to kernel
arguments. Hence every design must choose a **provenance horizon**: how far
back "all slots share one source" is proven before specializing.

Two shapes pin the ends of the spectrum:

- `lambda_mfem_style_launch` (production-critical, must specialize): closure
  assembled by per-slot stores whose values arrive as separately
  SROA-decomposed scalars; after inlining/SROA the common source may only be
  provable at the kernel-argument level.
- `lambda_interior_slot_overwrite` (miscompile, must not specialize): slot 0
  re-stored from the authoritative argument, slot 1 stored from a donor
  argument via type-erased pointers. Currently specializes (pinned
  expected-fail).

## The options

### A. Assume — witness resolution plus frontend contract (current default)

Candidate collection admits only writes covering the tracked byte; the
witness chain is resolved; all other slots are *assumed* to be parallel
copies of the same source. The functor extent (`proteus.functor_size`, from
`sizeof` in the wrapper annotation) is consumed only by targeted guards:
extent-bounded fallback query locations (fixes spurious declines when writes
land past the functor; `lambda_size_test`), full-extent-or-decline memory
transfers (the wrong-`sizeof` partial-copy bug), memset overlap declines.

- Soundness boundary: the **frontend**. The assumption holds for all C++
  reachable without type erasure (captures are private; the AST checker
  forbids mutation and type-erased access). Required remaining work: the
  `LambdaASTChecker` diagnostic for type-erased pointer writes into
  wrapper-containing storage, `mem*` builtins on wrapper storage, and unions
  over wrappers; `lambda_interior_slot_overwrite` then becomes an AST
  rejection test.
- Residual exposure: IR that never passed the checker — LTO-linked foreign
  bitcode, non-Clang frontends, hand-written or JIT-composed IR. (Optimizer
  transforms are not a realistic exposure: splitting/merging stores of the
  *same* source preserves the assumption; synthesizing a store from a
  different object is not a legal transform.)
- Cost: none at compile time, no completeness loss.

### B. Within-frame proof — self-contained, loses shapes without a local root

Widen candidate collection to extent-overlap, then accept only candidate
sets whose destinations partition the extent and whose value operands share
one SSA root at consistent offsets (`getValueBaseWithConstantOffset`
partition in `resolveCollectedClobbers`). The analysis becomes
self-contained: the interior test declines with no frontend help.

- What survives: assembly whose slot values extract from one in-frame SSA
  aggregate (the enclosure shape in `lambda_size_test`; factory-style field
  loads from one source object).
- What declines: assembly whose pieces arrive as **separate formal
  parameters or SROA'd scalars** — no in-frame common root exists even
  though the caller passed one object. This is precisely the mfem/RAJA
  lowering at -O1+, i.e. the flagship production integration. Losing it is
  presumed unacceptable (`skills/lambda-analysis/SKILL.md` lists
  `lambda_mfem_style_launch` as the required regression guard).
- Note: the current tip already implements this partition for the case where
  *multiple byte-0-covering stores* collide; the open question is only about
  widening collection to the full extent, which is what drags in the
  no-local-root shapes.

### C. Kernel-argument-level proof — per-piece merge (complete, expensive)

When the extent is assembled piecewise, resolve **each piece's** value
operand backward independently (the existing `LambdaArgVisitor` walk) to a
`(KernelFunction, ArgIndex, Offset)` and accept iff all pieces agree on
function and argument with offsets affine-consistent with their destination
displacement — the same merge rule `visitPHINode` applies to control-flow
arms, applied to spatial pieces. mfem passes by proof (all pieces reach one
kernel argument); interior declines by proof (pieces reach different
arguments).

- Cost: N backward walks per callsite (each potentially interprocedural),
  forward→backward plumbing for piece sets (prototyped as
  `PartialEntityWrite`/`resolvePartialEntityWrites`, rolled back 2026-10-13
  after regressions), piece-level failure modes (any piece declining declines
  the closure), work-budget interaction. The two prior regressions came from
  pairing this with extent-wide *anchoring*; the merge itself was never fully
  wired. Treat the cost as real but unmeasured.
- Benefit: the only option that is simultaneously self-contained and keeps
  mfem. Also subsumes the within-frame partition as its cheap first tier.

## Decision drivers

1. How much do we trust the frontend boundary? If Proteus must stay sound on
   IR that never saw the AST checker (LTO'd libraries, other frontends),
   option A is insufficient *by policy* regardless of its practical safety.
2. Compile-time budget per lambda callsite (C multiplies backward walks).
3. mfem/RAJA coverage is non-negotiable (rules out B as an end state).
4. Decline determinism: A and C give stable outcomes; B's outcome depends on
   how far SROA happened to run.

## Resolution (2026-10-15): option A

A, hardened: witness resolution + extent guards + in-frame partition for
byte-0 collisions, with interior-store soundness delegated to the frontend
contract (AST diagnostic still TODO).

Rationale. The decisive observation is that C re-proves what A assumes
rather than accepting anything new: mfem-style shapes already specialize
under A because the witness chain resolves and the parallel slots ride the
parallel-copy assumption. C's only marginal value is *checking* that
assumption -- and the programs that would fail the check are ones that
dissect closure memory and reassemble it from different kernel arguments
through type erasure. No portability layer does this (RAJA/Kokkos/mfem
forward closures whole, by value; that is the point of their abstractions),
the Proteus usage contract already forbids it, and in C++ the closure
interior is unreachable without the type erasure the AST checker flags. C
would therefore price N interprocedural walks per callsite, on every compile
of the hot production path, to defend against code nobody writes and the
contract excludes. B remains ruled out by driver 3 (loses mfem).

Framing for the PR: this analysis work buys no new production acceptance and
is not meant to -- its value is that every known unsoundness changed failure
category, from silent wrong-constant miscompiles in production kernels to
conservative declines with logged reasons (or frontend errors), plus
compile-time guarantees (no hangs, bounded work, deterministic outcomes
across compiler versions). The one genuinely production-plausible new
protection is the wrong-sizeof partial-memcpy guard.

Independent of this decision, the in-frame partition must still enforce
**affine delta consistency** (equal `ValueOffset - DestOffset` across
candidates), without which a field-swapping copy from one source passes the
same-base check with transposed slots.

Changing the position requires updating: this file, gate 8 in
`docs/dev/reviews/clobbering-aware-use-analysis.md`, the NOTE at
`getPreciseTrackedLocation`, and the final form of
`lambda_interior_slot_overwrite` (AST rejection test under A; IR decline
test under B/C).
