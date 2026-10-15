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

## Appendix: searching for the A-accepts / C-rejects miscompile (2026-10-15)

The decisive question for option A was whether **strictly-conforming C++**
(no `reinterpret_cast`, no byte pointers, no `memcpy` with a wrong length)
can lower to an IR shape that the witness model accepts while per-piece
kernel-arg merging would reject -- i.e. an executed closure whose bytes come
from two different kernel arguments. We attempted to construct it
empirically (LLVM 19.1.7, `-O1`, amdgcn device pipeline, pseudo-GPU
harness). Result: **every attempt either declines conservatively or is
inexpressible**; no accepting splice was found.

### The IR shape we are trying (and failing) to produce

A frame in which the executed closure's storage receives *partial* writes
whose values trace to different kernel arguments, with the byte-0 chain
resolving cleanly:

```llvm
; HYPOTHETICAL -- never produced by conforming C++
define void @helper(ptr %fromArg0, ptr %fromArg1) {
  %closure = alloca %class.anon            ; 2 x i32 registered slots
  %slot0 = load i32, ptr %fromArg0
  store i32 %slot0, ptr %closure           ; byte 0 <- kernel arg 0 (witness)
  %p1 = getelementptr %class.anon, ptr %closure, i32 0, i32 1
  %slot1src = getelementptr i8, ptr %fromArg1, i64 4
  %v1 = load i32, ptr %slot1src
  store i32 %v1, ptr %p1                   ; byte 4 <- kernel arg 1 (foreign)
  call void @operator()(ptr %closure)      ; witness accepts; slot 1 is stale
  ret void
}
```

The analysis resolves the byte-0 store to kernel argument 0 and specializes
both slots from it; slot 1 executes with argument 1's bytes. This shape IS
reachable with type erasure (`tests/gpu/lambda_interior_slot_overwrite.cpp`
produces exactly it via `reinterpret_cast<uint32_t*>`), which is why the
frontend contract exists. The question was whether it is reachable without.

### Why conforming C++ cannot emit it

Closure members are unnamable and private; the only writers of closure
storage the language offers are the closure's own special member functions,
and those copy **whole objects**. Field-granular heterogeneity -- the load
at `%slot1src` above -- requires naming an interior byte of a closure, which
requires erasure. Corroborating friction encountered while building the
probes: the registered wrapper's copy-assignment is implicitly deleted
(reference-capturing lambdas delete it; our probe needed placement-new just
to overwrite union storage), and device-side placement new/delete had to be
hand-declared. The object model actively resists even whole-object
overwrites.

### Probe 1: union pun (the only cast-free storage aliasing)

```cpp
template <typename F> union Pun {
  F A;
  F B;
  __device__ Pun(const F &Init) : A(Init) {}
  __device__ ~Pun() {}
};

template <typename F>
__device__ __attribute__((noinline)) static void punInvoke(F X, F Y, bool Flip) {
  Pun<F> U(X);                              // whole copy of kernel arg X
  if (Flip)
    ::new (static_cast<void *>(&U.B)) F(Y); // whole copy of kernel arg Y
  U.A();                                    // UB read of inactive member
}
```

Lowered IR of the overwrite (unconditional variant), demonstrating the
whole-object property -- note both writes cover the full 8-byte closure:

```llvm
call void @_ZN3Pun...C2ERKS4_(ptr %11, ptr %8)   ; ctor: whole copy of X
call void @llvm.memcpy.p0.p0.i64(ptr align 4 %11, ptr align 4 %9,
                                 i64 8, i1 false) ; whole copy of Y
call void @...clI...(ptr %11)                     ; invoke
```

Outcomes:

- Conditional overwrite (`if (Flip)`): **declines** -- the reaching
  definition is path-dependent, MemorySSA returns a MemoryPhi, and the
  analysis hard-fails on Phi.
- Unconditional overwrite: **declines** -- the ctor call plus covering
  memcpy form a candidate set whose reconciliation fails ("selected clobber
  is not an instruction").
- Structurally, even a mis-acceptance here could not splice: the union
  selects *which argument's whole bytes* win, never *which bytes from which
  argument*. Either outcome is single-source-consistent.

### Probe 2: aggregate-of-closures member splice (control)

```cpp
template <typename F> struct PairBox { F First; F Second; };

template <typename F>
__device__ __attribute__((noinline)) static void
pairInvoke(PairBox<F> P, PairBox<F> Q) {
  PairBox<F> Mixed{P.First, Q.Second};  // member-wise whole-closure copies
  Mixed.Second();
}
```

**Specializes, correctly.** The enclosing aggregate is spliced from two
kernel arguments, but each *closure's* bytes have exactly one source; the
analysis keys on the closure object, not its container. Confirms A does not
over-reject this family.

### Conclusion

Within conforming C++ the A/C gap appears to be empty: every route is
blocked by the object model, is whole-object (hence single-source), already
declines (MemoryPhi, unresolvable candidate sets), or requires the type
erasure the frontend contract governs. This is an empirical sweep of the
enumerable constructions, not a proof -- but the union probe is the worst
legal case, and it cannot splice. The practical residue of the frontend
contract is therefore exactly the erasure constructs already listed for the
AST diagnostic: casts/byte pointers into wrapper storage, `mem*` builtins on
wrapper storage, and unions over wrappers (the last as a diagnosable
courtesy, since even its miscompile potential is whole-object switching, and
reads of inactive members are UB regardless).
