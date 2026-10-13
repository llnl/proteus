# Consolidated Review: clobbering-aware lambda use analysis

PR: llnl/proteus#526 ("feature: make LambdaInstUseVisitor clobbering aware")
Branch: `feature/bowen/clobbering-aware-use-analysis` (base `2a8f200a`)
Synthesis of two independent reviews (A: code-analysis review; B: ROCm 7.2.1
re-review) plus local pseudo-GPU validation on LLVM 19 and 22.
Date: 2026-10-12

## Verdict

**Do not merge yet — but this is repair, not redesign.** The reviews disagreed
on the central question: review B held that the proof core is unsound ("the
MemorySSA location being queried is not reliably the closure byte"); close
reading shows the design is sound *conditionally*, via an invariant that is
real but implicit — and one blocker (context-scoping corruption) breaks exactly
that condition. The correct read is: **sound architecture, implicit proof,
four genuine miscompile/hang channels in the plumbing.** All are fixable with
targeted changes.

## The load-bearing invariant

Acceptance of a specialization requires that MemorySSA's selected clobber be
**a member of the collected candidate set** (`KernelArgPtrUseVisitor.h:282`
single-candidate; `:614-626` otherwise), and every store/memtransfer candidate
**covers the tracked byte** (`:889`, `:1234`). Therefore: if the walker reports
X as the latest definition aliasing a range containing the tracked byte, and X
is a covering candidate, X defines that byte. Querying the first candidate's
`MemoryLocation::get(SI)` instead of the tracked byte (B-1), the zero-remaining
GEP retry (B-2), and the aggregate four-case block (B-3) can each only push the
clobber *out* of the candidate set, producing a conservative decline, not a
wrong specialization.

Three caveats keep this from being a clean bill of health:

1. The invariant holds **only if `ValueOffsetMap` is trustworthy** — broken by
   Blocker 1 below.
2. It is **stated nowhere** and enforced structurally nowhere; it survives by
   the accident of two checks in different functions.
3. Which candidate location gets queried depends on `Value::users()` order,
   giving **nondeterministic completeness**: identical code can specialize
   under one compiler build and decline under another.

**Structural remedy (top of the merge gate):** carry the exact tracked byte
range in `ClobberQuery` and use it for every MemorySSA query. This makes the
proof explicit and deterministic, and discharges B-1/2/3 by construction.

## Blockers

1. **Call-context corruption of `ValueOffsetMap` — wrong byte offset, silent
   miscompile.** Second entry into the same callee overwrites the formal's
   offset (`KernelArgPtrUseVisitor.h:962`, `:578`); `Seen` being
   continuation-keyed forces callee-internal GEPs to be revisited and their
   map entries recomputed under the new context, corrupting `WriteOffset`
   (`:899`) for still-pending first-context candidates. Reconciliation
   validates the *instruction*, not the offset, so this escapes to
   `Result.Offset`. Minimum fix: fail on conflicting re-entry (mirror the
   guard at `:1050`); full fix: context-key the map.

2. **Unbounded recursion — compile-time hang.** A pointer-returning helper
   recursing on the tracked pointer mints a fresh `ReturnContinuation` per
   entry (`:963`); `Seen` keyed on `(Value, continuation)` structurally
   defeats cycle detection. Fix: reject repeated callees along the `Parent`
   chain, plus a global work-item budget.

3. **`visitArgument` counts calls that merely reference the function.**
   `KernelArgVisitor.h:623-624` lacks `CB->isCallee(&U)`; with exactly one
   non-callee `CallBase` user, the walk follows an unrelated operand. Treat
   address-escaping functions as "not exactly one caller."

4. **`analyzeFunction` trusts the first `ReturnInst` only.** `getRetInst`
   (`KernelArgVisitor.h:38`, used at `:298`) analyzes one return path of a
   multi-return callee and specializes all paths. The forward visitor already
   rejects multi-return callees (`KernelArgPtrUseVisitor.h:558`); make the
   backward side symmetric.

## High

5. **`PendingWorkItems` underflow**: `WorkList.append` (`:706`) bypasses
   `pushBack`'s increment; `popBack` (`:479`) decrements unconditionally, so
   the unsigned counter wraps and wedges the continuation.
6. **Seen-path skips `finishCurrentWorkItem`** (`:1293-1295`): a duplicate
   final work item strands a completed continuation.
7. **`Seen` holds dangling `ReturnContinuation*`** (`:221`): allocator reuse
   makes nondeterministic false "seen" results possible. Use stable context
   IDs or visitor-lifetime ownership.
8. **Nondeterministic query-location selection** (`:239-253`): the
   completeness residue of B-1 after the soundness downgrade; subsumed by the
   merge-gate remedy. `OffsetOfWriteToBasePtr` participates in cache identity
   but not in the queried location; the size field is unused.

## Medium

9.  **`getStateBefore` manufactures `LiveOnEntry`** at multi-predecessor
    blocks with no MemoryPhi, and on cycles
    (`PointerClobberAnalysis.h:205-212`). Conservative today (every consumer
    of a no-instruction result declines; the one semantic `LiveOnEntry`
    consumer queries at a load), but a latent trap and a completeness hole for
    multi-block helpers. Walk the idom chain; document the invariant.
10. **Aggregate reconciliation block** (`:343-434`): the double insertion of
    the clobbering store (`:393` seed + `:403` loop) is real but
    behavior-preserving — both arms return the clobbering store, and a
    non-candidate clobber still fails the downstream membership check. The
    real defect is intent/behavior mismatch: the four-case comment describes
    logic that does not execute. Replace the block with the explicit
    membership rule (plus optional heterogeneous-source rejection).
11. **`callCannotModifyTrackedLocal` one-level asm scan**
    (`PointerClobberAnalysis.h:115-123`) is incoherent as written: redundant
    if the capture argument is trusted (never-escaped allocas are not
    semantically reachable by asm at any depth — the same reasoning DSE uses),
    insufficient if it is not. Keep the capture-based override, delete the
    scan, document the argument; config-gate if extra margin is wanted.
12. **Legacy `dbg.value` users abort the walk** in `visitCallBase`
    (`:932-940`), so `-g` builds on older toolchains silently lose
    specialization. Exempt `DbgInfoIntrinsic`.
13. **`getPreciseTrackedLocation`** (`:747-784`): the feared unsound
    acceptance cannot occur (zero-remaining entries point *at* the tracked
    byte by construction; the retry is gated on an exact candidate match at
    `:267`) — but the invariant lives entirely in map construction. Assert
    and document it.
14. **Per-callsite oracle construction** (`KernelArgVisitor.h:341`):
    MemorySSA/DT/AA are rebuilt per lambda callsite; hoist to module scope.
    Cache the callee asm scan.
15. **IR-shape test assertions**: runtime-output-only checks can stay green
    while optimization erases the shape under test. The pseudo-GPU harness
    (`tools/pseudo-gpu/`, separate branch) emits raw and post-pass device IR
    locally; add FileCheck-over-IR companions for correctness-critical
    shapes. New tests needed for Blockers 1-4: double-entered helper at
    different offsets, recursive helper (hang must become decline),
    address-taken helper, multi-return callee, plus a `-g` build.

## Cleanups

Dead code: reaching-pointer-stores family (`KernelArgVisitor.h:56-176`),
`getCallBaseIdentityArgOperand` (`:227`), `isSelectedClobber` plus its
commented call (`KernelArgPtrUseVisitor.h:786`, `:886`),
`StoreInstToOffsetBasePtrMap`, `SizeOfWriteToBasePtr`, `DTree`. Hygiene:
`DEBUG(... << ValueOffsetMap[&SI])` default-inserts a key — debug/release
state divergence (`:902`); non-`inline` header definitions
(`getReturnInstructions`, `getRetInst`); visitor-global `Offset` coexisting
with per-value offsets (`:994`); `AnalysisSuccess`/`AnalysisFailed` dual
flags; stale `-passes=proteus-pass` comment (`ProteusPass.cpp:2078` — no
parsing callback exists); `emitModuleDevice` non-idempotence fatals under
upstream clang's double EP registration for HIP device compiles (`:980`).
Document the intentional `memmove` support regression; reword the
`MergedSlot` diagnostic (`LambdaASTChecker.cpp:531`).

## Validation performed

- Review B: `ProteusPass` builds on the ROCm 7.2.1 / LLVM 22 tree; AST checker
  test passes; GPU runtime tests not run (no device available).
- This review: branch builds clean on Homebrew LLVM 19.1.7 **and** 22.1.0
  (macOS arm64). The pseudo-GPU harness drove `lambda_clobber_shapes.cpp`
  through the real amdgcn device pipeline with the pass loaded on both LLVM
  versions: identical decisions, matching all 12 FileCheck expectations
  (5 specialize / 7 decline). Runtime specialization values and caching remain
  unvalidated — final sign-off requires a ROCm machine.

## Missing functor-extent tracking (upgraded to gate item 8)

Identified by the author post-review; both reviews under-stated it. The
semantic unit the analysis must protect is the registered functor's byte
range [tracked offset, tracked offset + sizeof(functor)): runtime
specialization reads every registered slot relative to the resolved closure
offset, so a foreign write to *any* functor byte between the authoritative
write and the callsite invalidates specialization. The analysis is
byte-granular instead: candidates must cover byte 0 only
(`KernelArgPtrUseVisitor.h:889`), non-covering sibling GEPs are skipped
(`:1109`) making interior-slot writes invisible to candidate collection, and
`getPreciseTrackedLocation` anchors the MemorySSA query at the *smallest*
zero-remaining GEP -- whatever size that happens to have.

Evidence from `lambda_clobber_query_suffix.cpp` (passes, 631 specializes):
the anchor is `gep %LambdaEnvelopeWithSuffix, 0, 0` with size 4, which equals
the functor extent *only because* that GEP's result type happens to be the
functor struct. Two consequences:

- Soundness: with a multi-slot functor and a narrower zero-remaining GEP
  (e.g. to slot 0), a foreign write to a later slot is invisible to both the
  anchored query and candidate collection -> stale interior-slot
  specialization. `lambda_mixed_field_overwrite` declines today only because
  its IR lacks a narrow anchor, so the broad whole-alloca query happens to
  see the interior store -- anchor luck, not proof.
- Completeness: the suffix test's pass depends on Clang emitting that typed
  GEP; a byte-GEP lowering flips it to a decline.

At functor-range granularity, review B's "checked affine source/destination
mapping with coverage" (its finding 3) is the correct spec after all -- the
byte-granular membership rule is sound only for the first byte. Fix design:
thread `TrackedEntitySize` (from the lambda callsite's typed GEP/alloca, or
conservatively unknown) into `ClobberQuery`; clamp precise anchors to cover
the full range; classify candidates by range overlap (full-cover resolvable,
partial overlap -> decline, same-source field-wise writes jointly covering
the range -> resolvable via the aggregate reconciliation path); decline on
unknown size + interior ambiguity. Regression tests: multi-slot functor with
foreign interior write plus a narrow slot-0 GEP (must decline); 2-slot suffix
variant (must specialize); functor at nonzero envelope offset.

## Fix progress (on-branch, verified via pseudo-GPU harness)

- Gate 2 (Blocker 1): conflicting callee re-entry offsets now decline, at both
  entry points. DONE (minimum fix; context-keying remains follow-up).
- Gate 3 (Blocker 2): recursion through pointer-returning helpers rejected via
  the continuation parent chain; 128K work-item budget in `popBack`; plus an
  `AnalysisDepth` cap fixing pre-existing unbounded recursion in the backward
  visitor's PHI clones and nested function analyses. DONE.
- Gate 4 (Blockers 3-4): `visitArgument` requires direct-callee uses and
  rejects address-escaping functions; `analyzeFunction` requires exactly one
  value-returning `ReturnInst`. DONE.
- Gate 1: reconciliation queries are now deterministic and byte-anchored --
  `ClobberQuery::TrackedByteOffset` is the query identity (cache key),
  `SizeOfWriteToBasePtr` removed, every query routes through
  `getQueryLocation` (exact tracked-byte location when a visited pointer
  denotes it, covering superset otherwise), and the proven-returned-alias
  skip applies on every query path. Remaining: field-size threading (above).
- Verification: decisions identical across all 15 harness-compilable lambda
  suites before/after each fix stage, on LLVM 19 and 22;
  `lambda_mfem_style_launch` specializes. Findings 5-7 (continuation
  accounting) and regression tests for Blockers 1-4 not yet started.

## Merge gate

1. Represent the tracked byte range explicitly in `ClobberQuery`; use it for
   every MemorySSA query (discharges B-1/2/3; removes decline nondeterminism).
2. Fix context scoping: conflict guard on callee re-entry now; context-keyed
   facts as follow-up (Blocker 1).
3. Add recursion rejection and a work budget (Blocker 2).
4. `isCallee` check in `visitArgument`; single-return requirement in
   `analyzeFunction` (Blockers 3-4).
5. Fix the continuation accounting trio (findings 5-7, one commit).
6. Replace the aggregate four-case block with the explicit membership rule;
   keep capture-based call skipping but delete the asm scan and document the
   argument.
7. Land the regression tests from finding 15 with IR-shape assertions via the
   harness; then the full `lambda_*` suite plus `lambda_mfem_style_launch` on
   ROCm hardware.
8. Track the registered functor extent and enforce range-granular write
   coverage (see "Missing functor-extent tracking").
   RESOLVED DESIGN (implemented on-branch): the extent travels as a frontend
   constant -- sizeof in the proteus.wrapper_call annotation payload, folded
   per-target by Sema, stamped as proteus.functor_size metadata in both
   modules -- and is consumed by three surgical IR checks: (a) the fallback
   MemorySSA query location is bounded to [0, offset + size), so writes past
   the functor can no longer be selected as clobbers (pinned by
   lambda_size_test, kernelEnvelopeSuffix, which exposed exactly that
   spurious decline at -O1); (b) a transfer covering the tracked byte must
   cover the full extent or decline, closing the no-cast partial-memcpy
   splice (the classic wrong-sizeof bug), in both visitors; (c) memsets
   decline on any extent overlap instead of ignoring interior-only hits.
   Deliberately NOT implemented in IR: extent-wide clobber visibility for
   interior GEP stores -- slot-wise assembly (the mfem shape) makes interior
   stores legitimate parallel copies, and accepting them would require
   per-piece backward reconciliation (prototyped, rolled back, twice
   regressed mfem). Interior-store soundness is instead a FRONTEND CONTRACT:
   such stores require type erasure visible to the AST checker. REMAINING
   WORK: the LambdaASTChecker diagnostic (type-erased writes into
   wrapper-containing storage, mem* builtins on wrapper storage, unions over
   wrappers) and repurposing lambda_interior_slot_overwrite as an AST
   rejection test; until it lands, interior-store declines remain
   anchor-dependent (lambda_mixed_field_overwrite declines by luck, not
   proof).
