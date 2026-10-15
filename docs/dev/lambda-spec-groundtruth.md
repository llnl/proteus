# Lambda GPU suite: specialization ground truth

`tests/gpu/lambda-spec-groundtruth.json` records, for every lambda GPU test,
which kernels the analysis specializes and which functors it conservatively
declines. It is the reviewable source of truth for analysis behavior: any
change to the pass that shifts a decision must either be a bug fix (update
the JSON in the same commit, explaining the flip) or is a regression (CI
fails).

## Running the check

Logs come from the ProteusPass debug trace, which is identical for real ROCm
builds and the pseudo-GPU pipeline:

    # 1. produce one log per test (any build that runs the pass), e.g.:
    PROTEUS_DEBUG_OUTPUT=1 ctest -R 'lambda_.*HIP' ...   # ROCm
    #    or compile each test --offload-device-only with the pass plugin
    #    (pseudo-GPU), then collect .proteus-logs/proteus-pass.*.log into
    #    <logs-dir>/<test>.log

    # 2. diff against ground truth:
    python3 tools/lambda-spec-ci/lambda_spec_ci.py check \
        --logs <logs-dir> --truth tests/gpu/lambda-spec-groundtruth.json

    # re-capture (after an intended behavior change):
    python3 tools/lambda-spec-ci/lambda_spec_ci.py capture \
        --logs <logs-dir> --truth tests/gpu/lambda-spec-groundtruth.json \
        --pipeline <rocm-x.y|pseudo-gpu-llvmNN> --note "<why>"

## Provenance caveat

The current JSON was captured from the pseudo-GPU pipeline (LLVM 19, -O1,
gfx90a) at the branch tip. Compile-time decisions have matched ROCm in spot
checks, but this file MUST be re-captured from a ROCm run before being
treated as hardware ground truth -- pseudo-GPU "green" has disagreed with
ROCm before. Tests with no entry produce no analysis decisions in this
pipeline: lambda_dominating_store (ill-formed, unregistered in CMake) and
lambda_large_warning (needs __syncthreads in the pseudo-GPU stub).

Known expected-fail pinned in the suite: lambda_interior_slot_overwrite
currently SPECIALIZES (type-erased interior write; soundness delegated to
the frontend contract -- see docs/dev/design/lambda-closure-provenance.md).
Its runtime FileCheck fails by design until the AST diagnostic lands.

## Current decisions

| Test | Specializes | Declines |
| --- | --- | --- |
| lambda | kernel | — |
| lambda-two | kernel | — |
| lambda_aggregate_store_reconciliation | — | makeBody |
| lambda_call_shapes | kernelSelectSame | makeIndirectBody, makeReturnBody, makeSameReturnBody, makeSelectSameBody |
| lambda_clobber_query_range | kernelUnrelatedPrefixOverwrite | — |
| lambda_clobber_query_suffix | kernelUnrelatedSuffixOverwrite | — |
| lambda_clobber_shapes | kernelCallOverwrite, kernelLocalSupersedesCall, kernelNestedCallOverwrite, kernelNonoverlapOverwrite, kernelPredecessorOverwrite | makeAmbiguousLoopBody, makeDifferentBranchBody, makeDifferentCallBranchBody, makeSameBranchBody, makeSameCallBranchBody, makeSameLoopBody, makeUnrelatedCallBody |
| lambda_declaration_clobber | — | makeBody |
| lambda_def | kernel | — |
| lambda_factory | kernel | — |
| lambda_factory_forwarded | kernel | — |
| lambda_factory_multi | — | — |
| lambda_factory_so_runner | — | — |
| lambda_factory_tu1 | _GLOBAL__N_1 | — |
| lambda_factory_tu2 | _GLOBAL__N_1 | — |
| lambda_host_device | kernel | — |
| lambda_index_first_wrapper | kernel | — |
| lambda_insertvalue_offsets | kernelDoubleShift, kernelPadded, kernelShift | — |
| lambda_inst_visitor_call_clobber | kernelInstVisitorCallClobber, kernelInstVisitorCallDoesntClobber, kernelInstVisitorOrderedMemcpys | makeUntraceableBody |
| lambda_interior_slot_overwrite | — | makeInteriorBody |
| lambda_memory_intrinsic_shapes | kernelNonoverlapMemset | makeDynamicBody, makeMoveBody |
| lambda_mfem_style_launch | MockRajaInterface | — |
| lambda_mixed_field_overwrite | — | makeBody |
| lambda_multiple | kernel | — |
| lambda_multiple_callers | — | makeBody |
| lambda_pathological | kernel_case1, kernel_case2_a, kernel_case2_b, kernel_case4, kernel_case5 | — |
| lambda_phi_analysis | kernel_phi, kernel_phi_arg1, kernel_phi_changed_rc | — |
| lambda_pointer_field_load | kernelPointerFieldLoad | — |
| lambda_portability_aggregate_shapes | kernelAggregateFieldCall, kernelCallContext, kernelNestedReturn, kernelReturnedAlias, kernelReturnedFieldAlias | — |
| lambda_provenance_shapes | kernelArrayMember, kernelMemcpySource, kernelNegativeGep, kernelOffsetMemcpy, kernelPartialMemcpy, kernelReturnedInteriorPointer | — |
| lambda_reconciliation_adversarial | — | makeBytePair, makeConditionalPair, makeDuplicatePair, makeHeterogeneousPair, makePermutedPair, makeTriple |
| lambda_size_test | kernelEnclosedFunctor, kernelEnvelopeSuffix | — |
| lambda_slot_selection_supported | kernelCallAfterInvoke, kernelPhiSlot, kernelSelectedSlot, kernelTrailingReadOnlyCall | — |
| lambda_spec_test | kernel | — |
| lambda_store_order | kernelCaptureFromArg, kernelLatestStore, kernelStoreOrder | makeAmbiguousBody, makeMutatedBody |
| lambda_thread_privatizer | kernel_thread_privatizer | — |
| lambda_unknown_clobber | — | makeAggregateHiddenBody, makeAtomicBody, makeCompareExchangeBody, makeIndirectCallBody, makeOpaqueCallBody, makeSelectedStoreBody |
| lambda_unknown_clobber_decl | — | — |
