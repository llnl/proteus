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

The current JSON is the baseline of MAIN's analysis over main's lambda
suite, captured from the pseudo-GPU pipeline (LLVM 19, -O1, gfx90a).
Compile-time decisions have matched ROCm in spot checks, but this file
should be re-captured from a ROCm run before being treated as hardware
ground truth. Tests with no entry produce no analysis decisions in this
pipeline: lambda_dominating_store (ill-formed, unregistered in CMake) and
lambda_large_warning (needs __syncthreads in the pseudo-GPU stub).

## Current decisions

| Test | Specializes | Declines |
| --- | --- | --- |
| lambda | kernel | — |
| lambda-two | kernel | — |
| lambda_call_shapes | kernelSameReturns, kernelSelectSame | makeIndirectBody, makeReturnBody, makeSelectSameBody |
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
| lambda_memory_intrinsic_shapes | kernelNonoverlapMemset, kernelOffsetMemmove | makeDynamicBody |
| lambda_mfem_style_launch | MockRajaInterface | — |
| lambda_multiple | kernel | — |
| lambda_pathological | kernel_case1, kernel_case2_a, kernel_case2_b, kernel_case4, kernel_case5 | — |
| lambda_phi_analysis | kernel_phi, kernel_phi_arg1, kernel_phi_changed_rc | — |
| lambda_pointer_field_load | kernelPointerFieldLoad | — |
| lambda_provenance_shapes | kernelArrayMember, kernelMemcpySource, kernelNegativeGep, kernelOffsetMemcpy, kernelPartialMemcpy, kernelReturnedInteriorPointer | — |
| lambda_spec_test | kernel | — |
| lambda_store_order | kernelCaptureFromArg, kernelLatestStore, kernelStoreOrder | makeAmbiguousBody, makeMutatedBody |
| lambda_thread_privatizer | kernel_thread_privatizer | — |
