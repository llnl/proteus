# Design: Proteus with NVCC as the AOT compiler (shadow device pipeline)

Status: PROTOTYPE (option (a) below). Owner: NVCC interop.

## Problem

Proteus requires its LLVM pass pipeline on both host and device IR, which
today means Clang compiles the whole TU. Production CUDA codes often require
NVCC as the AOT compiler. NVCC fatbinaries carry SASS/PTX only -- no LLVM
bitcode -- so the existing runtime path (extract `_jit_bitcode_*` globals
from the loaded CUmodule) has nothing to extract, and the Proteus passes
never see the IR.

## Approach: NVCC compiles the product, Clang compiles a shadow

- The user's TU is compiled by NVCC exactly as before, with
  `-ccbin clang++` so the HOST half is Clang and can load the Proteus
  plugins (`-Xcompiler -fplugin=...`). The host pass instruments
  `__cudaRegisterFatBinary` / `__cudaRegisterFunction` / launches as it
  already does for Clang-CUDA builds; NVCC's generated registration glue
  calls the same CUDA runtime entry points, so the interception is
  compiler-agnostic.
- A SHADOW device-only Clang compile of the same source
  (`clang++ -x cuda --cuda-device-only`) runs the full Proteus device
  pipeline -- lambda analysis, manifests, module extraction -- and produces
  the specialization bitcode NVCC cannot.
- The shadow bitcode is embedded in a generated host object and registered
  at program startup into a ShadowBitcodeRegistry keyed by module ID. At
  runtime, bitcode extraction consults the fatbin first (clang-CUDA builds,
  unchanged) and falls back to the shadow registry (NVCC builds).
- Kernels Proteus cannot or should not specialize keep launching NVCC's own
  SASS through the original device pointer: the design degrades to "no
  specialization," never "wrong answer."

## Why the bitcode cannot come from the NVCC fatbin

nvcc does not embed NVVM/LLVM IR in release objects; `cuModuleGetGlobal` on
`_jit_bitcode_*` fails by construction. The shadow compile is not an
optimization -- it is the only source of IR.

## ABI boundary (the real risk)

A Proteus-specialized kernel is Clang-codegened but launched with an
argument buffer laid out by NVCC-generated host code, at offsets computed
from the SHADOW host stub. This is sound where both toolchains agree on
NVIDIA's kernel parameter ABI -- true in practice for the scalar /
pointer / trivially-copyable types Proteus restricts itself to, but it is an
assumption shared by two compilers, not checked by either. Mitigations, in
order of adoption:

1. v1 restricts specialization to kernels whose parameters are scalars,
   pointers, and trivially-copyable aggregates (already the Proteus
   contract).
2. Build- or load-time parameter audit: the shadow pipeline knows each
   kernel's parameter layout from IR; `cuFuncGetParamInfo` (CUDA >= 12.4)
   exposes NVCC's view per loaded kernel. Mismatch demotes the kernel to
   the NVCC fallback with a logged reason.
3. Module `__device__`/`__constant__` globals: NVCC's module and the
   specialized Clang module each have their own copies; host writes
   (cudaMemcpyToSymbol) land in NVCC's. v1 policy: kernels referencing
   module globals are NOT specialized (detectable in shadow IR); lifting
   this requires binding the specialized module's globals to NVCC's device
   addresses at load, which the existing registerVar plumbing supports in
   principle.

## Known constraints

- cudafe output: with `-ccbin clang++ -Xcompiler -fplugin=...`, the host
  plugins see NVCC's cudafe-processed host code, not user source. The IR
  pass is shape-based and tolerant; the AST checker's wrapper-storage
  contract needs vetting against cudafe output (unvetted; the checker can
  be omitted from the NVCC host compile until then).
- TU divergence: the shadow compile may reject NVCC-specific constructs or
  diverge under `__CUDACC__`-conditional code. A TU that fails to
  shadow-compile degrades to fallback-only; the build must treat shadow
  failure as a warning, not an error.
- Template instantiation skew between NVCC and Clang device halves is
  possible under compiler-conditional code; kernels missing from the shadow
  module degrade to fallback.
- Device-side launches and CUDA graphs bypass the host launch hook (same
  limitation as the existing engines).
- RDC: shadow modules must be linked per the same -rdc grouping; v1 targets
  whole-TU (non-RDC) kernels first.

## Pieces

- `tools/nvcc-shadow/ProteusNvccShadow.cmake`: `add_proteus_nvcc(target)` --
  per CUDA source, adds the shadow device-only Clang compile with the
  Proteus pass, converts the bitcode to a generated registration C file
  (`xxd`-style byte array + constructor calling
  `__proteus_register_shadow_bitcode`), and adds it to the target.
- `src/include/proteus/impl/ShadowBitcodeRegistry.h`: process-wide registry
  mapping module ID -> bitcode buffer; populated by constructors, consulted
  by the CUDA engine.
- `src/runtime/CompilerInterfaceDevice.cpp`:
  `__proteus_register_shadow_bitcode` entry point.
- `JitEngineDeviceCUDA::{extractModules,getModuleHash}`: consult the shadow
  registry for module IDs whose globals are absent from the CUmodule.

## Validation status

Developed on a machine with neither NVCC nor a CUDA toolkit: the shadow
device pipeline and registry compile and are exercised via clang NVPTX
device-only compiles with a header stub (mirroring the pseudo-GPU harness);
the NVCC host half and the runtime CUmodule fallback are written to the
documented CUDA driver API behavior but are UNTESTED until run on a CUDA
machine. Treat this branch as a prototype pending that validation.
