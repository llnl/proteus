# Pseudo-GPU device-compile targets.
#
# For every GPU test, adds a `<test>.HIP` target that runs the real HIP
# device pipeline (clang -x hip targeting amdgcn, ProteusPass loaded) and
# emits the post-pass LLVM module to pseudo-gpu/<test>.device.ll -- the
# equivalent of -S -emit-llvm, with no ROCm or GPU required. Executables and
# runtime behavior are out of scope; use these targets to inspect pass
# decisions (with PROTEUS_DEBUG_OUTPUT=1) and IR shapes ahead of time.
#
#   cmake -B build -DPROTEUS_PSEUDO_GPU=ON -DLLVM_INSTALL_DIR=... \
#         -DCMAKE_C_COMPILER=.../clang -DCMAKE_CXX_COMPILER=.../clang++
#   make -C build lambda_clobber_shapes.HIP
#   make -C build pseudo-gpu-all

set(PSEUDO_GPU_CLANG "${LLVM_INSTALL_DIR}/bin/clang++")
if(NOT EXISTS "${PSEUDO_GPU_CLANG}")
  message(FATAL_ERROR "PROTEUS_PSEUDO_GPU requires clang++ in LLVM_INSTALL_DIR"
                      " (${PSEUDO_GPU_CLANG} not found)")
endif()

set(PSEUDO_GPU_OUT_DIR "${CMAKE_BINARY_DIR}/pseudo-gpu")
file(MAKE_DIRECTORY "${PSEUDO_GPU_OUT_DIR}")

set(PSEUDO_GPU_STUB_HEADER
  "${PROJECT_SOURCE_DIR}/tools/pseudo-gpu/include/hip/hip_runtime.h")

set(PSEUDO_GPU_FLAGS
  -x hip --offload-arch=gfx90a --offload-device-only
  -nogpulib -nogpuinc -std=c++17 -O1
  -DPROTEUS_ENABLE_HIP=1
  -I "${PROJECT_SOURCE_DIR}/tools/pseudo-gpu/include"
  -I "${PROJECT_SOURCE_DIR}/include"
  # Real HIP force-includes clang's runtime wrapper before user code; mirror
  # that so target attribute macros exist regardless of include order.
  -include hip/hip_runtime.h)

file(GLOB PSEUDO_GPU_TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/gpu/*.cpp")

set(PSEUDO_GPU_ALL_OUTPUTS)
foreach(PseudoSource IN LISTS PSEUDO_GPU_TEST_SOURCES)
  get_filename_component(PseudoTestName "${PseudoSource}" NAME_WE)
  set(PseudoOutput "${PSEUDO_GPU_OUT_DIR}/${PseudoTestName}.device.ll")
  add_custom_command(
    OUTPUT "${PseudoOutput}"
    COMMAND "${PSEUDO_GPU_CLANG}" ${PSEUDO_GPU_FLAGS}
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
            -S -emit-llvm -o "${PseudoOutput}" "${PseudoSource}"
    DEPENDS "${PseudoSource}" ProteusPass "${PSEUDO_GPU_STUB_HEADER}"
    WORKING_DIRECTORY "${PSEUDO_GPU_OUT_DIR}"
    COMMENT "Pseudo-GPU device compile: ${PseudoTestName}"
    VERBATIM)
  add_custom_target("${PseudoTestName}.HIP" DEPENDS "${PseudoOutput}")
  list(APPEND PSEUDO_GPU_ALL_OUTPUTS "${PseudoOutput}")
endforeach()

add_custom_target(pseudo-gpu-all DEPENDS ${PSEUDO_GPU_ALL_OUTPUTS})
