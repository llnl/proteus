if(PROTEUS_ENABLE_HIP)
  set(PROTEUS_GPU_LANG HIP)
  enable_language(HIP)
elseif(PROTEUS_ENABLE_CUDA)
  set(PROTEUS_GPU_LANG CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
  endif()

  enable_language(CUDA)
  message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

  if(PROTEUS_ENABLE_NVCC_SHADOW)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      message(FATAL_ERROR
        "PROTEUS_ENABLE_NVCC_SHADOW requires NVCC as the CUDA compiler")
    endif()
  elseif(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
  endif()
else()
  message(FATAL_ERROR "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA")
endif()

function(CREATE_MLIR_GPU_LIT_TEST exe check_source)
  add_executable(${exe}.${PROTEUS_GPU_LANG} ${check_source} ${ARGN})
  target_link_libraries(${exe}.${PROTEUS_GPU_LANG} PUBLIC proteusFrontend)
  set_source_files_properties(${check_source} ${ARGN} PROPERTIES
                              LANGUAGE ${PROTEUS_GPU_LANG})

  add_test(
    NAME ${exe}.${PROTEUS_GPU_LANG}
    COMMAND ${LIT} -vv -D EXT=${PROTEUS_GPU_LANG} -D FILECHECK=${FILECHECK}
            ${check_source})
  set_property(TEST ${exe}.${PROTEUS_GPU_LANG}
               PROPERTY LABELS "frontend-mlir;frontend-mlir-gpu")
endfunction()

file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT MLIR GPU tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
# Create a unique temp exec_root to avoid races on lit_test_times.txt
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

ext = lit_config.params['EXT']
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%ext', ext))
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
")

CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_kernel_execute mlir_gpu_kernel_execute.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_adam mlir_gpu_adam.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_adam_runconst mlir_gpu_adam_runconst.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_add_vectors_runconst mlir_gpu_add_vectors_runconst.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_atomics mlir_gpu_atomics.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_convert mlir_gpu_convert.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_floydwarshall mlir_gpu_floydwarshall.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_for mlir_gpu_for.cpp)
# WIP: The MLIR backend currently doesn't honor the loop unroll hint or emit
# stable LLVM loop-unroll metadata, so this test is disabled until unroll
# support is implemented.
# CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_for_unroll mlir_gpu_for_unroll.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_functional_blocks mlir_gpu_functional_blocks.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_get_address mlir_gpu_get_address.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_if mlir_gpu_if.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_index_lowering mlir_gpu_index_lowering.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_internal_call mlir_gpu_internal_call.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_intrinsics mlir_gpu_intrinsics.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_jit_module_source
             mlir_gpu_jit_module_source.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_kernel_2d mlir_gpu_kernel_2d.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_kernel_3d mlir_gpu_kernel_3d.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_multi_module mlir_gpu_multi_module.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_operators mlir_gpu_operators.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_shared_memory mlir_gpu_shared_memory.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_vars mlir_gpu_vars.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_while mlir_gpu_while.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_add_vectors mlir_gpu_add_vectors.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_builtins mlir_gpu_builtins.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_device_function_calls
             mlir_gpu_device_function_calls.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_kernel_representation
             mlir_gpu_kernel_representation.cpp)
CREATE_MLIR_GPU_LIT_TEST(mlir_gpu_launch_bounds mlir_gpu_launch_bounds.cpp)
