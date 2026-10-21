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

function(CREATE_LLVMIR_GPU_LIT_TEST exe check_source)
  add_executable(${exe}.${PROTEUS_GPU_LANG} ${check_source} ${ARGN})
  target_link_libraries(${exe}.${PROTEUS_GPU_LANG} PUBLIC proteusFrontend)
  set_source_files_properties(${check_source} ${ARGN} PROPERTIES
                              LANGUAGE ${PROTEUS_GPU_LANG})

  add_test(
    NAME ${exe}.${PROTEUS_GPU_LANG}
    COMMAND ${LIT} -vv -D EXT=${PROTEUS_GPU_LANG} -D FILECHECK=${FILECHECK}
            -D LLVM_AS=${LLVM_AS} ${check_source})
  set_property(TEST ${exe}.${PROTEUS_GPU_LANG}
               PROPERTY LABELS "frontend-llvmir;frontend-llvmir-gpu")
endfunction()

find_program(LLVM_AS llvm-as
  PATHS ${LLVM_BINARY_DIR}/bin ${LLVM_TOOLS_BINARY_DIR})

if(NOT LLVM_AS)
  message(FATAL_ERROR "Testing requires llvm-as to be installed")
endif()

file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT LLVMIR GPU tests'
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
LLVM_AS = lit_config.params['LLVM_AS']
config.substitutions.append(('%ext', ext))
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%llvm_as', LLVM_AS))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
")

CREATE_LLVMIR_GPU_LIT_TEST(llvmir_gpu_jit_module_source
             llvmir_gpu_jit_module_source.cpp)
CREATE_LLVMIR_GPU_LIT_TEST(llvmir_gpu_jit_module_bitcode
             llvmir_gpu_jit_module_bitcode.cpp)
