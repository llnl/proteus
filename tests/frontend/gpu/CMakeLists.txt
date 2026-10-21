if(PROTEUS_ENABLE_HIP)
    enable_language(HIP)
elseif(PROTEUS_ENABLE_CUDA)
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
endif()

function(CREATE_FRONTEND_TEST_GPU exe check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    else()
        SET(lang CUDA)
    endif()

    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PUBLIC proteusFrontend)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
    )

    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
    set_property(TEST ${exe}.${lang} PROPERTY LABELS "frontend-gpu")
endfunction()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT tests'
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
"
)

CREATE_FRONTEND_TEST_GPU(operators operators.cpp)
CREATE_FRONTEND_TEST_GPU(if if.cpp)
CREATE_FRONTEND_TEST_GPU(for for.cpp)
CREATE_FRONTEND_TEST_GPU(for_unroll for_unroll.cpp)
CREATE_FRONTEND_TEST_GPU(add_vectors add_vectors.cpp)
CREATE_FRONTEND_TEST_GPU(add_vectors_runconst add_vectors_runconst.cpp)
CREATE_FRONTEND_TEST_GPU(get_address get_address.cpp)
CREATE_FRONTEND_TEST_GPU(adam adam.cpp)
CREATE_FRONTEND_TEST_GPU(adam_runconst adam_runconst.cpp)
CREATE_FRONTEND_TEST_GPU(kernel_2d kernel_2d.cpp)
CREATE_FRONTEND_TEST_GPU(kernel_3d kernel_3d.cpp)
CREATE_FRONTEND_TEST_GPU(floydwarshall floydwarshall.cpp)
CREATE_FRONTEND_TEST_GPU(multi_module multi_module.cpp)
CREATE_FRONTEND_TEST_GPU(internal_call internal_call.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_source cpp_source.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_adam cpp_adam.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_instantiate cpp_instantiate.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_instantiate_shmem cpp_instantiate_shmem.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_store_handle cpp_store_handle.cpp)
CREATE_FRONTEND_TEST_GPU(cpp_same_symbol cpp_same_symbol.cpp)
if(PROTEUS_ENABLE_CUDA)
    CREATE_FRONTEND_TEST_GPU(cpp_device_nvcc cpp_device_nvcc.cpp)
    CREATE_FRONTEND_TEST_GPU(cpp_instantiate_nvcc cpp_instantiate_nvcc.cpp)
    CREATE_FRONTEND_TEST_GPU(cpp_instantiate_nvcc_shmem cpp_instantiate_nvcc_shmem.cpp)
endif()
CREATE_FRONTEND_TEST_GPU(shared_memory shared_memory.cpp)
CREATE_FRONTEND_TEST_GPU(convert convert.cpp)
CREATE_FRONTEND_TEST_GPU(launch_bounds launch_bounds.cpp)
CREATE_FRONTEND_TEST_GPU(atomics atomics.cpp)
CREATE_FRONTEND_TEST_GPU(while while.cpp)
CREATE_FRONTEND_TEST_GPU(intrinsics intrinsics.cpp)
CREATE_FRONTEND_TEST_GPU(vars vars.cpp)
CREATE_FRONTEND_TEST_GPU(functional_blocks functional_blocks.cpp)
