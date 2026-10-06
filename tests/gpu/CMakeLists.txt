if(PROTEUS_ENABLE_HIP)
    SET(lang HIP)
elseif(PROTEUS_ENABLE_CUDA)
    SET(lang CUDA)
else()
    message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
endif()

# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    target_link_libraries(${exe}.${lang} PUBLIC proteus)
    proteus_attach_pass_plugin_rebuild_dep(${exe}.${lang} ${check_source} ${ARGN})

    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe}.${lang} PROPERTIES LABELS "gpu;gpu-basic")
endfunction()

function(CREATE_GPU_TEST_RDC exe check_source)
    add_executable(${exe}.${lang}.rdc ${check_source} ${ARGN})
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    target_link_libraries(${exe}.${lang}.rdc PUBLIC proteus)
    proteus_attach_pass_plugin_rebuild_dep(${exe}.${lang}.rdc ${check_source} ${ARGN})

    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

        # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            --hip-link
            -Xoffload-linker --load-pass-plugin=$<TARGET_FILE:ProteusPassOffload>
        )
        add_dependencies(${exe}.${lang}.rdc ProteusPassOffload)
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(${exe}.${lang}.rdc PRIVATE --no-offload-new-driver)

        set_target_properties(${exe}.${lang}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_test(NAME ${exe}.${lang}.rdc COMMAND ${LIT} -vv -D EXT=${lang}.rdc -DFILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES LABELS "gpu;gpu-rdc")
endfunction()

function(CREATE_GPU_TEST_RDC_LIBS exe libs check_source)
    #message(FATAL_ERROR "libs ${libs} check_source ${check_source} ARGN ${ARGN}")

    CREATE_GPU_TEST_RDC(${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang}.rdc PRIVATE ${libs})
endfunction()

function(CREATE_GPU_TEST_LIBS exe libs check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    CREATE_GPU_TEST(${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PRIVATE ${libs})
endfunction()

# Creates a device library, which can be linked with another test. The library does not
# apply the pass
function(CREATE_GPU_LIBRARY lib source)
    add_library(${lib} ${source})

    set_source_files_properties(${source} PROPERTIES LANGUAGE ${lang})

    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(${lib} PRIVATE --no-offload-new-driver)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()
endfunction()

function(CREATE_PROTEUS_GPU_LIBRARY lib source)
    add_library(${lib} ${source} ${ARGN})

    set_source_files_properties(${source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    target_link_libraries(${lib} PUBLIC proteus)
    proteus_attach_pass_plugin_rebuild_dep(${lib} ${source} ${ARGN})

    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(${lib} PRIVATE --no-offload-new-driver)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()
endfunction()

function(CREATE_PROTEUS_GPU_SHARED_LIBRARY lib source)
    add_library(${lib} SHARED ${source} ${ARGN})

    set_source_files_properties(${source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    target_link_libraries(${lib} PUBLIC proteus)
    proteus_attach_pass_plugin_rebuild_dep(${lib} ${source} ${ARGN})

    if(PROTEUS_ENABLE_HIP)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(${lib} PRIVATE --no-offload-new-driver)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()
endfunction()

function(CREATE_GPU_TEST_FORCE exe check_source)
    CREATE_GPU_TEST(${exe} ${check_source} ${ARGN})
    # Add flags for forced annotations.
    target_compile_options(
        ${exe}.${lang}
        PUBLIC
            "-fplugin=$<TARGET_FILE:ProteusPass>"
            "SHELL:-Xclang -mllvm -Xclang -force-proteus-jit-annotate-all"
    )
endfunction()

function(CREATE_GPU_TEST_RDC_FORCE exe check_source)
    CREATE_GPU_TEST_RDC(${exe} ${check_source} ${ARGN})
    # Add flags for forced annotations.
    target_compile_options(
        ${exe}.${lang}.rdc
        PUBLIC
            "-fplugin=$<TARGET_FILE:ProteusPass>"
            "SHELL:-Xclang -mllvm -Xclang -force-proteus-jit-annotate-all"
    )
endfunction()

if(PROTEUS_ENABLE_HIP)
    enable_language(HIP)
elseif(PROTEUS_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
    endif()

    enable_language(CUDA)
    message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

    if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
        message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
    endif()
endif()

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
config.substitutions.append(('%device_lang', '${lang}'))
"
)

CREATE_GPU_TEST(dim_spec_cache_test dim_spec_cache_test.cpp)
CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_GPU_TEST(kernel_metadata kernel_metadata.cpp)
CREATE_GPU_TEST(kernel_pass_pipeline kernel_pass_pipeline.cpp)
CREATE_GPU_TEST(kernel_pass_plugin kernel_pass_plugin.cpp)
target_compile_definitions(kernel_pass_plugin.${lang} PRIVATE
    PROTEUS_TEST_JIT_PASS_PLUGIN_PATH="$<TARGET_FILE:JITTestPass>")
add_dependencies(kernel_pass_plugin.${lang} JITTestPass)
CREATE_GPU_TEST(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST(kernel_args kernel_args.cpp)
CREATE_GPU_TEST(kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST(kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST(kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST(indirect_fallthrough indirect_fallthrough.cpp)
CREATE_GPU_TEST(inlining_regression inlining_regression.cpp)
CREATE_GPU_TEST(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(daxpy daxpy.cpp)
CREATE_GPU_TEST(daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST(annot_nested annot_nested.cpp)
CREATE_GPU_TEST(types types.cpp)
CREATE_GPU_TEST(types_api types_api.cpp)
CREATE_GPU_TEST(kernel_unused_gvar kernel_unused_gvar.cpp kernel_unused_gvar_def.cpp)
CREATE_GPU_TEST(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST(kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST(block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST(block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST(block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST(lambda lambda.cpp)
CREATE_GPU_TEST(lambda-two lambda-two.cpp)
CREATE_GPU_TEST(shared_array shared_array.cpp)
CREATE_GPU_TEST(enable_disable enable_disable.cpp)
CREATE_GPU_TEST(lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST(lambda_def lambda_def.cpp)
CREATE_GPU_TEST(lambda_factory lambda_factory.cpp)
CREATE_GPU_TEST(lambda_factory_forwarded lambda_factory_forwarded.cpp)
CREATE_GPU_TEST(lambda_thread_privatizer lambda_thread_privatizer.cpp)
CREATE_GPU_TEST(lambda_index_first_wrapper lambda_index_first_wrapper.cpp)
CREATE_GPU_TEST(lambda_phi_analysis lambda_phi_analysis.cpp)
CREATE_GPU_TEST(lambda_insertvalue_offsets lambda_insertvalue_offsets.cpp)
CREATE_GPU_TEST(lambda_provenance_shapes lambda_provenance_shapes.cpp)
CREATE_GPU_TEST(lambda_store_order lambda_store_order.cpp)
CREATE_GPU_TEST(lambda_mixed_field_overwrite lambda_mixed_field_overwrite.cpp)
CREATE_GPU_TEST(lambda_call_shapes lambda_call_shapes.cpp)
CREATE_GPU_TEST(lambda_memory_intrinsic_shapes lambda_memory_intrinsic_shapes.cpp)
CREATE_GPU_TEST(lambda_pointer_field_load lambda_pointer_field_load.cpp)
CREATE_GPU_TEST(lambda_clobber_shapes lambda_clobber_shapes.cpp)
CREATE_GPU_TEST(lambda_portability_aggregate_shapes
                lambda_portability_aggregate_shapes.cpp)
CREATE_GPU_TEST(lambda_slot_selection_supported
                lambda_slot_selection_supported.cpp)
CREATE_GPU_TEST(lambda_unsupported_memory_shapes
                lambda_unsupported_memory_shapes.cpp)
CREATE_GPU_TEST(lambda_slot_selection_unsupported
                lambda_slot_selection_unsupported.cpp)
CREATE_GPU_TEST(lambda_multiple_callers lambda_multiple_callers.cpp)
CREATE_GPU_TEST(lambda_unknown_clobber lambda_unknown_clobber.cpp)
CREATE_GPU_TEST_RDC(lambda_declaration_clobber lambda_declaration_clobber.cpp
                    lambda_unknown_clobber_decl.cpp)
CREATE_GPU_TEST(lambda_inst_visitor_call_clobber lambda_inst_visitor_call_clobber.cpp)
CREATE_PROTEUS_GPU_SHARED_LIBRARY(lambda_factory_so_tu1 lambda_factory_tu1.cpp)
CREATE_PROTEUS_GPU_SHARED_LIBRARY(lambda_factory_so_tu2 lambda_factory_tu2.cpp)
CREATE_GPU_TEST_LIBS(lambda_factory_so_runner "lambda_factory_so_tu1;lambda_factory_so_tu2" lambda_factory_so_runner.cpp)
CREATE_GPU_TEST(lambda_pathological lambda_pathological.cpp)
CREATE_GPU_TEST(lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST(lambda_large_warning lambda_large_warning.cpp)
CREATE_GPU_TEST(lambda_spec_test lambda_spec_test.cpp)
CREATE_GPU_TEST(lambda_mfem_style_launch lambda_mfem_style_launch.cpp)
CREATE_GPU_TEST(builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST(kernel_calls_indirect kernel_calls_indirect.cpp)
CREATE_GPU_TEST(global_var_register global_var_register.cpp)
if(PROTEUS_ENABLE_CUDA)
    # Only CUDA caps dynamic shared memory at 48KB without a per-kernel opt-in.
    CREATE_GPU_TEST(kernel_dynamic_shmem kernel_dynamic_shmem.cpp)
endif()
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST(alias_func alias_func.cpp)
    CREATE_GPU_TEST(alias_gvar alias_gvar.cpp)
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA support alias on func only on LLVM >=18 and <22.
    if(LLVM_VERSION_MAJOR VERSION_GREATER_EQUAL 18 AND
       LLVM_VERSION_MAJOR VERSION_LESS 22)
        CREATE_GPU_TEST(alias_func alias_func.cpp)
    endif()
    # CUDA does not support alias on global variables.
endif()
CREATE_GPU_TEST(mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST(types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST(dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST(jit_struct jit_struct.cpp)
CREATE_GPU_TEST(kernel_tuning kernel_tuning.cpp)
CREATE_GPU_TEST(gvar_tracking gvar_driver.cpp gvar1.cpp gvar2.cpp)
CREATE_GPU_TEST(kernel_lambda kernel_lambda.cpp)
CREATE_GPU_TEST(kernel_lambda_api kernel_lambda_api.cpp)
CREATE_GPU_TEST_FORCE(force_annotations force_annotations.cpp)

CREATE_GPU_TEST_RDC(kernel kernel.cpp)
CREATE_GPU_TEST_RDC(kernel_metadata kernel_metadata.cpp)
CREATE_GPU_TEST_RDC(kernel_pass_pipeline kernel_pass_pipeline.cpp)
CREATE_GPU_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST_RDC(kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST_RDC(kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_arg_api indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi_arg_api indirect_launcher_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST_RDC(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST_RDC(annot_nested annot_nested.cpp)
CREATE_GPU_TEST_RDC(types types.cpp)
CREATE_GPU_TEST_RDC(types_api types_api.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_func kernel_calls_func.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_func_api kernel_calls_func_api.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST_RDC(kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST_RDC(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST_RDC(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST_RDC(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST_RDC(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST_RDC(lambda lambda.cpp)
CREATE_GPU_TEST_RDC(shared_array shared_array.cpp)
CREATE_GPU_TEST_RDC(enable_disable enable_disable.cpp)
CREATE_GPU_TEST_RDC(lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST_RDC(lambda_def lambda_def.cpp)
CREATE_GPU_TEST_RDC(lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST_RDC(builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_indirect kernel_calls_indirect.cpp)
CREATE_GPU_TEST_RDC(global_var_register global_var_register.cpp)
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST_RDC(alias_func alias_func.cpp)
    # HIP does not support alias on global variables in RDC.
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA does not support alias in RDC.
endif()

CREATE_GPU_LIBRARY(device_func_lib device_func.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib device_func_lib kernel_calls_func_lib.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib_api device_func_lib kernel_calls_func_lib_api.cpp)
CREATE_GPU_TEST_RDC(mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST_RDC(types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST_RDC(dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST_RDC(jit_struct jit_struct.cpp)
CREATE_GPU_TEST_RDC(kernel_tuning kernel_tuning.cpp)
CREATE_GPU_TEST_RDC(kernel_lambda kernel_lambda.cpp)
CREATE_GPU_TEST_RDC(kernel_lambda_api kernel_lambda_api.cpp)
CREATE_GPU_TEST_RDC_FORCE(force_annotations force_annotations.cpp)

if(PROTEUS_ENABLE_HIP)
    if(LLVM_VERSION_MAJOR GREATER_EQUAL 18)
        function(CREATE_GPU_TEST_COMPRESS exe check_source)
            CREATE_GPU_TEST(${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP PRIVATE --offload-compress)
        endfunction()

        function(CREATE_GPU_TEST_RDC_COMPRESS exe check_source)
            CREATE_GPU_TEST_RDC(${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
        endfunction()

        CREATE_GPU_TEST_COMPRESS(kernel kernel.cpp)
        CREATE_GPU_TEST_RDC_COMPRESS(kernel kernel.cpp)
    endif()
endif()

add_subdirectory(scale100)
add_subdirectory(scale100-gvar)
