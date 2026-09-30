find_program(LIT lit
    PATHS ${LLVM_BINARY_DIR}/bin)

if(NOT LIT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import lit"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    if(${EXIT_CODE} EQUAL 0)
        set(LIT ${CMAKE_CURRENT_SOURCE_DIR}/lit-main.py)
        message(STATUS "Using ${LIT} for lit testing")
    else()
        message(FATAL_ERROR "Testing requires lit to be installed, e.g., python -m pip install lit>=15.0.7")
    endif()
else()
    message(STATUS "Found lit at ${LIT}")
endif()

find_program(FILECHECK FileCheck
    PATHS ${LLVM_BINARY_DIR}/bin ${LLVM_BINARY_DIR}/libexec/llvm)

if(NOT FILECHECK)
    message(FATAL_ERROR "Testing requires FileCheck to be installed")
else()
    message(STATUS "Found Filecheck at ${FILECHECK}")
endif()

add_library(JITTestPass MODULE JITTestPass.cpp)
target_compile_definitions(JITTestPass PRIVATE ${LLVM_DEFINITIONS})
target_include_directories(JITTestPass SYSTEM PRIVATE "${LLVM_INSTALL_DIR}/include")

if(LLVM_LINK_LLVM_DYLIB)
    set(jit_test_pass_llvm_libs LLVM)
else()
    llvm_map_components_to_libnames(jit_test_pass_llvm_libs
        Core
        Passes
        Support
    )
endif()
target_link_libraries(JITTestPass PRIVATE ${jit_test_pass_llvm_libs})
target_link_options(JITTestPass PRIVATE
    "$<$<PLATFORM_ID:Darwin>:LINKER:SHELL:-undefined dynamic_lookup>"
    "$<$<AND:$<NOT:$<PLATFORM_ID:Darwin>>,$<BOOL:${UNIX}>>:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/JITTestPass.exports.map>"
)

if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(JITTestPass PRIVATE -fno-rtti)
endif()

function(proteus_attach_pass_plugin_rebuild_dep target)
    set(proteus_pass_stamp "${CMAKE_CURRENT_BINARY_DIR}/${target}.proteuspass.stamp")

    # Rebuild test sources when the pass plugin is rebuilt, since the compiler
    # consumes the plugin during compilation via -fpass-plugin.
    add_custom_command(
        OUTPUT "${proteus_pass_stamp}"
        COMMAND ${CMAKE_COMMAND} -E touch "${proteus_pass_stamp}"
        DEPENDS ProteusPass ProteusASTChecker
        VERBATIM
    )

    set_source_files_properties(${ARGN} PROPERTIES
        OBJECT_DEPENDS "${proteus_pass_stamp}"
    )
    target_compile_options(
        ${target}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusPass>
        -fplugin=$<TARGET_FILE:ProteusASTChecker>
    )
    add_dependencies(${target} ProteusASTChecker)
endfunction()

add_subdirectory(cpu)
add_subdirectory(ast)

if(PROTEUS_ENABLE_HIP OR PROTEUS_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()

add_subdirectory(frontend)
