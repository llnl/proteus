function(add_proteus target)
    set(options FORCE_JIT_ANNOTATE_ALL)
    cmake_parse_arguments(PROTEUS_PASS "${options}" "" "" ${ARGN})

    if (PROTEUS_PASS_FORCE_JIT_ANNOTATE_ALL)
        target_compile_options(${target} PRIVATE
            # -fplugin loading is needed to register the plugin command line
            # option.
            "-fplugin=$<TARGET_FILE:ProteusPass>"
            "-fpass-plugin=$<TARGET_FILE:ProteusPass>"
            "SHELL:-Xclang -mllvm -Xclang -force-proteus-jit-annotate-all"
            "-fplugin=\$<TARGET_FILE:ProteusASTChecker>"
        )
    else()
        target_compile_options(${target} PRIVATE
            "-fpass-plugin=\$<TARGET_FILE:ProteusPass>"
            "-fplugin=\$<TARGET_FILE:ProteusASTChecker>"
        )
    endif()

    add_dependencies(${target} ProteusPass ProteusASTChecker)

    target_link_options(${target} PRIVATE
        "SHELL:\$<\$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=\$<TARGET_FILE:ProteusPassOffload>>")

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        target_link_libraries(${target} PRIVATE proteus)
    else()
        target_link_libraries(${target} PUBLIC proteus)
    endif()

    message(STATUS "Linked target '${target}' with libproteus")
endfunction()

function(proteus_register_jit_pass_plugin target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Target '${target}' does not exist")
    endif()

    set(one_value_args PLUGIN_TARGET PLUGIN_PATH PIPELINE POSITION)
    cmake_parse_arguments(PROTEUS_JIT_PASS "" "${one_value_args}" "" ${ARGN})

    if(PROTEUS_JIT_PASS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "proteus_register_jit_pass_plugin received unknown arguments: ${PROTEUS_JIT_PASS_UNPARSED_ARGUMENTS}")
    endif()

    if(PROTEUS_JIT_PASS_POSITION AND NOT PROTEUS_JIT_PASS_PIPELINE)
        message(FATAL_ERROR
            "proteus_register_jit_pass_plugin POSITION requires a nonempty PIPELINE")
    endif()

    if(PROTEUS_JIT_PASS_POSITION AND
       NOT PROTEUS_JIT_PASS_POSITION STREQUAL "PREPEND" AND
       NOT PROTEUS_JIT_PASS_POSITION STREQUAL "APPEND")
        message(FATAL_ERROR
            "proteus_register_jit_pass_plugin POSITION must be PREPEND or APPEND")
    endif()

    if(PROTEUS_JIT_PASS_PLUGIN_TARGET AND PROTEUS_JIT_PASS_PLUGIN_PATH)
        message(FATAL_ERROR
            "proteus_register_jit_pass_plugin accepts either PLUGIN_TARGET or PLUGIN_PATH, not both")
    endif()

    if(NOT PROTEUS_JIT_PASS_PLUGIN_TARGET AND NOT PROTEUS_JIT_PASS_PLUGIN_PATH)
        message(FATAL_ERROR
            "proteus_register_jit_pass_plugin requires PLUGIN_TARGET or PLUGIN_PATH")
    endif()

    if(PROTEUS_JIT_PASS_PLUGIN_TARGET)
        if(NOT TARGET ${PROTEUS_JIT_PASS_PLUGIN_TARGET})
            message(FATAL_ERROR
                "Plugin target '${PROTEUS_JIT_PASS_PLUGIN_TARGET}' does not exist")
        endif()
        set(_proteus_jit_pass_plugin_path "$<TARGET_FILE:${PROTEUS_JIT_PASS_PLUGIN_TARGET}>")
        add_dependencies(${target} ${PROTEUS_JIT_PASS_PLUGIN_TARGET})
    else()
        set(_proteus_jit_pass_plugin_path "${PROTEUS_JIT_PASS_PLUGIN_PATH}")
    endif()

    if(PROTEUS_JIT_PASS_PIPELINE)
        if(PROTEUS_JIT_PASS_POSITION STREQUAL "PREPEND")
            set(_proteus_jit_pass_position "Prepend")
        else()
            set(_proteus_jit_pass_position "Append")
        endif()
        set(_proteus_jit_pass_registration
"    proteus::registerJITPassPlugin(
        R\"(${_proteus_jit_pass_plugin_path})\",
        R\"(${PROTEUS_JIT_PASS_PIPELINE})\",
        proteus::JITPassPluginPosition::${_proteus_jit_pass_position});")
    else()
        set(_proteus_jit_pass_position "LoadOnly")
        set(_proteus_jit_pass_registration
"    proteus::registerJITPassPlugin(
        R\"(${_proteus_jit_pass_plugin_path})\");")
    endif()

    string(MD5 _proteus_jit_pass_key
        "${target};${_proteus_jit_pass_plugin_path};${_proteus_jit_pass_position};${PROTEUS_JIT_PASS_PIPELINE}")
    set(_proteus_jit_pass_source
        "${CMAKE_CURRENT_BINARY_DIR}/${target}.proteus_jit_pass_${_proteus_jit_pass_key}.cpp")

    file(GENERATE OUTPUT "${_proteus_jit_pass_source}" CONTENT
"#include <proteus/Init.h>

namespace {
struct AutoRegisterProteusJITPassPlugin {
  AutoRegisterProteusJITPassPlugin() {
${_proteus_jit_pass_registration}
  }
};

AutoRegisterProteusJITPassPlugin AutoRegisterProteusJITPassPluginInstance;
} // namespace
")

    if(ENABLE_COVERAGE)
        # This generated TU only registers the JIT pass plugin and does not
        # represent product code. Exclude it from gcov instrumentation so
        # gcovr does not try to resolve a synthetic build-tree-only source.
        # Relies on coverage being enabled with the explicit
        # -fprofile-arcs/-ftest-coverage flags (see CMakeLists.txt), not the
        # --coverage alias, which these -fno- flags cannot cancel.
        set_source_files_properties("${_proteus_jit_pass_source}" PROPERTIES
            COMPILE_OPTIONS "-fno-profile-arcs;-fno-test-coverage")
    endif()

    target_sources(${target} PRIVATE "${_proteus_jit_pass_source}")
endfunction()
