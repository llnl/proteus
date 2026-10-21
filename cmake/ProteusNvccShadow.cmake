# Shadow device pipeline for NVCC-compiled targets.
#
# PROTOTYPE -- see docs/dev/design/nvcc-interop.md. Unvalidated on a real
# CUDA machine; the shadow compile and bitcode embedding run anywhere Clang
# with an NVPTX backend exists, the NVCC host half does not.
#
# add_proteus_nvcc(<target>
#     CLANG <path-to-clang++>          # Clang used for the shadow compile
#     [CUDA_ARCH <sm_XX>]              # defaults to sm_80
#     [CUDA_PATH <toolkit root>]       # passed as --cuda-path when set
# )
#
# For a CUDA target whose .cu sources are compiled by NVCC (with
# -ccbin <clang++> so the host half is Clang and can load the Proteus host
# plugins), this function adds, per source:
#
#   1. a shadow device-only Clang compile: the same source through
#      `clang++ -x cuda --cuda-device-only` with the Proteus pass plugin,
#      emitting the extracted LLVM bitcode and Proteus device-analysis
#      manifests,
#   2. a generated C registration file embedding that bitcode and a
#      constructor calling __proteus_register_shadow_bitcode(module-id, ...),
#      compiled into the target.
#
# CMake generates a stable compilation ID for each target/source pair and
# passes it to both compiler invocations. This avoids relying on NVCC's
# temporary cudafe source name, source-file inodes, or a shared parent PID.

function(add_proteus_nvcc target)
    set(oneValueArgs CLANG CUDA_ARCH CUDA_PATH)
    set(multiValueArgs OPTIONS)
    cmake_parse_arguments(SHADOW "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT TARGET ${target})
        message(FATAL_ERROR "add_proteus_nvcc target does not exist: ${target}")
    endif()

    if(NOT SHADOW_CLANG)
        message(FATAL_ERROR "add_proteus_nvcc requires CLANG <path-to-clang++>")
    endif()
    if(NOT SHADOW_CUDA_ARCH)
        set(SHADOW_CUDA_ARCH sm_80)
    endif()

    set(SHADOW_CUDA_PATH_FLAG "")
    if(SHADOW_CUDA_PATH)
        set(SHADOW_CUDA_PATH_FLAG "--cuda-path=${SHADOW_CUDA_PATH}")
    endif()

    # Host half: NVCC must use Clang as host compiler and load the Proteus
    # host plugins there. The IR pass instruments fatbin registration and
    # launches; the AST checker is omitted until vetted against cudafe
    # output (see the design doc).
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${SHADOW_CLANG}>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fplugin=$<TARGET_FILE:ProteusPass>>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fpass-plugin=$<TARGET_FILE:ProteusPass>>
    )
    add_dependencies(${target} ProteusPass)
    target_link_libraries(${target} PRIVATE proteus)

    get_target_property(SHADOW_SOURCES ${target} SOURCES)
    set(SHADOW_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target}.proteus-shadow")
    file(MAKE_DIRECTORY "${SHADOW_GEN_DIR}")

    foreach(Source IN LISTS SHADOW_SOURCES)
        get_filename_component(SourceExt "${Source}" LAST_EXT)
        get_source_file_property(SourceLanguage "${Source}" LANGUAGE)
        if(NOT SourceExt STREQUAL ".cu" AND
           NOT SourceLanguage STREQUAL "CUDA")
            continue()
        endif()
        get_filename_component(SourceAbs "${Source}" ABSOLUTE)
        get_filename_component(SourceName "${Source}" NAME_WE)
        string(SHA256 SourceHash "${target}|${SourceAbs}")
        string(SUBSTRING "${SourceHash}" 0 12 SourceHash)

        # Include a path hash so sources with the same basename do not share
        # generated outputs.
        set(ShadowStem "${SourceName}-${SourceHash}")
        set(ShadowBitcode "${SHADOW_GEN_DIR}/${ShadowStem}.shadow.bc")
        set(ShadowCompilerBitcode
            "${SHADOW_GEN_DIR}/${ShadowStem}.shadow.compiler.bc")
        set(ShadowRegC "${SHADOW_GEN_DIR}/${ShadowStem}.shadow.reg.c")
        set(CompilationId "shadow_${SourceHash}")
        set(ModuleId "_jit_bitcode_tu_${CompilationId}")

        set(TargetIncludeFlags
            "$<$<BOOL:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>>")
        set(TargetDefinitionFlags
            "$<$<BOOL:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>>:-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,;-D>>")
        set(ProteusIncludeFlags
            "$<$<BOOL:$<TARGET_PROPERTY:proteus,INTERFACE_INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_PROPERTY:proteus,INTERFACE_INCLUDE_DIRECTORIES>,;-I>>")

        # NVCC invokes Clang on its generated host source. Supply the same
        # explicit identity and manifest directory used by the independent
        # shadow device compile. The target-name condition keeps the source
        # property correct when a source is reused by multiple targets.
        set(HostIdentityOption
            "-Xcompiler=-mllvm,-proteus-compilation-id=${CompilationId},-mllvm,-proteus-manifest-dir=${SHADOW_GEN_DIR}")
        set_property(SOURCE "${Source}" APPEND PROPERTY COMPILE_OPTIONS
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<STREQUAL:$<TARGET_PROPERTY:NAME>,${target}>>:${HostIdentityOption}>")

        # 1. Shadow device-only compile with the Proteus pass.
        add_custom_command(
            OUTPUT "${ShadowBitcode}"
            BYPRODUCTS "${ShadowCompilerBitcode}"
            COMMAND "${SHADOW_CLANG}" -x cuda --cuda-device-only
                    --cuda-gpu-arch=${SHADOW_CUDA_ARCH}
                    ${SHADOW_CUDA_PATH_FLAG}
                    -std=c++17 -O1 -emit-llvm -c
                    -DPROTEUS_ENABLE_CUDA
                    -fplugin=$<TARGET_FILE:ProteusPass>
                    -fpass-plugin=$<TARGET_FILE:ProteusPass>
                    "${TargetIncludeFlags}"
                    "${TargetDefinitionFlags}"
                    "${ProteusIncludeFlags}"
                    -mllvm -proteus-compilation-id=${CompilationId}
                    -mllvm -proteus-manifest-dir=${SHADOW_GEN_DIR}
                    -mllvm -proteus-shadow-bitcode-output=${ShadowBitcode}
                    "$<TARGET_PROPERTY:${target},PROTEUS_NVCC_SHADOW_OPTIONS>"
                    ${SHADOW_OPTIONS}
                    -o "${ShadowCompilerBitcode}" "${SourceAbs}"
            DEPENDS "${SourceAbs}" ProteusPass
            COMMENT "Proteus shadow device compile: ${SourceName}"
            COMMAND_EXPAND_LISTS
            VERBATIM)

        # Compile the host object only after the shadow pass has emitted the
        # annotation and lambda manifests it consumes.
        set_property(SOURCE "${Source}" APPEND PROPERTY OBJECT_DEPENDS
            "${ShadowBitcode}")

        # 2. Embed + register under the exact ID recorded by the host pass.
        add_custom_command(
            OUTPUT "${ShadowRegC}"
            COMMAND ${CMAKE_COMMAND}
                    -DBITCODE_FILE=${ShadowBitcode}
                    -DMODULE_ID=${ModuleId}
                    -DOUTPUT_FILE=${ShadowRegC}
                    -P ${PROTEUS_NVCC_SHADOW_EMBED_SCRIPT}
            DEPENDS "${ShadowBitcode}" ${PROTEUS_NVCC_SHADOW_EMBED_SCRIPT}
            COMMENT "Proteus shadow bitcode embedding: ${SourceName}"
            VERBATIM)

        target_sources(${target} PRIVATE "${ShadowRegC}")
    endforeach()
endfunction()

# Resolve paths relative to this module file for installed and in-tree use.
set(PROTEUS_NVCC_SHADOW_EMBED_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/ProteusNvccShadowEmbed.cmake")
