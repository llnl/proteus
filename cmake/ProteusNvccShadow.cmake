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
#      emitting LLVM bitcode that contains the Proteus device analysis
#      results (_jit_bitcode_* embedding, lambda manifests),
#   2. a generated C registration file embedding that bitcode and a
#      constructor calling __proteus_register_shadow_bitcode(module-id, ...),
#      compiled into the target.
#
# The module ID must match what the host pass records at the fatbinary
# registration site: Proteus derives it from the source file's unique ID, so
# the shadow compile passes the same source path. Kernels absent from the
# shadow module (TU divergence, unsupported constructs) degrade to NVCC's
# own binaries at runtime.

function(add_proteus_nvcc target)
    set(oneValueArgs CLANG CUDA_ARCH CUDA_PATH)
    cmake_parse_arguments(SHADOW "" "${oneValueArgs}" "" ${ARGN})

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
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fpass-plugin=$<TARGET_FILE:ProteusPass>>
    )
    add_dependencies(${target} ProteusPass)
    target_link_libraries(${target} PRIVATE proteus)

    get_target_property(SHADOW_SOURCES ${target} SOURCES)
    set(SHADOW_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target}.proteus-shadow")
    file(MAKE_DIRECTORY "${SHADOW_GEN_DIR}")

    foreach(Source IN LISTS SHADOW_SOURCES)
        get_filename_component(SourceExt "${Source}" LAST_EXT)
        if(NOT SourceExt STREQUAL ".cu")
            continue()
        endif()
        get_filename_component(SourceAbs "${Source}" ABSOLUTE)
        get_filename_component(SourceName "${Source}" NAME_WE)

        set(ShadowBitcode "${SHADOW_GEN_DIR}/${SourceName}.shadow.bc")
        set(ShadowRegC "${SHADOW_GEN_DIR}/${SourceName}.shadow.reg.c")

        # 1. Shadow device-only compile with the Proteus pass.
        add_custom_command(
            OUTPUT "${ShadowBitcode}"
            COMMAND "${SHADOW_CLANG}" -x cuda --cuda-device-only
                    --cuda-gpu-arch=${SHADOW_CUDA_ARCH}
                    ${SHADOW_CUDA_PATH_FLAG}
                    -std=c++17 -O1 -emit-llvm -c
                    -fpass-plugin=$<TARGET_FILE:ProteusPass>
                    -I "${PROTEUS_INCLUDE_DIR}"
                    -o "${ShadowBitcode}" "${SourceAbs}"
            DEPENDS "${SourceAbs}" ProteusPass
            COMMENT "Proteus shadow device compile: ${SourceName}"
            VERBATIM)

        # 2. Embed + register. The module id mirrors the host pass's
        # getUniqueFileID-derived id for this TU (source path based).
        add_custom_command(
            OUTPUT "${ShadowRegC}"
            COMMAND ${CMAKE_COMMAND}
                    -DBITCODE_FILE=${ShadowBitcode}
                    -DSOURCE_FILE=${SourceAbs}
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
if(NOT DEFINED PROTEUS_INCLUDE_DIR)
    get_filename_component(PROTEUS_INCLUDE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)
endif()
