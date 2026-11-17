set(SOURCES
  VariantRegistry.cpp
  CompilerInterfaceHost.cpp
  CompilerInterfaceRuntimeConstantInfo.cpp
  CompiledLibrary.cpp
  Error.cpp
  KernelMetadata.cpp
  JitEngine.cpp
  JITPassPluginRegistry.cpp
  JitEngineHost.cpp
  Init.cpp
  TimeTracing.cpp
  Caching/ObjectCacheChain.cpp
  Caching/StorageCache.cpp
  Frontend/CppJitCompiler.cpp
  Frontend/CppJitCompilerClang.cpp
  Frontend/CUDAToolchain.cpp
  Frontend/HIPToolchain.cpp
  Frontend/JitFuncAttribute.cpp
  Frontend/CppJitModule.cpp
  Frontend/LLVMIRJitModule.cpp
  Frontend/CppJitCompilerNvcc.cpp
  Frontend/Builtins.cpp
  Frontend/Dispatcher.cpp
  Frontend/Func.cpp
  Frontend/JitFrontend.cpp
  Frontend/LLVMCodeBuilder.cpp
  Frontend/TargetModel.cpp
  Frontend/TypeMap.cpp
  Frontend/LLVMTypeMap.cpp
)

set(PROTEUS_PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

if(PROTEUS_ENABLE_CUDA)
  if(PROTEUS_PYTHON_WHEEL AND UNIX AND NOT APPLE)
    list(APPEND SOURCES CUDADriverAPI.cpp)
  endif()

  list(APPEND SOURCES
    CompilerInterfaceDevice.cpp
    JitEngineDeviceCUDA.cpp
  )
endif()

if(PROTEUS_ENABLE_HIP)
  list(APPEND SOURCES
    HIPRuntimeAPI.cpp
    CompilerInterfaceDevice.cpp
    JitEngineDeviceHIP.cpp
  )
endif()

if(PROTEUS_ENABLE_MLIR)
  list(APPEND SOURCES
    Frontend/MLIRJitModule.cpp
    Frontend/MLIRLower.cpp
    Frontend/MLIRCodeBuilder.cpp
  )
endif()

if(PROTEUS_ENABLE_MPI)
  list(APPEND SOURCES
    Caching/MPIHelpers.cpp
    Caching/MPIStorageCache.cpp
    Caching/MPILocalLookupCache.cpp
    Caching/MPIRemoteLookupCache.cpp
  )
endif()

if(BUILD_SHARED)
  add_library(proteus SHARED ${SOURCES})
else()
  add_library(proteus STATIC ${SOURCES})
  set_target_properties(proteus PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(proteusCore INTERFACE)
add_library(proteusFrontend INTERFACE)

target_compile_definitions(proteus PRIVATE ${LLVM_DEFINITIONS})
target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

if(DEFINED LLVM_INSTALL_PREFIX AND NOT LLVM_INSTALL_PREFIX STREQUAL "")
  target_compile_definitions(
    proteus PRIVATE PROTEUS_LLVM_TOOLS_BINDIR="${LLVM_INSTALL_PREFIX}/bin")
endif()

if(PROTEUS_ENABLE_MPI)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_MPI)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_MPI)
endif()

if(PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_HIP)
  string(REGEX MATCH "^[0-9]+" PROTEUS_HIP_VERSION_MAJOR "${hip_VERSION}")
  string(REGEX MATCH "^[0-9]+" PROTEUS_HIPRTC_VERSION_MAJOR "${hiprtc_VERSION}")
  get_filename_component(PROTEUS_HIP_INSTALL_ROOT
    "${hip_DIR}/../../.." REALPATH)
  target_compile_definitions(
    proteus PRIVATE
    PROTEUS_HIP_VERSION="${hip_VERSION}"
    PROTEUS_HIP_INSTALL_ROOT="${PROTEUS_HIP_INSTALL_ROOT}"
    PROTEUS_HIP_RUNTIME_LIBRARY_SONAME="libamdhip64.so.${PROTEUS_HIP_VERSION_MAJOR}"
    PROTEUS_HIPRTC_LIBRARY_SONAME="libhiprtc.so.${PROTEUS_HIPRTC_VERSION_MAJOR}")
elseif(PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_CUDA)
  target_compile_definitions(
    proteus PRIVATE PROTEUS_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}")
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_CUDA)
endif()

if(PROTEUS_ENABLE_MLIR)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_MLIR)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_MLIR)
endif()

target_include_directories(proteus SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(proteusCore SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})

if(PROTEUS_ENABLE_MLIR)
  target_include_directories(proteus SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})
  target_include_directories(proteusCore SYSTEM INTERFACE ${MLIR_INCLUDE_DIRS})
endif()

target_include_directories(proteus
  PUBLIC
  $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${PROJECT_SOURCE_DIR}/src/include
)

target_include_directories(proteusCore
  INTERFACE
  $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link with LLVM libraries dynamically or statically based on the
# LLVM_LINK_LLVM_DYLIB option provided by the LLVM cmake setup.
if(LLVM_LINK_LLVM_DYLIB)
  if(NOT TARGET LLVM)
    message(FATAL_ERROR "Missing LLVM target. Check LLVM cmake targets")
  endif()
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs
    Analysis
    BitReader
    BitWriter
    Core
    ExecutionEngine
    IPO
    IRReader
    Linker
    MC
    ${LLVM_TARGETS_TO_BUILD}
    Object
    OrcJit
    Passes
    Support
    Target
  )
endif()
message(STATUS "Found LLVM libraries: ${llvm_libs}")
target_link_libraries(proteus PRIVATE ${llvm_libs})

if(PROTEUS_ENABLE_MLIR)
	set(mlir_libs
	  MLIRConvertToLLVMPass
	  MLIRArithToLLVM
	  MLIRFuncDialect
	  MLIRFuncToLLVM
	  MLIRArithDialect
	  MLIRGPUDialect
	  MLIRControlFlowDialect
	  MLIRControlFlowToLLVM
	  MLIRLLVMDialect
	  MLIRIndexToLLVM
	  MLIRMathDialect
	  MLIRMathToLLVM
	  MLIRMemRefDialect
	  MLIRMemRefToLLVM
	  MLIRPass
	  MLIRReconcileUnrealizedCasts
	  MLIRSCFDialect
	  MLIRSCFToControlFlow
	  MLIRBuiltinToLLVMIRTranslation
	  MLIRLLVMToLLVMIRTranslation
	  MLIRIR
	  MLIRSupport
	  MLIRTargetLLVM
	  MLIRTargetLLVMIRExport
  )

  # MLIR GPU lowering/translation for CUDA/HIP requires target-specific
  # dialect + conversion + LLVM IR translation libraries to be explicitly
  # linked so symbols are available in libproteus at runtime.
  if(PROTEUS_ENABLE_CUDA)
    list(APPEND mlir_libs
      MLIRNVVMDialect
      MLIRGPUToNVVMTransforms
      MLIRNVVMToLLVM
      MLIRGPUToLLVMIRTranslation
      MLIRNVVMToLLVMIRTranslation
    )
  endif()

  if(PROTEUS_ENABLE_HIP)
    list(APPEND mlir_libs
      MLIRROCDLDialect
      MLIRGPUToROCDLTransforms
      MLIRAMDGPUToROCDL
      MLIRGPUToLLVMIRTranslation
      MLIRROCDLToLLVMIRTranslation
    )
  endif()

  message(STATUS "Linking MLIR libraries: ${mlir_libs}")
  target_link_libraries(proteus PRIVATE ${mlir_libs})
endif()

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(proteus PRIVATE -fno-rtti)
  target_compile_options(proteusCore INTERFACE -fno-rtti)
endif()

if(PROTEUS_ENABLE_HIP)
  target_include_directories(proteus SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_include_directories(proteusCore SYSTEM INTERFACE ${hip_INCLUDE_DIRS})
  target_compile_definitions(
    proteus PRIVATE
    $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_DEFINITIONS>)
  target_compile_options(
    proteus PRIVATE
    $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_OPTIONS>)
  target_compile_definitions(
    proteusCore INTERFACE
    $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_DEFINITIONS>)
  target_compile_options(
    proteusCore INTERFACE
    $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_OPTIONS>)
  if(UNIX)
    target_link_libraries(proteus PRIVATE ${CMAKE_DL_LIBS})
  endif()

  # NOTE: HIP compilation (-x hip) is needed to include the HIP_SYMBOL macro,
  # which depends on the target architecture AMD or NVIDIA.  We define this
  # macro for HIP compilation in our headers to avoid forcing HIP compilation.
  # target_compile_options(proteus PRIVATE -x hip)
  # target_compile_options(proteusCore INTERFACE -x hip)


  if(NOT TARGET lldCommon)
    message(FATAL_ERROR "Missing lldCommon target. Check LLD cmake targets")
  endif()
  if(NOT TARGET lldELF)
    message(FATAL_ERROR "Missing lldELF target. Check LLD cmake targets")
  endif()

  set(lld_libs lldCommon lldELF)
  message(STATUS "Found LLD libraries: ${lld_libs}")
  target_link_libraries(proteus PRIVATE ${lld_libs})
endif()

if(PROTEUS_ENABLE_CUDA)
  add_library(proteus_cudart_builtins STATIC ProteusCUDARuntimeBuiltins.cpp)
  # Compiling builtins needs only the headers.
  target_include_directories(proteus_cudart_builtins PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(proteus_cudart_builtins PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(proteus INTERFACE proteus_cudart_builtins)

  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # target_link_libraries(... CUDA::nvptxcompiler_static)
  find_library(LIBNVPTXCOMPILER
    NAMES nvptxcompiler nvptxcompiler_static libnvptxcompiler_static
    REQUIRED
    NO_DEFAULT_PATH
    PATHS ${CUDAToolkit_LIBRARY_DIR}
  )
  message(STATUS "Found libnvptxcompiler: ${LIBNVPTXCOMPILER}")

  target_include_directories(proteus SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  if(PROTEUS_PYTHON_WHEEL AND UNIX AND NOT APPLE)
    target_link_libraries(proteus PRIVATE ${CMAKE_DL_LIBS})
  else()
    # Link just with the CUDA driver, leave the CUDA runtime to be linked by
    # the application static or shared. Proteus resolves its CUDA runtime
    # dependencies at runtime through proteus builtins.
    target_link_libraries(proteus PUBLIC CUDA::cuda_driver)
  endif()
  target_link_libraries(proteus PRIVATE "${LIBNVPTXCOMPILER}")
  target_include_directories(proteusCore SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

endif()

if(PROTEUS_ENABLE_MPI)
  target_link_libraries(proteus PRIVATE MPI::MPI_CXX)
endif()

# Find required clang libraries in the LLVM installation.
if(CLANG_LINK_CLANG_DYLIB)
    list(APPEND clang_libs clang-cpp)
else()
  if(NOT TARGET clangFrontend)
    message(FATAL_ERROR "Missing clangCodeGen target. Check Clang cmake targets")
  endif()

  if(NOT TARGET clangCodeGen)
    message(FATAL_ERROR "Missing clangCodeGen target. Check Clang cmake targets")
  endif()

  if(NOT TARGET clangDriver)
    message(FATAL_ERROR "Missing clangDriver target. Check Clang cmake targets")
  endif()

  list(APPEND clang_libs clangDriver clangFrontend clangCodeGen)
endif()
message(STATUS "Found Clang libraries: ${clang_libs}")
target_link_libraries(proteus PRIVATE ${clang_libs})

find_package(Threads REQUIRED)
target_link_libraries(proteus PRIVATE Threads::Threads)

if(BUILD_SHARED)
# Use a version script to hide internal symbols, especially from LLVM static
# libraries, on Linux. Only effective for shared proteus builds.
  if(UNIX AND NOT APPLE)
    target_link_options(proteus PRIVATE
      "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
      "-Wl,-z,defs")
  endif()
else()
  # When proteus is static and links other libraries statically, propagate
  # --exclude-libs for those specific archives so that consuming shared
  # libraries don't export duplicate global state from those libraries.
  # Possible static library deps: LLVM, Clang, LLD, nvptxcompiler.
  if(UNIX AND NOT APPLE)
    set(_exclude_archives "")

    if(NOT LLVM_LINK_LLVM_DYLIB)
      foreach(lib ${llvm_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(NOT CLANG_LINK_CLANG_DYLIB)
      foreach(lib ${clang_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(PROTEUS_ENABLE_HIP)
      foreach(lib ${lld_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(PROTEUS_ENABLE_CUDA)
      get_filename_component(_nvptxcompiler_name ${LIBNVPTXCOMPILER} NAME)
      list(APPEND _exclude_archives ${_nvptxcompiler_name})
    endif()

    if(_exclude_archives)
      list(JOIN _exclude_archives ":" _exclude_libs_str)
      target_link_options(proteus INTERFACE
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:$<HOST_LINK:LINKER:SHELL:--exclude-libs ${_exclude_libs_str}>>"
      )
    endif()
  endif()
endif()

target_link_libraries(proteusFrontend INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_link_libraries(proteusCore INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

set_target_properties(proteus PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,\$ORIGIN>"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

# TODO: Teach wheel builds to optionally ship the exported C++ SDK surface
# (headers, interface targets, and related install metadata) once that becomes
# a supported packaging contract.
if(PROTEUS_PYTHON_WHEEL)
  install(
    TARGETS proteus
    LIBRARY DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
    RUNTIME DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
  )
else()
  set(PROTEUS_INSTALL_TARGETS proteus proteusCore proteusFrontend)
  if(PROTEUS_ENABLE_CUDA)
      list(APPEND PROTEUS_INSTALL_TARGETS proteus_cudart_builtins)
  endif()

  install(
    TARGETS ${PROTEUS_INSTALL_TARGETS}
    EXPORT proteusTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  # Install headers preserving directory structure.
  set(PROTEUS_PUBLIC_HEADER_INSTALL_EXCLUDES)
  if(NOT PROTEUS_ENABLE_MLIR)
    # Exclude MLIR-specific headers from installation if MLIR support is not
    # enabled.
    list(APPEND PROTEUS_PUBLIC_HEADER_INSTALL_EXCLUDES
      PATTERN "proteus/Frontend/MLIRCodeBuilder.h" EXCLUDE
      PATTERN "proteus/MLIRJitModule.h" EXCLUDE
    )
  endif()
  install(DIRECTORY ${PROTEUS_PROJECT_INCLUDE_DIR}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
    ${PROTEUS_PUBLIC_HEADER_INSTALL_EXCLUDES}
  )

  if(PROTEUS_INSTALL_IMPL_HEADERS)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include/
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
      FILES_MATCHING PATTERN "*.h"
    )
  endif()
endif()
