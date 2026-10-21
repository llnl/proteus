cmake_minimum_required(VERSION 3.18)

project(Proteus
  VERSION 2026.10.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
option(PROTEUS_ENABLE_HIP "Enable HIP" OFF)
option(PROTEUS_ENABLE_CUDA "Enable CUDA" OFF)
option(PROTEUS_ENABLE_NVCC_SHADOW
  "Use NVCC for CUDA AOT compilation and Clang for a shadow device pipeline"
  OFF)
option(PROTEUS_ENABLE_MPI "Enable MPI support for shared caching" OFF)
option(PROTEUS_ENABLE_MLIR "Enable MLIR backend" OFF)
option(BUILD_SHARED "Builds the JIT library as shared" OFF)
option(ENABLE_TESTS "Enable tests" OFF)
option(ENABLE_COVERAGE "Enable host-side coverage instrumentation" OFF)
option(ENABLE_DEVELOPER_COMPILER_FLAGS "Enable developer compiler flags: -Wall -Wextra -Werror" OFF)
option(PROTEUS_INSTALL_IMPL_HEADERS "Install implementation headers" OFF)
option(PROTEUS_ENABLE_PYTHON "Build Python bindings" OFF)
option(PROTEUS_PYTHON_WHEEL "Build wheel-oriented Python package layout" OFF)

set(PROTEUS_PYTHON_PACKAGE_DIR "proteus" CACHE STRING
  "Install-relative path for the Proteus Python package")
set(PROTEUS_PYTHON_PACKAGE_SOURCE_DIR "python/proteus_backend" CACHE STRING
  "Source-relative path to the Python package template used for native backend builds")

if(NOT IS_ABSOLUTE "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}")
  set(PROTEUS_PYTHON_PACKAGE_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}")
endif()

# Wheel builds provide LLVM_INSTALL_DIR through the environment, so mirror it
# into the CMake cache when the caller did not pass -DLLVM_INSTALL_DIR.
if((NOT DEFINED LLVM_INSTALL_DIR OR LLVM_INSTALL_DIR STREQUAL "")
   AND DEFINED ENV{LLVM_INSTALL_DIR}
   AND NOT "$ENV{LLVM_INSTALL_DIR}" STREQUAL "")
  set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH
    "Path to the LLVM installation used to build Proteus" FORCE)
endif()

# Enforce LLVM_INSTALL_DIR requirement.
if(NOT DEFINED LLVM_INSTALL_DIR OR LLVM_INSTALL_DIR STREQUAL "")
  message(FATAL_ERROR
    "LLVM_INSTALL_DIR is required but not defined. "
    "Please specify it with: cmake -DLLVM_INSTALL_DIR=/path/to/llvm/install ...")
endif()

# Verify LLVM_INSTALL_DIR exists.
if(NOT EXISTS "${LLVM_INSTALL_DIR}")
  message(FATAL_ERROR
    "LLVM_INSTALL_DIR points to non-existent directory: ${LLVM_INSTALL_DIR}")
endif()

message(STATUS "Using LLVM installation: ${LLVM_INSTALL_DIR}")

if (ENABLE_DEVELOPER_COMPILER_FLAGS)
  add_compile_options(-Wall -Wextra -Werror "-Wno-error=\#warnings" -Wno-error=unknown-cuda-version -Wno-error=invalid-feature-combination)
endif()

if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:C>:-g>"
      "$<$<COMPILE_LANGUAGE:CXX>:-g>"
      # Explicit form, not the --coverage alias: only these let a later
      # per-source -fno-profile-arcs cancel instrumentation (see
      # proteus_register_jit_pass_plugin in cmake/ProteusFunctions.cmake).
      "$<$<COMPILE_LANGUAGE:C>:SHELL:-fprofile-arcs -ftest-coverage>"
      "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-fprofile-arcs -ftest-coverage>"
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xarch_host -g>"
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xarch_host --coverage>"
      "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -g>"
      "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host --coverage>"
    )
    add_link_options(--coverage)
  else()
    message(FATAL_ERROR
      "ENABLE_COVERAGE is only supported with Clang compilers.")
  endif()
endif()

if(PROTEUS_ENABLE_HIP)
  add_definitions("-DPROTEUS_ENABLE_HIP")
  find_package(hip REQUIRED CONFIG)
  find_package(hiprtc REQUIRED CONFIG)
  message(STATUS "HIP Version: ${hip_VERSION}")
  message(STATUS "hiprtc Version: ${hiprtc_VERSION}")
  if(hip_VERSION VERSION_LESS "6.2.0")
    message(FATAL_ERROR "HIP found: ${hip_VERSION} is less than minimum required version 6.2.0.")
  endif()
endif()

if(PROTEUS_ENABLE_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "MPI Version: ${MPI_CXX_VERSION}")
endif()

if(PROTEUS_ENABLE_CUDA)
  add_definitions("-DPROTEUS_ENABLE_CUDA")

  find_package(CUDAToolkit 12 REQUIRED)
endif()

if(PROTEUS_ENABLE_NVCC_SHADOW)
  if(NOT PROTEUS_ENABLE_CUDA)
    message(FATAL_ERROR
      "PROTEUS_ENABLE_NVCC_SHADOW requires PROTEUS_ENABLE_CUDA=ON")
  endif()

  set(PROTEUS_NVCC_SHADOW_CLANG "${LLVM_INSTALL_DIR}/bin/clang++" CACHE FILEPATH
    "Clang compiler used by the NVCC shadow device pipeline")
  set(PROTEUS_NVCC_SHADOW_CUDA_ARCH "sm_80" CACHE STRING
    "CUDA architecture passed to the Clang shadow device compile")
  set(PROTEUS_NVCC_SHADOW_CUDA_PATH "${CUDAToolkit_ROOT}" CACHE PATH
    "CUDA toolkit root passed to the Clang shadow device compile")

  include(cmake/ProteusNvccShadow.cmake)
endif()

include(cmake/SetupLLVM.cmake)
include(cmake/ProteusFunctions.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src)

if(ENABLE_TESTS)
  enable_testing()
endif()

if(PROTEUS_ENABLE_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# TODO: Teach wheel builds to optionally ship the exported CMake package and
# headers once the Python wheel is ready to support C++ SDK consumers too.
if(NOT PROTEUS_PYTHON_WHEEL)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/proteusConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

  install(EXPORT proteusTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)

  install(FILES
    "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/ProteusFunctions.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/ProteusNvccShadow.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/ProteusNvccShadowEmbed.cmake"
    DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)
endif()
