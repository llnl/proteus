# Integration

## CMake

To integrate Proteus with CMake, add the install prefix to `CMAKE_PREFIX_PATH`,
or set `proteus_DIR` to the directory containing `proteusConfig.cmake`, typically
`<install-prefix>/lib/cmake/proteus` or `<install-prefix>/lib64/cmake/proteus`.

Then, in your project's `CMakeLists.txt` add:
```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(<target>)
```

`find_package(proteus)` also resolves the dependencies recorded in the Proteus
installation. In practice, this means your CMake environment must be able to
find the matching LLVM and Clang packages, plus any backend-specific
dependencies that Proteus was built with, such as `CUDAToolkit`, `hip`,
`hiprtc`, `LLD`, or `MPI`.

If you only need the DSL, C++ frontend, or MLIR frontend APIs, you can link directly against
`proteusFrontend`.
In this case, you don't need to compile your target with Clang:

```cmake
find_package(proteus CONFIG REQUIRED)

target_link_libraries(<target> ... proteusFrontend ...)
```

## Make

With `make`, annotation-based integration requires adding compilation and
linking flags, for example:

```bash
CXXFLAGS += -I<install_path>/include \
    -fplugin=<install_path>/<libdir>/libProteusASTChecker.so \
    -fpass-plugin=<install_path>/<libdir>/libProteusPass.so

LDFLAGS += -L<install_path>/<libdir> \
    -Wl,-rpath,<install_path>/<libdir> \
    -lproteus $(llvm-config --libs) -lclang-cpp
```

The frontend checker loaded by `-fplugin` diagnoses unsupported uses of
`proteus::register_lambda` and `proteus::jit_variable` before LLVM IR is
emitted. If you don't use code annotations or these lambda APIs, you can omit
both plugin options.

!!! note
    The example above is the simplest case and is closest to a host-only build.
    If Proteus was built with CUDA or HIP enabled, you will also need to link
    the corresponding backend libraries required by your application and by the
    installed Proteus library. For example, CUDA builds typically also need CUDA
    driver/runtime libraries, while HIP builds typically also need HIP runtime
    libraries such as `hiprtc`.

!!! note
    The example above links against `-lclang-cpp`, which is the monolithic
    Clang C++ library available in most LLVM distributions.
    If your toolchain instead provides static Clang component archives
    (e.g. `libclangFrontend.a`, `libclangDriver.a`, etc.), you'll need to link
    those explicitly instead of `-lclang-cpp`. On ELF platforms, remember to wrap
    them in `-Wl,--start-group ... -Wl,--end-group` to resolve circular
    dependencies.

## Link with `-rdynamic`

When using Proteus and your JIT-compiled code needs to call functions defined in
your main program, you must link your main application with the `-rdynamic`
flag.
This flag ensures that symbols from your main program are exported and visible
to the Proteus JIT, allowing external function calls from JIT code to resolve
correctly at runtime.
