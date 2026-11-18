add_executable(contextual_use_def contextual_use_def.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/Error.cpp)
target_include_directories(contextual_use_def PRIVATE
    ${PROJECT_SOURCE_DIR}/src/pass
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include)
target_include_directories(contextual_use_def SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
if(LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(contextual_use_def PRIVATE LLVM)
else()
    llvm_map_components_to_libnames(contextual_use_def_libs
        Analysis AsmParser Core Support TransformUtils)
    target_link_libraries(contextual_use_def PRIVATE ${contextual_use_def_libs})
endif()
if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(contextual_use_def PRIVATE -fno-rtti)
endif()
add_test(NAME contextual_use_def COMMAND contextual_use_def
    ${CMAKE_CURRENT_SOURCE_DIR}/contextual_use_def.ll)

foreach(case pointer_slot nested_holder pointer_array returned_pointer)
    add_test(NAME contextual_use_def.${case} COMMAND contextual_use_def
        ${CMAKE_CURRENT_SOURCE_DIR}/contextual_reachable_storage.ll ${case})
endforeach()
