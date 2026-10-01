file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py CONTENT "
import lit.formats
import os

config.name = 'Proteus AST checker tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()
config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
config.substitutions.append(('%clang', '${LLVM_TOOLS_BINARY_DIR}/clang++'))
config.substitutions.append(('%plugin', '$<TARGET_FILE:ProteusASTChecker>'))
config.substitutions.append(('%proteus_include', '${PROJECT_SOURCE_DIR}/include'))
")

# Exercise the same add_proteus() path used by installed CMake consumers, in
# addition to the direct driver invocations in the lit tests below.
add_library(proteus_ast_checker_cmake_test OBJECT lambda_valid.cpp)
add_proteus(proteus_ast_checker_cmake_test)

add_test(NAME lambda_ast_checker
  COMMAND ${LIT} -vv .)
set_tests_properties(lambda_ast_checker PROPERTIES LABELS "ast")
