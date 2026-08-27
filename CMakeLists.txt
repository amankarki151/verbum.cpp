cmake_minimum_required(VERSION 3.20)
project(verbum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

add_library(verbum_core
  core/src/config.cpp
  core/src/safetensors.cpp
  core/src/tokenizer.cpp
  core/src/tensor.cpp
  core/src/ops.cpp
)

target_include_directories(verbum_core PUBLIC
  ${CMAKE_SOURCE_DIR}/core/include
  ${CMAKE_SOURCE_DIR}/third_party
)

enable_testing()

add_executable(test_safetensors tests/test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE verbum_core)
add_test(NAME safetensors COMMAND test_safetensors)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE verbum_core)
add_test(NAME tokenizer COMMAND test_tokenizer)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE verbum_core)
add_test(NAME ops COMMAND test_ops)