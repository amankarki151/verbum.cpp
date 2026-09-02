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
  core/src/nn.cpp
  core/src/attention.cpp
  core/src/model.cpp
  core/src/generate.cpp
  core/src/quant.cpp
  
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

add_executable(test_nn tests/test_nn.cpp)
target_link_libraries(test_nn PRIVATE verbum_core)
add_test(NAME nn COMMAND test_nn)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE verbum_core)
add_test(NAME attention COMMAND test_attention)

add_executable(test_block tests/test_block.cpp)
target_link_libraries(test_block PRIVATE verbum_core)
add_test(NAME block COMMAND test_block)

add_executable(test_logits tests/test_logits.cpp)
target_link_libraries(test_logits PRIVATE verbum_core)

add_executable(test_cache tests/test_cache.cpp)
target_link_libraries(test_cache PRIVATE verbum_core)
add_test(NAME cache COMMAND test_cache)

add_executable(generate tools/generate.cpp)
target_link_libraries(generate PRIVATE verbum_core)

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE verbum_core)
add_test(NAME quant COMMAND test_quant)