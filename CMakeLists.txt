cmake_minimum_required(VERSION 3.18)
project(TinyInfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TINYINFER_NATIVE "Compile for the host CPU (enables AVX2/FMA kernels when available)" ON)
option(TINYINFER_BUILD_TESTS "Build GoogleTest suite" ON)
option(TINYINFER_BUILD_PYTHON "Build pybind11 module" ON)

find_package(OpenMP REQUIRED)

add_library(tinyinfer
  src/onnx_reader.cpp
  src/gemm.cpp
  src/ops.cpp
  src/session.cpp)
target_include_directories(tinyinfer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tinyinfer PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(tinyinfer PRIVATE -Wall -Wextra -O3)
if(TINYINFER_NATIVE)
  target_compile_options(tinyinfer PRIVATE -march=native)
else()
  target_compile_options(tinyinfer PRIVATE -mavx2 -mfma)
endif()

add_executable(ti_bench tools/ti_bench.cpp)
target_link_libraries(ti_bench PRIVATE tinyinfer)

add_executable(ti_run tools/ti_run.cpp)
target_link_libraries(ti_run PRIVATE tinyinfer)

if(TINYINFER_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_tinyinfer python/bindings.cpp)
  target_link_libraries(_tinyinfer PRIVATE tinyinfer)
  set_target_properties(_tinyinfer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/tinyinfer)
endif()

if(TINYINFER_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  add_executable(tinyinfer_tests
    tests/test_gemm.cpp
    tests/test_ops.cpp
    tests/test_onnx_models.cpp)
  target_link_libraries(tinyinfer_tests PRIVATE tinyinfer GTest::gtest_main)
  target_compile_definitions(tinyinfer_tests PRIVATE
    TINYINFER_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  include(GoogleTest)
  gtest_discover_tests(tinyinfer_tests)
endif()
