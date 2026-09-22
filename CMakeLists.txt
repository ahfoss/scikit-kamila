cmake_minimum_required(VERSION 3.15...3.31)
project(kamila LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_kamila_cpp NB_STATIC src/bindings.cpp src/kamila_core.cpp)

option(ENABLE_COVERAGE "Enable C++ coverage instrumentation" OFF)
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_kamila_cpp PRIVATE --coverage -O0 -g)
    target_link_options(_kamila_cpp PRIVATE --coverage)
  endif()
endif()

install(TARGETS _kamila_cpp DESTINATION kamila)


