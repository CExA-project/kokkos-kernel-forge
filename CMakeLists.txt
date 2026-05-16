cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)
option(KKF_BUILD_TESTS "Build kokkos-kernel-forge tests" OFF)

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_subdirectory(src)

if(KKF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KKF_BUILD_TESTS)
  add_subdirectory(tests)
endif()
