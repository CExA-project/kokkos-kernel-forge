cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Kokkos Serial backend")
option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)

add_subdirectory(external/kokkos)

include(CTest)

add_subdirectory(src)

if(KKF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
