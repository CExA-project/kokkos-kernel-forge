cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)
option(KKF_BUILD_TESTS "Build kokkos-kernel-forge tests" OFF)
option(KKF_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(KKF_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)

if(KKF_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

if(KKF_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined)
endif()

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_subdirectory(src)

include(CTest)

if(KKF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KKF_BUILD_TESTS)
  add_subdirectory(tests)
endif()
