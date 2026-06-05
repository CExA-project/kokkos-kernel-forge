cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)
option(KKF_BUILD_TESTS "Build kokkos-kernel-forge tests" OFF)
option(KKF_ENABLE_WARNINGS "Enable compiler warning options for kokkos-kernel-forge targets" ON)
option(KKF_WARNINGS_AS_ERRORS "Treat kokkos-kernel-forge compiler warnings as errors" OFF)
option(KKF_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(KKF_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)

add_library(kkf_warnings INTERFACE)

if(KKF_ENABLE_WARNINGS)
  target_compile_options(
    kkf_warnings INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  )
endif()

function(kkf_enable_warnings TARGET_NAME)
  target_link_libraries("${TARGET_NAME}" PRIVATE kkf_warnings)

  if(KKF_WARNINGS_AS_ERRORS)
    set_property(TARGET "${TARGET_NAME}" PROPERTY COMPILE_WARNING_AS_ERROR ON)
  endif()
endfunction()

add_library(kkf_sanitizers INTERFACE)

if(KKF_ENABLE_ASAN OR KKF_ENABLE_UBSAN)
  target_compile_options(kkf_sanitizers INTERFACE -fno-omit-frame-pointer)
endif()

if(KKF_ENABLE_ASAN)
  target_compile_options(kkf_sanitizers INTERFACE -fsanitize=address)
  target_link_options(kkf_sanitizers INTERFACE -fsanitize=address)
endif()

if(KKF_ENABLE_UBSAN)
  target_compile_options(kkf_sanitizers INTERFACE -fsanitize=undefined -fno-sanitize-recover=undefined)
  target_link_options(kkf_sanitizers INTERFACE -fsanitize=undefined)
endif()

function(kkf_enable_sanitizers TARGET_NAME)
  if(KKF_ENABLE_ASAN OR KKF_ENABLE_UBSAN)
    target_link_libraries("${TARGET_NAME}" PRIVATE kkf_sanitizers)
  endif()
endfunction()

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
