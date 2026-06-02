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
  list(APPEND KKF_SANITIZER_COMPILE_OPTIONS -fsanitize=address -fno-omit-frame-pointer)
  list(APPEND KKF_SANITIZER_LINK_OPTIONS -fsanitize=address)
endif()

if(KKF_ENABLE_UBSAN)
  list(APPEND KKF_SANITIZER_COMPILE_OPTIONS -fsanitize=undefined -fno-sanitize-recover=undefined
       -fno-omit-frame-pointer
  )
  list(APPEND KKF_SANITIZER_LINK_OPTIONS -fsanitize=undefined)
endif()

function(kkf_enable_sanitizers TARGET_NAME)
  if(KKF_SANITIZER_COMPILE_OPTIONS)
    target_compile_options("${TARGET_NAME}" PRIVATE ${KKF_SANITIZER_COMPILE_OPTIONS})
  endif()

  if(KKF_SANITIZER_LINK_OPTIONS)
    get_target_property(KKF_TARGET_TYPE "${TARGET_NAME}" TYPE)
    if(KKF_TARGET_TYPE STREQUAL "EXECUTABLE" OR KKF_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
      target_link_options("${TARGET_NAME}" PRIVATE ${KKF_SANITIZER_LINK_OPTIONS})
    endif()
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
