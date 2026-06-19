cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)
option(KKF_BUILD_TESTS "Build kokkos-kernel-forge tests" OFF)
option(KKF_ENABLE_WARNINGS "Enable compiler warning options for kokkos-kernel-forge targets" ON)
option(KKF_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(KKF_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)

add_library(kkf_warnings INTERFACE)

if(KKF_ENABLE_WARNINGS)
  target_compile_options(
    kkf_warnings INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  )
endif()

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

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src/kkf)

include(CTest)

if(KKF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KKF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

write_basic_package_version_file(kkfConfigVersion.cmake COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kkfConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/kkfConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kkf"
)

install(
  EXPORT kkfTargets
  FILE kkfTargets.cmake
  NAMESPACE kkf::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kkf"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/kkfConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/kkfConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kkf"
)
