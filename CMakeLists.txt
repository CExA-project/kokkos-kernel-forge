cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(kokkos_kernel_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Kokkos Serial backend")
option(KKF_BUILD_EXAMPLES "Build kokkos-kernel-forge examples" ON)

add_subdirectory(external/kokkos)

function(kkf_add_hdf5_submodule)
  set(BUILD_TESTING OFF)
  set(BUILD_SHARED_LIBS OFF)
  set(HDF5_BUILD_EXAMPLES OFF)
  set(HDF5_BUILD_HL_LIB OFF)
  set(HDF5_BUILD_TOOLS OFF)
  set(HDF5_DISABLE_COMPILER_WARNINGS ON)
  set(HDF5_ENABLE_ALL_WARNINGS OFF)
  add_subdirectory(external/hdf5 EXCLUDE_FROM_ALL)
endfunction()

kkf_add_hdf5_submodule()

include(CTest)

add_subdirectory(src)

if(KKF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
