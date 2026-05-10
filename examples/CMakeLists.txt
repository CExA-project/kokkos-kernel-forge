add_executable(kokkos_kernel_example kokkos_kernel_example.cpp)

find_package(HDF5 REQUIRED COMPONENTS C HL)

target_link_libraries(kokkos_kernel_example PRIVATE Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl)
target_compile_features(kokkos_kernel_example PRIVATE cxx_std_20)

add_executable(kernel_replay kernel_replay.cpp)
target_link_libraries(kernel_replay PRIVATE Kokkos::kokkos kernel_replayer)

if(BUILD_TESTING)
  add_test(NAME kokkos_kernel_example COMMAND kokkos_kernel_example test.h5)
endif()
