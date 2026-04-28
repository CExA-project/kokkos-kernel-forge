add_executable(kokkos_kernel_example kokkos_kernel_example.cpp)

target_link_libraries(kokkos_kernel_example PRIVATE Kokkos::kokkos)
target_compile_features(kokkos_kernel_example PRIVATE cxx_std_20)

if(BUILD_TESTING)
  add_test(NAME kokkos_kernel_example COMMAND kokkos_kernel_example)
endif()
