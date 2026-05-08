find_package(Kokkos REQUIRED)

find_package(HDF5 REQUIRED COMPONENTS C HL)

add_library(kernel_replayer kernel_replayer.cpp)
target_link_libraries(kernel_replayer PRIVATE Kokkos::kokkos HDF5::HDF5)
