add_library(kernel_replayer kernel_replayer.cpp)
target_include_directories(kernel_replayer PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(kernel_replayer PRIVATE Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl)
