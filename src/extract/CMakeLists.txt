add_library(kernel_extractor INTERFACE)
target_include_directories(kernel_extractor INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(kernel_extractor INTERFACE Kokkos::kokkos)
