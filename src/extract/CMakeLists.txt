add_library(kernel_extractor kernel_extractor.cpp)
target_include_directories(kernel_extractor PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(kernel_extractor PUBLIC Kokkos::kokkos)
