function(kkf_add_kernel_replayer TARGET_NAME)
  add_library("${TARGET_NAME}" kernel_replayer.cpp)
  target_include_directories("${TARGET_NAME}" PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries("${TARGET_NAME}" PRIVATE kkf_warnings Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl)
endfunction()

kkf_add_kernel_replayer(kernel_replayer)

if(KKF_ENABLE_ASAN OR KKF_ENABLE_UBSAN)
  kkf_add_kernel_replayer(kernel_replayer_sanitized)
  target_link_libraries(kernel_replayer_sanitized PRIVATE kkf_sanitizers)
else()
  add_library(kernel_replayer_sanitized ALIAS kernel_replayer)
endif()
