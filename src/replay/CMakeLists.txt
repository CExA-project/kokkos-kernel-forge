function(kkf_add_kernel_replayer TARGET_NAME)
  add_library("${TARGET_NAME}" kernel_replayer.cpp)
  target_include_directories("${TARGET_NAME}" PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries("${TARGET_NAME}" PRIVATE Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl)
endfunction()

kkf_add_kernel_replayer(kernel_replayer)

if(KKF_SANITIZER_COMPILE_OPTIONS OR KKF_SANITIZER_LINK_OPTIONS)
  kkf_add_kernel_replayer(kernel_replayer_sanitized)
  kkf_enable_sanitizers(kernel_replayer_sanitized)
else()
  add_library(kernel_replayer_sanitized ALIAS kernel_replayer)
endif()
