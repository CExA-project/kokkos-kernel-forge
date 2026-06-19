function(kkf_add_kernel_replayer TARGET_NAME)
  add_library("${TARGET_NAME}" kernel_replayer.cpp)
  target_sources(
    "${TARGET_NAME}"
    PRIVATE kernel_replayer.cpp
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           "${PROJECT_SOURCE_DIR}/src/"
           "${CMAKE_CURRENT_LIST_DIR}"
           FILES
           hdf5_utils.hpp
           allocation.hpp
           cuda_allocation.hpp
           hip_allocation.hpp
           host_allocation.hpp
           kernel_replayer.hpp
           memory_space_type.hpp
  )
  target_link_libraries("${TARGET_NAME}" PRIVATE Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl)
  target_link_libraries("${TARGET_NAME}" PRIVATE $<BUILD_LOCAL_INTERFACE:kkf_warnings>)
endfunction()

kkf_add_kernel_replayer(kernel_replayer)

if(KKF_ENABLE_ASAN OR KKF_ENABLE_UBSAN)
  kkf_add_kernel_replayer(kernel_replayer_sanitized)
  target_link_libraries(kernel_replayer_sanitized PRIVATE kkf_sanitizers)
else()
  add_library(kernel_replayer_sanitized ALIAS kernel_replayer)
endif()

install(TARGETS kernel_replayer EXPORT kkfTargets FILE_SET HEADERS)
