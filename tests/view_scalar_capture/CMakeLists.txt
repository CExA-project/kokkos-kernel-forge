add_executable(view_scalar_capture main.cpp)
kkf_enable_sanitizers(view_scalar_capture)
target_link_libraries(view_scalar_capture PRIVATE Kokkos::kokkos kernel_extractor)

add_executable(view_scalar_capture_replay replay_main.cpp)
target_link_libraries(view_scalar_capture_replay PRIVATE Kokkos::kokkos kernel_replayer)

add_replay_test(view_scalar_capture "")
