kkf_add_test_executable(view_capture main.cpp)
target_link_libraries(view_capture PRIVATE kkf_sanitizers Kokkos::kokkos kernel_extractor)

kkf_add_test_executable(view_capture_replay replay_main.cpp)
target_link_libraries(view_capture_replay PRIVATE Kokkos::kokkos kernel_replayer)

add_replay_test(view_capture "")
