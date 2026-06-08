kkf_add_test_executable(no_capture main.cpp)
target_link_libraries(no_capture PRIVATE kkf_sanitizers Kokkos::kokkos kernel_extractor)

kkf_add_test_executable(no_capture_replay replay_main.cpp)
target_link_libraries(no_capture_replay PRIVATE kkf_sanitizers Kokkos::kokkos kernel_replayer_sanitized)

add_replay_test(no_capture TRUE)
