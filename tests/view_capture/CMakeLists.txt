add_executable(view_capture main.cpp)
target_link_libraries(view_capture PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(view_capture_replay replay_main.cpp)
target_link_libraries(view_capture_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(view_capture FALSE)
