add_executable(no_capture main.cpp)
kkf_enable_sanitizers(no_capture)
target_link_libraries(no_capture PRIVATE Kokkos::kokkos kernel_extractor)

add_executable(no_capture_replay replay_main.cpp)
target_link_libraries(no_capture_replay PRIVATE Kokkos::kokkos kernel_replayer)

add_replay_test(no_capture TRUE)
