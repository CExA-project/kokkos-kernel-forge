add_executable(empty_view main.cpp)
target_link_libraries(empty_view PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(empty_view_replay replay_main.cpp)
target_link_libraries(empty_view_replay PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_replayer_sanitized)

add_replay_test(empty_view FALSE)
