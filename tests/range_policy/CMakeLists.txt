add_executable(range_policy main.cpp)
target_link_libraries(range_policy PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(range_policy_replay replay_main.cpp)
target_link_libraries(range_policy_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(range_policy FALSE)
