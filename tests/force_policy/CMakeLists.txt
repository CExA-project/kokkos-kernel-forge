add_executable(force_policy main.cpp)
target_link_libraries(force_policy PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(force_policy_replay replay_main.cpp)
target_link_libraries(force_policy_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(force_policy "")
