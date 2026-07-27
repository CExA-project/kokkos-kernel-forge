add_executable(mdrange_policy main.cpp)
target_link_libraries(mdrange_policy PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(mdrange_policy_replay replay_main.cpp)
target_link_libraries(mdrange_policy_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(mdrange_policy FALSE)
