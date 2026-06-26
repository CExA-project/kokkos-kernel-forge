add_executable(repeated_kernel_invocation main.cpp)
target_link_libraries(repeated_kernel_invocation PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(repeated_kernel_invocation_replay replay_main.cpp)
target_link_libraries(repeated_kernel_invocation_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(repeated_kernel_invocation FALSE "--kkf-dump-kernel-invocation=2")
