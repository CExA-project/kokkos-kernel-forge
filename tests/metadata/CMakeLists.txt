add_executable(metadata main.cpp)
target_link_libraries(metadata PRIVATE kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(metadata_replay replay_main.cpp)
target_link_libraries(metadata_replay PRIVATE kkf_sanitizers Kokkos::kokkos kernel_replayer_sanitized)

add_replay_test(metadata TRUE)
