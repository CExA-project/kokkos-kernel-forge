add_executable(metadata main.cpp)
kkf_enable_sanitizers(metadata)
target_link_libraries(metadata PRIVATE Kokkos::kokkos kernel_extractor)

add_executable(metadata_replay replay_main.cpp)
kkf_enable_sanitizers(metadata_replay)
target_link_libraries(metadata_replay PRIVATE Kokkos::kokkos kernel_replayer_sanitized)

add_replay_test(metadata TRUE)
