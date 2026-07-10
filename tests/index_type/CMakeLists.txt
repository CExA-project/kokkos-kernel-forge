add_executable(index_type main.cpp)
target_link_libraries(index_type PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(index_type_replay replay_main.cpp)
target_link_libraries(index_type_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(index_type FALSE)
