add_executable(manual_extraction main.cpp)
target_link_libraries(manual_extraction PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(manual_extraction_replay replay_main.cpp)
target_link_libraries(manual_extraction_replay PRIVATE kkf_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(manual_extraction "")
