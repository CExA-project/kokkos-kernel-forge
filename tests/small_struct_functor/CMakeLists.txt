add_executable(small_struct_functor main.cpp)
target_link_libraries(small_struct_functor PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(small_struct_functor_replay replay_main.cpp)
target_link_libraries(
  small_struct_functor_replay PRIVATE kkf_warnings kkf_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(small_struct_functor TRUE)
