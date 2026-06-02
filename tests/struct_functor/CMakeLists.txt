add_executable(struct_functor main.cpp)
kkf_enable_sanitizers(struct_functor)
target_link_libraries(struct_functor PRIVATE Kokkos::kokkos kernel_extractor)

add_executable(struct_functor_replay replay_main.cpp)
kkf_enable_sanitizers(struct_functor_replay)
target_link_libraries(struct_functor_replay PRIVATE Kokkos::kokkos kernel_replayer)

add_replay_test(struct_functor "")
