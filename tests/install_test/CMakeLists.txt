cmake_minimum_required(VERSION 3.26)

project(kkf-test)

find_package(Kokkos REQUIRED)
find_package(kkf REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE Kokkos::kokkos kkf::kernel_extractor)

add_executable(replay_test replay_main.cpp)
target_link_libraries(replay_test PRIVATE Kokkos::kokkos kkf::kernel_replayer)
