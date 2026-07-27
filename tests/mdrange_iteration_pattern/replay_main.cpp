#include <Kokkos_Core.hpp>

#include <kkf/replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int M = 32;
  const int N = 32;
  Kokkos::View<int**> values;
  // Kokkos::parallel_for(
  //     "init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}),
  //     KOKKOS_LAMBDA(int i, int j) { values(i, j) = i * j; });

  using exec_space = Kokkos::DefaultExecutionSpace;
  cexa::kernel_replayer::parallel_for(
      "test_kernel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1, 1}),
      KOKKOS_LAMBDA(int i, int j) {
        values(i, j) *= 2;
#if defined(KOKKOS_ENABLE_SERIAL)
        if constexpr (std::is_same_v<exec_space, Kokkos::Serial>) {
          Kokkos::printf("%d %d\n", i, j);
        }
#endif
      });
  Kokkos::fence();

  // auto h_values =
  //     Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // Kokkos::printf("values(1, 2) = %d\n", h_values(1, 2));

  cexa::kernel_replayer::compare_views<int**>(
      "values", std::make_tuple(M, N), [](auto ref_values, auto replay_values) {
        auto h_replay_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), replay_values);

        Kokkos::printf("values(1, 2) = %d\n", h_replay_values(1, 2));

        auto h_ref_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ref_values);

        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            if (h_replay_values(i, j) != h_ref_values(i, j)) {
              Kokkos::printf("At index (%d, %d), expected %d but got %d\n", i,
                             j, h_ref_values(i, j), h_replay_values(i, j));
              std::exit(1);
            }
          }
        }
      });

  return 0;
}
