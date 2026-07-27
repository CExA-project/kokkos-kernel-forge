#include <Kokkos_Core.hpp>

#include <kkf/replayer.hpp>

struct MultiplyFunctor {
  int factor;
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(const int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values;

  cexa::kernel_replayer::parallel_for("test_kernel", N,
                                      MultiplyFunctor{0, values});
  Kokkos::fence();

  cexa::kernel_replayer::compare_views<int*>(
      "values", std::make_tuple(N), [](auto ref_values, auto replay_values) {
        auto h_replay_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), replay_values);

        auto h_ref_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ref_values);

        for (int i = 0; i < N; i++) {
          if (h_replay_values(i) != h_ref_values(i)) {
            Kokkos::printf("At index %d, expected %d but got %d\n", i,
                           h_ref_values(i), h_replay_values(i));
            std::exit(1);
          }
        }
      });

  return 0;
}
