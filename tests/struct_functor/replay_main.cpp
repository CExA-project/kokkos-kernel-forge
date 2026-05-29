#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

struct Functor {
 private:
  int factor;
  Kokkos::View<int*> values;

 public:
  Functor(int factor, Kokkos::View<int*> values)
      : factor(factor), values(values) {}

  KOKKOS_FUNCTION void operator()(int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values;
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  int factor = 0;
  Kokkos::parallel_for(
      "test_kernel", N,
      cexa::kernel_replayer::replay_functor(Functor(factor, values)));
  Kokkos::fence();

  // auto h_values =
  //     Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // Kokkos::printf("values(5) = %d\n", h_values(5));

  cexa::kernel_replayer::compare_views<int*>(
      "values", std::make_tuple(1024), [](auto ref_values, auto replay_values) {
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
