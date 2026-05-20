#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", 1);
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  Kokkos::parallel_for("test_kernel", N,
                       cexa::kernel_replayer::replay_functor(
                           KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
  Kokkos::fence();

  auto replay_values = cexa::kernel_replayer::get_view<int*>("values", N);
  auto h_replay_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replay_values);

  auto ref_values = cexa::kernel_replayer::get_out_view<int*>("values", N);
  auto h_ref_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_values);

  for (int i = 0; i < N; i++) {
    if (h_replay_values(i) != h_ref_values(i)) {
      Kokkos::printf("At index %d, expected %d but got %d\n", i,
                     h_ref_values(i), h_replay_values(i));
      return 1;
    }
  }

  Kokkos::printf("values(5) = %d\n", h_replay_values(5));

  return 0;
}
