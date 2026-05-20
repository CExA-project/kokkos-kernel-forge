#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A("A", 1);
  Kokkos::View<int*> B("B", 1);
  Kokkos::View<int*> C("C", 1);
  // Kokkos::parallel_for(
  //     "init", N, KOKKOS_LAMBDA(int i) {
  //       A(i) = i;
  //       B(i) = i % 32;
  // });

  Kokkos::parallel_for("test_kernel", N,
                       cexa::kernel_replayer::replay_functor(
                           KOKKOS_LAMBDA(int i) { C(i) = A(i) + B(i); }));
  Kokkos::fence();

  auto replay_C = cexa::kernel_replayer::get_view<int*>("C", N);
  auto h_replay_C =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replay_C);

  auto ref_C = cexa::kernel_replayer::get_out_view<int*>("C", N);
  auto h_ref_C =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_C);

  for (int i = 0; i < N; i++) {
    if (h_replay_C(i) != h_ref_C(i)) {
      Kokkos::printf("At index %d, expected %d but got %d\n", i, h_ref_C(i),
                     h_replay_C(i));
      return 1;
    }
  }

  Kokkos::printf("values(5) = %d\n", h_replay_C(5));

  return 0;
}
