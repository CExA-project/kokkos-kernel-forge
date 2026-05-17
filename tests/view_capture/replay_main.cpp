#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", 1);
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  auto functor        = KOKKOS_LAMBDA(int i) { values(i) *= 2; };
  auto replay_functor = cexa::kernel_replayer::get_functor(functor);

  Kokkos::parallel_for("test_kernel", N, *replay_functor);
  Kokkos::fence();

  using device_space = Kokkos::DefaultExecutionSpace::memory_space;
  void* replay_values_alloc =
      cexa::kernel_replayer::get_allocation<device_space>("values");
  Kokkos::View<int*, Kokkos::MemoryUnmanaged> replay_values(
      static_cast<int*>(replay_values_alloc), N);

  auto h_replay_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replay_values);

  void* ref_values_alloc =
      cexa::kernel_replayer::get_out_allocation<device_space>("values");
  Kokkos::View<int*, Kokkos::MemoryUnmanaged> ref_values(
      static_cast<int*>(ref_values_alloc), N);

  auto h_ref_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_values);

  for (int i = 0; i < N; i++) {
    if (h_replay_values(i) != h_ref_values(i)) {
      Kokkos::printf("At index %d, expected %d but got %d\n", i,
                     h_ref_values(i), h_replay_values(i));
      return 1;
    }
  }

  Kokkos::printf("values(5) = %d", h_replay_values(5));

  return 0;
}
