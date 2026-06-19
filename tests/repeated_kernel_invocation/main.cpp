#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>

struct MultiplyFunctor {
  int factor;
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(const int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  Kokkos::parallel_for(
      "test_kernel", N,
      cexa::kernel_replayer::replay_functor(MultiplyFunctor{2, values}));
  Kokkos::fence();

  Kokkos::parallel_for(
      "test_kernel", N,
      cexa::kernel_replayer::replay_functor(MultiplyFunctor{3, values}));
  Kokkos::fence();

  Kokkos::parallel_for(
      "test_kernel", N,
      cexa::kernel_replayer::replay_functor(MultiplyFunctor{4, values}));
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
