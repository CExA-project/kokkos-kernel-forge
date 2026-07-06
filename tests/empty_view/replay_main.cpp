#include <Kokkos_Core.hpp>

#include <kkf/replayer.hpp>

#include <cstdlib>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  Kokkos::parallel_for(
      "test_kernel", Kokkos::RangePolicy<>(0, 1),
      cexa::kernel_replayer::replay_functor(KOKKOS_LAMBDA(int) {
        if (values.extent(0) != 0) {
          values(0) = 1;
        }
      }));
  Kokkos::fence();

  cexa::kernel_replayer::compare_views<int*>(
      "empty_values", std::make_tuple(0),
      [](auto ref_values, auto replay_values) {
        if (ref_values.extent(0) != 0 || replay_values.extent(0) != 0) {
          Kokkos::printf(
              "Expected empty replay views, got extents %zu and %zu\n",
              ref_values.extent(0), replay_values.extent(0));
          std::exit(1);
        }
      });

  return 0;
}
