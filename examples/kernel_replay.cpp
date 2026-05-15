#include <Kokkos_Core.hpp>

#include <iostream>

#include <kernel_replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  constexpr int number_of_values = 1024;
  Kokkos::View<int*> values("view", 1);

  // Kokkos::parallel_for(
  //     "fill_values", Kokkos::RangePolicy<>(0, number_of_values),
  //     KOKKOS_LAMBDA(const int i) { values(i) = i; });

  int factor   = 0;
  auto functor = KOKKOS_LAMBDA(const int i, int& partial_sum) {
    partial_sum += values(i) * factor;
  };

  std::unique_ptr replay_functor = cexa::kernel_replayer::get_functor(functor);

  int sum = 0;
  Kokkos::parallel_reduce("sum_values",
                          Kokkos::RangePolicy<>(0, number_of_values),
                          *replay_functor, sum);

  constexpr int expected_sum =
      2 * number_of_values * (number_of_values - 1) / 2;
  if (sum != expected_sum) {
    std::cerr << "Unexpected Kokkos reduction result: got " << sum
              << ", expected " << expected_sum << '\n';
    return 1;
  }

  std::cout << "Kokkos kernel reduction succeeded: " << sum << '\n';
  return 0;
}
