#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int answer   = 0;
  auto functor = KOKKOS_LAMBDA(int) {
    Kokkos::printf("Hello from the kernel! The answer is %d\n", answer);
  };
  auto replay_functor = cexa::kernel_replayer::get_functor(functor);

  Kokkos::parallel_for("test_kernel", 1, *replay_functor);
  Kokkos::fence();

  return 0;
}
