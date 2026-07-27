#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::kernel_replayer::parallel_for(
      "test_kernel", 0,
      KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); });
  Kokkos::fence();

  return 0;
}
