#include <Kokkos_Core.hpp>

#include <kernel_extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::parallel_for(
      "test_kernel", 1,
      cexa::kernel_replayer::replay_functor(
          KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); }));
  Kokkos::fence();

  return 0;
}
