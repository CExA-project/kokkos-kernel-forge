#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::kernel_replayer::parallel_for(
      "test_kernel", 1,
      KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); });
  Kokkos::fence();

  return 0;
}
