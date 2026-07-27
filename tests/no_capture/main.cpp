#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  cexa::kernel_replayer::parallel_for(
      "test_kernel", 1,
      KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); });
  Kokkos::fence();

  return 0;
}
