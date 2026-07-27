#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int answer = 42;
  cexa::kernel_replayer::parallel_for(
      "test_kernel", 1, KOKKOS_LAMBDA(int) {
        Kokkos::printf("Hello from the kernel! The answer is %d\n", answer);
      });
  Kokkos::fence();

  return 0;
}
