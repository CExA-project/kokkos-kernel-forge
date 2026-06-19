#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>
#include <string>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  cexa::kernel_replayer::add_metadata("foo", "bar");
  cexa::kernel_replayer::add_metadata("with null bytes",
                                      std::string("hello\0world", 12));
  int n_iter = 10;
  cexa::kernel_replayer::add_metadata("n_iter", std::to_string(n_iter));

  int sum = 0;
  Kokkos::parallel_reduce("test_kernel", n_iter,
                          cexa::kernel_replayer::replay_functor(
                              KOKKOS_LAMBDA(int, int& sum) { sum++; }),
                          sum);

  Kokkos::printf("Result is %d\n", sum);

  return 0;
}
