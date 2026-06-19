#include <Kokkos_Core.hpp>

#include <kkf/extractor.hpp>

struct Functor {
 private:
  int factor;
  Kokkos::View<int*> values;

 public:
  Functor(int factor, Kokkos::View<int*> values)
      : factor(factor), values(values) {}

  KOKKOS_FUNCTION void operator()(int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  int factor = 3;
  Kokkos::parallel_for(
      "test_kernel", N,
      cexa::kernel_replayer::replay_functor(Functor(factor, values)));
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
