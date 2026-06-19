#include <Kokkos_Core.hpp>

#include <kkf/replayer.hpp>

struct Functor {
 private:
  int x;
  int y;

 public:
  Functor(int x, int y) : x(x), y(y) {}

  KOKKOS_FUNCTION void operator()(int) const {
    Kokkos::printf("x is %d, y is %d\n", x, y);
  }
};

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int x = 0;
  int y = 0;
  Kokkos::parallel_for("test_kernel", 1,
                       cexa::kernel_replayer::replay_functor(Functor(x, y)));
  Kokkos::fence();

  return 0;
}
