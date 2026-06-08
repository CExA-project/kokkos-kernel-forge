#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>
#include <sstream>
#include <stdexcept>

template <class T, class U>
void cexa_expect(const T& a, const U& b, const char* expr_a,
                 const char* expr_b) {
  if (a != b) {
    std::ostringstream os;
    os << "Assertion " << expr_a << " == " << expr_b << " failed, got " << a
       << " but expected " << b << '\n';
    throw std::runtime_error(os.str());
  }
}

#define CEXA_EXPECT(a, b) cexa_expect((a), (b), #a, #b)

int main(int argc, char* argv[]) {
  cexa::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  CEXA_EXPECT(cexa::kernel_replayer::get_metadata("foo").value(), "bar");
  CEXA_EXPECT(cexa::kernel_replayer::get_metadata("with null bytes").value(),
              std::string("hello\0world", 12));
  CEXA_EXPECT(cexa::kernel_replayer::get_metadata("n_iter").value(), "10");
  CEXA_EXPECT(cexa::kernel_replayer::get_metadata("missing").has_value(),
              false);

  int n_iter = std::stoi(cexa::kernel_replayer::get_metadata("n_iter").value());

  int sum = 0;
  Kokkos::parallel_reduce("test_kernel", n_iter,
                          cexa::kernel_replayer::replay_functor(
                              KOKKOS_LAMBDA(int, int& sum) { sum++; }),
                          sum);

  Kokkos::printf("Result is %d\n", sum);

  return 0;
}
