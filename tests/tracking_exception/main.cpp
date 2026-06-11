#include <Kokkos_Core.hpp>

#include <kernel_replayer.hpp>

#include <stdexcept>

namespace {

struct ThrowOnCopy {
  ThrowOnCopy() = default;

  ThrowOnCopy(const ThrowOnCopy&) { throw std::runtime_error("copy failed"); }
};

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  using Tracking = Kokkos::Impl::SharedAllocationRecord<void, void>;

  if (!Tracking::tracking_enabled()) {
    return 1;
  }

  try {
    (void)cexa::kernel_replayer::replay_functor(ThrowOnCopy{});
    return 2;
  } catch (const std::runtime_error&) {
  }

  return Tracking::tracking_enabled() ? 0 : 3;
}
