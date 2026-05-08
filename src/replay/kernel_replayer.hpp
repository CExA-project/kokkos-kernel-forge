#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
namespace cexa::kernel_replayer {
namespace impl {
void init_functor(char* buffer, std::size_t size);
}

class ScopeGuard {
 public:
  ScopeGuard(int& argc, char* argv[]);
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();
};

template <class Functor>
auto get_functor(const Functor& functor) {
  constexpr int N = sizeof(Functor);
  char* buffer[N];
  std::memcpy(buffer, &functor, N);
  impl::init_functor(buffer, N);
  Functor* new_functor = std::malloc(sizeof(Functor));
  std::memcpy(new_functor, buffer, N);

  return std::unique_ptr(new_functor, [](Functor* ptr) { free(ptr); });
}
}  // namespace cexa::kernel_replayer
