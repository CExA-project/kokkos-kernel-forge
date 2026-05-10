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
  char buffer[N];
  std::memcpy(buffer, &functor, N);
  impl::init_functor(buffer, N);
  Functor* new_functor = static_cast<Functor*>(std::malloc(sizeof(Functor)));
  std::memcpy(static_cast<void*>(new_functor), buffer, N);

  return std::unique_ptr<Functor, decltype([](Functor* ptr) { std::free(ptr); })>(new_functor);
}
}  // namespace cexa::kernel_replayer
