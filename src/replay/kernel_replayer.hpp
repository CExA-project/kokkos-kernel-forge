#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <Kokkos_Core.hpp>
#include "allocation.hpp"

namespace cexa::kernel_replayer {
namespace impl {
void init_functor(char* buffer, std::size_t size);
void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label);
}  // namespace impl

class ScopeGuard {
 private:
  std::unordered_map<std::string, impl::Allocation> host_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  std::unordered_map<std::string, impl::Allocation> device_allocations;
#endif

  void allocate(std::string label, std::string_view memory_space, char* address,
                char* data, std::size_t size);

 public:
  ScopeGuard(int& argc, char* argv[]);
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();
};

template <class MemorySpace>
void* get_allocation(const std::string& label) {
  return impl::get_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class Functor>
auto get_functor(const Functor& functor) {
  constexpr int N = sizeof(Functor);
  char buffer[N];
  std::memcpy(buffer, &functor, N);
  impl::init_functor(buffer, N);
  Functor* new_functor = static_cast<Functor*>(std::malloc(sizeof(Functor)));
  std::memcpy(static_cast<void*>(new_functor), buffer, N);

  return std::unique_ptr<Functor, decltype([](Functor* ptr) {
                           std::free(ptr);
                         })>(new_functor);
}
}  // namespace cexa::kernel_replayer
