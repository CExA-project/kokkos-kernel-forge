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

  // We create a temporary copy of the functor using placement new and the copy
  // constructor, as lambdas with a capture don't have a default contructor or
  // assignment operators. We then copy the data of the extracted functor into
  // it.
  void* tmp_buffer = std::aligned_alloc(alignof(Functor), N);
  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_disable();
  Functor* tmp_functor = new (tmp_buffer) Functor(functor);
  impl::init_functor(static_cast<char*>(tmp_buffer), N);

  // We create a copy of the extracted functor with allocation tracking
  // disabled, so that when the destructors of views contained in the extracted
  // funcor are called, we won't get a segfault.
  void* buffer         = std::aligned_alloc(alignof(Functor), N);
  Functor* new_functor = new (buffer) Functor(*tmp_functor);
  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_enable();
  std::free(tmp_buffer);

  return std::unique_ptr<Functor, decltype([](Functor* ptr) {
                           std::free(ptr);
                         })>(new_functor);
}
}  // namespace cexa::kernel_replayer
