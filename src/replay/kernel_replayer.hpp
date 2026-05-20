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
void* get_out_allocation(impl::MemorySpaceType memory_space,
                         const std::string& label);
}  // namespace impl

class ScopeGuard {
 private:
  std::vector<impl::Allocation> host_raw_allocations;
  std::unordered_map<std::string, void*> host_allocations;
  std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>
      host_output_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  std::vector<impl::Allocation> device_raw_allocations;
  std::unordered_map<std::string, void*> device_allocations;
  std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>
      device_output_allocations;
#endif

  void allocate(impl::MemorySpaceType memory_space, char* address,
                std::size_t size);

  void allocate_output(std::string label, std::string_view memory_space,
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

template <class MemorySpace>
void* get_out_allocation(const std::string& label) {
  return impl::get_out_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class DataType, class... Properties, class... ViewCtorArgs>
auto get_view(const std::string& label, ViewCtorArgs... ctor_args) {
  using ViewType          = Kokkos::View<DataType, Properties...>;
  using value_type        = typename ViewType::value_type;
  using memory_space_type = typename ViewType::memory_space;

  value_type* data =
      static_cast<value_type*>(get_allocation<memory_space_type>(label));
  return ViewType(data, ctor_args...);
}

template <class DataType, class... Properties, class... ViewCtorArgs>
auto get_out_view(const std::string& label, ViewCtorArgs... ctor_args) {
  using ViewType          = Kokkos::View<DataType, Properties...>;
  using value_type        = typename ViewType::value_type;
  using memory_space_type = typename ViewType::memory_space;

  value_type* data =
      static_cast<value_type*>(get_out_allocation<memory_space_type>(label));
  return ViewType(data, ctor_args...);
}

template <class Functor>
Functor replay_functor(const Functor& functor) {
  constexpr int N = sizeof(Functor);

  // In order to revive the to-be-replayed functor, we:
  // - Create a temporary functor and copy construct it from an existing
  //   functor (we have to use the copy constructor since its the only valid
  //   method to construct a lambda)
  // - Save the binary representation of this temporary functor
  // - memcpy the target functor into the temporary functor's memory
  // - Copy construct a new functor f from the patched temporary functor, since
  //   tracking is disabled, if the target functor contained Views, we won't get
  //   a segfault once they are destroyed
  // - memcpy the old data of the temporary functor back and properly destroy it
  // - return f
  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_disable();

  void* tmp_buffer         = std::aligned_alloc(alignof(Functor), N);
  Functor* tmp_functor     = new (tmp_buffer) Functor(functor);
  void* tmp_functor_buffer = std::malloc(N);
  std::memcpy(tmp_functor_buffer, tmp_buffer, N);

  std::size_t copy_length = N;
#if defined(KOKKOS_COMPILER_NVCC)
  // extended lambdas on nvcc also have a "data" pointer
  if constexpr (__nv_is_extended_host_device_lambda_closure_type(Functor)) {
    copy_length -= sizeof(void*);
  }
#endif
  impl::init_functor(static_cast<char*>(tmp_buffer), copy_length);
  Functor f(*tmp_functor);

  std::memcpy(tmp_buffer, tmp_functor_buffer, N);
  tmp_functor->~Functor();
  std::free(tmp_buffer);

  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_enable();

  return f;
}
}  // namespace cexa::kernel_replayer
