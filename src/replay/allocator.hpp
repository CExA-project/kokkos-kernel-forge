#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <Kokkos_Core.hpp>
#include <cerrno>
#include <cstddef>
#include <cstring>

#include "memory_space_type.hpp"
#include "host_allocation.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#elif defined(KOKKOS_ENABLE_CUDA)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#elif defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC) || \
    defined(KOKKOS_ENABLE_NEXTSILICON)
#define KERNEL_REPLAYER_HAS_UNSUPPORTED_DEVICE_SPACE
#endif

#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
#include "device_allocation.hpp"
#endif

namespace cexa::kernel_replayer {

namespace impl {

// FIXME: use multimaps to accomodate for duplicate labels
inline std::unordered_map<std::string, Allocation<MemorySpaceType::HOST>>
    host_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
inline std::unordered_map<std::string, Allocation<MemorySpaceType::DEVICE>>
    device_allocations;
#endif

void allocate(std::string label, std::string_view memory_space, char* address,
              char* data, std::size_t size) {
  if (impl::memory_space_type_from_string(memory_space) ==
      impl::MemorySpaceType::HOST) {
    impl::host_allocations[label] =
        impl::Allocation<impl::MemorySpaceType::HOST>(label, address, data,
                                                      size);
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    impl::device_allocations[label] =
        impl::Allocation<impl::MemorySpaceType::DEVICE>(label, address, data,
                                                        size);
#endif
  }
}

}  // namespace impl

template <Kokkos::MemorySpace MemorySpace>
void* get_allocation(const std::string& label) {
  if constexpr (impl::memory_space_type_from_string(MemorySpace::name()) ==
                impl::MemorySpaceType::HOST) {
    if (!impl::host_allocations.contains(label)) {
      return nullptr;
    }
    return impl::host_allocations[label].address;
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    if (!impl::device_allocations.contains(label)) {
      return nullptr;
    }
    return impl::device_allocations[label].address;
#endif
  }
}

}  // namespace cexa::kernel_replayer
