#pragma once

#include <string>
#include <tuple>

#include "memory_space_type.hpp"
#include "host_allocation.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#include "cuda_allocation.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#include "hip_allocation.hpp"
#elif defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC) || \
    defined(KOKKOS_ENABLE_NEXTSILICON)
#define KERNEL_REPLAYER_HAS_UNSUPPORTED_DEVICE_SPACE
#endif

namespace cexa::kernel_replayer::impl {
struct Allocation {
  std::string label;
  void* address    = nullptr;
  std::size_t size = 0;
  MemorySpaceType memory_space;

  Allocation() = default;

  Allocation(MemorySpaceType memory_space, std::string label,
             char* target_address, char* data, std::size_t requested_size)
      : label(label), memory_space(memory_space) {
    if (memory_space == MemorySpaceType::HOST) {
      std::tie(address, size) =
          host_allocate(target_address, requested_size, data);
    } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
      std::tie(address, size) =
          device_allocate(target_address, requested_size, data);
#elif defined(KERNEL_REPLAYER_HAS_UNSUPPORTED_DEVICE_SPACE)
      throw std::runtime_error(
          "Trying to allocate on an unsupported device space");
#else
      throw std::runtime_error(
          "Trying to allocate on device but no device backend has been "
          "enabled");
#endif
    }
  }

  ~Allocation() {
    if (address == nullptr) {
      return;
    }

    if (memory_space == MemorySpaceType::HOST) {
      host_deallocate(address, size);
    } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
      device_deallocate(address, size);
#endif
    }
  }

  Allocation(const Allocation&)            = delete;
  Allocation& operator=(const Allocation&) = delete;

  Allocation(Allocation&& other)
      : label(std::move(other.label)),
        address(other.address),
        size(other.size) {
    other.address = nullptr;
    other.size    = 0;
  }

  Allocation& operator=(Allocation&& other) {
    label   = std::move(other.label);
    address = other.address;
    size    = other.size;

    other.address = nullptr;
    other.size    = 0;

    return *this;
  }
};
}  // namespace cexa::kernel_replayer::impl
