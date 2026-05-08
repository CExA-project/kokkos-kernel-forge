#pragma once

#include <Kokkos_Core.hpp>

#include "memory_space_type.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include "cuda_allocation.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
#include "hip_allocation.hpp"
#else
#error This file should not be included when no device backend has been enabled
#endif

namespace cexa::kernel_replayer::impl {
template <MemorySpaceType>
struct Allocation;

template <>
struct Allocation<MemorySpaceType::DEVICE> {
  std::string label;
  void* address    = nullptr;
  std::size_t size = 0;

  Allocation() = default;

  Allocation(std::string label, char* target_address, char* data,
             std::size_t requested_size)
      : label(label) {
    auto [address, size] =
        device_allocate(target_address, data, requested_size);
    this->address = address;
    this->size    = size;
  }

  ~Allocation() {
    if (address != nullptr) {
      // FIXME: check return value ?
      device_deallocate(address, size);
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
