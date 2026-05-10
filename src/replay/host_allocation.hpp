#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <Kokkos_Core.hpp>
#include <cerrno>
#include <cstddef>
#include <cstring>

#include "memory_space_type.hpp"

namespace cexa::kernel_replayer::impl {

template <MemorySpaceType>
struct Allocation;

template <>
struct Allocation<MemorySpaceType::HOST> {
  std::string label;
  void* address    = nullptr;
  std::size_t size = 0;

  Allocation() = default;

  Allocation(std::string label, char* target_address, char* data,
             std::size_t requested_size)
      : label(label), size(requested_size) {
    static const int page_size = getpagesize();

    char* aligned_address = reinterpret_cast<char*>(
        ((reinterpret_cast<std::uintptr_t>(target_address)) / page_size) *
        page_size);
    size += target_address - aligned_address;

    address = mmap(aligned_address, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);

    if (address == MAP_FAILED) {
      if (errno == EEXIST) {
        // FIXME: better error message
        throw std::runtime_error(
            "Failed to allocate to a predefined host address");
      } else {
        throw std::runtime_error(
            "Unknown error when trying to allocate to a predefined host "
            "address: " +
            std::string(std::strerror(errno)));
      }
    } else if (address != aligned_address) {
      // We only reach this branch when the target address already belongs to a
      // reservation on older kernels which don't handle the MAP_FIXED_NOREPLACE
      // case. In that case the kernel allocates on another available address.
      munmap(address, size);
      throw std::runtime_error(
          "Failed to allocate to a predefined host address");
    }

    memcpy(target_address, data, requested_size);
  }

  ~Allocation() {
    if (address != nullptr) {
      // FIXME: check return value ?
      munmap(address, size);
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
