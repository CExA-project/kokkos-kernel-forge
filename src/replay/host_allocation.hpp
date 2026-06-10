#pragma once

#include <unistd.h>
#include <sys/mman.h>

#if defined(__APPLE__) && defined(__MACH__)
#define KKF_USE_MACH_VM 1
#else
#define KKF_USE_MACH_VM 0
#endif

#if KKF_USE_MACH_VM
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <mach/mach_vm.h>
#include <mach/vm_statistics.h>
#endif

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cexa::kernel_replayer::impl {

inline auto regular_host_allocate(std::size_t size, char* data) {
  void* ptr = std::malloc(size);
  std::memcpy(ptr, data, size);
  return std::unique_ptr<void, void (*)(void*)>(ptr, std::free);
}

inline std::size_t host_allocation_granularity() { return getpagesize(); }

inline void host_copy_data(char* address, char* data, std::size_t size) {
  std::memcpy(address, data, size);
}

inline std::tuple<void*, std::size_t> host_allocate(void* aligned_address,
                                                    std::size_t size) {
  assert(reinterpret_cast<std::uintptr_t>(aligned_address) %
             host_allocation_granularity() ==
         0);

#if KKF_USE_MACH_VM
  // The replayer must recreate Host allocations at the exact captured virtual
  // address because Kokkos::View pointers are stored inside the captured
  // functor. On macOS, the target range may already be reserved by the process
  // allocator, so use Mach overwrite semantics for this short-lived replay
  // mapping.
  mach_vm_address_t address = static_cast<mach_vm_address_t>(
      reinterpret_cast<std::uintptr_t>(aligned_address));
  const mach_vm_address_t requested_address = address;
  const mach_vm_size_t mach_size            = static_cast<mach_vm_size_t>(size);

  kern_return_t status =
      mach_vm_map(mach_task_self(), &address, mach_size, 0,
                  VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, MEMORY_OBJECT_NULL, 0,
                  FALSE, VM_PROT_READ | VM_PROT_WRITE,
                  VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_DEFAULT);

  if (status != KERN_SUCCESS) {
    throw std::runtime_error(
        "Failed to allocate to a predefined host address: " +
        std::string(mach_error_string(status)));
  }

  if (address != requested_address) {
    mach_vm_deallocate(mach_task_self(), address, mach_size);
    throw std::runtime_error("Failed to allocate to a predefined host address");
  }

  return std::make_tuple(reinterpret_cast<void*>(address), size);
#else
  void* address =
      mmap(aligned_address, size, PROT_READ | PROT_WRITE,
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
    throw std::runtime_error("Failed to allocate to a predefined host address");
  }

  return std::make_tuple(address, size);
#endif
}

inline void host_deallocate(void* address, std::size_t size) {
#if KKF_USE_MACH_VM
  mach_vm_deallocate(
      mach_task_self(),
      static_cast<mach_vm_address_t>(reinterpret_cast<std::uintptr_t>(address)),
      static_cast<mach_vm_size_t>(size));
#else
  munmap(address, size);
#endif
}

}  // namespace cexa::kernel_replayer::impl

#undef KKF_USE_MACH_VM
