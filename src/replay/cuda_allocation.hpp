#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <cuda.h>
#include <Kokkos_Core.hpp>

namespace cexa::kernel_replayer::impl {
inline void throw_error(CUresult error, const char* expr, const char* file,
                        int line) {
  if (error == CUDA_SUCCESS) {
    return;
  }
  const char* name        = nullptr;
  const char* description = nullptr;
  cuGetErrorName(error, &name);
  cuGetErrorString(error, &description);
  std::ostringstream os;
  if (file) {
    os << file << ":" << line << ": ";
  }
  os << "Call " << expr << " failed with:\n";
  if (name) {
    os << name;
    if (description) {
      os << ": " << description;
    }
  }
  throw std::runtime_error(os.str());
}

#define CHECK_CUDA_CALL(expr) \
  ::cexa::kernel_replayer::impl::throw_error((expr), #expr, __FILE__, __LINE__)

inline auto regular_device_allocate(std::size_t size, char* data) {
  void* ptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&ptr, size));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
  return std::unique_ptr<void, void (*)(void*)>(
      ptr, [](void* ptr) { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(ptr)); });
}

inline std::tuple<void*, std::size_t> device_allocate(void* address,
                                                      std::size_t size,
                                                      char* data) {
  CUdevice device;
  CHECK_CUDA_CALL(cuCtxGetDevice(&device));

  CUdeviceptr real_address   = 0;
  CUdeviceptr target_address = reinterpret_cast<CUdeviceptr>(address);

  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAllocationProp prop  = {};
  prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id          = device;
  prop.requestedHandleTypes = handleType;

  size_t granularity = 0;
  CHECK_CUDA_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // FIXME: investigate, on jean-zay the allocated addresses are multiple of
  // this, even though granularity is way smaller
  granularity = 0x2000000;

  // Compute the starting address of the virtual memory range, so that the
  // address is a multiple of the granularity and the range contains the
  // original memory range we want to load.
  CUdeviceptr starting_address = (target_address / granularity) * granularity;
  CUdeviceptr ending_address   = target_address + size;
  std::size_t virtual_range_size =
      ((ending_address - starting_address + granularity - 1) / granularity) *
      granularity;

  // Allocate physical memory
  CUmemGenericAllocationHandle allocHandle;
  CHECK_CUDA_CALL(cuMemCreate(&allocHandle, virtual_range_size, &prop, 0));

  // TODO: check if the alignment parameter should be equal to the granularity
  CHECK_CUDA_CALL(cuMemAddressReserve(&real_address, virtual_range_size, 64,
                                      starting_address, 0));
  if (real_address != starting_address) {
    throw std::runtime_error(
        "Failed to allocate data at a specific device address");
  }
  CHECK_CUDA_CALL(
      cuMemMap(real_address, virtual_range_size, 0, allocHandle, 0));

  // Make the address range accessible on device
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id     = device;
  accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_CUDA_CALL(
      cuMemSetAccess(real_address, virtual_range_size, &accessDesc, 1));

  CHECK_CUDA_CALL(cuMemRelease(allocHandle));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(target_address),
                                        data, size, cudaMemcpyHostToDevice));

  return std::make_tuple(reinterpret_cast<void*>(real_address),
                         virtual_range_size);
}

inline void device_deallocate(void* address, std::size_t size) {
  CHECK_CUDA_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(address), size));
  CHECK_CUDA_CALL(
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(address), size));
}
}  // namespace cexa::kernel_replayer::impl
