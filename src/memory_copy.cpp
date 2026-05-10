#include "memory_copy.hpp"

#include <cstring>
#include <exception>
#include <limits>

#if defined(KKF_ENABLE_CUDA_DUMP)
#include <cuda_runtime_api.h>
#endif

#if defined(KKF_ENABLE_HIP_DUMP)
#include <hip/hip_runtime_api.h>
#endif

#if !defined(KKF_KOKKOS_ALLOCATION_HEADER_SIZE)
#error "KKF_KOKKOS_ALLOCATION_HEADER_SIZE must be defined by CMake"
#endif

static_assert(KKF_KOKKOS_ALLOCATION_HEADER_SIZE == 128 ||
                  KKF_KOKKOS_ALLOCATION_HEADER_SIZE == 256,
              "unexpected Kokkos allocation header size");

namespace kkf {
namespace {

bool is_host_accessible_space(const std::string& space) {
  return space == "Host" || space == "HostSpace" ||
         space == "CudaHostPinned" || space == "CudaHostPinnedSpace" ||
         space == "CudaUVM" || space == "CudaUVMSpace" ||
         space == "HIPHostPinned" || space == "HIPHostPinnedSpace" ||
         space == "HIPManaged" || space == "HIPManagedSpace" ||
         space == "SYCLHostUSM" || space == "SYCLHostUSMSpace" ||
         space == "SYCLSharedUSM" || space == "SYCLSharedUSMSpace";
}

bool is_cuda_device_space(const std::string& space) {
  return space == "Cuda" || space == "CudaSpace";
}

bool is_hip_device_space(const std::string& space) {
  return space == "HIP" || space == "HIPSpace";
}

std::string allocate_staging_buffer(const ActiveAllocation& allocation,
                                    std::vector<unsigned char>& bytes) {
  const std::uint64_t size = allocation.record.size;
  if (size >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return "allocation is too large to copy on this platform";
  }

  try {
    bytes.resize(static_cast<std::size_t>(size));
  } catch (const std::exception& error) {
    return std::string("could not allocate host staging buffer: ") +
           error.what();
  }

  return {};
}

std::size_t kokkos_allocation_header_size() {
  return KKF_KOKKOS_ALLOCATION_HEADER_SIZE;
}

const void* allocation_data_pointer(const ActiveAllocation& allocation) {
  return static_cast<const unsigned char*>(allocation.ptr) +
         kokkos_allocation_header_size();
}

}  // namespace

std::string copy_allocation_bytes(const ActiveAllocation& allocation,
                                  std::vector<unsigned char>& bytes) {
  const std::string& space = allocation.record.space;
  const bool host_accessible = is_host_accessible_space(space);
  const bool cuda_device     = is_cuda_device_space(space);
  const bool hip_device      = is_hip_device_space(space);

  if (!host_accessible && !cuda_device && !hip_device) {
    return "memory space is not supported for raw byte dumps";
  }
#if !defined(KKF_ENABLE_CUDA_DUMP)
  if (cuda_device) {
    return "CUDA dump support was not enabled at build time";
  }
#endif
#if !defined(KKF_ENABLE_HIP_DUMP)
  if (hip_device) {
    return "HIP dump support was not enabled at build time";
  }
#endif

  std::string reason = allocate_staging_buffer(allocation, bytes);
  if (!reason.empty() || bytes.empty()) {
    return reason;
  }

  const void* data = allocation_data_pointer(allocation);

  if (host_accessible) {
    std::memcpy(bytes.data(), data, bytes.size());
    return {};
  }

#if defined(KKF_ENABLE_CUDA_DUMP)
  if (cuda_device) {
    const cudaError_t error =
        cudaMemcpy(bytes.data(), data, bytes.size(), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      bytes.clear();
      return std::string("cudaMemcpyDeviceToHost failed: ") +
             cudaGetErrorString(error);
    }
    return {};
  }
#endif

#if defined(KKF_ENABLE_HIP_DUMP)
  if (hip_device) {
    const hipError_t error =
        hipMemcpy(bytes.data(), data, bytes.size(), hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
      bytes.clear();
      return std::string("hipMemcpyDeviceToHost failed: ") +
             hipGetErrorString(error);
    }
    return {};
  }
#endif

  return "internal error: no copy path selected";
}

}  // namespace kkf
