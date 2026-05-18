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

namespace kkf {
namespace {

// SYCLDeviceUSM and OpenACCSpace, need dedicated device-to-host 
// copy paths before they can be dumped safely.
bool is_host_accessible_space(const std::string& space) {
  return space == "Host" || space == "CudaHostPinned" ||
         space == "CudaUVM" || space == "HIPHostPinned" ||
         space == "HIPManaged" || space == "SYCLHostUSM" ||
         space == "SYCLSharedUSM" || space == "NextSiliconSharedSpace";
}

bool is_cuda_device_space(const std::string& space) {
  return space == "Cuda";
}

bool is_hip_device_space(const std::string& space) {
  return space == "HIP";
}

std::string allocate_staging_buffer(const ActiveAllocation& allocation,
                                    std::vector<unsigned char>& bytes) {
  try {
    bytes.resize(static_cast<std::size_t>(allocation.record.size));
  } catch (const std::exception& error) {
    return std::string("could not allocate host staging buffer: ") +
           error.what();
  }

  return {};
}

}  // namespace

std::string copy_allocation_bytes(const ActiveAllocation& allocation,
                                  std::vector<unsigned char>& bytes) {
  const std::string& space   = allocation.record.space;
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

  const void* data = allocation.record.p_data;
  if (data == nullptr) {
    bytes.clear();
    return "allocation data pointer is null";
  }

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
