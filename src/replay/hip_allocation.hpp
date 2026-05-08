#include <cstring>
#include <utility>
#include <hip/hip_runtime.h>
#include <Kokkos_Core.hpp>

// void throw_error(hipError_t error, const char* expr, const char* file, int
// line) {
//   if (error == hipSuccess) {
//     return;
//   }
//   const char* name        = hipGetErrorName(error);
//   const char* description = hipGetErrorString(error);
//   std::ostringstream os;
//   if (file) {
//     os << file << ":" << line << ": ";
//   }
//   os << "Call " << expr << " failed with:\n";
//   if (name) {
//     os << name;
//     if (description) {
//       os << ": " << description;
//     }
//   }
//   throw std::runtime_error(os.str());
// }

#define CHECK_HIP_CALL(expr) KOKKOS_IMPL_HIP_SAFE_CALL(expr)

namespace cexa::kernel_replayer::impl {
inline std::pair<void*, std::size_t> device_allocate(void* target_address,
                                              std::size_t size, char* data) {
  int device;
  CHECK_HIP_CALL(hipGetDevice(&device));

  void* real_address = nullptr;

  hipMemAllocationProp prop = {};
  prop.type                 = hipMemAllocationTypePinned;
  prop.location.type        = hipMemLocationTypeDevice;
  prop.location.id          = device;

  size_t granularity = 0;
  CHECK_HIP_CALL(hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityMinimum));

  // Compute the starting address of the virtual memory range, so that the
  // address is a multiple of the granularity and the range contains the
  // original memory range we want to load.
  void* starting_address = reinterpret_cast<void*>(
      (reinterpret_cast<std::uintptr_t>(target_address) / granularity) *
      granularity);
  void* ending_address = reinterpret_cast<char*>(target_address) + size;
  std::size_t virtual_range_size =
      ((reinterpret_cast<std::uintptr_t>(ending_address) -
        reinterpret_cast<std::uintptr_t>(starting_address) + granularity - 1) /
       granularity) *
      granularity;

  // Allocate physical memory
  hipMemGenericAllocationHandle_t allocHandle;
  CHECK_HIP_CALL(hipMemCreate(&allocHandle, virtual_range_size, &prop, 0));

  CHECK_HIP_CALL(hipMemAddressReserve(&real_address, virtual_range_size, 64,
                                      starting_address, 0));
  CHECK_HIP_CALL(
      hipMemMap(real_address, virtual_range_size, 0, allocHandle, 0));

  // Make the address range accessible on device
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type    = hipMemLocationTypeDevice;
  accessDesc.location.id      = device;
  accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
  CHECK_HIP_CALL(
      hipMemSetAccess(real_address, virtual_range_size, &accessDesc, 1));

  CHECK_HIP_CALL(hipMemRelease(allocHandle));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemcpy(reinterpret_cast<void*>(target_address),
                                      data, size, hipMemcpyHostToDevice));

  return std::make_pair(real_address, virtual_range_size);
}

inline void device_deallocate(void* address, std::size_t size) {
  if (address) {
    CHECK_HIP_CALL(hipMemUnmap(address, size));
    CHECK_HIP_CALL(hipMemAddressFree(address, size));
  }
}
}  // namespace cexa::kernel_replayer::impl
