#pragma once

#include <any>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>

#include <level_zero/ze_api.h>

namespace cexa::kernel_replayer::impl {
inline std::string ze_result_to_string(const ze_result_t result) {
  switch (result) {
    case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
    case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
    case ZE_RESULT_ERROR_UNKNOWN: return "ZE_RESULT_ERROR_UNKNOWN";
    default:
      return "Unknown ze_result_t value: " +
             std::to_string(static_cast<int>(result));
  }
}

inline void throw_error(ze_result_t error, const char* expr, const char* file,
                        int line) {
  if (error == ZE_RESULT_SUCCESS) {
    return;
  }

  std::ostringstream os;
  if (file) {
    os << file << ":" << line << ": ";
  }
  os << "Call " << expr << " failed with " << ze_result_to_string(error)
     << "\n";
  throw std::runtime_error(os.str());
}

#define CHECK_SYCL_CALL(expr) \
  ::cexa::kernel_replayer::impl::throw_error((expr), #expr, __FILE__, __LINE__)

inline static ze_driver_handle_t driver_handle             = nullptr;
inline static ze_device_handle_t device_handle             = nullptr;
inline static ze_context_handle_t context_handle           = nullptr;
inline static ze_command_list_handle_t command_list_handle = nullptr;
inline static ze_event_pool_handle_t pool_handle           = nullptr;
inline static ze_event_handle_t event_handle               = nullptr;

inline void level_zero_init() {
  std::uint32_t nb_drivers = 0;
  CHECK_SYCL_CALL(zeDriverGet(&nb_drivers, nullptr));
  std::vector<ze_driver_handle_t> drivers(nb_drivers);
  CHECK_SYCL_CALL(zeDriverGet(&nb_drivers, drivers.data()));

  // ze_driver_handle_t driver_handle = nullptr;
  // ze_device_handle_t device_handle = nullptr;
  std::vector<ze_device_handle_t> devices;
  for (auto driver : drivers) {
    std::uint32_t nb_devices = 0;
    CHECK_SYCL_CALL(zeDeviceGet(driver, &nb_devices, nullptr));
    devices.resize(nb_devices);
    CHECK_SYCL_CALL(zeDeviceGet(driver, &nb_devices, devices.data()));

    for (auto device : devices) {
      ze_device_properties_t device_properties = {};
      device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      CHECK_SYCL_CALL(zeDeviceGetProperties(device, &device_properties));
      if (device_properties.type == ZE_DEVICE_TYPE_GPU) {
        device_handle = device;
        break;
      }
    }

    if (device_handle) {
      driver_handle = driver;
      break;
    }
  }

  if (!device_handle) {
    throw std::runtime_error("Failed to find a level zero GPU device");
  }

  ze_context_desc_t context_desc = {};
  context_desc.stype             = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  CHECK_SYCL_CALL(
      zeContextCreate(driver_handle, &context_desc, &context_handle));

  ze_command_queue_desc_t command_queue_desc = {};
  command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  CHECK_SYCL_CALL(zeCommandListCreateImmediate(context_handle, device_handle,
                                               &command_queue_desc,
                                               &command_list_handle));

  ze_event_pool_desc_t pool_desc = {};
  pool_desc.stype                = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  pool_desc.count                = 1;
  pool_desc.flags                = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  CHECK_SYCL_CALL(zeEventPoolCreate(context_handle, &pool_desc, 1,
                                    &device_handle, &pool_handle));

  ze_event_desc_t event_desc = {};
  event_desc.stype           = ZE_STRUCTURE_TYPE_EVENT_DESC;
  event_desc.signal          = ZE_EVENT_SCOPE_FLAG_HOST;
  event_desc.wait            = ZE_EVENT_SCOPE_FLAG_HOST;
  CHECK_SYCL_CALL(zeEventCreate(pool_handle, &event_desc, &event_handle));
}

inline void level_zero_finalize() {
  CHECK_SYCL_CALL(zeContextDestroy(context_handle));
  CHECK_SYCL_CALL(zeCommandListDestroy(command_list_handle));
  CHECK_SYCL_CALL(zeEventDestroy(event_handle));
  CHECK_SYCL_CALL(zeEventPoolDestroy(pool_handle));
}

inline auto regular_device_allocate(std::size_t size, char* data) {
  void* ptr;
  ze_device_mem_alloc_desc_t alloc_desc = {};
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  CHECK_SYCL_CALL(zeMemAllocDevice(context_handle, &alloc_desc, size, 64,
                                   device_handle, &ptr));

  CHECK_SYCL_CALL(zeCommandListAppendMemoryCopy(
      command_list_handle, ptr, data, size, event_handle, 0, nullptr));
  CHECK_SYCL_CALL(zeEventHostSynchronize(event_handle, UINT64_MAX));

  return std::unique_ptr<void, void (*)(void*)>(
      ptr, [](void* ptr) { CHECK_SYCL_CALL(zeMemFree(context_handle, ptr)); });
}

inline void device_copy_data(char* address, char* data, std::size_t size) {
  CHECK_SYCL_CALL(zeCommandListAppendMemoryCopy(
      command_list_handle, address, data, size, event_handle, 0, nullptr));
  CHECK_SYCL_CALL(zeEventHostSynchronize(event_handle, UINT64_MAX));
}

inline std::size_t device_allocation_granularity(std::size_t size) {
  std::size_t granularity = 0;
  CHECK_SYCL_CALL(zeVirtualMemQueryPageSize(context_handle, device_handle, size,
                                            &granularity));
  return granularity;
}

inline std::tuple<void*, std::size_t, std::any> device_allocate(
    void* address, std::size_t size) {
  void* real_address = nullptr;
  CHECK_SYCL_CALL(
      zeVirtualMemReserve(context_handle, address, size, &real_address));
  if (real_address != address) {
    throw std::runtime_error(
        "Failed to allocate data at a specific device address");
  }

  ze_physical_mem_handle_t physical_mem_handle = nullptr;
  ze_physical_mem_desc_t physical_mem_desc     = {};
  physical_mem_desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
  physical_mem_desc.size  = size;
  CHECK_SYCL_CALL(zePhysicalMemCreate(
      context_handle, device_handle, &physical_mem_desc, &physical_mem_handle));

  CHECK_SYCL_CALL(zeVirtualMemMap(context_handle, real_address, size,
                                  physical_mem_handle, 0,
                                  ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));

  return std::make_tuple(reinterpret_cast<void*>(real_address), size,
                         std::any{physical_mem_handle});
}

inline void device_deallocate(void* address, std::size_t size,
                              const std::any& data) {
  auto physical_mem_handle = std::any_cast<ze_physical_mem_handle_t>(data);
  CHECK_SYCL_CALL(zeVirtualMemUnmap(context_handle, address, size));
  CHECK_SYCL_CALL(zePhysicalMemDestroy(context_handle, physical_mem_handle));
  CHECK_SYCL_CALL(zeVirtualMemFree(context_handle, address, size));
}
}  // namespace cexa::kernel_replayer::impl
