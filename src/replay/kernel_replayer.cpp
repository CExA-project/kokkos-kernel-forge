#include <Kokkos_Core.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "kernel_replayer.hpp"
#include "allocation.hpp"
#include "memory_space_type.hpp"

void remove_cli_args(int& argc, char* argv[], int pos, int n) {
  if (pos + n < argc) {
    for (int i = pos; i < argc - n; i++) {
      std::swap(argv[i], argv[i + n]);
    }
  }
  argc -= n;
}

std::string_view find_flag_argument(int& argc, char* argv[],
                                    std::string_view flag) {
  std::string_view argument;
  for (int i = 1; i < argc; i++) {
    std::string_view current_arg{argv[i]};
    // --flag=argument
    if (current_arg.size() > flag.size() && current_arg.starts_with(flag) &&
        current_arg[flag.size()] == '=') {
      auto equal = current_arg.find_first_of('=');
      argument   = current_arg.substr(equal + 1);
      remove_cli_args(argc, argv, i, 1);
      break;
    }

    // --flag argument
    if (current_arg == flag) {
      if (i + 1 == argc) {
        break;
      }
      argument = argv[i + 1];
      remove_cli_args(argc, argv, i, 2);
      break;
    }
  }

  return argument;
}

void device_init() {
  int device_supports_virtual_address = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  CHECK_CUDA_CALL(cuInit(0));

  CUdevice cuDevice;
  CHECK_CUDA_CALL(cuDeviceGet(&cuDevice, 0));
  CUcontext ctx;
  CHECK_CUDA_CALL(cuCtxCreate(&ctx, 0, cuDevice));

  CHECK_CUDA_CALL(cuDeviceGetAttribute(
      &device_supports_virtual_address,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));
#elif defined(KOKKOS_ENABLE_HIP)
  int hipDevice;
  CHECK_HIP_CALL(hipGetDevice(&hipDevice));

  CHECK_HIP_CALL(hipDeviceGetAttribute(
      &device_supports_virtual_address,
      hipDeviceAttributeVirtualMemoryManagementSupported, hipDevice));
#else
  device_supports_virtual_address = 1;
#endif
  if (!device_supports_virtual_address) {
    throw std::runtime_error(
        "The selected device does not support virtual memory management");
  }
}

static std::optional<std::vector<char>> functor_data;

namespace cexa::kernel_replayer {

namespace impl {

static std::unordered_map<std::string, void*>* host_allocations;
static std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>*
    host_output_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
static std::unordered_map<std::string, void*>* device_allocations;
static std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>*
    device_output_allocations;
#endif

void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    if (!host_allocations->contains(label)) {
      return nullptr;
    }
    return (*host_allocations)[label];
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    if (!device_allocations->contains(label)) {
      return nullptr;
    }
    return (*device_allocations)[label];
#endif
  }
}

void* get_out_allocation(impl::MemorySpaceType memory_space,
                         const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    auto it = host_output_allocations->find(label);
    if (it == host_output_allocations->end()) {
      return nullptr;
    }
    return it->second.get();
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    auto it = device_output_allocations->find(label);
    if (it == device_output_allocations->end()) {
      return nullptr;
    }
    return it->second.get();
#endif
  }
}

void init_functor(char* buffer, std::size_t size) {
  std::memcpy(buffer, functor_data->data(), size);
}

char* allocate_host_buffer(std::size_t size, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    return new char[size];
  } else {
    void* ptr = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocHost(&ptr, size));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipHostMalloc(&ptr, size));
#endif
    return static_cast<char*>(ptr);
  }
}

void free_host_buffer(char* ptr, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    delete[] ptr;
  } else {
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(ptr));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipHostFree(ptr));
#endif
  }
}

void check_hdf5_call(herr_t status, const char* expr, const char* file,
                     int line) {
  if (status >= 0) {
    return;
  }

  std::stringstream os;
  os << file << ":" << line << ": "
     << "The call " << expr << " failed";
  throw std::runtime_error(os.str());
}

#define CHECK_HDF5_CALL(expr) \
  cexa::kernel_replayer::impl::check_hdf5_call(expr, #expr, __FILE__, __LINE__);

using hdf5_iterate_fun_t = std::function<void(std::string, std::string_view,
                                              char*, char*, std::size_t)>;

std::string get_hdf5_string_attribute(hid_t group, const char* name,
                                      const char* attr_name) {
  hid_t attr_id =
      H5Aopen_by_name(group, name, attr_name, H5P_DEFAULT, H5P_DEFAULT);
  if (attr_id == H5I_INVALID_HID) {
    throw std::runtime_error(
        "Error when reading the dump file: group does not contain a " +
        std::string(attr_name) + "attribute");
  }

  H5A_info_t attr_info;
  CHECK_HDF5_CALL(H5Aget_info(attr_id, &attr_info));
  hid_t type = H5Aget_type(attr_id);
  std::string attribute(attr_info.data_size, '\0');
  CHECK_HDF5_CALL(H5Aread(attr_id, type, attribute.data()));

  // remove the null-terminator
  attribute.pop_back();
  CHECK_HDF5_CALL(H5Aclose(attr_id));

  return attribute;
}

herr_t get_hdf5_dataset_alloc_info(hid_t group, const char* name,
                                   const H5L_info_t*, void* allocation_sets) {
  std::string label = get_hdf5_string_attribute(group, name, "label");
  std::string space = get_hdf5_string_attribute(group, name, "space");
  char* address     = reinterpret_cast<char*>(std::stoull(
      get_hdf5_string_attribute(group, name, "p_data"), nullptr, 16));

  MemorySpaceType memory_space = memory_space_type_from_string(space);
  if (memory_space == MemorySpaceType::HOST &&
      label == "kernel_replayer_functor") {
    return 0;
  }

  int rank = 0;
  std::string dataset_name(name);
  dataset_name += "/bytes";
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(group, dataset_name.c_str(), &rank));
  std::vector<hsize_t> dims(rank);
  H5T_class_t datatype;
  std::size_t datatype_size;

  CHECK_HDF5_CALL(H5LTget_dataset_info(group, dataset_name.c_str(), dims.data(),
                                       &datatype, &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  std::size_t buffer_size =
      std::reduce(dims.begin(), dims.end(), 1, std::multiplies<>{});

  auto allocations = reinterpret_cast<std::set<std::pair<char*, std::size_t>>*>(
      allocation_sets);
  std::pair<char*, std::size_t> allocation_info =
      get_allocation_address(address, buffer_size, memory_space);
  if (memory_space == MemorySpaceType::HOST) {
    allocations[0].insert(allocation_info);
  } else {
    allocations[1].insert(allocation_info);
  }

  return 0;
}

herr_t allocate_hdf5_dataset(hid_t group, const char* name, const H5L_info_t*,
                             void* allocate_fun) {
  std::string label = get_hdf5_string_attribute(group, name, "label");
  std::string space = get_hdf5_string_attribute(group, name, "space");
  char* address     = reinterpret_cast<char*>(std::stoull(
      get_hdf5_string_attribute(group, name, "p_data"), nullptr, 16));

  int rank = 0;
  std::string dataset_name(name);
  dataset_name += "/bytes";
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(group, dataset_name.c_str(), &rank));
  std::vector<hsize_t> dims(rank);
  H5T_class_t datatype;
  std::size_t datatype_size;

  CHECK_HDF5_CALL(H5LTget_dataset_info(group, dataset_name.c_str(), dims.data(),
                                       &datatype, &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  std::size_t buffer_size =
      std::reduce(dims.begin(), dims.end(), 1, std::multiplies<>{});

  MemorySpaceType memory_space = memory_space_type_from_string(space);
  if (memory_space == MemorySpaceType::HOST &&
      label == "kernel_replayer_functor" && !functor_data.has_value()) {
    functor_data.emplace(buffer_size);
    CHECK_HDF5_CALL(H5LTread_dataset(group, dataset_name.c_str(),
                                     H5T_NATIVE_UCHAR, functor_data->data()));
  } else {
    char* buffer = allocate_host_buffer(buffer_size, memory_space);
    CHECK_HDF5_CALL(H5LTread_dataset(group, dataset_name.c_str(),
                                     H5T_NATIVE_UCHAR, buffer));

    (*reinterpret_cast<hdf5_iterate_fun_t*>(allocate_fun))(
        label, space, address, buffer, buffer_size);

    free_host_buffer(buffer, memory_space);
  }

  return 0;
}

std::vector<std::pair<char*, std::size_t>> compute_allocations(
    const std::set<std::pair<char*, std::size_t>>& allocations) {
  if (allocations.empty()) {
    return {};
  }

  std::vector<std::pair<char*, std::size_t>> allocs;
  auto it                              = allocations.begin();
  auto [current_address, current_size] = *(it++);
  for (; it != allocations.end(); ++it) {
    auto [address, size] = *it;
    if (current_address + current_size > address) {
      std::size_t new_size = (address + size) - current_address;
      current_size         = new_size > current_size ? new_size : current_size;
    } else {
      allocs.emplace_back(current_address, current_size);
      current_address = address;
      current_size    = size;
    }
  }

  allocs.emplace_back(current_address, current_size);

  return allocs;
}
}  // namespace impl

ScopeGuard::ScopeGuard(int& argc, char* argv[]) {
  device_init();

  std::string_view hdf5_filename =
      find_flag_argument(argc, argv, "--kernel-replayer-dump");
  if (hdf5_filename.data() == nullptr) {
    throw std::runtime_error(
        "The kernel replayer expects the flag --kernel-replayer-dump");
  }

  hid_t file = H5Fopen(hdf5_filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file == H5I_INVALID_HID) {
    throw std::runtime_error("Failed to open hdf5 file " +
                             std::string(hdf5_filename));
  }
  std::set<std::pair<char*, std::size_t>> allocation_sets[2];

  hsize_t idx = 0;
  H5Literate_by_name(file, "views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                     impl::get_hdf5_dataset_alloc_info, allocation_sets,
                     H5P_DEFAULT);

  for (auto [address, size] : impl::compute_allocations(allocation_sets[0])) {
    allocate(impl::MemorySpaceType::HOST, address, size);
  }

  for (auto [address, size] : impl::compute_allocations(allocation_sets[1])) {
    allocate(impl::MemorySpaceType::DEVICE, address, size);
  }

  impl::hdf5_iterate_fun_t copy_data_wrapper =
      [this](std::string label, std::string_view memory_space, char* address,
             char* data, std::size_t size) {
        impl::MemorySpaceType space =
            impl::memory_space_type_from_string(memory_space);
        impl::copy_data(space, address, data, size);
        if (space == impl::MemorySpaceType::HOST) {
          host_allocations[label] = address;
        } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
          device_allocations[label] = address;
#endif
        }
      };

  idx = 0;
  H5Literate_by_name(file, "views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                     impl::allocate_hdf5_dataset, &copy_data_wrapper,
                     H5P_DEFAULT);

  H5Fclose(file);

  std::string_view hdf5_output_filename =
      find_flag_argument(argc, argv, "--kernel-replayer-out-dump");
  if (hdf5_output_filename.data() != nullptr) {
    file = H5Fopen(hdf5_output_filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file == H5I_INVALID_HID) {
      throw std::runtime_error("Failed to open hdf5 file " +
                               std::string(hdf5_output_filename));
    }

    impl::hdf5_iterate_fun_t allocate_wrapper =
        [this](std::string label, std::string_view memory_space, char*,
               char* data, std::size_t size) {
          allocate_output(label, memory_space, data, size);
        };

    hsize_t idx = 0;
    H5Literate_by_name(file, "views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                       impl::allocate_hdf5_dataset, &allocate_wrapper,
                       H5P_DEFAULT);

    H5Fclose(file);
  }

  if (!functor_data.has_value()) {
    throw std::runtime_error("No functor found in the provided hdf5 file");
  }

  impl::host_allocations        = &host_allocations;
  impl::host_output_allocations = &host_output_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  impl::device_allocations        = &device_allocations;
  impl::device_output_allocations = &device_output_allocations;
#endif
}

ScopeGuard::~ScopeGuard() {}

void ScopeGuard::allocate(impl::MemorySpaceType memory_space, char* address,
                          std::size_t size) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    host_raw_allocations.emplace_back(memory_space, address, size);
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    device_raw_allocations.emplace_back(memory_space, address, size);
#endif
  }
}

void ScopeGuard::allocate_output(std::string label,
                                 std::string_view memory_space, char* data,
                                 std::size_t size) {
  if (impl::memory_space_type_from_string(memory_space) ==
      impl::MemorySpaceType::HOST) {
    host_output_allocations.insert(
        {label, impl::regular_host_allocate(size, data)});
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    device_output_allocations.insert(
        {label, impl::regular_device_allocate(size, data)});
#endif
  }
}

}  // namespace cexa::kernel_replayer
