#include <Kokkos_Core.hpp>
#include <cassert>
#include <numeric>
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
      std::swap(argv[pos], argv[pos + n]);
    }
  }
  argc -= n;
}

std::string_view find_hdf5_filename(int& argc, char* argv[]) {
  std::string_view hdf5_filename;
  for (int i = 1; i < argc; i++) {
    std::string_view current_arg{argv[i]};
    if (current_arg.starts_with("--kernel-replayer-dump=")) {
      auto equal    = current_arg.find_first_of('=');
      hdf5_filename = current_arg.substr(equal + 1);
      remove_cli_args(argc, argv, i, 1);
      break;
    }

    if (current_arg == "--kernel-replayer-dump") {
      if (i + 1 == argc) {
        break;
      }
      hdf5_filename = argv[i + 1];
      remove_cli_args(argc, argv, i, 2);
      break;
    }
  }

  return hdf5_filename;
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

static std::unordered_map<std::string, impl::Allocation>* host_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
static std::unordered_map<std::string, impl::Allocation>* device_allocations;
#endif

void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    if (!host_allocations->contains(label)) {
      return nullptr;
    }
    return (*host_allocations)[label].address;
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    if (!device_allocations->contains(label)) {
      return nullptr;
    }
    return (*device_allocations)[label].address;
#endif
  }
}

void init_functor(char* buffer, std::size_t size) {
  std::size_t N = size;
#if defined(KOKKOS_ENABLE_CUDA)
  // functors on nvcc also have a "data" pointer
  N -= sizeof(void*);
#endif
  std::memcpy(buffer, functor_data->data(), N);
}

char* allocate_host_buffer(std::size_t size, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    return new char[size];
  } else {
    char* ptr = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocHost(&ptr, size));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipMallocHost(&ptr, size));
#endif
    return ptr;
  }
}

void free_host_buffer(char* ptr, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    delete[] ptr;
  } else {
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(ptr));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipFreeHost(ptr));
#endif
  }
}

void check_hdf5_call(herr_t status, const char* expr, const char* file,
                     int line) {
  if (status >= 0) {
    return;
  }

  std::stringstream os;
  os << file << ":" << line << ": " << "The call " << expr << " failed";
  throw std::runtime_error(os.str());
}

#define CHECK_HDF5_CALL(expr) \
  cexa::kernel_replayer::impl::check_hdf5_call(expr, #expr, __FILE__, __LINE__);

using allocate_fun_t = std::function<void(std::string, std::string_view, char*,
                                          char*, std::size_t)>;

herr_t allocate_hdf5_dataset(hid_t root, const char* name,
                             const H5O_info_t* info, void* allocate_fun) {
  if (info->type != H5O_TYPE_DATASET) {
    return 0;
  }

  // The dataset name is of the form "address;space;label"
  std::string_view dataset_name = name;
  auto first_sep                = dataset_name.find_first_of(';');
  std::string_view address_str  = dataset_name.substr(0, first_sep);
  char* address =
      reinterpret_cast<char*>(std::stoull(std::string(address_str)));
  auto second_sep = dataset_name.find_first_of(';', first_sep + 1);
  std::string_view space =
      dataset_name.substr(first_sep + 1, second_sep - (first_sep + 1));
  std::string_view label = dataset_name.substr(second_sep + 1);

  int rank = 0;
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(root, name, &rank));
  std::vector<hsize_t> dims(rank);
  H5T_class_t datatype;
  std::size_t datatype_size;

  CHECK_HDF5_CALL(
      H5LTget_dataset_info(root, name, dims.data(), &datatype, &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  std::size_t buffer_size =
      std::reduce(dims.begin(), dims.end(), 1, std::multiplies<>{});

  MemorySpaceType memory_space = memory_space_type_from_string(space);
  if (memory_space == MemorySpaceType::HOST &&
      label == "kernel_replay_functor") {
    functor_data.emplace(buffer_size);
    CHECK_HDF5_CALL(H5LTread_dataset_char(root, name, functor_data->data()));
  } else {
    char* buffer = allocate_host_buffer(buffer_size, memory_space);
    CHECK_HDF5_CALL(H5LTread_dataset_char(root, name, buffer));

    (*reinterpret_cast<allocate_fun_t*>(allocate_fun))(
        std::string(label), space, address, buffer, buffer_size);

    free_host_buffer(buffer, memory_space);
  }

  return 0;
}
}  // namespace impl

ScopeGuard::ScopeGuard(int& argc, char* argv[]) {
  device_init();

  std::string_view hdf5_filename = find_hdf5_filename(argc, argv);
  if (hdf5_filename.data() == nullptr) {
    throw std::runtime_error(
        "The kernel replayer expects the flag --kernel-replayer-dump");
  }

  hid_t file = H5Fopen(hdf5_filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file == H5I_INVALID_HID) {
    throw std::runtime_error("Failed to open hdf5 file " +
                             std::string(hdf5_filename));
  }

  using namespace std::placeholders;
  impl::allocate_fun_t allocate_wrapper =
      std::bind(&ScopeGuard::allocate, this, _1, _2, _3, _4, _5);

  H5Ovisit(file, H5_INDEX_NAME, H5_ITER_NATIVE, impl::allocate_hdf5_dataset,
           &allocate_wrapper
#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 12
           ,
           H5O_INFO_BASIC
#endif
  );

  H5Fclose(file);

  if (!functor_data.has_value()) {
    throw std::runtime_error("No functor found in the provided hdf5 file");
  }

  impl::host_allocations = &host_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  impl::device_allocations = &device_allocations;
#endif
}

ScopeGuard::~ScopeGuard() {}

void ScopeGuard::allocate(std::string label, std::string_view memory_space,
                          char* address, char* data, std::size_t size) {
  if (impl::memory_space_type_from_string(memory_space) ==
      impl::MemorySpaceType::HOST) {
    host_allocations[label] = impl::Allocation(impl::MemorySpaceType::HOST,
                                               label, address, data, size);
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    device_allocations[label] = impl::Allocation(impl::MemorySpaceType::DEVICE,
                                                 label, address, data, size);
#endif
  }
}

}  // namespace cexa::kernel_replayer
