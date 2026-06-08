#include "kernel_extractor.hpp"
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

namespace cexa::kernel_replayer {
namespace impl {

void (*add_metadata_function)(const char*, const char*, std::size_t) = nullptr;
void (*copy_functor_function)(const unsigned char*, std::size_t)     = nullptr;
std::optional<bool> has_kernel_dump_tool = std::nullopt;

bool init_internal_functions_from_tool() {
#if defined(KOKKOS_ENABLE_LIBDL)
  dlerror();

  void* handle = dlsym(RTLD_DEFAULT, "cexa_kernel_dump_add_metadata");
  if (dlerror() != nullptr) {
    return false;
  }
  add_metadata_function =
      reinterpret_cast<void (*)(const char*, const char*, std::size_t)>(handle);

  handle = dlsym(RTLD_DEFAULT, "cexa_kernel_dump_copy_functor");
  if (dlerror() != nullptr) {
    return false;
  }
  copy_functor_function =
      reinterpret_cast<void (*)(const unsigned char*, std::size_t)>(handle);

  return true;
#else
  return false;
#endif
}

void init_internal_functions() {
  if (!has_kernel_dump_tool.has_value()) {
    if (init_internal_functions_from_tool()) {
      has_kernel_dump_tool = true;
    } else {
      add_metadata_function = [](const char*, const char*, std::size_t) {};
      copy_functor_function = [](const unsigned char*, std::size_t) {};
      has_kernel_dump_tool  = false;
    }
  }
}

void copy_functor(const unsigned char* data, std::size_t size) {
  init_internal_functions();
  copy_functor_function(data, size);
}

}  // namespace impl

void add_metadata(const std::string& key, const std::string& value) {
  impl::init_internal_functions();
  impl::add_metadata_function(key.c_str(), value.c_str(), value.size());
}

}  // namespace cexa::kernel_replayer
