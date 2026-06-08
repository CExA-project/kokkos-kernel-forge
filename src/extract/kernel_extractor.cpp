#include "kernel_extractor.hpp"
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

namespace cexa::kernel_replayer {
namespace impl {

void (*add_metadata_function)(const char*, const char*, std::size_t) = nullptr;
std::optional<bool> has_kernel_dump_tool = std::nullopt;

bool init_kernel_tool_functions() {
  if (!has_kernel_dump_tool.has_value()) {
#if defined(KOKKOS_ENABLE_LIBDL)
    dlerror();
    void* handle = dlsym(RTLD_DEFAULT, "cexa_kernel_dump_add_metadata");
    char* error  = dlerror();
    if (error == nullptr) {
      add_metadata_function =
          reinterpret_cast<void (*)(const char*, const char*, std::size_t)>(
              handle);
      has_kernel_dump_tool = true;
    } else
#endif
    {
      add_metadata_function = [](const char*, const char*, std::size_t) {};
      has_kernel_dump_tool  = false;
    }
  }
  return *has_kernel_dump_tool;
}
}  // namespace impl

void add_metadata(const std::string& key, const std::string& value) {
  [[maybe_unused]] static const bool init_once =
      impl::init_kernel_tool_functions();
  impl::add_metadata_function(key.c_str(), value.c_str(), value.size());
}

}  // namespace cexa::kernel_replayer
