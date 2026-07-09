#include "kernel_extractor.hpp"
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

#include <optional>

namespace cexa::kernel_replayer {
namespace impl {

void (*add_metadata_function)(const char*, const char*, std::size_t) = nullptr;
void (*copy_functor_function)(const unsigned char*, std::size_t)     = nullptr;
void (*register_scalar_policy_function)(std::size_t, bool,
                                        std::uint64_t)               = nullptr;
void (*register_range_policy_function)(const char*, std::size_t, bool,
                                       std::uint64_t, std::uint64_t) = nullptr;
void (*register_mdrange_policy_function)(const char*, std::size_t, std::size_t,
                                         bool, std::uint64_t*, std::uint64_t*,
                                         std::uint64_t*)             = nullptr;
bool (*next_invocation_will_dump_function)(const char*)              = nullptr;
std::optional<bool> has_kernel_dump_tool = std::nullopt;

#if defined(KOKKOS_ENABLE_LIBDL)
template <class FuncPtr>
bool load_function_from_tool(const char* name, FuncPtr& ptr) {
  void* handle = dlsym(RTLD_DEFAULT, name);
  if (dlerror() != nullptr) {
    return false;
  }

  ptr = reinterpret_cast<FuncPtr>(handle);
  return true;
}

bool init_internal_functions_from_tool() {
  dlerror();

  // TODO: print a warning when some (at least 1 but not all) functions cannot
  // be loaded to indicate that there is an issue
  return load_function_from_tool("cexa_kernel_dump_add_metadata",
                                 add_metadata_function) &&
         load_function_from_tool("cexa_kernel_dump_copy_functor",
                                 copy_functor_function) &&
         load_function_from_tool("cexa_kernel_dump_register_scalar_policy",
                                 register_scalar_policy_function) &&
         load_function_from_tool("cexa_kernel_dump_register_range_policy",
                                 register_range_policy_function) &&
         load_function_from_tool("cexa_kernel_dump_register_mdrange_policy",
                                 register_mdrange_policy_function) &&
         load_function_from_tool("cexa_kernel_dump_next_invocation_will_dump",
                                 next_invocation_will_dump_function);
}
#else
bool init_internal_functions_from_tool() { return false; }
#endif

void init_internal_functions() {
  if (!has_kernel_dump_tool.has_value()) {
    if (init_internal_functions_from_tool()) {
      has_kernel_dump_tool = true;
    } else {
      add_metadata_function = [](const char*, const char*, std::size_t) {};
      copy_functor_function = [](const unsigned char*, std::size_t) {};
      register_scalar_policy_function = [](std::size_t, bool, std::uint64_t) {};
      register_range_policy_function  = [](const char*, std::size_t, bool,
                                          std::uint64_t, std::uint64_t) {};
      register_mdrange_policy_function   = [](const char*, std::size_t,
                                            std::size_t, bool, std::uint64_t*,
                                            std::uint64_t*, std::uint64_t*) {};
      next_invocation_will_dump_function = [](const char*) { return false; };
      has_kernel_dump_tool               = false;
    }
  }
}

void copy_functor(const unsigned char* data, std::size_t size) {
  init_internal_functions();
  copy_functor_function(data, size);
}

void register_scalar_policy(std::size_t index_type_size, bool index_type_signed,
                            std::uint64_t N) {
  init_internal_functions();
  register_scalar_policy_function(index_type_size, index_type_signed, N);
}

void register_range_policy(const char* space, std::size_t index_type_size,
                           bool index_type_signed, std::uint64_t begin,
                           std::uint64_t end) {
  init_internal_functions();
  register_range_policy_function(space, index_type_size, index_type_signed,
                                 begin, end);
}

void register_mdrange_policy(const char* space, std::size_t rank,
                             std::size_t index_type_size,
                             bool index_type_signed, std::uint64_t* begin,
                             std::uint64_t* end, std::uint64_t* tile) {
  init_internal_functions();
  register_mdrange_policy_function(space, rank, index_type_size,
                                   index_type_signed, begin, end, tile);
}

bool next_invocation_will_dump(const char* kernel_name) {
  init_internal_functions();
  return next_invocation_will_dump_function(kernel_name);
}

}  // namespace impl

void add_metadata(const std::string& key, const std::string& value) {
  impl::init_internal_functions();
  impl::add_metadata_function(key.c_str(), value.c_str(), value.size());
}

}  // namespace cexa::kernel_replayer
