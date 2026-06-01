#include "kernel_extractor.hpp"
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

namespace cexa::kernel_replayer::impl {
bool check_kernel_dump_tool_is_present() {
#if defined(KOKKOS_ENABLE_LIBDL)
  dlerror();
  [[maybe_unused]] void* handle =
      dlsym(RTLD_DEFAULT, "kernel_dump_tool_is_present");
  char* error = dlerror();
  if (error == nullptr) {
    return true;
  }
#endif

  return false;
}
}  // namespace cexa::kernel_replayer::impl
