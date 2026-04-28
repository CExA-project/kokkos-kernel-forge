/*
 * Kokkos Tools hook for observing user-visible Kokkos activity.
 *
 * This file is compiled as a shared library and loaded at runtime through
 * KOKKOS_TOOLS_LIBS. Kokkos discovers the exported kokkosp_* symbols and calls
 * them when kernels start/end and when data is allocated.
 *
 * The hook keeps the output focused on application code by filtering internal
 * Kokkos labels. For user labels, it logs View allocations and the begin/end of
 * parallel_for, parallel_reduce, and parallel_scan kernels. End callbacks also
 * request a Kokkos fence.
 */

#include <impl/Kokkos_Profiling_C_Interface.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#define KOKKOS_HOOKS_EXPORT __attribute__((visibility("default")))

namespace {

std::atomic<std::uint64_t> next_kernel_id{1};

std::mutex log_mutex;
std::mutex state_mutex;

struct KernelState {
  std::string label;
  std::uint32_t device_id;
};

std::unordered_map<std::uint64_t, KernelState> kernel_states;
Kokkos_Tools_toolInvokedFenceFunction tool_fence = nullptr;

std::string label_or_unknown(const char* label) {
  return label != nullptr ? label : "<unknown>";
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

bool is_internal_label(std::string_view label) {
  return starts_with(label, "Kokkos::") || starts_with(label, "HostSpace::") ||
         starts_with(label, "CudaSpace::") ||
         starts_with(label, "HIPSpace::") ||
         starts_with(label, "SYCLSpace::") ||
         starts_with(label, "OpenACCSpace::") ||
         starts_with(label, "NextSiliconSpace::") ||
         starts_with(label, "KOKKOS_") || starts_with(label, "kokkos.");
}

bool is_user_label(const char* label) {
  return !is_internal_label(label_or_unknown(label));
}

std::string bounded_string(const char* value, std::size_t max_size) {
  if (value == nullptr) {
    return "<null>";
  }

  const char* end = std::find(value, value + max_size, '\0');
  return std::string(value, end);
}

std::string space_name(const Kokkos_Profiling_SpaceHandle& handle) {
  return bounded_string(handle.name, sizeof(handle.name));
}

template <typename... Args>
void log_line(Args&&... args) {
  std::lock_guard<std::mutex> lock(log_mutex);
  std::cerr << "[kokkos-hooks] ";
  (std::cerr << ... << args);
  std::cerr << '\n';
}

void begin_kernel(const char* hook_name, const char* label,
                  const std::uint32_t device_id, std::uint64_t* kernel_id) {
  const std::uint64_t id = next_kernel_id.fetch_add(1);
  if (kernel_id != nullptr) {
    *kernel_id = id;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex);
    kernel_states[id] = {label_or_unknown(label), device_id};
  }

  if (is_user_label(label)) {
    log_line(hook_name, " label=\"", label_or_unknown(label),
             "\" device=", device_id, " id=", id);
  }
}

void end_kernel(const char* hook_name, const std::uint64_t kernel_id) {
  std::string label                           = "<unknown>";
  std::uint32_t device_id                     = 0;
  Kokkos_Tools_toolInvokedFenceFunction fence = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    fence   = tool_fence;
    auto it = kernel_states.find(kernel_id);
    if (it != kernel_states.end()) {
      label     = it->second.label;
      device_id = it->second.device_id;
      kernel_states.erase(it);
    }
  }

  if (fence != nullptr) {
    fence(device_id);
  }

  if (is_user_label(label.c_str())) {
    log_line(hook_name, " label=\"", label, "\" id=", kernel_id);
  }
}

}  // namespace

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_for(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel("kokkosp_begin_parallel_for", label, device_id, kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_for(
    const std::uint64_t kernel_id) {
  end_kernel("kokkosp_end_parallel_for", kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_reduce(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel("kokkosp_begin_parallel_reduce", label, device_id, kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_reduce(
    const std::uint64_t kernel_id) {
  end_kernel("kokkosp_end_parallel_reduce", kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_scan(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel("kokkosp_begin_parallel_scan", label, device_id, kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_scan(
    const std::uint64_t kernel_id) {
  end_kernel("kokkosp_end_parallel_scan", kernel_id);
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_allocate_data(
    const Kokkos_Profiling_SpaceHandle space, const char* label,
    const void* ptr, const std::uint64_t size) {
  if (is_user_label(label)) {
    log_line("kokkosp_allocate_data view_create label=\"",
             label_or_unknown(label), "\" space=\"", space_name(space),
             "\" ptr=", ptr, " size=", size);
  }
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_provide_tool_programming_interface(
    const std::uint32_t, Kokkos_Tools_ToolProgrammingInterface interface) {
  std::lock_guard<std::mutex> lock(state_mutex);
  tool_fence = interface.fence;
}

extern "C" KOKKOS_HOOKS_EXPORT void kokkosp_request_tool_settings(
    const std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  if (settings != nullptr) {
    *settings                         = {};
    settings->requires_global_fencing = false;
  }
}
