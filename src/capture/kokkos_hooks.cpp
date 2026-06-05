/*
 * Kokkos Tools hook for observing user-visible Kokkos activity.
 *
 * This file is compiled as a shared library and loaded at runtime through
 * KOKKOS_TOOLS_LIBS. Kokkos discovers the exported kokkosp_* symbols and calls
 * them when kernels start/end and when data is allocated or deallocated.
 *
 * The hook keeps the output focused on application code by filtering internal
 * Kokkos labels. For the selected kernel label, it dumps active Kokkos data
 * allocations to HDF5 before and after the selected kernel
 */

#include <impl/Kokkos_Profiling_C_Interface.h>

#include "allocation_tracker.hpp"
#include "view_dump.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>

#define KOKKOS_HOOKS_EXPORT __attribute__((visibility("default")))

#if !defined(KKF_KOKKOS_ALLOCATION_HEADER_SIZE)
#error "KKF_KOKKOS_ALLOCATION_HEADER_SIZE must be defined by CMake"
#endif

static_assert(KKF_KOKKOS_ALLOCATION_HEADER_SIZE == 128 ||
                  KKF_KOKKOS_ALLOCATION_HEADER_SIZE == 256,
              "unexpected Kokkos allocation header size");

namespace {

std::atomic<std::uint64_t> next_kernel_id{1};

std::mutex log_mutex;
std::mutex state_mutex;

struct KernelState {
  std::string label;
  std::uint32_t device_id;
  std::uint64_t invocation;
  bool dump_views;
};

std::unordered_map<std::uint64_t, KernelState> kernel_states;
std::unordered_map<std::string, std::uint64_t> kernel_invocations;
Kokkos_Tools_toolInvokedFenceFunction tool_fence = nullptr;

kkf::AllocationTracker allocation_tracker;

std::string dump_kernel_label;
std::optional<std::uint64_t> dump_kernel_invocation = 1;

constexpr std::string_view dump_kernel_label_option =
    "--kkf-dump-kernel-label=";
constexpr std::string_view dump_kernel_invocation_option =
    "--kkf-dump-kernel-invocation=";

std::string label_or_unknown(const char* label) {
  return label != nullptr ? label : "<unknown>";
}

bool is_internal_label(std::string_view label) {
  return label.starts_with("Kokkos::") || label.starts_with("HostSpace::") ||
         label.starts_with("CudaSpace::") || label.starts_with("HIPSpace::") ||
         label.starts_with("SYCLSpace::") ||
         label.starts_with("OpenACCSpace::") ||
         label.starts_with("NextSiliconSpace::") ||
         label.starts_with("KOKKOS_") || label.starts_with("kokkos.");
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
  std::cerr << "[kkf-capture] ";
  (std::cerr << ... << args);
  std::cerr << '\n';
}

bool should_dump_views_for_label(std::string_view label) {
  return !dump_kernel_label.empty() && label == dump_kernel_label;
}

bool should_dump_views_for_invocation(std::string_view label,
                                      const std::uint64_t invocation) {
  return dump_kernel_invocation.has_value() &&
         should_dump_views_for_label(label) &&
         invocation == dump_kernel_invocation.value();
}

std::optional<std::uint64_t> parse_positive_uint64(std::string_view value) {
  std::uint64_t parsed    = 0;
  const auto* const begin = value.begin();
  const auto* const end   = value.end();
  const auto result       = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end || parsed == 0) {
    return std::nullopt;
  }
  return parsed;
}

std::string obtain_dump_path(const std::string& filename) {
  std::error_code error;
  const std::filesystem::path path = std::filesystem::absolute(filename, error);
  return error ? filename : path.string();
}

void dump_views(const char* phase, const std::string& label,
                const std::uint64_t kernel_id, const std::uint64_t invocation) {
  const kkf::AllocationSnapshot snapshot = allocation_tracker.snapshot();
  const kkf::ViewDumpResult result =
      kkf::dump_view_snapshot(snapshot, phase, label, kernel_id, invocation);
  const std::string dump_path = obtain_dump_path(result.filename);
  if (!result.ok) {
    log_line("dump_failed phase=", phase, " path=\"", dump_path,
             "\" kernel_id=", kernel_id, " invocation=", invocation);
    return;
  }

  log_line("dump_written phase=", phase, " path=\"", dump_path,
           "\" kernel_id=", kernel_id, " invocation=", invocation,
           " active_allocations=", snapshot.allocations.size(),
           " active_bytes=", snapshot.active_bytes);
}

bool should_track_allocation(const char* label, const void* ptr) {
  return ptr != nullptr && is_user_label(label);
}

const void* allocation_data_pointer(const void* ptr) {
  return static_cast<const unsigned char*>(ptr) +
         KKF_KOKKOS_ALLOCATION_HEADER_SIZE;
}

void begin_kernel(const char* label, const std::uint32_t device_id,
                  std::uint64_t* kernel_id) {
  const std::uint64_t id         = next_kernel_id.fetch_add(1);
  const std::string kernel_label = label_or_unknown(label);
  std::uint64_t invocation       = 0;
  bool dump_this_kernel          = false;

  if (kernel_id != nullptr) {
    *kernel_id = id;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex);
    invocation = ++kernel_invocations[kernel_label];
    dump_this_kernel =
        should_dump_views_for_invocation(kernel_label, invocation);
    kernel_states[id] = {kernel_label, device_id, invocation, dump_this_kernel};
  }

  if (dump_this_kernel) {
    log_line("kernel_selected label=\"", kernel_label,
             "\" invocation=", invocation, " kernel_id=", id);
    Kokkos_Tools_toolInvokedFenceFunction fence = nullptr;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      fence = tool_fence;
    }
    if (fence != nullptr) {
      fence(device_id);
    }
    dump_views("in", kernel_label, id, invocation);
  }
}

void end_kernel(const std::uint64_t kernel_id) {
  std::string label                           = "<unknown>";
  std::uint32_t device_id                     = 0;
  std::uint64_t invocation                    = 0;
  bool dump_this_kernel                       = false;
  Kokkos_Tools_toolInvokedFenceFunction fence = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    fence   = tool_fence;
    auto it = kernel_states.find(kernel_id);
    if (it != kernel_states.end()) {
      label            = it->second.label;
      device_id        = it->second.device_id;
      invocation       = it->second.invocation;
      dump_this_kernel = it->second.dump_views;
      kernel_states.erase(it);
    }
  }

  if (dump_this_kernel) {
    if (fence != nullptr) {
      fence(device_id);
    }
    dump_views("out", label, kernel_id, invocation);
  }
}

}  // namespace

extern "C" {
// Variable used by the kernel extractor to detect the presence of the dumping
// tool, we don't care about its value.
KOKKOS_HOOKS_EXPORT int kernel_dump_tool_is_present = 0;

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_for(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_for(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_reduce(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_reduce(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_scan(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_scan(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_parse_args(const int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view argument = argv[i] != nullptr ? argv[i] : "";
    if (argument.starts_with(dump_kernel_label_option)) {
      dump_kernel_label =
          std::string(argument.substr(dump_kernel_label_option.size()));
    } else if (argument.starts_with(dump_kernel_invocation_option)) {
      const std::string_view value =
          argument.substr(dump_kernel_invocation_option.size());
      dump_kernel_invocation = parse_positive_uint64(value);
    }
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_allocate_data(
    const Kokkos_Profiling_SpaceHandle space, const char* label,
    const void* ptr, const std::uint64_t size) {
  const std::string allocation_label = label_or_unknown(label);
  const std::string allocation_space = space_name(space);
  const void* p_data = ptr != nullptr ? allocation_data_pointer(ptr) : nullptr;

  if (should_track_allocation(label, ptr)) {
    allocation_tracker.record_allocation(allocation_label, allocation_space,
                                         ptr, p_data, size);
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_deallocate_data(
    const Kokkos_Profiling_SpaceHandle space, const char*, const void* ptr,
    const std::uint64_t) {
  const std::string deallocation_space = space_name(space);

  if (ptr != nullptr) {
    allocation_tracker.record_deallocation(deallocation_space, ptr);
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_finalize_library() {}

KOKKOS_HOOKS_EXPORT void kokkosp_provide_tool_programming_interface(
    const std::uint32_t, Kokkos_Tools_ToolProgrammingInterface interface) {
  std::lock_guard<std::mutex> lock(state_mutex);
  tool_fence = interface.fence;
}

KOKKOS_HOOKS_EXPORT void kokkosp_request_tool_settings(
    const std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  if (settings != nullptr) {
    *settings                         = {};
    settings->requires_global_fencing = false;
  }
}
}
