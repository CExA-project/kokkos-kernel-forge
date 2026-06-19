#pragma once

#include <string_view>

namespace cexa::kernel_replayer::impl {

enum class MemorySpaceType {
  HOST,
  DEVICE,
};

inline constexpr MemorySpaceType memory_space_type_from_string(
    std::string_view space_name) {
  return space_name == "Host" ? MemorySpaceType::HOST : MemorySpaceType::DEVICE;
}
}  // namespace cexa::kernel_replayer::impl
