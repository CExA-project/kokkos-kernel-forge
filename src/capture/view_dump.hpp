#pragma once

#include "allocation_tracker.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace kkf {

struct ViewDumpResult {
  bool ok = false;
  std::string filename;
};

ViewDumpResult dump_view_snapshot(
    const AllocationSnapshot& snapshot,
    const std::unordered_map<std::string, std::string>& metadata,
    std::string_view phase, std::string_view label, std::uint64_t kernel_id,
    std::uint64_t kernel_invocation);

}  // namespace kkf
