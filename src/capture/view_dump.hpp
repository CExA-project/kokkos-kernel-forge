#pragma once

#include "allocation_tracker.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace kkf {

struct ViewDumpResult {
  bool ok = false;
  std::string filename;
};

ViewDumpResult dump_view_snapshot(const AllocationSnapshot& snapshot,
                                  std::string_view phase,
                                  std::string_view label,
                                  std::uint64_t kernel_id,
                                  std::uint64_t kernel_invocation);

}  // namespace kkf
