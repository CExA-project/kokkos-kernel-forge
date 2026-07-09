#pragma once

#include "allocation_tracker.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace kkf {

struct ViewDumpResult {
  bool ok = false;
  std::string filename;
  std::string error;
};

struct IndexTypeDesc {
  std::size_t size;
  bool is_signed;
};

struct NoPolicyDesc {};

struct ScalarPolicyDesc {
  IndexTypeDesc index_type_desc;
  std::uint64_t end;
};

struct RangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  std::uint64_t begin;
  std::uint64_t end;
};

struct MDRangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  std::size_t rank;
  std::vector<std::uint64_t> begin;
  std::vector<std::uint64_t> end;
  std::vector<std::uint64_t> tile;
};

ViewDumpResult dump_view_snapshot(
    const AllocationSnapshot& snapshot,
    const std::vector<unsigned char>& functor_data,
    const std::unordered_map<std::string, std::string>& metadata,
    const std::variant<kkf::NoPolicyDesc, kkf::ScalarPolicyDesc,
                       kkf::RangePolicyDesc, kkf::MDRangePolicyDesc>& policy,
    std::string_view phase, std::string_view label, std::uint64_t kernel_id,
    std::uint64_t kernel_invocation);

}  // namespace kkf
