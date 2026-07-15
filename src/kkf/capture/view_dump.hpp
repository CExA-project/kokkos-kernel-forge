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
  std::uint64_t end;
};

struct RangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  std::uint64_t begin;
  std::uint64_t end;
  int chunk_size;
};

struct MDRangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  std::size_t rank;
  const char* outer_dir;
  const char* inner_dir;
  std::vector<std::int64_t> begin;
  std::vector<std::int64_t> end;
  std::vector<std::int64_t> tile;
};

struct TeamPolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  int team_size;
  int league_size;
  int vector_length;
  int team_scratch_0;
  int team_scratch_1;
  int thread_scratch_0;
  int thread_scratch_1;
  int chunk_size;
};

ViewDumpResult dump_view_snapshot(
    const AllocationSnapshot& snapshot,
    const std::vector<unsigned char>& functor_data,
    const std::unordered_map<std::string, std::string>& metadata,
    const std::variant<kkf::NoPolicyDesc, kkf::ScalarPolicyDesc,
                       kkf::RangePolicyDesc, kkf::MDRangePolicyDesc,
                       kkf::TeamPolicyDesc>& policy,
    std::string_view phase, std::string_view label, std::uint64_t kernel_id,
    std::uint64_t kernel_invocation);

}  // namespace kkf
