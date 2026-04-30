#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace kkf {

struct AllocationRecord {
  std::string label;
  std::string space;
  std::uint64_t size;
};

struct ActiveAllocation {
  const void* ptr;
  AllocationRecord record;
};

struct AllocationSnapshot {
  std::uint64_t active_bytes = 0;
  std::vector<ActiveAllocation> allocations;
};

class AllocationTracker {
 public:
  void record_allocation(std::string label, std::string space, const void* ptr,
                         std::uint64_t size);
  void record_deallocation(std::string space, const void* ptr);

  AllocationSnapshot snapshot() const;

 private:
  mutable std::mutex mutex_;

  std::unordered_map<std::string,
                     std::unordered_map<const void*, AllocationRecord>>
      active_allocations_;
};

}  // namespace kkf
