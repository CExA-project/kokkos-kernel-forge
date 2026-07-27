#include "allocation_tracker.hpp"

#include <utility>

namespace krepe {

void AllocationTracker::record_allocation(std::string label, std::string space,
                                          const void* ptr, const void* p_data,
                                          const std::uint64_t size) {
  if (ptr == nullptr) {
    return;
  }

  AllocationRecord record{std::move(label), space, p_data, size};
  std::lock_guard<std::mutex> lock(mutex_);
  active_allocations_[space].insert_or_assign(ptr, std::move(record));
}

void AllocationTracker::record_deallocation(std::string space,
                                            const void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto space_it = active_allocations_.find(space);
  if (space_it == active_allocations_.end()) {
    return;
  }

  space_it->second.erase(ptr);
  if (space_it->second.empty()) {
    active_allocations_.erase(space_it);
  }
}

AllocationSnapshot AllocationTracker::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  AllocationSnapshot snapshot;
  for (const auto& [space, allocations] : active_allocations_) {
    for (const auto& [ptr, record] : allocations) {
      snapshot.active_bytes += record.size;
      snapshot.allocations.push_back({ptr, record});
    }
  }

  return snapshot;
}

}  // namespace krepe
