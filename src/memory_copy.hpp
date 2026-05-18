#pragma once

#include "allocation_tracker.hpp"

#include <string>
#include <vector>

namespace kkf {

std::string copy_allocation_bytes(const ActiveAllocation& allocation,
                                  std::vector<unsigned char>& bytes);

}  // namespace kkf
