#pragma once

#include <cstddef>
#include <string>
#include <Kokkos_Core.hpp>

namespace cexa::kernel_replayer {
namespace impl {
void copy_functor(const unsigned char* data, std::size_t size);
}  // namespace impl

/**
 * @brief Copies the functor inside the dump so that the replayer can use it
 * later
 */
template <class Functor>
Functor replay_functor(Functor&& functor) {
  impl::copy_functor(reinterpret_cast<const unsigned char*>(&functor),
                     sizeof(Functor));
  return functor;
}

void add_metadata(const std::string& key, const std::string& value);
}  // namespace cexa::kernel_replayer
