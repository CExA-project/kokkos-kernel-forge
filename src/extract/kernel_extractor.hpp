#pragma once

#include <optional>
#include <cstring>
#include <Kokkos_Core.hpp>

namespace cexa::kernel_replayer {
/**
 * @brief Stores a functor inside a view with a special label, so that the
 * replayer can use it later
 */
template <class Functor>
Functor extract_functor(Functor&& functor) {
  static std::optional<Kokkos::View<unsigned char*, Kokkos::HostSpace>>
      functor_data;
  if (!functor_data.has_value()) {
    functor_data.emplace("kernel_replayer_functor", sizeof(Functor));
    Kokkos::push_finalize_hook([&]() { functor_data.reset(); });
  }

  std::memcpy(functor_data->data(), reinterpret_cast<const void*>(&functor),
              sizeof(Functor));

  return functor;
}
}  // namespace cexa::kernel_replayer
