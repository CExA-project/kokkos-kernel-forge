#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <Kokkos_Core.hpp>

namespace cexa::kernel_replayer {
namespace impl {
void copy_functor(const unsigned char* data, std::size_t size);
void register_scalar_policy(std::size_t index_type_size, bool index_type_signed,
                            std::uint64_t N);
void register_range_policy(const char* space, std::size_t index_type_size,
                           bool index_type_signed, std::uint64_t begin,
                           std::uint64_t end);
void register_mdrange_policy(const char* space, std::size_t rank,
                             std::size_t index_type_size,
                             bool index_type_signed, std::uint64_t* begin,
                             std::uint64_t* end, std::uint64_t* tile);
bool next_invocation_will_dump(const char* kernel_name);

template <std::integral IndexType>
constexpr std::pair<std::size_t, bool> get_index_type_props() {
  return {sizeof(IndexType), std::is_signed_v<IndexType>};
}

template <std::integral IndexType>
constexpr std::uint64_t index_type_to_u64(IndexType N) {
  if constexpr (std::is_signed_v<IndexType>) {
    std::int64_t lN = N;
    return Kokkos::bit_cast<std::uint64_t>(lN);
  } else {
    return N;
  }
}

template <std::integral T>
void register_bounds(const T& upper_bound) {
  const auto [idx_type_size, idx_type_signed] = get_index_type_props<T>();
  register_scalar_policy(idx_type_size, idx_type_signed,
                         index_type_to_u64(upper_bound));
}

template <class... Args>
void register_bounds(const Kokkos::RangePolicy<Args...>& policy) {
  using Policy = Kokkos::RangePolicy<Args...>;
  const auto [idx_type_size, idx_type_signed] =
      get_index_type_props<typename Policy::index_type>();
  const std::uint64_t begin = index_type_to_u64(policy.begin());
  const std::uint64_t end   = index_type_to_u64(policy.end());
  register_range_policy(Policy::execution_space::name(), idx_type_size,
                        idx_type_signed, begin, end);
}

template <class... Args>
void register_bounds(const Kokkos::MDRangePolicy<Args...>& policy) {
  using Policy               = Kokkos::MDRangePolicy<Args...>;
  constexpr std::size_t rank = Policy::rank;

  // point_type and tile_type are always std::int64_t
  static_assert(
      std::is_same_v<typename Policy::point_type::value_type, std::int64_t>);
  static_assert(
      std::is_same_v<typename Policy::tile_type::value_type, std::int64_t>);

  const auto [idx_type_size, idx_type_signed] =
      get_index_type_props<std::int64_t>();
  std::array<std::uint64_t, rank> begin, end, tile;
  for (std::size_t i = 0; i < rank; i++) {
    begin[i] = index_type_to_u64(policy.m_lower[i]);
    end[i]   = index_type_to_u64(policy.m_upper[i]);
    tile[i]  = index_type_to_u64(policy.m_tile[i]);
  }
  register_mdrange_policy(policy.m_space.name(), policy.rank, idx_type_size,
                          idx_type_signed, begin.data(), end.data(),
                          tile.data());
}

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

/**
 * @brief Calls a parallel_for and saves the execution policy and the functor.
 */
template <class Policy, class Functor>
void parallel_for(const std::string& label, const Policy& policy,
                  Functor&& functor) {
  if (impl::next_invocation_will_dump(label.c_str())) {
    impl::register_bounds(policy);
  }
  Kokkos::parallel_for(label, policy, replay_functor(functor));
}

void add_metadata(const std::string& key, const std::string& value);
}  // namespace cexa::kernel_replayer
