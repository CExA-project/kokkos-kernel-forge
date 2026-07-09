#pragma once

#include <cassert>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <Kokkos_Core.hpp>
#include "allocation.hpp"

namespace cexa::kernel_replayer {
namespace impl {
void init_functor(char* buffer, std::size_t size);
void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label);
void* get_out_allocation(impl::MemorySpaceType memory_space,
                         const std::string& label);

template <class T>
struct add_unmanaged_trait;

template <unsigned int N>
struct add_unmanaged_trait<Kokkos::MemoryTraits<N>> {
  using type = Kokkos::MemoryTraits<N | Kokkos::Unmanaged>;
};

template <class Policy>
struct ForcePolicy {
  Policy policy;
};

template <class Policy>
constexpr bool is_force_policy_v = false;

template <class Policy>
constexpr bool is_force_policy_v<ForcePolicy<Policy>> = true;

// Variant which is visited at runtime to handle the compile time index type of
// execution policies
using index_type_var_t =
    std::variant<std::vector<std::int8_t>, std::vector<std::uint8_t>,
                 std::vector<std::int16_t>, std::vector<std::uint16_t>,
                 std::vector<std::int32_t>, std::vector<std::uint32_t>,
                 std::vector<std::int64_t>, std::vector<std::uint64_t>>;

// Variant which is visited at runtime to handle the compile time rank of
// MDRangePolicy
using mdrange_rank_var_t = std::variant<
#if KOKKOS_VERSION_GREATER_EQUAL(5, 2, 0)
    std::integral_constant<int, 1>,
#endif
    std::integral_constant<int, 2>, std::integral_constant<int, 3>,
    std::integral_constant<int, 4>, std::integral_constant<int, 5>,
    std::integral_constant<int, 6>>;

enum class PolicyType { none, scalar, range, mdrange };

inline PolicyType policy_type;

inline index_type_var_t policy_start;
inline index_type_var_t policy_end;
inline index_type_var_t policy_tile;
inline mdrange_rank_var_t mdrange_policy_rank;

template <class IndexType>
auto get_scalar_policy(const std::vector<IndexType>& end) {
  return end[0];
}

// NOTE: in practice, IndexType1 and IndexType2 will always be the
// same type. But since we visit a variant holding every possible index type,
// the compiler will instantiate every combination of index types, thus the need
// to use multiple template parameters and std::common_type.
template <class IndexType1, class IndexType2>
auto get_range_policy(const std::vector<IndexType1>& start,
                      const std::vector<IndexType2>& end) {
  using IndexType = std::common_type_t<IndexType1, IndexType2>;
  return Kokkos::RangePolicy(static_cast<IndexType>(start[0]),
                             static_cast<IndexType>(end[0]));
}

template <int rank, class IndexType>
auto get_mdrange_policy(const std::vector<IndexType>& start,
                        const std::vector<IndexType>& end,
                        const std::vector<IndexType>& tile) {
  Kokkos::Array<IndexType, rank> start_arr;
  Kokkos::Array<IndexType, rank> end_arr;
  Kokkos::Array<IndexType, rank> tile_arr;
  for (int i = 0; i < rank; i++) {
    start_arr[i] = static_cast<IndexType>(start[i]);
    end_arr[i]   = static_cast<IndexType>(end[i]);
    tile_arr[i]  = static_cast<IndexType>(tile[i]);
  }
  return Kokkos::MDRangePolicy(start_arr, end_arr, tile_arr);
}

// NOTE: Since we resolve the rank using a variant, when calling visit, the
// compiler will instantiate the parallel_for call with every possible rank,
// which is why we wrap the functor with a functor able to take 1 to 6
// arguments.
template <class Functor>
struct MDFunctorWrapper {
  Functor f;

  MDFunctorWrapper(Functor f) : f(f) {}

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0) const {
    if constexpr (requires(Functor f) { f(0); }) {
      f(i0);
      return;
    }
    KOKKOS_ASSERT(false);
  }

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0,
                                              IndexType i1) const {
    if constexpr (requires(Functor f) { f(0, 0); }) {
      f(i0, i1);
      return;
    }
    KOKKOS_ASSERT(false);
  }

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0, IndexType i1,
                                              IndexType i2) const {
    if constexpr (requires(Functor f) { f(0, 0, 0); }) {
      f(i0, i1, i2);
      return;
    }
    KOKKOS_ASSERT(false);
  }

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0, IndexType i1,
                                              IndexType i2,
                                              IndexType i3) const {
    if constexpr (requires(Functor f) { f(0, 0, 0, 0); }) {
      f(i0, i1, i2, i3);
      return;
    }
    KOKKOS_ASSERT(false);
  }

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0, IndexType i1,
                                              IndexType i2, IndexType i3,
                                              IndexType i4) const {
    if constexpr (requires(Functor f) { f(0, 0, 0, 0, 0); }) {
      f(i0, i1, i2, i3, i4);
      return;
    }
    KOKKOS_ASSERT(false);
  }

  template <std::integral IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(IndexType i0, IndexType i1,
                                              IndexType i2, IndexType i3,
                                              IndexType i4,
                                              IndexType i5) const {
    if constexpr (requires(Functor f) { f(0, 0, 0, 0, 0, 0); }) {
      f(i0, i1, i2, i3, i4, i5);
      return;
    }
    KOKKOS_ASSERT(false);
  }
};
}  // namespace impl

class ScopeGuard {
 private:
  std::vector<impl::Allocation> host_raw_allocations;
  std::unordered_map<std::string, void*> host_allocations;
  std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>
      host_output_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  std::vector<impl::Allocation> device_raw_allocations;
  std::unordered_map<std::string, void*> device_allocations;
  std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>
      device_output_allocations;
#endif

  void allocate(impl::MemorySpaceType memory_space, char* address,
                std::size_t size);

  void allocate_output(std::string label, std::string_view memory_space,
                       char* data, std::size_t size);

 public:
  ScopeGuard(int& argc, char* argv[]);
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();
};

std::optional<std::string> get_metadata(const std::string& key);

template <class MemorySpace>
void* get_allocation(const std::string& label) {
  return impl::get_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class MemorySpace>
void* get_out_allocation(const std::string& label) {
  return impl::get_out_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class View, class Functor, class Tuple>
void compare_views(View const& view, Tuple args, Functor&& f) {
  using memory_space = View::memory_space;
  using value_type   = View::value_type;

  value_type* data = static_cast<value_type*>(
      cexa::kernel_replayer::get_allocation<memory_space>(view.label()));
  value_type* ref_data = static_cast<value_type*>(
      cexa::kernel_replayer::get_out_allocation<memory_space>(view.label()));

  using ViewType = Kokkos::View<
      typename View::data_type, typename View::array_layout, memory_space,
      typename impl::add_unmanaged_trait<typename View::memory_traits>::type>;

  ViewType actual = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(data), args));
  ViewType expected = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(ref_data), args));

  f(expected, actual);
}

template <class DataType, class... Properties, class Functor, class Tuple>
void compare_views(const std::string& label, Tuple args, Functor&& f) {
  using View         = Kokkos::View<DataType, Properties...>;
  using memory_space = View::memory_space;
  using value_type   = View::value_type;

  value_type* data = static_cast<value_type*>(
      cexa::kernel_replayer::get_allocation<memory_space>(label));
  value_type* ref_data = static_cast<value_type*>(
      cexa::kernel_replayer::get_out_allocation<memory_space>(label));

  using ViewType = Kokkos::View<
      typename View::data_type, typename View::array_layout, memory_space,
      typename impl::add_unmanaged_trait<typename View::memory_traits>::type>;

  ViewType actual = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(data), args));
  ViewType expected = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(ref_data), args));

  f(expected, actual);
}

template <class Functor>
Functor replay_functor(const Functor& functor) {
  constexpr int N = sizeof(Functor);

  // In order to revive the to-be-replayed functor, we:
  // - Create a temporary functor and copy construct it from an existing
  //   functor (we have to use the copy constructor since its the only valid
  //   method to construct a lambda)
  // - Save the binary representation of this temporary functor
  // - memcpy the target functor into the temporary functor's memory
  // - Copy construct a new functor f from the patched temporary functor, since
  //   tracking is disabled, if the target functor contained Views, we won't get
  //   a segfault once they are destroyed
  // - memcpy the old data of the temporary functor back and properly destroy it
  // - return f
  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_disable();

  void* tmp_buffer         = std::aligned_alloc(alignof(Functor), N);
  Functor* tmp_functor     = new (tmp_buffer) Functor(functor);
  void* tmp_functor_buffer = std::malloc(N);
  std::memcpy(tmp_functor_buffer, tmp_buffer, N);

  std::size_t copy_length = N;
#if defined(KOKKOS_COMPILER_NVCC)
  // extended lambdas on nvcc also have a "data" pointer
  if constexpr (__nv_is_extended_host_device_lambda_closure_type(Functor)) {
    copy_length -= sizeof(void*);
  }
#endif
  impl::init_functor(static_cast<char*>(tmp_buffer), copy_length);
  Functor f(*tmp_functor);

  std::memcpy(tmp_buffer, tmp_functor_buffer, N);
  std::free(tmp_functor_buffer);
  tmp_functor->~Functor();
  std::free(tmp_buffer);

  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_enable();

  return f;
}

/**
 * @brief Wraps an execution policy and allows to override the saved one in a
 * parallel_for.
 */
template <class Policy>
impl::ForcePolicy<Policy> force_policy(Policy&& policy) {
  return impl::ForcePolicy{policy};
}

/**
 * @brief Replays a parallel_for using the execution policy and functors stored
 * in a replay dump.
 *
 * @param label A label forwarded to the underlying Kokkos::parallel_for
 * @param p The policy to use, if it is `force_policy(other_policy)` then
 * `other_policy` will be used, otherwise this parameter will be ignored and the
 * policy stored in the replay dump will be used
 * @param functor A functor with the same signature and the same captures as the
 * functor to replay, its operator() will be used but its data will be replaced
 * by the one stored in the replay dump
 */
template <class Policy, class Functor>
void parallel_for(const std::string& label, [[maybe_unused]] const Policy& p,
                  Functor&& functor) {
  if constexpr (impl::is_force_policy_v<Policy>) {
    Kokkos::parallel_for(label, p.policy, replay_functor(functor));
  } else {
    if (impl::policy_type == impl::PolicyType::none) {
      throw std::runtime_error(
          "Trying to use a replay parallel_for but the replay dump does not "
          "contain an execution policy");
    }

    // We need to use the wrapper even with RangePolicy, as when replaying
    // an MDRangePolicy the compiler will try to compile the scalar and
    // RangePolicy branches with a functor taking multiple arguments
    auto f = impl::MDFunctorWrapper{replay_functor(functor)};

    if (impl::policy_type == impl::PolicyType::scalar) {
      std::visit(
          [&](auto&& end) {
            Kokkos::parallel_for(label, impl::get_scalar_policy(end), f);
          },
          impl::policy_end);
    } else if (impl::policy_type == impl::PolicyType::range) {
      std::visit(
          [&](auto&& start, auto&& end) {
            Kokkos::parallel_for(label, impl::get_range_policy(start, end), f);
          },
          impl::policy_start, impl::policy_end);
    } else {
      std::visit(
          [&]<int i>(std::integral_constant<int, i>) {
            // MDRangePolicy's point_type and tile_type are always std::int64_t
            auto& start =
                std::get<std::vector<std::int64_t>>(impl::policy_start);
            auto& end  = std::get<std::vector<std::int64_t>>(impl::policy_end);
            auto& tile = std::get<std::vector<std::int64_t>>(impl::policy_tile);

            auto policy = impl::get_mdrange_policy<i>(start, end, tile);
            Kokkos::parallel_for(label, policy, f);
          },
          impl::mdrange_policy_rank);
    }
  }
}
}  // namespace cexa::kernel_replayer
