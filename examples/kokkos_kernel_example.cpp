#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <iostream>

template <class MemorySpace>
std::string make_dataset_name(const void* address, const std::string& label) {
  return std::to_string(reinterpret_cast<std::uintptr_t>(address)) + ";" +
         MemorySpace::name() + ";" + label;
}

template <class View, class Functor>
void save_state(std::string_view filename, const std::string& label,
                const View& view, const Functor& functor) {
  hid_t file =
      H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  if (file == H5I_INVALID_HID) {
    throw std::runtime_error("Failed to open hdf5 file " +
                             std::string(filename));
  }

  hsize_t dim = view.size() * sizeof(typename View::value_type);
  std::string dataset_name =
      make_dataset_name<typename View::memory_space>(view.data(), label);
  auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
  H5LTmake_dataset_char(file, dataset_name.data(), 1, &dim,
                        reinterpret_cast<char*>(mirror.data()));

  hsize_t functor_dim = sizeof(Functor);
  std::string functor_dataset_name =
      make_dataset_name<Kokkos::HostSpace>(&functor, "kernel_replay_functor");
  H5LTmake_dataset_char(file, functor_dataset_name.data(), 1, &functor_dim,
                        reinterpret_cast<const char*>(&functor));

  H5Fclose(file);
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  if (argc < 2) {
    std::cerr
        << "The program expects the name of the output hdf5 file as argument\n";
    return 1;
  }

  constexpr int number_of_values = 1024;
  int* ptr                       = static_cast<int*>(
      Kokkos::kokkos_malloc("view", number_of_values * sizeof(int)));
  Kokkos::View<int*, Kokkos::MemoryUnmanaged> values(ptr, number_of_values);

  Kokkos::parallel_for(
      "fill_values", Kokkos::RangePolicy<>(0, number_of_values),
      KOKKOS_LAMBDA(const int i) { values(i) = i; });

  int factor   = 2;
  auto functor = KOKKOS_LAMBDA(const int i, int& partial_sum) {
    partial_sum += values(i) * factor;
  };

  save_state(argv[1], "view", values, functor);

  int sum = 0;
  Kokkos::parallel_reduce(
      "sum_values", Kokkos::RangePolicy<>(0, number_of_values), functor, sum);

  constexpr int expected_sum =
      2 * number_of_values * (number_of_values - 1) / 2;
  if (sum != expected_sum) {
    std::cerr << "Unexpected Kokkos reduction result: got " << sum
              << ", expected " << expected_sum << '\n';
    return 1;
  }

  std::cout << "Kokkos kernel reduction succeeded: " << sum << '\n';

  Kokkos::kokkos_free(ptr);
  return 0;
}
