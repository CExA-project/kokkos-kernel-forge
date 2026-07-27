# KREPE

KREPE (Kernel Replay, Execution for Performance Evaluation) captures the
execution context of Kokkos kernels and replays them in standalone programs for
controlled performance evaluation.

## Build

Clone the repo, then configure and build with CMake

```sh
git clone https://github.com/CExA-project/KREPE.git
```

The library can be included in a CMake project:
```cmake
find_package(krepe REQUIRED)
target_link_libraries(my_program PRIVATE krepe::kernel_extractor)
```
The replayed program should link with `krepe::kernel_replayer`
```cmake
find_package(krepe REQUIRED)
target_link_libraries(my_program PRIVATE krepe::kernel_replayer)
```

## Extraction

In order to extract a kernel from a program, you have to:
1. replace the desired `Kokkos::parallel_for` with `krepe::kernel_replayer::parallel_for`,
   this will allow to save the execution policy and the kernel's data
2. execute with the `libkrepe.so` kokkos tool

For example, the following program:
```cpp
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  Kokkos::parallel_for("scale", N,
                       KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```
Will become
```cpp
#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // We replace the Kokkos parallel_for with the one from krepe::kernel_replayer
  krepe::kernel_replayer::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

The program has to be linked with `krepe::kernel_extractor`, it then has to be
executed with the following environment variables in order to extract the first
invocation of the kernel named "scale"
```sh
KOKKOS_TOOLS_LIBS=/path/to/libkrepe.so \
KOKKOS_TOOLS_ARGS="--krepe-dump-kernel-label=scale
--krepe-dump-kernel-invocation=1" \
./prog
```

This will generate two hdf5 files named `krepe_scale_2_{in,out}.h5`, see [HDF5 dump format](docs/hdf5-dumps.md) for the file naming scheme and
stored metadata.

### Parallel_for wrapper

If you are trying to extract a kernel used in a parallel_for wrapper provided
by another library, you can use the `krepe::kernel_replayer::replay_functor`
function to save the functor's data

```cpp
#include <krepe/extractor.hpp>

int main() {
  Kokkos::View<int*> values("view", N);
  my_funky_parallel_for("kernel", N,
                        krepe::kernel_replayer::replay_functor(
                            KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
}
```

Note that the execution policy cannot be saved in this case.

## Replay

Once the program dump has been generated, the kernel can be replayed in a
separate program. The new program should include the parallel construct call as
well as the functor declaration from the original program and any variable it
depends on. The replayer should also be initialized before Kokkos, using
`krepe::kernel_replayer::ScopeGuard`.

The program above becomes
```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>  // <krepe/extractor.hpp> -> <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  // We initialize the replayer before Kokkos
  krepe::kernel_replayer::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  // The execution policy could be ignored, as it will be restored from the dump
  const int N = 1024;
  // We don't care about the values inside the view, we only need it to have the
  // same type as in the original program
  Kokkos::View<int*> values;
  // No need to initialize, the initialized view from the original program is
  // captured in the dump Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // we still replace with krepe::kernel_replayer::parallel_for
  krepe::kernel_replayer::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

The program has to be linked with `krepe::kernel_replayer`, the dumps are passed using command line flags
```sh
./replay_prog --kernel-replayer-dump=krepe_scale_2_in.h5 --kernel-replayer-out-dump=krepe_scale_2_out.h5
```

### Modifying the execution policy

By default, `krepe::kernel_replayer::parallel_for` will use the execution policy
that was saved in the dump. You can override this by passing
`krepe::kernel_replayer::force_policy(your_policy)` as the execution policy
argument.

### Accessing the allocations

The value of allocations from the original program can be accessed using the
`get_allocation` and `get_out_allocation` for the values before and after the
kernel respectively.

```cpp
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
// Value of `values` before the kernel
int* initial_values_ptr = static_cast<int*>(krepe::kernel_replayer::get_allocation<memory_space>("values");
Kokkos::View<int*> intial_values(initial_values_ptr, 1024);
// Value of `values` after the kernel
int* result_values_ptr = static_cast<int*>(krepe::kernel_replayer::get_out_allocation<memory_space>("values");
Kokkos::View<int*> result_values(initial_values_ptr, 1024);
```

## Limitations

- Only the memory allocations going through Kokkos are captured (e.g. Views, `Kokkos::malloc`)
- Kernels taking a single generic argument are not supported
  (`KOKKOS_LAMBDA(auto i) { ... }`), if you still want to keep your kernel
  generic, you can use concepts to constrain the argument to either an integer or
  a Kokkos team handle:
  - For `RangePolicy`: `KOKKOS_LABMDA(std::integral auto i) { ... }`
  - For `TeamPolicy`: `KOKKOS_LAMBDA(Kokkos::TeamHandle auto team) { ... }`
  But note that generic kernels are not supported by nvcc
- The `LaunchBounds` and `WorkTag` template arguments for execution policies
  cannot be automatically restored in the replayed program
- Currently, the scratch memory parameters for `TeamPolicy` cannot be
  automatically restored in the replayed program
