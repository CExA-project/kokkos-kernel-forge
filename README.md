Tools for extracting, profiling, and auto-tuning Kokkos kernels from large HPC applications.

## Build

Fetch submodules, then configure and build the Kokkos example:

```sh
git clone https://github.com/CExA-project/kokkos-kernel-forge.git
git submodule update --init --recursive
```

## Targeted view dumps

OpenMP backend:

```sh
cmake -S . -B build/openmp -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ENABLE_OPENMP=ON
cmake --build build/openmp -j
KOKKOS_TOOLS_LIBS=$PWD/build/openmp/libkkf.so \
  KOKKOS_TOOLS_ARGS="--kkf-dump-kernel-label=sum_values" \
  ./build/openmp/examples/kokkos_kernel_example
```

CUDA backend:

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
cmake --build build/cuda -j
KOKKOS_TOOLS_LIBS=$PWD/build/cuda/libkkf.so \
  KOKKOS_TOOLS_ARGS="--kkf-dump-kernel-label=sum_values" \
  ./build/cuda/examples/kokkos_kernel_example
```

See [HDF5 dump format](docs/hdf5-dumps.md) for the file naming scheme and
stored metadata.
