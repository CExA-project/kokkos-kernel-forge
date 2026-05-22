Tools for extracting, profiling, and auto-tuning Kokkos kernels from large HPC applications.

## Build

Clone the repo, then configure and build the Kokkos example:

```sh
git clone https://github.com/CExA-project/kokkos-kernel-forge.git
```

## Targeted view dumps

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=<kokkos_install_dir>
cmake --build build -j
KOKKOS_TOOLS_LIBS=$PWD/build/openmp/libkkf.so \
  KOKKOS_TOOLS_ARGS="--kkf-dump-kernel-label=sum_values" \
  ./build/examples/kokkos_kernel_example
```

See [HDF5 dump format](docs/hdf5-dumps.md) for the file naming scheme and
stored metadata.
