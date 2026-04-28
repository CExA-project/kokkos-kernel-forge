Tools for extracting, profiling, and auto-tuning Kokkos kernels from large HPC applications.

## Build

Fetch submodules, then configure and build the Kokkos example:

```sh
git submodule update --init --recursive
cmake -S . -B build/serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build/serial -j
KOKKOS_TOOLS_LIBS=$PWD/build/serial/libkkf.so \
  ./build/serial/examples/kokkos_kernel_example
```

Examples are built by default. Disable them with `-DKKF_BUILD_EXAMPLES=OFF`.

OpenMP backend:

```sh
cmake -S . -B build/openmp -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ENABLE_OPENMP=ON
cmake --build build/openmp -j
KOKKOS_TOOLS_LIBS=$PWD/build/openmp/libkkf.so \
  ./build/openmp/examples/kokkos_kernel_example
```

CUDA backend:

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
cmake --build build/cuda -j
KOKKOS_TOOLS_LIBS=$PWD/build/cuda/libkkf.so \
  ./build/cuda/examples/kokkos_kernel_example
```
