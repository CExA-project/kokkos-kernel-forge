Tools for extracting, profiling, and auto-tuning Kokkos kernels from large HPC applications.

## Build

Fetch submodules, then configure and build the Kokkos example:

```sh
git submodule update --init --recursive
cmake -S . -B build/serial -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build/serial --target kokkos_kernel_example -j
ctest --test-dir build/serial --output-on-failure
```

OpenMP backend:

```sh
cmake -S . -B build/openmp -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ENABLE_OPENMP=ON
cmake --build build/openmp --target kokkos_kernel_example -j
ctest --test-dir build/openmp --output-on-failure
```

CUDA backend:

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
cmake --build build/cuda --target kokkos_kernel_example -j
ctest --test-dir build/cuda --output-on-failure
```
