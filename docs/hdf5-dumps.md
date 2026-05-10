# HDF5 Dump Format

The `--kkf-dump-kernel-label=<label>` argument selects one Kokkos kernel label
to dump. For example, `--kkf-dump-kernel-label=sum_values` matches a kernel
launched with the label `"sum_values"`.

For each matching kernel invocation, the tool writes two HDF5 files:

```text
kkf_<kernel-label>_<kernel-id>_in.h5
kkf_<kernel-label>_<kernel-id>_out.h5
```

The `in` file is written before the kernel runs, and the `out` file is written
after it completes. Each file contains one snapshot of all active user-visible
Kokkos allocations.

The dump stores:

```text
phase              # "in" before the kernel runs, "out" after it completes
kernel_label       # Kokkos label of the matched kernel
kernel_id          # tool-local id assigned to this kernel invocation
active_allocations # tracked user allocations alive at dump time
active_bytes       # total size, in bytes, of those active allocations
```

The `/views` group contains one subgroup per active allocation. Each allocation
group stores:

```text
label        # Kokkos allocation label
space        # Kokkos memory space name reported by the profiling hook
ptr          # allocation pointer value, stored as text
size         # allocation size in bytes
bytes_dumped
```
