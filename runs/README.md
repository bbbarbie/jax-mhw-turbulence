# Runs Directory

`runs/` is reserved for large local simulation outputs such as HDF5 production
files and NPZ detached-spin-up states.

These files are intentionally ignored by Git:

- `*.h5`
- `*.h50`
- `*.npz`

Historical production logs show that four finite-difference HDF5 files were
once written here:

- `runs/mhw_256_T1000_a018.h5`
- `runs/mhw_256_T1000_a022.h5`
- `runs/mhw_256_T1000_a078.h5`
- `runs/mhw_256_T1000_a082.h5`

Those original HDF5 files were unavailable during final project archiving.
Derived summaries, compact evidence logs, and generated tables are preserved
under `results/`.
