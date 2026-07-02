# Reproducibility

## Available Evidence

The archive contains:

- solver source files `mhw_jax.py` and `mhw_jax_stage2_ad.py`;
- completed finite-difference production logs under
  `results/raw_logs/finite_difference/`;
- direct-AD sweep logs under `results/raw_logs/direct_ad/`;
- ensemble AD logs under `results/raw_logs/ensemble_ad/`;
- failure logs under `results/raw_logs/failures/`;
- postprocessed Stage 1 and Stage 2 summaries under `results/stage1/` and
  `results/stage2/`;
- generated machine-readable summary tables under `results/summary/tables/`.

## Missing Evidence

The original HDF5 files from the four long production simulations are not
available in this copy. The finite-difference values are preserved through
completed production logs and an existing postprocessed summary, not through
independent reprocessing during wrap-up.

The original SLURM scripts are also unavailable. This prevents exact recovery
of job commands, NERSC resource directives, and ensemble seed mapping.

## Software Dependencies

The solvers require:

- Python 3
- JAX
- `h5py` for HDF5 output
- NumPy

Both solver files enable JAX float64 with:

```python
config.update("jax_enable_x64", True)
```

The lightweight archive scripts in `scripts/` use only the Python standard
library. The figure generator writes simple PNG files directly and does not
require matplotlib or Pillow.

## CPU/GPU Caveats

The archived logs report:

```text
Running with JAX backend on cpu:0
```

Different hardware, JAX versions, FFT implementations, or accelerator
backends may produce small numerical differences. Long turbulent averages may
also vary with initial condition, averaging window, and transient handling.

## Validated Archive Commands

These commands are lightweight and were used for final archive validation:

```bash
python3 -m py_compile mhw_jax.py mhw_jax_stage2_ad.py scripts/parse_ad_logs.py scripts/parse_ensemble_logs.py scripts/build_run_manifest.py scripts/generate_summary_figures.py
python3 scripts/parse_ad_logs.py
python3 scripts/parse_ensemble_logs.py
python3 scripts/build_run_manifest.py
python3 scripts/generate_summary_figures.py
```

The solvers were checked for import/runtime dependencies before attempting
runtime validation. If JAX or `h5py` are unavailable locally, do not install
large dependencies automatically for archive validation.

During final project wrap-up on this local machine, `jax`, `numpy`, and `h5py`
were unavailable. Solver execution was therefore not performed locally; only
static compilation of the Python source and lightweight archive-script
validation were run.

## NERSC Reproducibility Limits

The following command shape is reconstructed from logs and CLI support:

```bash
python mhw_jax.py --res 256 --dt 0.0025 --nframes 1000 --nts 400 --solver rk4 --alpha 0.18 --kappa 1.0 --diffw 1e-3 --diffn 1e-3 --diffop 6 --out runs/mhw_256_T1000_a018.h5
```

Because the original SLURM scripts are missing, this is not an exact recovered
`sbatch` command. It is the Python command shape consistent with the available
logs.

## Random-Seed Provenance

Both solvers accept `--seed` and initialize:

```python
np.random.seed(args.seed)
```

However, the ensemble logs do not print seed values, and the array script that
would map array index to seed is missing. Generated ensemble tables therefore
leave `seed` blank and mark the mapping as not recoverable from available
artifacts.
