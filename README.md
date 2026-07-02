# MHW JAX Sensitivity Archive

This repository records a concluded research project on sensitivities of
turbulent particle transport in Hasegawa-Wakatani (HW) and modified
Hasegawa-Wakatani (MHW) plasma-edge simulations using JAX.

The project established a working JAX solver and explored whether automatic
differentiation can compute sensitivities of long-time-averaged turbulent flux
with respect to physical parameters such as `alpha` and `kappa`. The archive is
organized to preserve the scientific evidence honestly, including successful
runs, failed validation attempts, and missing raw data limitations.

## Implemented Capabilities

- JAX-based HW/MHW simulation with float64 enabled.
- RK4 time stepping.
- Predictor-corrector time stepping, historically exposed as `leapfrog`.
- Modified/unmodified HW switch through zonal-mode coupling subtraction.
- Arakawa nonlinear Poisson bracket.
- Optional simple centered-difference Poisson bracket.
- Fourier-space inversion from vorticity to electrostatic potential.
- Hyperdiffusion using spectral damping factors.
- Turbulent particle-flux diagnostic.
- Historical direct AD with respect to `alpha` and `kappa`.
- HDF5 output for forward production runs.
- Stage 2 detached spin-up with NPZ state save/load and CSV output.

## Solver Versions

### `mhw_jax.py`

Historical baseline solver. It reproduces the original forward HDF5 workflow
and the archived direct-AD trajectory sweep. Its direct-AD objective returns
`xp.sum(metrics)`, accumulating per-step turbulent flux over the AD trajectory.
Raw objectives and gradients therefore scale in part with the number of AD
steps and must be normalized before comparing different trajectory lengths.

### `mhw_jax_stage2_ad.py`

Final Stage 2 detached-spin-up solver. It keeps the same baseline numerical
model but adds forward spin-up outside the differentiated window, mean-flux
averaging over the AD window, NPZ state save/load, CSV output, finite-value
checks, and separate spin-up and averaging durations. This is the recommended
starting point for future statistical-sensitivity experiments.

## Representative Commands

Very small local forward smoke command:

```bash
python mhw_jax.py --res 8 --dt 0.0025 --nframes 1 --nts 1 --solver rk4 --diffw 1e-3 --diffn 1e-3 --diffop 6 --out runs/smoke.h5
```

Historical direct-AD command shape:

```bash
python mhw_jax.py --test_ad --ad_steps 500 --res 64 --dt 0.0025 --solver rk4 --alpha 0.2 --kappa 1.0 --diffw 1e-3 --diffn 1e-3 --diffop 6
```

Stage 2 detached-spin-up command shape:

```bash
python mhw_jax_stage2_ad.py --test_ad --res 64 --dt 0.0025 --solver rk4 --alpha 0.8 --kappa 1.0 --diffw 1e-3 --diffn 1e-3 --diffop 6 --spinup_time 300 --avg_time 25 --result_csv results/stage2/example.csv
```

NERSC production command shape reconstructed from available logs:

```bash
python mhw_jax.py --res 256 --dt 0.0025 --nframes 1000 --nts 400 --solver rk4 --alpha 0.18 --kappa 1.0 --diffw 1e-3 --diffn 1e-3 --diffop 6 --out runs/mhw_256_T1000_a018.h5
```

The original SLURM scripts were not present in the local archive, so exact
`sbatch` commands and array-index seed mapping are not recoverable.

## Experimental Record

### Long finite-difference production simulations

Four production simulations completed at `alpha = 0.18`, `0.22`, `0.78`, and
`0.82`. The archived logs confirm `256x256`, `dt = 0.0025`, `kappa = 1.0`,
`diffw = diffn = 0.001`, `diffop = 6`, modified HW, Arakawa bracket, RK4,
`400000` total steps, and final time `T = 1000`.

The production logs confirm that all four simulations completed, and the
existing postprocessed summary records the following mean-flux and
finite-difference values. The original HDF5 files are no longer available, so
these values could not be independently recomputed during project wrap-up.

- `Gamma_bar(0.18) = 0.02776784`
- `Gamma_bar(0.22) = 0.03121801`
- `Gamma_bar(0.78) = 0.01744796`
- `Gamma_bar(0.82) = 0.01702120`
- `dGamma/dalpha near alpha=0.2 = 0.08625416`
- `dGamma/dalpha near alpha=0.8 = -0.01066905`

### Historical direct-AD trajectory sweep

Ten standalone logs record direct AD at `alpha = 0.2` and `alpha = 0.8` for
`500`, `1000`, `2000`, `5000`, and `10000` AD steps. These runs used the
historical `mhw_jax.py` accumulated objective and started from their configured
initial conditions. They did not use the later detached saturated-state
spin-up workflow.

The historical direct-AD sweep demonstrated that JAX could differentiate
through short HW/MHW trajectories. The raw objective accumulated per-step flux,
so comparisons across trajectory lengths require normalization. The normalized
gradients did not show clear convergence over the tested windows. These runs
also did not use the later detached saturated-state spin-up workflow, so they
should not be interpreted as a definitive comparison with the long-time
saturated finite-difference sensitivities.

### Ensemble AD study

Forty compact logs are preserved from job family `ens_ad-54341982`. Available
logs verify array indices `0-19` used `alpha = 0.2` and indices `20-39` used
`alpha = 0.8`. The existing summary states `AD steps = 10000` and `20` seeds
per alpha. The exact array-index-to-seed mapping is not recoverable from the
available artifacts, so generated tables leave seed values unknown.

### Stage 2 detached-spin-up development

`mhw_jax_stage2_ad.py` was developed to separate forward spin-up from
mean-flux differentiation through a later averaging window. It supports NPZ
state reuse and CSV output. A full production-scale Stage 2 validation was not
completed before the project changed direction. The available Stage 2
validation log failed before execution because no architecture was specified
for resource allocation; it is not evidence of an AD, numerical, memory, or
OOM failure.

## Generated Summary Tables and Figures

Tables generated from archived logs and summaries:

- `results/summary/tables/direct_ad_window_sweep.csv`
- `results/summary/tables/ensemble_ad_results.csv`
- `results/summary/tables/finite_difference_summary.csv`
- `results/summary/tables/run_manifest.csv`

Figures generated from those tables:

- `results/summary/figures/direct_ad_grad_alpha_vs_steps.png`
- `results/summary/figures/direct_ad_grad_kappa_vs_steps.png`
- `results/summary/figures/ensemble_grad_alpha_distribution.png`
- `results/summary/figures/ensemble_grad_kappa_distribution.png`
- `results/summary/figures/finite_difference_sensitivity.png`

Direct-AD window figures plot normalized per-step gradients. They should not
be read as apples-to-apples validation against saturated finite-difference
sensitivities because the trajectory preparation and averaging protocols differ.

## Missing Data

The original production HDF5 files are unavailable in this local copy. The
finite-difference values are preserved through completed production logs and
the existing postprocessed summary, not through independent reprocessing during
this cleanup.

The original SLURM job scripts are also unavailable, which prevents exact
reconstruction of some NERSC command lines and ensemble seed mappings.

Local solver execution was not performed during final wrap-up because `jax`,
`numpy`, and `h5py` were unavailable in this environment. The wrap-up
validation covered static Python compilation, log parsing, table generation,
and figure regeneration only.

## Future Directions

- Run statistically converged Stage 2 detached-spin-up experiments with saved
  nonlinear states and explicit seed metadata.
- Preserve command lines and SLURM scripts with every production run.
- Consider shadowing-based long-time sensitivity methods such as LSS or NILSS
  for chaotic turbulent averages.
