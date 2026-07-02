# Project Summary

## 1. Motivation

This project investigated whether automatic differentiation can be used to
estimate sensitivities of turbulent particle transport in Hasegawa-Wakatani
(HW) and modified Hasegawa-Wakatani (MHW) plasma-edge simulations. The main
target quantity was the long-time-averaged turbulent particle flux and its
sensitivity to parameters such as the adiabaticity parameter `alpha`.

The project is archived here because the researcher moved to a different
project. This repository is therefore a reproducible wrap-up of what was
implemented, what was run, and what remains unresolved.

## 2. Physical Model

The code evolves vorticity and density fields on a periodic two-dimensional
domain. Vorticity is inverted spectrally to obtain electrostatic potential.
The implementation supports the modified HW coupling, where the zonal mean of
the coupling term is subtracted, as well as an unmodified mode. Nonlinear
advection can be computed with an Arakawa bracket or a simpler centered
Poisson bracket.

The flux diagnostic used throughout the archive is the spatial mean of
`-kappa * n * dphi/dy`.

## 3. JAX Implementation

The solver uses JAX arrays and enables 64-bit floating point arithmetic. It
contains RK4 time stepping and a predictor-corrector method historically named
`leapfrog` in the CLI. Hyperdiffusion is applied through spectral damping
factors. Forward production runs write HDF5 files, while later Stage 2 work
added NPZ state save/load and CSV result output.

Two solver histories are intentionally preserved:

- `mhw_jax.py`: historical baseline solver for forward HDF5 production and the
  original accumulated-objective direct-AD sweep.
- `mhw_jax_stage2_ad.py`: later detached-spin-up implementation intended for
  future mean-flux sensitivity studies from nonlinear states.

## 4. Forward Production Simulations

Four long production simulations completed near two target alpha values:

- `alpha = 0.18`
- `alpha = 0.22`
- `alpha = 0.78`
- `alpha = 0.82`

The logs show that these runs used `256x256` resolution, `dt = 0.0025`,
`kappa = 1.0`, `diffw = diffn = 0.001`, `diffop = 6`, modified HW, the
Arakawa bracket, RK4, `400000` total steps, and final time `T = 1000`.

The production logs confirm that all four simulations completed. The original
HDF5 files are not available in this archive.

## 5. Finite-Difference Sensitivity

The existing postprocessed Stage 1 summary records:

- `Gamma_bar(0.18) = 0.02776784`
- `Gamma_bar(0.22) = 0.03121801`
- `Gamma_bar(0.78) = 0.01744796`
- `Gamma_bar(0.82) = 0.01702120`
- `dGamma/dalpha near alpha=0.2 = 0.08625416`
- `dGamma/dalpha near alpha=0.8 = -0.01066905`

These values are supported by the completed production logs and the existing
postprocessed summary. They could not be independently recomputed during
project wrap-up because the HDF5 files are missing.

## 6. Historical Direct-AD Sweep

The historical direct-AD trajectory sweep used `mhw_jax.py` and differentiated
through short trajectories initialized from configured random initial
conditions. It was run at `alpha = 0.2` and `alpha = 0.8` for `500`, `1000`,
`2000`, `5000`, and `10000` AD steps.

The objective in this historical implementation returns `xp.sum(metrics)`, so
it accumulates per-step flux rather than averaging over the AD trajectory. Raw
objectives and gradients therefore require division by AD steps before
comparing different windows. After normalization, the gradients still showed
strong window dependence over the tested intervals.

Because these runs did not use detached saturated spin-up, they are not an
equivalent comparison to the long-time saturated finite-difference
calculation.

## 7. Ensemble AD Study

A 40-member ensemble AD study is preserved in compact logs. Available evidence
shows that array indices `0-19` used `alpha = 0.2` and `20-39` used
`alpha = 0.8`. The existing summary records `10000` AD steps and `20` seeds
per alpha.

The original job array script is missing, and the logs do not print seed
values. The exact mapping from array index to random seed is therefore not
recoverable from the available artifacts.

## 8. Stage 2 Detached-Spin-Up Development

Stage 2 introduced a more appropriate workflow for future sensitivity work:
forward spin-up is performed outside the differentiated trajectory, the final
nonlinear state can be saved as NPZ, and AD differentiates the mean flux over a
specified averaging window. It also adds CSV output and finite-value checks.

A full production-scale Stage 2 validation was not completed. The available
validation log failed before execution because resource allocation did not
specify an architecture. This is not evidence of an AD, numerical, memory, or
OOM failure.

## 9. Main Conclusions

This project established a JAX-based HW/MHW simulation and demonstrated
automatic differentiation through short model trajectories. Four long
production simulations supplied finite-difference estimates of saturated
mean-flux sensitivity near `alpha=0.2` and `alpha=0.8`. The historical
direct-AD sweep used an accumulated flux objective and did not include the
later detached nonlinear spin-up, so it does not constitute an equivalent
comparison with the saturated finite-difference calculation. After
normalization, the historical gradients still showed strong window dependence.
A Stage 2 implementation was developed to separate forward spin-up from
mean-flux differentiation, but production-scale validation was not completed
before the project direction changed. Future work would require statistically
converged Stage 2 experiments or a shadowing-based method such as LSS/NILSS.

## 10. Limitations

- Original production HDF5 files are missing.
- Original SLURM scripts are missing.
- Ensemble seed mapping is not recoverable from the available logs.
- Historical direct AD and finite-difference runs used different trajectory
  preparation and averaging protocols.
- Reverse-mode AD through long chaotic trajectories has severe memory and
  conditioning challenges.
- Stage 2 was implemented but not validated at production scale.

## 11. Future Directions

- Preserve exact commands, SLURM scripts, seed metadata, software environment,
  and raw-output provenance with every future production run.
- Run Stage 2 detached-spin-up experiments over statistically meaningful
  ensembles and averaging windows.
- Compare AD estimates only against finite differences generated with matching
  trajectory preparation and averaging protocols.
- Investigate long-time chaotic sensitivity methods such as least-squares
  shadowing or NILSS.
