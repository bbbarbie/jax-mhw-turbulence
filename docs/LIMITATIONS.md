# Limitations

## Missing HDF5 Production Data

The original HDF5 files from the four finite-difference production simulations
are unavailable in this local archive. The logs confirm that the runs
completed, and the existing postprocessed summary records flux and sensitivity
values, but the values could not be independently recomputed during wrap-up.

## Missing SLURM Scripts

The original NERSC job scripts and array scripts are not present. As a result:

- exact `sbatch` commands are not recoverable;
- resource directives are not recoverable;
- ensemble array-index-to-seed mapping is not recoverable.

## Unknown Ensemble Seed Mapping

The ensemble logs show the alpha mapping by array index, but they do not print
random seeds. Generated ensemble tables intentionally leave seed values blank.

## Transient Versus Saturated Protocol Mismatch

The historical direct-AD sweep started from configured initial conditions and
did not use the later detached saturated-state spin-up workflow. The
finite-difference calculation used long `T=1000` production simulations and a
saturation-window postprocessing protocol. These are not equivalent
trajectory-preparation or averaging procedures.

## Summed Versus Averaged Historical Objective

The direct-AD objective in `mhw_jax.py` returns an accumulated flux:

```text
xp.sum(metrics)
```

Raw objectives and gradients naturally scale in part with the number of AD
steps. Comparisons across AD trajectory lengths require division by AD steps.

## Finite Statistical Windows

The finite-difference sensitivities are based on the available production
outputs and a recorded saturation window. The HDF5 files are missing, so window
choice, statistical uncertainty, and alternate postprocessing cannot be
re-evaluated from raw data in this archive.

## Chaotic Sensitivity

Long-time turbulent sensitivities can be ill-conditioned for direct
trajectory-based AD because perturbations grow over chaotic trajectories. The
historical AD results should be treated as short-trajectory differentiation
evidence, not a validated long-time sensitivity method.

## Reverse-Mode Memory Scaling

Reverse-mode AD through long simulations requires retaining or reconstructing
trajectory information. This creates substantial memory and runtime pressure
for production-scale turbulent simulations.

## Incomplete Stage 2 Validation

Stage 2 implements detached spin-up and mean-flux AD, but production-scale
validation was not completed. The available validation failure was a resource
allocation configuration issue, not an AD or numerical failure.

## Shadowing Methods Not Implemented

Least-squares shadowing, NILSS, and related methods were not implemented in
this project. They remain plausible future approaches for chaotic long-time
sensitivity estimation.
