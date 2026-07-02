# Numerical Methods

## State Variables

The solver evolves two periodic two-dimensional fields:

- vorticity `w`
- density `n`

The spectral state is stored as `(w_hat, n_hat)`. Real-space fields are
recovered with inverse FFTs when evaluating nonlinear terms and diagnostics.

## Potential Inversion

Electrostatic potential is obtained from vorticity in Fourier space:

```text
phi_hat = -w_hat / k^2
```

The zero Fourier mode is handled by setting the inverse `k^2` factor to zero
where `k^2 = 0`. This avoids division by zero and fixes the arbitrary constant
component of potential.

## HW and MHW Coupling

The coupling term is formed from:

```text
coupling = phi - n
```

For the modified HW option, the zonal mean of this coupling is subtracted:

```text
coupling = coupling - mean(coupling, axis=1)
```

The code implements this through the `modified` flag in `MHWParams`.

## Nonlinear Bracket

Two nonlinear bracket implementations are available:

- `arakawa(f, g, dx, dy)`: conservative Arakawa bracket using periodic padding.
- `poisson_simple(f, g, dx, dy)`: centered finite-difference Poisson bracket.

The `arakawa` flag selects between these implementations. All archived
production and AD logs used Arakawa.

## Right-Hand Side

The implemented RHS computes:

```text
dw/dt = -[phi, w] + alpha * coupling
dn/dt = -[phi, n] + alpha * coupling - kappa * dphi/dy
```

The derivative dphi/dy is evaluated spectrally through multiplication by
`1j * KY`.

## Hyperdiffusion

Hyperdiffusion is applied as a spectral damping factor after each time step:

```text
exp(-diff * (k^2)^(diffop/2) * dt)
```

Separate damping factors are stored for vorticity and density.

## Time Stepping

### RK4

`step_rk4` implements a standard fourth-order Runge-Kutta update followed by
spectral damping. The long finite-difference runs and direct-AD logs used
`--solver rk4`.

### Predictor-Corrector Named `leapfrog`

`step_leapfrog` is historically exposed as `--solver leapfrog`, but the code
implements a predictor-corrector update:

1. predict with one forward Euler step;
2. evaluate the RHS at the predicted state;
3. update with the average of the initial and predicted RHS values;
4. apply spectral damping.

The historical CLI name is preserved for reproducibility.

## Turbulent Flux

The turbulent particle flux diagnostic is:

```text
Gamma = mean(-kappa * n * dphi/dy)
```

Forward HDF5 output stores this spatial mean at output frames.

## Historical Direct-AD Objective

In `mhw_jax.py`, `run_ad_optimization` differentiates with respect to both
`alpha` and `kappa` and returns:

```text
xp.sum(metrics)
```

where each metric is the spatially averaged flux after one time step. This is
an accumulated objective, not a time average. Raw objectives and gradients from
the historical direct-AD sweep therefore require division by AD steps before
comparing different trajectory lengths.

These historical runs start from their configured initial conditions and do
not use detached saturated-state spin-up.

## Stage 2 Mean-Flux Objective

In `mhw_jax_stage2_ad.py`, `run_ad_window` differentiates only through the
averaging window and returns:

```text
xp.mean(flux_series)
```

The nonlinear starting state is explicitly detached with
`jax.lax.stop_gradient`. This is the more appropriate implementation for
future experiments that separate turbulent spin-up from sensitivity averaging.

Stage 2 currently differentiates the mean flux with respect to `alpha` with
kappa fixed.

## Detached Spin-Up

Stage 2 supports:

- forward-only spin-up for a specified physical duration;
- finite-state checks after spin-up;
- optional NPZ save/load of the spectral nonlinear state;
- AD over a separate averaging duration;
- CSV output for result rows.

Production-scale validation of this workflow was not completed in the
available archive.
