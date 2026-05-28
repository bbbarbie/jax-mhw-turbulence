# MHW JAX Sensitivity

JAX-based plasma turbulence simulator for the Modified Hasegawa–Wakatani model with AD-ready research workflow.

## Current features

- Modified Hasegawa–Wakatani model simulation
- JAX/NumPy backend support
- RK4 and leapfrog time steppers
- Arakawa bracket option
- HDF5 output for production runs
- Preliminary automatic differentiation test for sensitivity analysis

## Example usage

```bash
python mhw_jax.py --res 128 --dt 0.01 --nframes 20 --nts 100 --out mhw_out.h5
