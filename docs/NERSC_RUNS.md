# NERSC Run Record

This file records run evidence recoverable from local logs. The original SLURM
scripts are not available in this archive.

## Successful Finite-Difference Production Runs

| Log | Job ID | Alpha | Configuration | Output recorded by log |
|---|---:|---:|---|---|
| `results/raw_logs/finite_difference/mhw_a018-54261699.out` | 54261699 | 0.18 | `256x256`, `dt=0.0025`, RK4, MHW, Arakawa, `400000` steps, `T=1000` | `runs/mhw_256_T1000_a018.h5` |
| `results/raw_logs/finite_difference/mhw_a022-54261701.out` | 54261701 | 0.22 | `256x256`, `dt=0.0025`, RK4, MHW, Arakawa, `400000` steps, `T=1000` | `runs/mhw_256_T1000_a022.h5` |
| `results/raw_logs/finite_difference/mhw_a078-54261704.out` | 54261704 | 0.78 | `256x256`, `dt=0.0025`, RK4, MHW, Arakawa, `400000` steps, `T=1000` | `runs/mhw_256_T1000_a078.h5` |
| `results/raw_logs/finite_difference/mhw_a082-54261705.out` | 54261705 | 0.82 | `256x256`, `dt=0.0025`, RK4, MHW, Arakawa, `400000` steps, `T=1000` | `runs/mhw_256_T1000_a082.h5` |

All four logs report `Running with JAX backend on cpu:0`.

The HDF5 files recorded by the logs are not present in this archive.

## Additional Forward Run Evidence

| Log | Job ID | Alpha | Notes |
|---|---:|---:|---|
| `results/raw_logs/finite_difference/mhw_a1_T1000-54225942.out` | 54225942 | 1.0 | Completed `T=1000`-style forward run; output path appears as `runs/mhw_256_T1000_a1.h50` in the log. |
| `results/raw_logs/failures/mhw_debug-54226559.out` | 54226559 | 1.0 | Small completed debug/smoke run at `64x64`, `20` frames, `100` steps per frame. |

## Historical Direct-AD Sweep

Standalone direct-AD logs:

- `results/raw_logs/direct_ad/ad_a02_s500-54333963.out`
- `results/raw_logs/direct_ad/ad_a02_s1000-54333964.out`
- `results/raw_logs/direct_ad/ad_a02_s2000-54333965.out`
- `results/raw_logs/direct_ad/ad_a02_s5000-54333967.out`
- `results/raw_logs/direct_ad/ad_a02_s10000-54333968.out`
- `results/raw_logs/direct_ad/ad_a08_s500-54334192.out`
- `results/raw_logs/direct_ad/ad_a08_s1000-54334194.out`
- `results/raw_logs/direct_ad/ad_a08_s2000-54334196.out`
- `results/raw_logs/direct_ad/ad_a08_s5000-54334198.out`
- `results/raw_logs/direct_ad/ad_a08_s10000-54334200.out`

All report `64x64`, `dt=0.0025`, RK4, MHW, Arakawa, `diffw = diffn = 0.001`,
`diffop = 6`, `kappa = 1.0`, and `Running with JAX backend on cpu:0`.

## Ensemble AD Job Family

The ensemble logs are:

- `results/raw_logs/ensemble_ad/ens_ad-54341982_0.out` through
  `results/raw_logs/ensemble_ad/ens_ad-54341982_39.out`

Available logs show:

- indices `0-19`: `alpha = 0.2`
- indices `20-39`: `alpha = 0.8`
- `64x64`, `dt=0.0025`, RK4, MHW, Arakawa
- `Running with JAX backend on cpu:0`

The existing summary states `AD steps = 10000` and `20` seeds per alpha. The
array script and exact seed mapping are not available.

## Failed Runs

### Root Forward Check

`results/raw_logs/failures/root/forward_check_a08_r128.log` failed because the
script path was missing:

```text
/global/u2/b/bbbarbie/jax-mhw-turbulence/check_forward_stationarity.py
```

The log reports `srun` exit code 2.

### Results Forward Check

`results/raw_logs/failures/results/forward_check_a08_r128.log` failed because
the script path was missing:

```text
/global/u2/b/bbbarbie/jax-mhw-turbulence/results/check_forward_stationarity.py
```

The log reports `srun` exit code 2.

### Stage 2 Validation

`results/raw_logs/failures/validation_a08_r128_t200.log` failed before
execution:

```text
srun: error: No architecture specified, cannot estimate job costs.
srun: error: Unable to allocate resources: Unspecified error
```

This is a resource-allocation configuration failure. The available logs do not
support classifying it as OOM.
