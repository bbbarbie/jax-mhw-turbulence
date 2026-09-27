"""Task 2: validate the Lyapunov spectrum of the MHW discrete map.

Usage:
  python experiments/task2_lyapunov.py --regime a0.2 --res 64 --tspin 300 \
      --T 100 --M 40 [--dt-check] [--res-check]

Every tangent is jax.jvp(step); QR renormalisation every ``renorm`` steps.
Writes results/sens/task2_<regime>_res<res>.json and figures.
"""
import argparse
import time

import numpy as np
import jax

from common import regime_kwargs, save_json, OUT  # noqa
from sens.mhw_system import MHWSystem, get_spun_state
from sens.lyapunov import lyapunov_spectrum, exponents_from_logR, batched_se, kaplan_yorke
from sens.hw_linear import gamma_max, n_unstable_modes

ap = argparse.ArgumentParser()
ap.add_argument("--regime", default="a0.2")
ap.add_argument("--res", type=int, default=64)
ap.add_argument("--tspin", type=float, default=300.0)
ap.add_argument("--T", type=float, default=100.0, help="integration time for each check")
ap.add_argument("--M", type=int, default=40, help="number of exponents for the spectrum")
ap.add_argument("--renorm", type=int, default=40, help="steps between QRs (0.1 t.u. at dt=0.0025)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dt-check", action="store_true")
ap.add_argument("--res-check", action="store_true")
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()

over = {} if args.nu is None else {"nu": args.nu}
kw = regime_kwargs(args.regime, args.res, **over)
S = MHWSystem(**kw)
alpha = S.alpha
name = f"task2_{args.regime}_res{args.res}{args.tag}"
res = dict(params=kw, T=args.T, renorm_steps=args.renorm, seed=args.seed)
print(f"== {name}: {kw}", flush=True)

u0, fl, en = get_spun_state(S, args.seed, args.tspin)
res["spinup"] = dict(tspin=args.tspin, flux_last100_mean=float(fl[-100:].mean()),
                     flux_last100_std=float(fl[-100:].std()),
                     energy_last100_mean=float(en[-100:].mean()))
nR = int(round(args.T / (args.renorm * S.dt)))


def run(sys_, u, M, renorm, nR, key):
    t = time.time()
    logR, _ = lyapunov_spectrum(sys_, u, sys_.alpha, M=M, renorm_every=renorm, n_renorm=nR,
                                key=jax.random.PRNGKey(key))
    lam, se = exponents_from_logR(logR, sys_.dt, renorm, discard=nR // 10)
    return logR, lam, se, time.time() - t


# ---- 1. running lambda_1 vs T, 3. three random directions (M=1) -------------
res["lambda1_dirs"] = []
running = []
for k in range(3):
    logR, lam, se, wall = run(S, u0, 1, args.renorm, nR, key=k)
    cum = np.cumsum(logR[:, 0]) / (np.arange(1, nR + 1) * args.renorm * S.dt)
    running.append(cum)
    res["lambda1_dirs"].append(dict(dir_key=k, lambda1=float(lam[0]), se=float(se[0]),
                                    se_batched=float(batched_se(logR, S.dt, args.renorm, discard=nR // 10)[0]),
                                    wall_s=wall))
    print(f"  lambda1 dir {k}: {lam[0]:.4f} +- {se[0]:.4f}  ({wall:.0f}s)", flush=True)
running = np.array(running)
tgrid = np.arange(1, nR + 1) * args.renorm * S.dt
res["running_T"] = tgrid[::max(1, nR // 200)].tolist()
res["running_lambda1"] = running[:, ::max(1, nR // 200)].tolist()

# ---- 2. renormalisation interval ------------------------------------------
res["lambda1_renorm"] = []
for rn in [args.renorm // 2, args.renorm, args.renorm * 2]:
    nR_ = int(round(args.T / (rn * S.dt)))
    logR, lam, se, wall = run(S, u0, 1, rn, nR_, key=0)
    res["lambda1_renorm"].append(dict(renorm_steps=rn, renorm_time=rn * S.dt,
                                      lambda1=float(lam[0]), se=float(se[0]), wall_s=wall))
    print(f"  lambda1 renorm {rn} steps ({rn*S.dt:g} t.u.): {lam[0]:.4f} +- {se[0]:.4f}", flush=True)

# ---- 4. dt halved ----------------------------------------------------------
if args.dt_check:
    S2 = MHWSystem(**{**kw, "dt": kw["dt"] / 2})
    nR2 = int(round(args.T / (2 * args.renorm * S2.dt)))
    logR, lam, se, wall = run(S2, u0, 1, 2 * args.renorm, nR2, key=0)
    res["lambda1_dt_half"] = dict(dt=S2.dt, lambda1=float(lam[0]), se=float(se[0]), wall_s=wall)
    print(f"  lambda1 dt/2: {lam[0]:.4f} +- {se[0]:.4f}", flush=True)

# ---- 6. full spectrum (M exponents) ---------------------------------------
logR, lam, se, wall = run(S, u0, args.M, args.renorm, nR, key=0)
seb = batched_se(logR, S.dt, args.renorm, discard=nR // 10)
Npos = int(np.sum(lam > 0))
res["spectrum"] = dict(M=args.M, lam=lam.tolist(), se=se.tolist(), se_batched=seb.tolist(),
                       N_plus=Npos, sum_M=float(lam.sum()),
                       kaplan_yorke=kaplan_yorke(lam) if lam[-1] < 0 else None,
                       wall_s=wall)
print(f"  spectrum M={args.M}: N+={Npos}, lambda1={lam[0]:.4f}, lambda_M={lam[-1]:.4f}, "
      f"sum={lam.sum():.3f}, D_KY={res['spectrum']['kaplan_yorke']}  ({wall:.0f}s)", flush=True)

# ---- 7. linear growth rate -------------------------------------------------
g, kx, ky = gamma_max(S.res, S.L, alpha, S.kappa, S.nu, S.diffop)
res["gamma_max"] = dict(gamma_max=g, kx=kx, ky=ky,
                        n_unstable_modes=n_unstable_modes(S.res, S.L, alpha, S.kappa, S.nu, S.diffop),
                        lambda1_over_gamma_max=float(lam[0] / g))
print(f"  gamma_max={g:.4f} at (kx,ky)=({kx:.3f},{ky:.3f}); lambda1/gamma_max={lam[0]/g:.3f}; "
      f"linearly unstable modes: {res['gamma_max']['n_unstable_modes']}", flush=True)

# ---- 5. resolution check ---------------------------------------------------
if args.res_check:
    res["res_check"] = []
    for r in [args.res // 2, args.res * 2]:
        try:
            Sr = MHWSystem(**{**kw, "res": r})
            ur, _, _ = get_spun_state(Sr, args.seed, args.tspin)
            Mr = args.M
            logR, lam_r, se_r, wall = run(Sr, ur, Mr, args.renorm, nR, key=0)
            res["res_check"].append(dict(res=r, M=Mr, lambda1=float(lam_r[0]), se=float(se_r[0]),
                                         N_plus=int(np.sum(lam_r > 0)), lam=lam_r.tolist(),
                                         wall_s=wall))
            print(f"  res {r}: lambda1={lam_r[0]:.4f} +- {se_r[0]:.4f}, N+={int(np.sum(lam_r>0))} ({wall:.0f}s)", flush=True)
        except FloatingPointError as e:
            res["res_check"].append(dict(res=r, error=str(e)))
            print(f"  res {r}: {e}", flush=True)

save_json(res, name + ".json")

# ---- figure -----------------------------------------------------------------
import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in range(3):
    ax[0].plot(tgrid, running[k], lw=1, label=f"direction {k}")
ax[0].set_xlabel("T"); ax[0].set_ylabel(r"running $\lambda_1$"); ax[0].legend()
ax[0].set_title(f"{args.regime} res {args.res}: running estimate")
ax[1].plot(np.arange(1, args.M + 1), lam, "o-", ms=3)
ax[1].errorbar(np.arange(1, args.M + 1), lam, yerr=seb, fmt="none", ecolor="gray")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("index j"); ax[1].set_ylabel(r"$\lambda_j$"); ax[1].set_title(f"spectrum, N+={Npos}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_{name}.png", dpi=130)
print("done", flush=True)
