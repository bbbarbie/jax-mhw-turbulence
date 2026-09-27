"""Task 1: T-sweep showing direct forward-mode AD of Gamma_T diverges at rate
lambda_1 while central finite differences stay bounded.

Usage:
  python experiments/task1_ad_vs_fd.py --regime a0.2 --res 64 --tspin 300 \
      --lambda1 0.5 --dalpha 0.02 --seeds 5

For each seed: spun-up state u0 (cached), one jax.jvp rollout through
lax.scan carrying (u, v, S=sum Gamma dt, dS=sum dGamma/dalpha dt); records
Gamma_T = S/T, dGamma_T/dalpha = dS/T and |v(t)| every ``rec`` steps.  Central
FD of Gamma_T at alpha +- dalpha from the same u0 on the same time grid.
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from common import regime_kwargs, save_json, OUT, autocorr_time, linfit  # noqa
from sens.mhw_system import MHWSystem, get_spun_state

ap = argparse.ArgumentParser()
ap.add_argument("--regime", default="a0.2")
ap.add_argument("--res", type=int, default=64)
ap.add_argument("--tspin", type=float, default=300.0)
ap.add_argument("--lambda1", type=float, required=True, help="Benettin lambda_1 (Task 2)")
ap.add_argument("--dalpha", type=float, default=0.02)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--Tmax-lyap", type=float, default=50.0, help="T_max in units of 1/lambda1")
ap.add_argument("--rec", type=int, default=40, help="record every this many steps")
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()

over = {} if args.nu is None else {"nu": args.nu}
kw = regime_kwargs(args.regime, args.res, **over)
S = MHWSystem(**kw)
alpha = S.alpha
dt = S.dt
name = f"task1_{args.regime}_res{args.res}{args.tag}"
Tmax = args.Tmax_lyap / args.lambda1
nchunk = int(np.ceil(Tmax / (args.rec * dt)))
Tmax = nchunk * args.rec * dt
print(f"== {name}: {kw}  lambda1={args.lambda1}  Tmax={Tmax:.1f} ({nchunk} chunks of {args.rec} steps)", flush=True)

step, flux = S.step, S.flux


@jax.jit
def ad_rollout(u0):
    """Forward-mode tangent of the running flux sum w.r.t. alpha."""
    def body(c, _):
        u, v, Ssum, dS = c
        # tangent of the map w.r.t. (u, alpha) with tangent (v, 1)
        u2, v2 = jax.jvp(step, (u, alpha), (v, 1.0))
        g, dg = jax.jvp(flux, (u2,), (v2,))
        return (u2, v2, Ssum + g * dt, dS + dg * dt), None

    def chunk(c, _):
        c, _ = jax.lax.scan(body, c, None, length=args.rec)
        u, v, Ssum, dS = c
        return c, (Ssum, dS, jnp.linalg.norm(v), flux(u))
    c0 = (u0, jnp.zeros_like(u0), 0.0, 0.0)
    _, out = jax.lax.scan(chunk, c0, None, length=nchunk)
    return out


@jax.jit
def fd_rollout(u0, a):
    def body(c, _):
        u, Ssum = c
        u2 = step(u, a)
        return (u2, Ssum + flux(u2) * dt), None

    def chunk(c, _):
        c, _ = jax.lax.scan(body, c, None, length=args.rec)
        return c, c[1]
    _, out = jax.lax.scan(chunk, (u0, 0.0), None, length=nchunk)
    return out


tgrid = np.arange(1, nchunk + 1) * args.rec * dt
per_seed = []
for seed in range(args.seeds):
    t0 = time.time()
    u0, fl, en = get_spun_state(S, seed, args.tspin)
    Ssum, dS, vnorm, ginst = [np.asarray(x) for x in ad_rollout(u0)]
    Sp = np.asarray(fd_rollout(u0, alpha + args.dalpha))
    Sm = np.asarray(fd_rollout(u0, alpha - args.dalpha))
    GT = Ssum / tgrid
    dGT_ad = dS / tgrid
    dGT_fd = (Sp - Sm) / (2 * args.dalpha) / tgrid
    tau_c = autocorr_time(fl[len(fl) // 2:], 1.0)          # from spin-up series (1 t.u. samples)
    per_seed.append(dict(seed=seed, GT=GT, dGT_ad=dGT_ad, dGT_fd=dGT_fd, vnorm=vnorm,
                         GT_plus=Sp / tgrid, GT_minus=Sm / tgrid,
                         sigma_G=float(fl[len(fl) // 2:].std()), tau_c=float(tau_c),
                         wall_s=time.time() - t0))
    print(f"  seed {seed}: Gamma_T(Tmax)={GT[-1]:.4e}  AD={dGT_ad[-1]:.3e}  FD={dGT_fd[-1]:.3e}  "
          f"|v(Tmax)|={vnorm[-1]:.3e}  sigma_G={per_seed[-1]['sigma_G']:.3e} tau_c={tau_c:.2f}  ({time.time()-t0:.0f}s)",
          flush=True)

# ---- aggregate --------------------------------------------------------------
AD = np.array([p["dGT_ad"] for p in per_seed])
FD = np.array([p["dGT_fd"] for p in per_seed])
VN = np.array([p["vnorm"] for p in per_seed])
ns = len(per_seed)
fd_mean, fd_se = FD.mean(0), FD.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(FD.mean(0))
ad_absmean = np.exp(np.log(np.abs(AD)).mean(0))            # geometric mean of |AD|
ratio = ad_absmean / np.maximum(np.abs(fd_mean), 1e-300)

# log-spaced T checkpoints 1/lambda1 .. 50/lambda1 (nearest grid points)
Tcheck = np.logspace(np.log10(1 / args.lambda1), np.log10(Tmax), 8)
idx = np.unique([np.argmin(np.abs(tgrid - T)) for T in Tcheck])
table = [dict(T=float(tgrid[i]), T_lyap=float(tgrid[i] * args.lambda1),
              AD_geomean_abs=float(ad_absmean[i]), AD_seeds=AD[:, i].tolist(),
              FD_mean=float(fd_mean[i]), FD_se=float(fd_se[i]), ratio=float(ratio[i]),
              vnorm_geomean=float(np.exp(np.log(VN[:, i]).mean()))) for i in idx]

# T at which AD first exceeds FD by 10, 1e3, 1e6 (log-interpolated, seed-geomean)
cross = {}
for lev in [1e1, 1e3, 1e6]:
    j = np.where(ratio >= lev)[0]
    cross[f"{lev:g}"] = float(tgrid[j[0]]) if len(j) else None

# slopes over the "clean growth" range: from first ratio>=10 to Tmax
j0 = np.where(ratio >= 10)[0]
j0 = j0[0] if len(j0) else nchunk // 4
sl_ad = [linfit(tgrid[j0:], np.log(np.abs(AD[k, j0:])))[0] for k in range(ns)]
sl_adc = [linfit(tgrid[j0:], np.log(np.abs(AD[k, j0:]) * tgrid[j0:]))[0] for k in range(ns)]  # 1/T removed
sl_v = [linfit(tgrid[j0:], np.log(VN[k, j0:]))[0] for k in range(ns)]
def ms(x):
    x = np.array(x); return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
res = dict(params=kw, lambda1_input=args.lambda1, dalpha=args.dalpha, Tmax=Tmax, tspin=args.tspin,
           seeds=ns, rec_steps=args.rec,
           fit_range=[float(tgrid[j0]), float(Tmax)],
           slope_logAD=dict(zip(["mean", "se"], ms(sl_ad))),
           slope_logAD_times_T=dict(zip(["mean", "se"], ms(sl_adc))),
           slope_logvnorm=dict(zip(["mean", "se"], ms(sl_v))),
           crossings=cross, table=table,
           sigma_G=float(np.mean([p["sigma_G"] for p in per_seed])),
           tau_c=float(np.mean([p["tau_c"] for p in per_seed])),
           GT_Tmax_mean=float(np.mean([p["GT"][-1] for p in per_seed])),
           GT_Tmax_se=float(np.std([p["GT"][-1] for p in per_seed], ddof=1) / np.sqrt(ns)) if ns > 1 else 0.0,
           FD_Tmax_mean=float(fd_mean[-1]), FD_Tmax_se=float(fd_se[-1]))
sig, tau = res["sigma_G"], res["tau_c"]
res["fd_noise_estimate_Tmax"] = float(sig * np.sqrt(tau / Tmax) / args.dalpha)
res["fd_noise_over_FD_Tmax"] = float(res["fd_noise_estimate_Tmax"] / max(abs(fd_mean[-1]), 1e-300))
print(f"  slopes: log|AD| {res['slope_logAD']}, log(|AD|T) {res['slope_logAD_times_T']}, log|v| {res['slope_logvnorm']}; "
      f"crossings {cross}; FD(Tmax)={fd_mean[-1]:.4e}+-{fd_se[-1]:.4e}; FD noise est {res['fd_noise_estimate_Tmax']:.3e}", flush=True)
save_json(res, name + ".json")
np.savez_compressed(f"{OUT}/{name}_curves.npz", tgrid=tgrid, AD=AD, FD=FD, VN=VN,
                    GT=np.array([p["GT"] for p in per_seed]))

# ---- figure -----------------------------------------------------------------
import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for k in range(ns):
    ax[0].semilogy(tgrid, np.abs(AD[k]), color="C0", lw=0.7, alpha=0.5)
ax[0].semilogy(tgrid, ad_absmean, color="C0", lw=2, label=r"AD $|d\Gamma_T/d\alpha|$ (geo-mean, 5 seeds)")
ax[0].errorbar(tgrid[idx], np.abs(fd_mean[idx]), yerr=fd_se[idx], fmt="s", color="C3", capsize=3,
               label=fr"central FD, $\Delta\alpha$={args.dalpha}, mean±SE (5 seeds)")
ax[0].semilogy(tgrid, np.abs(fd_mean), color="C3", lw=0.8, alpha=0.6)
xx = tgrid[j0:]; ax[0].semilogy(xx, np.abs(AD).mean(0)[j0] * np.exp(args.lambda1 * (xx - xx[0])), "k--", lw=1,
                                 label=fr"$e^{{\lambda_1 T}}$, $\lambda_1$={args.lambda1:.3f} (Benettin)")
ax[0].set_xlabel("T"); ax[0].set_ylabel(r"$|d\Gamma_T/d\alpha|$"); ax[0].legend(fontsize=7)
ax[0].set_title(f"{args.regime}, res {args.res}: direct AD vs FD")
for k in range(ns):
    ax[1].semilogy(tgrid, VN[k], color="C2", lw=0.7, alpha=0.6)
ax[1].semilogy(xx, np.exp(np.log(VN[:, j0]).mean()) * np.exp(args.lambda1 * (xx - xx[0])), "k--", lw=1,
               label=fr"$e^{{\lambda_1 t}}$")
ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\|v(t)\|$ (inhomogeneous tangent)"); ax[1].legend(fontsize=8)
ax[1].set_title(f"slope fit: {res['slope_logvnorm']['mean']:.3f} ± {res['slope_logvnorm']['se']:.3f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_{name}.png", dpi=130)
print("done", flush=True)
