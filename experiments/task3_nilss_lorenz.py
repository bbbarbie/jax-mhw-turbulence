"""Task 3: NILSS on Lorenz 63 (sigma=10, beta=8/3, rho=28), J = z, s = rho.

Produces results/sens/task3_lorenz.json and results/sens/fig_task3_lorenz.png.

Checks: M = 1 vs M = 2; convergence with T (error ~ 1/sqrt(T)); insensitivity
to segment length DT over a factor of 4; comparison with central FD of the
long-time mean with error bars from independent trajectories.
Run:  python experiments/task3_nilss_lorenz.py
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sens.lorenz import make_lorenz_step, qoi_z, rollout_mean, lorenz_rhs  # noqa: E402
from sens.nilss import nilss_segments, nilss_solve  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "sens")
os.makedirs(OUT, exist_ok=True)

DT = 0.01
RHO = 28.0
step = make_lorenz_step(DT)


def spun_ic(seed, tspin=200.0):
    u0 = jnp.array([1.0, 1.0, 25.0]) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(100 + seed), (3,))
    u, _ = rollout_mean(step, u0, RHO, int(tspin / DT), qoi_z)
    return u


PLOT_ONLY = "--plot-only" in sys.argv
if PLOT_ONLY:
    with open(os.path.join(OUT, "task3_lorenz.json")) as fh:
        res = json.load(fh)
    fd = np.array(res["fd"]["values"]); T_SC = res["nilss_scatter"]["T"]; T_DT = 4000.0
else:
    res = {"dt": DT, "rho": RHO, "sigma": 10.0, "beta": 8.0 / 3.0}

if not PLOT_ONLY:
    # ---------------------------------------------------------------- FD reference
    # <z>(rho +- drho) over T_fd per trajectory, n_fd independent trajectories.
    DRHO = 0.5
    T_FD = 20000.0
    N_FD = 20
    t0 = time.time()
    fd = []
    zbar = {"+": [], "-": []}
    for k in range(N_FD):
        u = spun_ic(1000 + k)
        _, zp = rollout_mean(step, u, RHO + DRHO, int(T_FD / DT), qoi_z)
        _, zm = rollout_mean(step, u, RHO - DRHO, int(T_FD / DT), qoi_z)
        zbar["+"].append(float(zp)); zbar["-"].append(float(zm))
        fd.append((float(zp) - float(zm)) / (2 * DRHO))
    fd = np.array(fd)
    res["fd"] = dict(drho=DRHO, T_per_traj=T_FD, n_traj=N_FD, values=fd.tolist(),
                     mean=float(fd.mean()), se=float(fd.std(ddof=1) / np.sqrt(N_FD)),
                     zbar_plus=zbar["+"], zbar_minus=zbar["-"],
                     wall_s=time.time() - t0)
    print(f"FD: d<z>/drho = {fd.mean():.4f} +- {fd.std(ddof=1)/np.sqrt(N_FD):.4f} "
          f"(drho={DRHO}, T={T_FD} x {N_FD} traj, {time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------ NILSS: M=1,2 ; T convergence
    NSEG = 200          # DT_seg = 2.0
    K_MAX = 10000       # T = 20000
    res["nilss"] = {}
    for M in [1, 2]:
        t0 = time.time()
        u0 = spun_ic(0)
        segs, _ = nilss_segments(step, u0, RHO, qoi_z, M=M, nseg_steps=NSEG,
                                 K=K_MAX, dt=DT, key=jax.random.PRNGKey(1),
                                 rec_every=10, K_pre=10, rhs=lorenz_rhs)
        Ks = [int(k) for k in np.unique(np.logspace(np.log10(25), np.log10(K_MAX), 25).astype(int))]
        conv = [(k * NSEG * DT, nilss_solve(segs, K_use=k)["dJds"]) for k in Ks]
        full = nilss_solve(segs)
        res["nilss"][f"M{M}"] = dict(
            M=M, DT_seg=NSEG * DT, T=full["T"], dJds=full["dJds"], Javg=full["Javg"],
            term_int=full["term_int"], term_dil=full["term_dil"],
            max_vperp=float(full["vperp_norm"].max()),
            mean_vperp=float(full["vperp_norm"].mean()),
            conv_T=[c[0] for c in conv], conv_dJds=[c[1] for c in conv],
            wall_s=time.time() - t0)
        print(f"NILSS M={M}: T={full['T']:.0f} dJds={full['dJds']:.4f} "
              f"(int {full['term_int']:.4f} + dil {full['term_dil']:.4f}) "
              f"max|vperp|={full['vperp_norm'].max():.2f} {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------ scatter over independent trajectories
    # 10 independent NILSS runs at T=2000 (M=1) -> empirical SE, and check ~1/sqrt(T)
    T_SC = 2000.0
    scat = []
    t0 = time.time()
    for k in range(10):
        u0 = spun_ic(10 + k)
        segs, _ = nilss_segments(step, u0, RHO, qoi_z, M=1, nseg_steps=NSEG,
                                 K=int(T_SC / (NSEG * DT)), dt=DT,
                                 key=jax.random.PRNGKey(50 + k), rec_every=10, K_pre=10, rhs=lorenz_rhs)
        scat.append(nilss_solve(segs)["dJds"])
    scat = np.array(scat)
    res["nilss_scatter"] = dict(T=T_SC, M=1, values=scat.tolist(), mean=float(scat.mean()),
                                std=float(scat.std(ddof=1)),
                                se=float(scat.std(ddof=1) / np.sqrt(len(scat))),
                                wall_s=time.time() - t0)
    print(f"NILSS scatter (T={T_SC}, 10 traj): {scat.mean():.4f} +- {scat.std(ddof=1):.4f} (std)", flush=True)

    # ------------------------------------------------------- DT_seg insensitivity
    T_DT = 4000.0
    res["dt_seg"] = {}
    for nseg in [100, 200, 400]:       # DT_seg = 1, 2, 4
        vals = []
        for k in range(4):
            u0 = spun_ic(30 + k)
            segs, _ = nilss_segments(step, u0, RHO, qoi_z, M=1, nseg_steps=nseg,
                                     K=int(T_DT / (nseg * DT)), dt=DT,
                                     key=jax.random.PRNGKey(70 + k), rec_every=10, K_pre=10, rhs=lorenz_rhs)
            vals.append(nilss_solve(segs)["dJds"])
        vals = np.array(vals)
        res["dt_seg"][f"{nseg*DT:g}"] = dict(DT_seg=nseg * DT, T=T_DT, values=vals.tolist(),
                                              mean=float(vals.mean()),
                                              se=float(vals.std(ddof=1) / 2))
        print(f"DT_seg={nseg*DT:g}: dJds = {vals.mean():.4f} +- {vals.std(ddof=1)/2:.4f}", flush=True)

    with open(os.path.join(OUT, "task3_lorenz.json"), "w") as fh:
        json.dump(res, fh, indent=1)

# ------------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for M, c in [(1, "C0"), (2, "C1")]:
    r = res["nilss"][f"M{M}"]
    ax[0].semilogx(r["conv_T"], r["conv_dJds"], "o-", color=c, ms=3, label=f"NILSS M={M}")
ax[0].axhspan(fd.mean() - res["fd"]["se"], fd.mean() + res["fd"]["se"], color="k", alpha=0.15,
              label=f"FD {fd.mean():.3f}±{res['fd']['se']:.3f}")
ax[0].axhline(1.01, color="r", ls="--", lw=0.8, label="lit. ≈1.01 (Ni & Wang 2017)")
ax[0].set_xlabel("T"); ax[0].set_ylabel(r"$d\langle z\rangle/d\rho$"); ax[0].legend(fontsize=8)
ax[0].set_title("convergence with T (DT_seg=2)")
r = res["nilss"]["M1"]
Tt = np.array(r["conv_T"]); cv = np.array(r["conv_dJds"])
err_stat = np.abs(cv[:-1] - cv[-1])
ax[1].loglog(Tt[:-1], err_stat, "o-", ms=3, label="|NILSS(T) − NILSS(T=20000)|")
ax[1].loglog(Tt, np.abs(cv - fd.mean()), "s-", ms=3, color="C3", label="|NILSS(T) − FD mean|")
ax[1].loglog(Tt, err_stat[0] * np.sqrt(Tt[0] / Tt), "k--", label=r"$T^{-1/2}$")
ax[1].axhline(res["fd"]["se"], color="gray", ls=":", label="FD SE")
ax[1].set_xlabel("T"); ax[1].set_ylabel("error"); ax[1].legend(fontsize=7)
ax[1].set_title("statistical convergence vs bias")
xs = [float(k) for k in res["dt_seg"]]
ys = [res["dt_seg"][k]["mean"] for k in res["dt_seg"]]
es = [res["dt_seg"][k]["se"] for k in res["dt_seg"]]
ax[2].errorbar(xs, ys, yerr=es, fmt="o", capsize=3, label=f"NILSS M=1, T={T_DT:.0f}")
ax[2].axhspan(fd.mean() - res["fd"]["se"], fd.mean() + res["fd"]["se"], color="k", alpha=0.15, label="FD ± SE")
ax[2].set_xscale("log"); ax[2].set_xlabel("segment length ΔT"); ax[2].set_ylabel(r"$d\langle z\rangle/d\rho$")
ax[2].legend(fontsize=8); ax[2].set_title("ΔT insensitivity")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_task3_lorenz.png"), dpi=130)
print("done")
