"""Task 4: NILSS on MHW at low resolution.

Usage:
  python experiments/task4_nilss_mhw.py --regime a0.2 --res 32 --nu 3e-2 \
      --tspin 300 --M 15 --DT 2.0 --K 100 --fd-seeds 5 --fd-T 1000 --dalpha 0.02

Runs:  (i) NILSS with M homogeneous tangents, segment length DT, K segments
       (ii) M+5 check, (iii) DT/2 check (same total T), (iv) convergence in T
       from the prefix solves of run (i), (v) central FD of the long-time mean
       flux over fd-seeds independent spun-up states.
Writes results/sens/task4_<regime>_res<res><tag>.json and a figure.
"""
import argparse
import resource
import time

import numpy as np
import jax
import jax.numpy as jnp

from common import regime_kwargs, save_json, OUT, autocorr_time  # noqa
from sens.mhw_system import MHWSystem, get_spun_state
from sens.nilss import nilss_segments, nilss_solve

ap = argparse.ArgumentParser()
ap.add_argument("--regime", default="a0.2")
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--tspin", type=float, default=300.0)
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--DT", type=float, default=2.0)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--K-pre", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fd-seeds", type=int, default=5)
ap.add_argument("--fd-T", type=float, default=1000.0)
ap.add_argument("--dalpha", type=float, default=0.02)
ap.add_argument("--neutral", default="rhs", choices=["rhs", "discrete"])
ap.add_argument("--skip-checks", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()

over = {} if args.nu is None else {"nu": args.nu}
kw = regime_kwargs(args.regime, args.res, **over)
S = MHWSystem(**kw)
alpha, dt = S.alpha, S.dt
name = f"task4_{args.regime}_res{args.res}{args.tag}"
nseg = int(round(args.DT / dt))
print(f"== {name}: {kw} M={args.M} DT={args.DT} ({nseg} steps) K={args.K} T={args.K*args.DT}", flush=True)
res = dict(params=kw, M=args.M, DT=args.DT, K=args.K, T=args.K * args.DT, neutral=args.neutral,
           seed=args.seed, tspin=args.tspin)

u0, fl, en = get_spun_state(S, args.seed, args.tspin)
rhs = S.rhs if args.neutral == "rhs" else None


def run_nilss(M, nseg, K, key, label):
    t0 = time.time()
    m0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    segs, _ = nilss_segments(S.step, u0, alpha, S.flux, M=M, nseg_steps=nseg, K=K, dt=dt,
                             key=jax.random.PRNGKey(key), rec_every=max(1, nseg // 20),
                             K_pre=args.K_pre, verbose=True, log_prefix=f"  [{label}] ", rhs=rhs)
    wall = time.time() - t0
    m1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sol = nilss_solve(segs)
    print(f"  [{label}] M={M} DT={nseg*dt:g} K={K}: dGamma/dalpha = {sol['dJds']:.5e} "
          f"(ibp form {sol['dJds_ibp']:.5e}; perp {sol['term_perp']:.4e} + dil {sol['term_dilB']:.4e}) "
          f"<Gamma>={sol['Javg']:.5f}  max|vperp|={sol['vperp_norm'].max():.3e} "
          f"mean|vperp|={sol['vperp_norm'].mean():.3e}  wall {wall:.0f}s  maxrss {m1/1e6:.2f} GB", flush=True)
    return segs, sol, wall, m1


segs, sol, wall, rss = run_nilss(args.M, nseg, args.K, 1, "main")
Ks = [int(k) for k in np.unique(np.logspace(np.log10(5), np.log10(args.K), 20).astype(int))]
conv = [(k * nseg * dt, nilss_solve(segs, K_use=k)["dJds"]) for k in Ks]
res["main"] = dict(dJds=sol["dJds"], dJds_ibp=sol["dJds_ibp"], Javg=sol["Javg"],
                   term_perp=sol["term_perp"], term_dil=sol["term_dilB"],
                   max_vperp=float(sol["vperp_norm"].max()), mean_vperp=float(sol["vperp_norm"].mean()),
                   max_abs_a=float(np.abs(sol["a"]).max()), max_abs_xi=float(np.abs(sol["xi"]).max()),
                   conv_T=[c[0] for c in conv], conv_dJds=[c[1] for c in conv],
                   wall_s=wall, wall_per_lyap_time=None, maxrss_GB=rss / 1e6)
vperp_t, vperp = sol["times"], sol["vperp_norm"]
vstar_perp = sol["vstar_perp_norm"]
# convergence error estimate: spread of the last half of the prefix estimates
late = np.array([c[1] for c in conv if c[0] >= res["T"] / 2])
res["main"]["conv_err_est"] = float(late.std(ddof=1)) if len(late) > 2 else float("nan")
save_json(res, name + ".json")

if not args.skip_checks:
    segs2, sol2, wall2, _ = run_nilss(args.M + 5, nseg, args.K, 2, "M+5")
    res["check_M_plus_5"] = dict(M=args.M + 5, dJds=sol2["dJds"], max_vperp=float(sol2["vperp_norm"].max()),
                                 wall_s=wall2)
    save_json(res, name + ".json")
    segs3, sol3, wall3, _ = run_nilss(args.M, nseg // 2, 2 * args.K, 3, "DT/2")
    res["check_DT_half"] = dict(DT=nseg // 2 * dt, K=2 * args.K, dJds=sol3["dJds"],
                                max_vperp=float(sol3["vperp_norm"].max()), wall_s=wall3)
    save_json(res, name + ".json")

# ---- FD of the long-time mean over independent seeds -------------------------
t0 = time.time()
fd_vals, gp_all, gm_all = [], [], []
nfd = int(round(args.fd_T / dt))
for sd in range(args.fd_seeds):
    us, _, _ = get_spun_state(S, 100 + sd, args.tspin)
    _, gp, _ = S.spinup(us, alpha + args.dalpha, nfd, record_every=400)
    _, gm, _ = S.spinup(us, alpha - args.dalpha, nfd, record_every=400)
    fd_vals.append((gp.mean() - gm.mean()) / (2 * args.dalpha))
    gp_all.append(gp.mean()); gm_all.append(gm.mean())
    print(f"  FD seed {sd}: <G>(+)={gp.mean():.5f} <G>(-)={gm.mean():.5f} -> {fd_vals[-1]:.5e}", flush=True)
fd_vals = np.array(fd_vals)
tau_c = autocorr_time(fl[len(fl) // 2:], 1.0)
sig = float(fl[len(fl) // 2:].std())
res["fd"] = dict(dalpha=args.dalpha, T_per_seed=args.fd_T, seeds=args.fd_seeds, values=fd_vals.tolist(),
                 mean=float(fd_vals.mean()), se=float(fd_vals.std(ddof=1) / np.sqrt(len(fd_vals))),
                 Gplus=gp_all, Gminus=gm_all, sigma_G=sig, tau_c=float(tau_c),
                 noise_est_per_seed=float(sig * np.sqrt(tau_c / args.fd_T) / args.dalpha),
                 wall_s=time.time() - t0)
print(f"  FD: {fd_vals.mean():.5e} +- {res['fd']['se']:.3e}  (sigma_G={sig:.3f}, tau_c={tau_c:.2f})", flush=True)
res["agreement"] = dict(nilss_minus_fd=float(sol["dJds"] - fd_vals.mean()),
                        in_sigmas=float((sol["dJds"] - fd_vals.mean()) / res["fd"]["se"]),
                        rel_percent=float(100 * (sol["dJds"] - fd_vals.mean()) / abs(fd_vals.mean())))
save_json(res, name + ".json")

# ---- figure -----------------------------------------------------------------
import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].semilogy(vperp_t, vperp, lw=0.8, label=r"$\|v_\perp(t)\|$ (shadowing)")
ax[0].semilogy(vperp_t, vstar_perp, lw=0.5, alpha=0.6, label=r"$\|v^*_\perp(t)\|$ (per-segment inhomogeneous)")
ax[0].set_xlabel("t"); ax[0].legend(fontsize=8); ax[0].set_title(f"{name}: tangent norms (M={args.M}, ΔT={args.DT})")
ax[1].semilogx([c[0] for c in conv], [c[1] for c in conv], "o-", ms=3, label=f"NILSS M={args.M}")
if "check_M_plus_5" in res:
    ax[1].axhline(res["check_M_plus_5"]["dJds"], color="C1", ls="--", label=f"M={args.M+5}")
if "check_DT_half" in res:
    ax[1].axhline(res["check_DT_half"]["dJds"], color="C2", ls=":", label="ΔT/2")
ax[1].axhspan(fd_vals.mean() - res["fd"]["se"], fd_vals.mean() + res["fd"]["se"], color="k", alpha=0.15,
              label=fr"FD {fd_vals.mean():.3e}±{res['fd']['se']:.1e} ({args.fd_seeds} seeds, T={args.fd_T:g})")
ax[1].set_xlabel("T"); ax[1].set_ylabel(r"$d\langle\Gamma\rangle/d\alpha$"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_{name}.png", dpi=130)
print("done", flush=True)
