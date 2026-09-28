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

import sys

import numpy as np
import jax
import jax.numpy as jnp

from common import regime_kwargs, save_json, load_json, OUT, autocorr_time  # noqa
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
ap.add_argument("--L", type=float, default=64.0)
ap.add_argument("--part", default="all", choices=["all", "main", "m5", "dt2", "fd", "plot"],
                help="run one part in its own process (parallelism); 'plot' merges the parts")
ap.add_argument("--tag", default="")
args = ap.parse_args()

over = {} if args.nu is None else {"nu": args.nu}
over["L"] = args.L
kw = regime_kwargs(args.regime, args.res, **over)
S = MHWSystem(**kw)
alpha, dt = S.alpha, S.dt
name = f"task4_{args.regime}_res{args.res}{args.tag}"
nseg = int(round(args.DT / dt))
print(f"== {name}: {kw} M={args.M} DT={args.DT} ({nseg} steps) K={args.K} T={args.K*args.DT}", flush=True)
res = dict(params=kw, M=args.M, DT=args.DT, K=args.K, T=args.K * args.DT, neutral=args.neutral,
           seed=args.seed, tspin=args.tspin)
PART = args.part


def part_file(part):
    return f"{name}_{part}.json"


def load_part(part):
    try:
        return load_json(part_file(part))
    except FileNotFoundError:
        return None


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


def summarize(sol, segs, wall, rss, M, nseg_, K_):
    Ks = [int(k) for k in np.unique(np.clip(np.logspace(np.log10(5), np.log10(max(K_, 5)), 20).astype(int), 1, K_))]
    conv = [(k * nseg_ * dt, nilss_solve(segs, K_use=k)["dJds"]) for k in Ks]
    late = np.array([c[1] for c in conv if c[0] >= K_ * nseg_ * dt / 2])
    return dict(M=M, DT=nseg_ * dt, K=K_, T=K_ * nseg_ * dt,
                dJds=sol["dJds"], dJds_ibp=sol["dJds_ibp"], Javg=sol["Javg"],
                term_perp=sol["term_perp"], term_dil=sol["term_dilB"],
                max_vperp=float(sol["vperp_norm"].max()), mean_vperp=float(sol["vperp_norm"].mean()),
                max_abs_a=float(np.abs(sol["a"]).max()), max_abs_xi=float(np.abs(sol["xi"]).max()),
                conv_T=[c[0] for c in conv], conv_dJds=[c[1] for c in conv],
                conv_err_est=float(late.std(ddof=1)) if len(late) > 2 else float("nan"),
                wall_s=wall, maxrss_GB=rss / 1e6,
                vperp_t=sol["times"].tolist(), vperp=sol["vperp_norm"].tolist(),
                vstar_perp=sol["vstar_perp_norm"].tolist())


if PART in ("all", "main"):
    segs, sol, wall, rss = run_nilss(args.M, nseg, args.K, 1, "main")
    res["main"] = summarize(sol, segs, wall, rss, args.M, nseg, args.K)
    save_json(res["main"], part_file("main"))
if PART in ("all", "m5"):
    segs2, sol2, wall2, rss2 = run_nilss(args.M + 5, nseg, args.K, 2, "M+5")
    res["check_M_plus_5"] = summarize(sol2, segs2, wall2, rss2, args.M + 5, nseg, args.K)
    save_json(res["check_M_plus_5"], part_file("m5"))
if PART in ("all", "dt2"):
    segs3, sol3, wall3, rss3 = run_nilss(args.M, nseg // 2, 2 * args.K, 3, "DT/2")
    res["check_DT_half"] = summarize(sol3, segs3, wall3, rss3, args.M, nseg // 2, 2 * args.K)
    save_json(res["check_DT_half"], part_file("dt2"))
if PART in ("main", "m5", "dt2"):
    sys.exit(0)

# ---- FD of the long-time mean over independent seeds -------------------------
if PART == "plot":
    res["main"] = load_part("main"); res["check_M_plus_5"] = load_part("m5")
    res["check_DT_half"] = load_part("dt2"); res["fd"] = load_part("fd")
    fd_vals = np.array(res["fd"]["values"])
    sol = None
t0 = time.time()
if PART != "plot":
    fd_vals, gp_all, gm_all = [], [], []
nfd = int(round(args.fd_T / dt))
for sd in (range(args.fd_seeds) if PART != "plot" else []):
    us, _, _ = get_spun_state(S, 100 + sd, args.tspin)
    _, gp, _ = S.spinup(us, alpha + args.dalpha, nfd, record_every=400)
    _, gm, _ = S.spinup(us, alpha - args.dalpha, nfd, record_every=400)
    fd_vals.append((gp.mean() - gm.mean()) / (2 * args.dalpha))
    gp_all.append(gp.mean()); gm_all.append(gm.mean())
    print(f"  FD seed {sd}: <G>(+)={gp.mean():.5f} <G>(-)={gm.mean():.5f} -> {fd_vals[-1]:.5e}", flush=True)
if PART != "plot":
  fd_vals = np.array(fd_vals)
  tau_c = autocorr_time(fl[len(fl) // 2:], 1.0)
  sig = float(fl[len(fl) // 2:].std())
  res["fd"] = dict(dalpha=args.dalpha, T_per_seed=args.fd_T, seeds=args.fd_seeds, values=fd_vals.tolist(),
                 mean=float(fd_vals.mean()), se=float(fd_vals.std(ddof=1) / np.sqrt(len(fd_vals))),
                 Gplus=gp_all, Gminus=gm_all, sigma_G=sig, tau_c=float(tau_c),
                 noise_est_per_seed=float(sig * np.sqrt(tau_c / args.fd_T) / args.dalpha),
                 wall_s=time.time() - t0)
  print(f"  FD: {fd_vals.mean():.5e} +- {res['fd']['se']:.3e}  (sigma_G={sig:.3f}, tau_c={tau_c:.2f})", flush=True)
  save_json(res["fd"], part_file("fd"))
  if PART == "fd":
      sys.exit(0)
main = res["main"]
res["agreement"] = dict(nilss_minus_fd=float(main["dJds"] - fd_vals.mean()),
                        in_sigmas=float((main["dJds"] - fd_vals.mean()) / res["fd"]["se"]),
                        rel_percent=float(100 * (main["dJds"] - fd_vals.mean()) / abs(fd_vals.mean())))
save_json(res, name + ".json")
vperp_t, vperp, vstar_perp = np.array(main["vperp_t"]), np.array(main["vperp"]), np.array(main["vstar_perp"])
conv = list(zip(main["conv_T"], main["conv_dJds"]))

# ---- figure -----------------------------------------------------------------
import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].semilogy(vperp_t, vperp, lw=0.8, label=r"$\|v_\perp(t)\|$ (shadowing)")
ax[0].semilogy(vperp_t, vstar_perp, lw=0.5, alpha=0.6, label=r"$\|v^*_\perp(t)\|$ (per-segment inhomogeneous)")
ax[0].set_xlabel("t"); ax[0].legend(fontsize=8); ax[0].set_title(f"{name}: tangent norms (M={args.M}, ΔT={args.DT})")
ax[1].semilogx([c[0] for c in conv], [c[1] for c in conv], "o-", ms=3, label=f"NILSS M={args.M}")
if res.get("check_M_plus_5"):
    ax[1].axhline(res["check_M_plus_5"]["dJds"], color="C1", ls="--", label=f"M={args.M+5}")
if res.get("check_DT_half"):
    ax[1].axhline(res["check_DT_half"]["dJds"], color="C2", ls=":", label="ΔT/2")
ax[1].axhspan(fd_vals.mean() - res["fd"]["se"], fd_vals.mean() + res["fd"]["se"], color="k", alpha=0.15,
              label=fr"FD {fd_vals.mean():.3e}±{res['fd']['se']:.1e} ({args.fd_seeds} seeds, T={args.fd_T:g})")
ax[1].set_xlabel("T"); ax[1].set_ylabel(r"$d\langle\Gamma\rangle/d\alpha$"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_{name}.png", dpi=130)
print("done", flush=True)
