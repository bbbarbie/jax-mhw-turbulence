"""Task 3b: resolve the ~1% NILSS-vs-FD gap on Lorenz 63.

FD with three step sizes drho (bias check) and NILSS with three dt (map
discretisation check), all at rho = 28, J = z.  Writes
results/sens/task3b_lorenz_checks.json.
"""
import json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sens.lorenz import make_lorenz_step, qoi_z, rollout_mean, lorenz_rhs
from sens.nilss import nilss_segments, nilss_solve
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "sens")
RHO = 28.0
out = {}
# --- FD at several drho, dt = 0.01, T = 20000 x 20 trajectories
dt = 0.01; step = make_lorenz_step(dt)
def ic(seed):
    u0 = jnp.array([1.0, 1.0, 25.0]) + 0.1 * jax.random.normal(jax.random.PRNGKey(100 + seed), (3,))
    return rollout_mean(step, u0, RHO, int(200 / dt), qoi_z)[0]
out["fd"] = {}
for drho in [0.25, 0.5, 1.0, 2.0]:
    vals = []
    for k in range(20):
        u = ic(1000 + k)
        zp = rollout_mean(step, u, RHO + drho, int(20000 / dt), qoi_z)[1]
        zm = rollout_mean(step, u, RHO - drho, int(20000 / dt), qoi_z)[1]
        vals.append((float(zp) - float(zm)) / (2 * drho))
    vals = np.array(vals)
    out["fd"][str(drho)] = dict(mean=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(20)))
    print(f"FD drho={drho}: {vals.mean():.4f} +- {vals.std(ddof=1)/np.sqrt(20):.4f}", flush=True)
# --- NILSS at several dt (RHS neutral direction), DT_seg = 2, T = 4000
out["nilss_dt"] = {}
for dt in [0.02, 0.01, 0.005, 0.0025]:
    step = make_lorenz_step(dt)
    u0 = jnp.array([1.0, 1.0, 25.0]); u0 = rollout_mean(step, u0, RHO, int(200 / dt), qoi_z)[0]
    nseg = int(round(2.0 / dt)); K = 2000
    segs, _ = nilss_segments(step, u0, RHO, qoi_z, M=1, nseg_steps=nseg, K=K, dt=dt,
                             key=jax.random.PRNGKey(1), rec_every=nseg // 20, K_pre=10, rhs=lorenz_rhs)
    sol = nilss_solve(segs)
    # FD on the same discrete map (drho=0.5, 10 x T=20000) for the dt=0.02 and 0.0025 maps
    out["nilss_dt"][str(dt)] = dict(dJds=sol["dJds"], dJds_ibp=sol["dJds_ibp"], Javg=sol["Javg"])
    print(f"NILSS dt={dt}: {sol['dJds']:.4f} (ibp {sol['dJds_ibp']:.4f})  <z>={sol['Javg']:.4f}", flush=True)
# --- FD on the dt=0.0025 map to separate map-discretisation from method bias
step = make_lorenz_step(0.0025); vals = []
for k in range(10):
    u0 = jnp.array([1.0, 1.0, 25.0]) + 0.1 * jax.random.normal(jax.random.PRNGKey(100 + k), (3,))
    u = rollout_mean(step, u0, RHO, int(200 / 0.0025), qoi_z)[0]
    zp = rollout_mean(step, u, RHO + 0.5, int(20000 / 0.0025), qoi_z)[1]
    zm = rollout_mean(step, u, RHO - 0.5, int(20000 / 0.0025), qoi_z)[1]
    vals.append((float(zp) - float(zm)))
vals = np.array(vals)
out["fd_dt0.0025"] = dict(mean=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(10)), drho=0.5)
print(f"FD on dt=0.0025 map (drho=0.5, 10 x T=20000): {vals.mean():.4f} +- {vals.std(ddof=1)/np.sqrt(10):.4f}", flush=True)
json.dump(out, open(os.path.join(OUT, "task3b_lorenz_checks.json"), "w"), indent=1)
print("done")
