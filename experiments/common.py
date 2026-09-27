"""Shared regime definitions and helpers for the sensitivity experiments."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "results", "sens")
os.makedirs(OUT, exist_ok=True)

# Archive parameters (results/stage1, stage2 logs): kappa=1, nu_w=nu_n=1e-3,
# k^6 hyperdiffusion, dt=0.0025, box 64x64, RK4, Arakawa, modified HW.
# alpha = 0.2 and alpha = 0.8 are the two regimes of the archived AD-vs-FD
# comparison.  Resolution is chosen per task (see results.md).
BASE = dict(kappa=1.0, nu=1e-3, diffop=6, dt=0.0025, L=64.0)
REGIMES = {"a0.2": dict(alpha=0.2), "a0.8": dict(alpha=0.8)}


def regime_kwargs(name, res, **over):
    kw = dict(BASE)
    kw.update(REGIMES[name])
    kw["res"] = res
    kw.update(over)
    return kw


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)


def load_json(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def autocorr_time(x, dt_sample, maxlag=None):
    """Integrated autocorrelation time tau_c = dt * (1 + 2 sum_k rho_k) with
    the sum truncated at the first zero crossing of rho."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    maxlag = maxlag or n // 4
    var = np.dot(x, x) / n
    if var == 0:
        return dt_sample
    rho = np.array([np.dot(x[:n - k], x[k:]) / (n - k) / var for k in range(maxlag)])
    zc = np.where(rho <= 0)[0]
    kmax = zc[0] if len(zc) else maxlag
    return dt_sample * (1 + 2 * rho[1:kmax].sum())


def linfit(x, y):
    """Least-squares slope with standard error."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    if n > 2:
        s2 = float(res[0]) / (n - 2) if len(res) else 0.0
        se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    else:
        se = np.nan
    return coef[0], coef[1], se
