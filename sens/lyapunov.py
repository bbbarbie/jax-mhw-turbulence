"""Benettin / QR Lyapunov-spectrum estimator on the discrete map.

Every tangent is ``jax.jvp(step, (u,), (v,))``.  ``M`` tangents are carried as
an array ``V`` of shape ``(M, *u.shape)`` and propagated with ``vmap``; every
``renorm_every`` steps the tangent matrix is QR-factorised (Gram-Schmidt) and
``log|diag R|`` is accumulated.  For ``M = 1`` this is Benettin's largest
exponent; for ``M > 1`` it is the standard discrete QR method
(Shimada-Nagashima / Benettin et al. 1980).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _flat(V):
    return V.reshape(V.shape[0], -1)


def qr_renorm(V):
    """QR of the tangent matrix; returns (Q reshaped like V, log|diag R|)."""
    M = V.shape[0]
    A = _flat(V).T                       # (ndof, M)
    Q, R = jnp.linalg.qr(A)              # reduced
    d = jnp.diag(R)
    sgn = jnp.sign(d)
    Q = Q * sgn[None, :]                 # make diag(R) positive
    return Q.T.reshape(V.shape), jnp.log(jnp.abs(d))


def lyapunov_spectrum(sys_, u0, s, M, renorm_every, n_renorm, key=None,
                      V0=None, return_Q=False):
    """Run ``n_renorm`` renormalisation intervals of ``renorm_every`` steps.

    Returns
    -------
    logR : (n_renorm, M) array of log-stretching per interval
    u    : final primal state
    Q    : final orthonormal tangents (if ``return_Q``)
    The running estimate after k intervals is
        lambda_j(k) = sum_{i<=k} logR[i, j] / (k * renorm_every * dt).
    """
    if V0 is None:
        key = jax.random.PRNGKey(0) if key is None else key
        V0 = jax.random.normal(key, (M,) + u0.shape, dtype=u0.dtype)
    V0, _ = qr_renorm(V0)
    step = sys_.step

    def one_step(c, _):
        u, V = c
        fu, Vn = jax.vmap(lambda v: jax.jvp(lambda uu: step(uu, s), (u,), (v,)),
                          out_axes=(None, 0))(V)
        return (fu, Vn), None

    @jax.jit
    def interval(c, _):
        (u, V), _ = jax.lax.scan(one_step, c, None, length=renorm_every)
        Q, lr = qr_renorm(V)
        return (u, Q), lr

    @jax.jit
    def run(u, V):
        return jax.lax.scan(interval, (u, V), None, length=n_renorm)

    (u, Q), logR = run(u0, V0)
    if return_Q:
        return np.asarray(logR), u, Q
    return np.asarray(logR), u


def exponents_from_logR(logR, dt, renorm_every, discard=0):
    """Mean exponents and standard error (over intervals) from ``logR``."""
    x = logR[discard:] / (renorm_every * dt)
    lam = x.mean(axis=0)
    # standard error assuming intervals are ~independent (conservative when
    # renorm_every*dt << correlation time; see results.md for the batched SE)
    se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return lam, se


def batched_se(logR, dt, renorm_every, nbatch=10, discard=0):
    """Standard error from ``nbatch`` block means (accounts for correlation)."""
    x = logR[discard:] / (renorm_every * dt)
    n = (x.shape[0] // nbatch) * nbatch
    b = x[:n].reshape(nbatch, -1, x.shape[1]).mean(axis=1)
    return b.std(axis=0, ddof=1) / np.sqrt(nbatch)


def kaplan_yorke(lam):
    lam = np.sort(np.asarray(lam))[::-1]
    c = np.cumsum(lam)
    j = np.where(c >= 0)[0]
    if len(j) == 0:
        return 0.0
    j = j[-1]
    if j + 1 >= len(lam):
        return float(len(lam))  # not enough exponents to close the sum
    return float(j + 1 + c[j] / abs(lam[j + 1]))
