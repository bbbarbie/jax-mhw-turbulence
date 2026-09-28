"""Thin wrapper around the archived MHW solver exposing ``step(u, s)``.

State layout used by the sensitivity code
-----------------------------------------
``u`` is a real float64 array of shape ``(2, N, N)``: ``u[0]`` is vorticity
``w = nabla^2 phi`` and ``u[1]`` is density ``n`` on the periodic grid.  The
archived solver works on the spectral pair ``(w_hat, n_hat)``; ``step`` below
is exactly ``ifft2 o step_rk4 o fft2`` so the primal, every tangent and the
Lyapunov/NILSS bookkeeping all see one and the same discrete map.  Working in
real space gives a plain Euclidean inner product (no Hermitian redundancy) for
the QR factorisations.

The differentiated parameter is ``s = alpha`` (adiabaticity).  ``kappa`` is
held fixed, matching ``mhw_jax_stage2_ad.run_ad_window``.

Flux convention (same as the archive):  Gamma = mean( -kappa * n * d(phi)/dy )
which, with kappa = 1, is < -n d_y phi > = < n v_x >  with v_x = -d_y phi.
Positive Gamma is outward (down the background gradient).
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mhw_jax_stage2_ad import MHWParams, get_jax_params, step_rk4, rhs_hw  # noqa: E402


class MHWSystem:
    """Discrete MHW map ``u_{n+1} = step(u_n, alpha)`` and its diagnostics."""

    def __init__(self, res=64, alpha=0.2, kappa=1.0, nu=1e-3, diffop=6,
                 dt=0.0025, L=64.0, arakawa=True, modified=True):
        self.res = res
        self.alpha = alpha
        self.kappa = kappa
        self.nu = nu
        self.diffop = diffop
        self.dt = dt
        self.L = L
        self.arakawa = arakawa
        self.modified = modified
        self.p = MHWParams(nx=res, ny=res, Lx=L, Ly=L, alpha=alpha,
                           kappa=kappa, diffw=nu, diffn=nu, diffop=diffop,
                           dt=dt, arakawa=arakawa, modified=modified)
        self.gp = get_jax_params(self.p)
        self.ndof = 2 * res * res

    # -- the discrete map ----------------------------------------------------
    def step(self, u, s):
        """One RK4 + spectral-damping step of the archived solver.

        ``u``: real (2, N, N).  ``s``: alpha (scalar, may be a tracer).
        """
        w_hat = jnp.fft.fft2(u[0])
        n_hat = jnp.fft.fft2(u[1])
        w2, n2 = step_rk4((w_hat, n_hat), (self.gp, (s, self.kappa)))
        return jnp.stack([jnp.real(jnp.fft.ifft2(w2)),
                          jnp.real(jnp.fft.ifft2(n2))])

    def f(self, u, s):
        """Discrete time-derivative direction f(u) = (step(u) - u)/dt.

        Used as the neutral (trajectory) direction in NILSS.  It is the exact
        first-order generator of the discrete map, so it is consistent with the
        tangents produced by ``jax.jvp(step)``.
        """
        return (self.step(u, s) - u) / self.dt

    def rhs(self, u, s):
        """Semi-discrete RHS of the model the solver integrates:
        ``rhs_hw`` (advection + coupling + kappa drive) minus the hyperdiffusion
        ``nu k^p`` that the stepper applies as an integrating factor.  Used as
        the neutral direction f in NILSS (alternative to ``f``)."""
        w_hat = jnp.fft.fft2(u[0])
        n_hat = jnp.fft.fft2(u[1])
        dw, dn = rhs_hw((w_hat, n_hat), self.gp, (s, self.kappa))
        rate = -jnp.log(self.gp["damp_w"]) / self.dt          # nu k^p
        dw = dw - rate * w_hat
        dn = dn - (-jnp.log(self.gp["damp_n"]) / self.dt) * n_hat
        return jnp.stack([jnp.real(jnp.fft.ifft2(dw)),
                          jnp.real(jnp.fft.ifft2(dn))])

    # -- diagnostics ---------------------------------------------------------
    def phi_hat(self, w):
        return -jnp.fft.fft2(w) * self.gp["inv_ksq"]

    def flux(self, u):
        """Gamma = mean(-kappa n d_y phi)   (archive convention, kappa fixed)."""
        ph = self.phi_hat(u[0])
        phi_y = jnp.real(jnp.fft.ifft2(1j * self.gp["KY"] * ph))
        return jnp.mean(-self.kappa * u[1] * phi_y)

    def energy(self, u):
        """(1/2) mean(n^2 + |grad phi|^2)."""
        ph = self.phi_hat(u[0])
        phi_x = jnp.real(jnp.fft.ifft2(1j * self.gp["KX"] * ph))
        phi_y = jnp.real(jnp.fft.ifft2(1j * self.gp["KY"] * ph))
        return 0.5 * jnp.mean(u[1] ** 2 + phi_x ** 2 + phi_y ** 2)

    # -- initial conditions / spin-up ---------------------------------------
    def init_state(self, seed):
        """Archive initialisation: w0 = n0 = 1e-4 * (U[0,1) - 0.5), np seed."""
        rng = np.random.RandomState(seed)
        w0 = 1e-4 * (rng.rand(self.res, self.res) - 0.5)
        return jnp.asarray(np.stack([w0, w0.copy()]))

    def spinup(self, u, s, nsteps, record_every=0):
        """Forward-only integration (never differentiated).

        If ``record_every > 0`` also returns (flux, energy) sampled every
        ``record_every`` steps.
        """
        step = self.step
        if record_every <= 0:
            @jax.jit
            def run(u, s):
                return jax.lax.scan(lambda c, _: (step(c, s), None), u, None,
                                    length=nsteps)[0]
            return run(u, s)

        nchunks = nsteps // record_every
        flux, energy = self.flux, self.energy

        @jax.jit
        def run(u, s):
            def chunk(c, _):
                c = jax.lax.scan(lambda cc, _: (step(cc, s), None), c, None,
                                 length=record_every)[0]
                return c, (flux(c), energy(c))
            return jax.lax.scan(chunk, u, None, length=nchunks)
        u, (fl, en) = run(u, s)
        return u, np.asarray(fl), np.asarray(en)

    # -- state cache ---------------------------------------------------------
    def tag(self):
        extra = ("" if self.arakawa else "_centered") + ("" if self.modified else "_unmod")
        extra += "" if self.L == 64.0 else f"_L{self.L:g}"
        return (f"res{self.res}_a{self.alpha:g}_k{self.kappa:g}_nu{self.nu:g}"
                f"_op{self.diffop}_dt{self.dt:g}{extra}")


def state_path(sys_, seed, tspin, root=None):
    root = root or os.path.join(_REPO, "results", "sens", "states")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{sys_.tag()}_seed{seed}_T{tspin:g}.npz")


def get_spun_state(sys_, seed, tspin, record_every=400, force=False):
    """Spin up from the archive IC for ``tspin`` time units, cached on disk.

    Returns ``(u, flux_series, energy_series)``; the series are sampled every
    ``record_every`` steps (1.0 time unit at dt = 0.0025).
    """
    path = state_path(sys_, seed, tspin)
    if os.path.exists(path) and not force:
        d = np.load(path)
        return jnp.asarray(d["u"]), d["flux"], d["energy"]
    nsteps = int(round(tspin / sys_.dt))
    u0 = sys_.init_state(seed)
    u, fl, en = sys_.spinup(u0, sys_.alpha, nsteps, record_every=record_every)
    if not bool(jnp.all(jnp.isfinite(u))):
        raise FloatingPointError(f"spin-up went non-finite: {sys_.tag()} seed {seed}")
    np.savez_compressed(path, u=np.asarray(u), flux=fl, energy=en,
                        alpha=sys_.alpha, kappa=sys_.kappa, nu=sys_.nu,
                        diffop=sys_.diffop, dt=sys_.dt, res=sys_.res,
                        tspin=tspin, seed=seed)
    return u, fl, en
