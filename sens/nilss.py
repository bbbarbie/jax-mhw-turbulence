"""Non-Intrusive Least Squares Shadowing (NILSS).

Reference: A. Ni & Q. Wang, "Sensitivity analysis on chaotic dynamical systems
by Non-Intrusive Least Squares Shadowing (NILSS)", J. Comput. Phys. 347 (2017)
56-77.  The authors' prototype (github.com/niangxiu/nilss, ``nilss.py``) was
used to cross-check every formula below; the paper itself was not reachable
from this machine, so equation numbers quoted in comments are the ones I am
confident of from the arXiv version (1611.00880) and are marked "(paper Sec.)"
where I only know the section.

Everything is discrete-time: the primal is ``u_{n+1} = step(u_n, s)``, every
tangent is ``jax.jvp(step, (u, s), (w, ds))`` of that same map, and all time
integrals are left-point Riemann sums ``sum(.) * dt``.  The neutral direction
is the discrete one, ``f(u_n) = (u_{n+1} - u_n)/dt``, which is free inside the
step body.

Notation (paper Sec. 3 / Alg. in Sec. 4):
  * K segments of N = ``nseg_steps`` steps, DT = N*dt, T = K*DT.
  * W_i(t)  : M homogeneous tangents on segment i   (dw/dt = df/du w)
  * v*_i(t) : inhomogeneous tangent on segment i     (dv*/dt = df/du v* + df/ds)
  * P(u) = I - f f^T/(f^T f);  W_perp = P W,  v*_perp = P v*
  * shadowing direction on segment i:  v_i = v*_i + W_i a_i
  * segment integrals  C_i = int W_perp^T W_perp dt,  d_i = int W_perp^T v*_perp dt
  * boundary:  W_perp(t_{i+1}) = Q_i R_i  (QR),   b_i = Q_i^T v*_perp(t_{i+1})
               restart W_{i+1}(t_{i+1}) = Q_i,   v*_{i+1}(t_{i+1}) = v*_perp - Q_i b_i
  * continuity of v_perp across boundaries  =>  a_{i+1} = R_i a_i + b_i
  * NILSS problem:  min_a  sum_i ( a_i^T C_i a_i + 2 d_i^T a_i )  s.t. the above
  * sensitivity with time dilation  (v = v_perp + xi f, eta = -dxi/dt):
        d<J>/ds = (1/T) sum_i [ int_i (dJ/du . v_i + dJ/ds) dt
                               + xi_i ( <J> - J(u(t_{i+1})) ) ]
    where xi_i = f . v_i / f . f at the END of segment i (it is 0 at the start
    of every segment because every restart is orthogonal to f).  This is the
    integrated-by-parts form of  (1/T) int (dJ/du v_perp + dJ/ds) dt
    + (1/T) int eta (J - <J>) dt  and is exactly what the authors' code does
    (terms ``t1`` and ``t2`` in their ``nilss.py``).
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _flat(x):
    return x.reshape(x.shape[0], -1)


def nilss_segments(step, u0, s, qoi, M, nseg_steps, K, dt, key=None,
                   rec_every=1, K_pre=2, verbose=False, log_prefix="",
                   rhs=None):
    """Integrate the primal, M homogeneous and 1 inhomogeneous tangent over
    ``K_pre`` discarded warm-up segments then ``K`` recorded segments.

    ``rhs(u, s)``, if given, is used as the neutral direction f instead of the
    discrete ``(step(u) - u)/dt``.  For an RK4 map the RHS is neutral to
    O(dt^5) per step whereas the discrete difference is neutral only to
    O(dt^2), which matters for the projections (see results.md, Task 3).

    Returns a dict of per-segment arrays (see ``nilss_solve``) plus the final
    state.  Nothing here depends on the least-squares solution, so the
    solve can be redone for any prefix of segments (convergence with T).
    """
    key = jax.random.PRNGKey(0) if key is None else key
    shape = u0.shape
    ds_vec = jnp.concatenate([jnp.zeros(M), jnp.ones(1)])       # d s / d s
    assert nseg_steps % rec_every == 0
    nchunk = nseg_steps // rec_every

    def project(T, f):
        """Return (T_perp, f-components) for tangents T (M+1, ...)."""
        ff = jnp.vdot(f, f)
        Tf = _flat(T) @ f.reshape(-1) / ff                        # (M+1,)
        Tperp = T - Tf.reshape((-1,) + (1,) * len(shape)) * f[None]
        return Tperp, Tf

    def body(carry, _):
        u, T, C, d, G, gs, Js = carry
        # --- one step of primal + all tangents through the SAME discrete map
        u_next, T_next = jax.vmap(
            lambda t, dsv: jax.jvp(step, (u, s), (t, dsv)),
            out_axes=(None, 0))(T, ds_vec)
        f = (u_next - u) / dt if rhs is None else rhs(u, s)       # neutral dir. at u_n
        Tperp, Tf = project(T, f)
        Wp, vp = _flat(Tperp[:M]), Tperp[M].reshape(-1)
        gram = Wp @ Wp.T                                          # W_perp^T W_perp
        cross = Wp @ vp                                           # W_perp^T v*_perp
        vv = vp @ vp
        dJ = jax.vmap(lambda t: jax.jvp(qoi, (u,), (t,))[1])(T)   # dJ/du . (W, v*)
        J, dJf = jax.jvp(qoi, (u,), (f,))                         # J, dJ/du . f
        carry = (u_next, T_next, C + gram * dt, d + cross * dt,
                 G + dJ[:M] * dt, gs + dJ[M] * dt, Js + J * dt)
        return carry, ((gram, cross, vv), (Tf, J, dJf))

    def chunk(carry, _):
        carry, (rec, per) = jax.lax.scan(body, carry, None, length=rec_every)
        return carry, (jax.tree_util.tree_map(lambda x: x[0], rec), per)

    @jax.jit
    def segment(u, T):
        carry0 = (u, T, jnp.zeros((M, M)), jnp.zeros(M), jnp.zeros(M), 0.0, 0.0)
        (u, T, C, d, G, gs, Js), (rec, per) = jax.lax.scan(
            chunk, carry0, None, length=nchunk)
        per = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), per)
        # --- segment end: project, QR, restart  (paper Sec. 4, steps 3-5)
        f = (step(u, s) - u) / dt if rhs is None else rhs(u, s)
        Tperp, Tf = project(T, f)
        Wp, vp = _flat(Tperp[:M]), Tperp[M].reshape(-1)
        Q, R = jnp.linalg.qr(Wp.T)                                # W_perp = Q R
        sgn = jnp.sign(jnp.diag(R))
        Q, R = Q * sgn[None, :], R * sgn[:, None]
        b = Q.T @ vp                                              # b_i = Q^T v*_perp
        p = vp - Q @ b                                            # restart for v*
        Tnew = jnp.concatenate([Q.T.reshape((M,) + shape),
                                p.reshape((1,) + shape)])
        seg = dict(R=R, b=b, C=C, d=d, G=G, gstar=gs, Jsum=Js, J_end=qoi(u),
                   fv=Tf[M], fW=Tf[:M],                           # xi pieces
                   p_norm=jnp.linalg.norm(p), rec=rec, per=per)
        return u, Tnew, seg

    # --- initial tangents: W random orthonormal, v* = 0 (authors' choice)
    W0 = jax.random.normal(key, (M,) + shape, dtype=u0.dtype)
    Q0, _ = jnp.linalg.qr(_flat(W0).T)
    T = jnp.concatenate([Q0.T.reshape((M,) + shape), jnp.zeros((1,) + shape)])
    u = u0

    t0 = time.time()
    for i in range(K_pre):                                        # warm-up
        u, T, _ = segment(u, T)
    if verbose:
        print(f"{log_prefix}warm-up {K_pre} segments: {time.time()-t0:.1f}s",
              flush=True)
    segs = []
    t0 = time.time()
    for i in range(K):
        u, T, seg = segment(u, T)
        segs.append(jax.tree_util.tree_map(np.asarray, seg))
        if verbose and (i + 1) % max(1, K // 10) == 0:
            print(f"{log_prefix}segment {i+1}/{K}  {time.time()-t0:.1f}s  "
                  f"|p|={float(seg['p_norm']):.3e}", flush=True)
    out = {k: np.stack([sg[k] for sg in segs]) for k in segs[0]
           if k not in ("rec", "per")}
    out["rec"] = tuple(np.stack([sg["rec"][j] for sg in segs]) for j in range(3))
    out["per"] = tuple(np.stack([sg["per"][j] for sg in segs]) for j in range(3))
    out["dt"] = dt
    out["nseg_steps"] = nseg_steps
    out["M"] = M
    out["rec_every"] = rec_every
    out["wall_s"] = time.time() - t0
    return out, u


def nilss_solve(segs, K_use=None, dJds_mean=0.0):
    """Solve the NILSS least-squares problem on the first ``K_use`` segments
    and assemble d<J>/ds (formula in the module docstring).

    Returns a dict: dJds, Javg, a (K,M), xi (K,), vperp_norm (K*nchunk,),
    times, plus the two contributions separately.
    """
    K = segs["R"].shape[0] if K_use is None else K_use
    M = segs["M"]
    dt = segs["dt"]
    N = segs["nseg_steps"]
    T = K * N * dt
    C = [segs["C"][i] for i in range(K)]
    d = np.concatenate([segs["d"][i] for i in range(K)])
    # constraint  a_{i+1} - R_i a_i = b_i ,  i = 0..K-2
    if K > 1:
        B = sp.lil_matrix(((K - 1) * M, K * M))
        for i in range(K - 1):
            B[i * M:(i + 1) * M, i * M:(i + 1) * M] = -segs["R"][i]
            B[i * M:(i + 1) * M, (i + 1) * M:(i + 2) * M] = np.eye(M)
        B = B.tocsr()
        b = np.concatenate([segs["b"][i] for i in range(K - 1)])
        Cb = sp.block_diag(C, format="csr")
        # KKT:  [C  B^T][a  ]   [-d]
        #       [B  0  ][lam] = [ b]
        KKT = sp.bmat([[Cb, B.T], [B, None]], format="csc")
        rhs = np.concatenate([-d, b])
        sol = spla.spsolve(KKT, rhs)
        a = sol[:K * M].reshape(K, M)
    else:
        a = np.linalg.solve(C[0], -d).reshape(1, M)

    Javg = segs["Jsum"][:K].sum() / T
    xi = segs["fv"][:K] + np.einsum("km,km->k", segs["fW"][:K], a)
    # --- form A (integration by parts, as in the authors' code):
    #     (1/T) sum_i [ int_i dJ/du . v_i dt + xi_i (<J> - J(u(t_{i+1}))) ]
    term_int = (segs["gstar"][:K] + np.einsum("km,km->k", segs["G"][:K], a)).sum() / T
    term_dil = (xi * (Javg - segs["J_end"][:K])).sum() / T
    dJds_ibp = term_int + term_dil + dJds_mean
    # --- form B (discretely consistent, no integration by parts):
    #     (1/T) [ sum_n dJ/du . v_perp_n dt  +  sum_n (xi_{n+1} - xi_n) (<J> - J_n) ]
    #     with xi_n = f_n . v_n / f_n . f_n inside each segment (0 at its start,
    #     xi_i at its end).  Identical to form A up to time-quadrature error.
    Tf, Jn, dJf = segs["per"]
    Tf, Jn, dJf = Tf[:K], Jn[:K], dJf[:K]                     # (K, N, M+1), (K, N), (K, N)
    xin = Tf[:, :, M] + np.einsum("knm,km->kn", Tf[:, :, :M], a)   # (K, N)
    xi_next = np.concatenate([xin[:, 1:], xi[:, None]], axis=1)
    term_perp = term_int - (xin * dJf).sum() / T * dt
    term_dilB = ((xi_next - xin) * (Javg - Jn)).sum() / T
    dJds = term_perp + term_dilB + dJds_mean

    # reconstruct |v_perp(t)|^2 = |v*_perp|^2 + 2 a.cross + a^T gram a
    gram, cross, vv = segs["rec"]
    gram, cross, vv = gram[:K], cross[:K], vv[:K]
    vperp2 = (vv + 2 * np.einsum("kcm,km->kc", cross, a)
              + np.einsum("kcmn,km,kn->kc", gram, a, a))
    nchunk = vv.shape[1]
    times = (np.arange(K * nchunk) * segs["rec_every"] + 0) * dt
    return dict(dJds=dJds, dJds_ibp=dJds_ibp, Javg=Javg, a=a, xi=xi,
                term_int=term_int, term_dil=term_dil, term_perp=term_perp,
                term_dilB=term_dilB, T=T, K=K,
                vperp_norm=np.sqrt(np.maximum(vperp2.reshape(-1), 0.0)),
                vstar_perp_norm=np.sqrt(vv.reshape(-1)), times=times)
