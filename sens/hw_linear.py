"""Linear growth rate of the (M)HW system on the solver's own k-grid.

Linearising the archived RHS (``rhs_hw``) about u = 0 for a single Fourier
mode (k_x, k_y) with k_y != 0 (zonal modes have no alpha coupling in MHW and
are purely damped):

    -k^2 d/dt phi = alpha (phi - n) + nu k^(p) k^2 phi        (from d/dt w, w = -k^2 phi)
          d/dt n = alpha (phi - n) - i kappa k_y phi - nu k^(p) n

With e^{-i omega t} and  omega~ = omega + i nu k^p  (p = diffop):

    omega~^2 + i omega~ alpha (1 + k^2)/k^2 - i alpha kappa k_y / k^2 = 0

which is the dispersion relation quoted in the task with the dissipation
shift.  gamma = Im(omega).  The solver applies the damping as an exact
integrating factor after each RK4 step, so this is the growth rate of the
semi-discrete linear system exactly (up to RK4 time-discretisation error,
negligible for gamma*dt ~ 1e-3).
"""
import numpy as np


def hw_growth_rates(res, L, alpha, kappa, nu, diffop):
    dx = L / res
    k = 2 * np.pi * np.fft.fftfreq(res, dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    gamma = np.full(K2.shape, -np.inf)
    mask = (KY != 0)
    k2 = K2[mask]
    ky = KY[mask]
    # omega~^2 + B omega~ + Cc = 0
    B = 1j * alpha * (1 + k2) / k2
    Cc = -1j * alpha * kappa * ky / k2
    disc = np.sqrt(B ** 2 - 4 * Cc)
    w1 = (-B + disc) / 2
    w2 = (-B - disc) / 2
    damp = nu * k2 ** (diffop / 2)
    g = np.maximum(w1.imag, w2.imag) - damp
    gamma[mask] = g
    return gamma, KX, KY


def gamma_max(res, L, alpha, kappa, nu, diffop):
    gamma, KX, KY = hw_growth_rates(res, L, alpha, kappa, nu, diffop)
    i = np.unravel_index(np.argmax(gamma), gamma.shape)
    return float(gamma[i]), float(KX[i]), float(KY[i])


def n_unstable_modes(res, L, alpha, kappa, nu, diffop):
    gamma, _, _ = hw_growth_rates(res, L, alpha, kappa, nu, diffop)
    return int(np.sum(gamma > 0))
