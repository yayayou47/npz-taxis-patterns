"""
Finite-volume IMEX solver for the truncated NPZ cross-diffusion system

    d_t c + u.grad c = F(Pi_K c) + div( D(Pi_K c) grad c ),   zero flux on the boundary,

on a box (0,L_1) x ... x (0,L_d), d = 1, 2 or 3, uniform cell-centred grid.

Scheme (Section 2.3 of the revised manuscript)
- space: cell averages; diffusive and cross-diffusive fluxes on interior faces
  by centred differences, sigma at the face taken as the mean of the two cell
  values; all boundary fluxes set to zero (exact zero-flux condition);
  advection by the first-order upwind flux with face-normal velocities
  obtained from a streamfunction, hence discretely divergence-free and zero
  on boundary faces;
- time: IMEX Euler. The diagonal self-diffusion delta_i Lap_h is implicit
  and inverted exactly by the type-II discrete cosine transform, which
  diagonalises the zero-flux five-point (seven-point) Laplacian; reaction,
  cross-diffusion and advection are explicit.

All fluxes are conservative, so sum_cells (c1+c2+c3) changes only through
the reaction source S (n0 - Pi_K c1); test_solver.py checks this to round-off.
"""
import numpy as np
from scipy.fft import dctn, idctn

import npz_model as m


class Grid:
    def __init__(self, L, N):
        self.L = np.asarray(L, float)
        self.N = tuple(int(n) for n in N)
        self.d = len(self.N)
        self.h = self.L / np.asarray(self.N)
        self.vol = float(np.prod(self.h))
        # eigenvalues of the zero-flux FV Laplacian along each axis (<= 0)
        self.lap_eig = []
        for a in range(self.d):
            mm = np.arange(self.N[a])
            lam = -(2 - 2 * np.cos(np.pi * mm / self.N[a])) / self.h[a] ** 2
            shape = [1] * self.d; shape[a] = self.N[a]
            self.lap_eig.append(lam.reshape(shape))

    def centres(self, a):
        return (np.arange(self.N[a]) + 0.5) * self.h[a]

    def wavenumbers(self):
        """Continuous Neumann wavenumber |k| of each DCT mode."""
        k2 = 0.0
        for a in range(self.d):
            shape = [1] * self.d; shape[a] = self.N[a]
            k2 = k2 + (np.pi * np.arange(self.N[a]) / self.L[a]).reshape(shape) ** 2
        return np.sqrt(k2)

    def discrete_k2(self):
        """-eigenvalue of the discrete Laplacian of each DCT mode."""
        return -sum(self.lap_eig)


def div_flux(c, a, grid, mask=None):
    """div_h( a grad_h c ) with zero flux on boundary faces; a cell-centred.

    mask[ax] = False switches off the term along axis ax."""
    out = np.zeros_like(c)
    for ax in range(grid.d):
        if mask is not None and not mask[ax]:
            continue
        sl_hi = [slice(None)] * grid.d; sl_hi[ax] = slice(1, None)
        sl_lo = [slice(None)] * grid.d; sl_lo[ax] = slice(None, -1)
        af = 0.5 * (a[tuple(sl_hi)] + a[tuple(sl_lo)])
        flux = af * (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / grid.h[ax]   # interior faces
        pad = [(0, 0)] * grid.d; pad[ax] = (1, 1)
        flux = np.pad(flux, pad)                                       # zero boundary flux
        sl_hi[ax] = slice(1, None); sl_lo[ax] = slice(None, -1)
        out += (flux[tuple(sl_hi)] - flux[tuple(sl_lo)]) / grid.h[ax]
    return out


class Streamfunction2D:
    """Overturning cell psi = A sin(pi x/Lx) sin(pi z/Lz) in the (x1, x3) plane.

    Face velocities are differences of psi at grid corners, so the discrete
    divergence vanishes identically and u.n = 0 on the boundary (psi = 0 there)."""

    def __init__(self, grid, A):
        Lx, Lz = grid.L
        xe = np.arange(grid.N[0] + 1) * grid.h[0]
        ze = np.arange(grid.N[1] + 1) * grid.h[1]
        psi = A * np.sin(np.pi * xe[:, None] / Lx) * np.sin(np.pi * ze[None, :] / Lz)
        # u_x = d psi/dz on x-faces (Nx+1, Nz); u_z = -d psi/dx on z-faces (Nx, Nz+1)
        self.ux = (psi[:, 1:] - psi[:, :-1]) / grid.h[1]
        self.uz = -(psi[1:, :] - psi[:-1, :]) / grid.h[0]
        self.Umax = max(np.abs(self.ux).max(), np.abs(self.uz).max())
        self.grid = grid

    def advect(self, c):
        """-div_h(u c) with first-order upwind face values (= -u.grad c, div u = 0)."""
        g = self.grid
        cx = np.pad(c, ((1, 1), (0, 0)), mode="edge")
        fx = np.where(self.ux > 0, cx[:-1, :], cx[1:, :]) * self.ux
        cz = np.pad(c, ((0, 0), (1, 1)), mode="edge")
        fz = np.where(self.uz > 0, cz[:, :-1], cz[:, 1:]) * self.uz
        return -((fx[1:, :] - fx[:-1, :]) / g.h[0] + (fz[:, 1:] - fz[:, :-1]) / g.h[1])


class Solver:
    def __init__(self, grid, dt, chip=0.0, chiz=0.0, delta=(m.DELTA,) * 3,
                 params=m.PARAMS, K=m.K_TRUNC, flow=None):
        self.g, self.dt, self.chip, self.chiz = grid, dt, chip, chiz
        self.delta, self.p, self.K, self.flow = delta, params, K, flow
        lap = sum(grid.lap_eig)
        self.inv = [1.0 / (1.0 - dt * d * lap) for d in delta]

    def implicit(self, rhs, i):
        return idctn(dctn(rhs, type=2, norm="ortho") * self.inv[i], type=2, norm="ortho")

    def explicit_rhs(self, c):
        c1, c2, c3 = c
        F1, F2, F3 = m.reaction(c1, c2, c3, self.p, self.K)
        r = [F1, F2, F3]
        if self.chip:
            r[1] = r[1] + div_flux(c1, self.chip * m.sigma(c1), self.g)
        if self.chiz:
            r[2] = r[2] + div_flux(c2, self.chiz * m.sigma(c2), self.g)
        if self.flow is not None:
            r = [ri + self.flow.advect(ci) for ri, ci in zip(r, c)]
        return r

    def step(self, c):
        r = self.explicit_rhs(c)
        return [self.implicit(ci + self.dt * ri, i) for i, (ci, ri) in enumerate(zip(c, r))]

    def run(self, c, T, callback=None, every=None):
        n = int(round(T / self.dt))
        for it in range(n):
            c = self.step(c)
            if callback is not None and every and (it + 1) % every == 0:
                callback((it + 1) * self.dt, c)
        return c


def smooth_perturbation(grid, rms, seed, kmax=20):
    """Seeded perturbation built from the first kmax Neumann modes per axis,
    so that the same continuous initial datum is sampled on every grid."""
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal((kmax,) * grid.d)
    field = np.zeros(grid.N)
    xs = [grid.centres(a) for a in range(grid.d)]
    modes = [np.cos(np.pi * np.arange(kmax)[:, None] * xs[a][None, :] / grid.L[a])
             for a in range(grid.d)]
    if grid.d == 1:
        field = modes[0].T @ coef
    elif grid.d == 2:
        field = modes[0].T @ coef @ modes[1]
    else:
        field = np.einsum("ijk,ia,jb,kc->abc", coef, *modes)
    field -= field.mean()
    return rms * field / np.sqrt(np.mean(field ** 2))


def dominant_wavenumber(field, grid):
    """Dominant wavenumber of a field on the Neumann (cosine) basis.

    Returns (k_peak, k_mean): the |k| of the DCT mode of largest power and the
    power-weighted mean |k|, both excluding the mean (k = 0) mode."""
    P = dctn(field - field.mean(), type=2, norm="ortho") ** 2
    k = grid.wavenumbers()
    P.flat[0] = 0.0
    return float(k.flat[np.argmax(P)]), float((k * P).sum() / P.sum())
