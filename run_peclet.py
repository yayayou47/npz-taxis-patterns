"""
Section 7.3: advection-taxis competition in the cross-shore/vertical plane
(x1, x3) in (0,20) x (0,8), system (2.1) in two dimensions with the
overturning cell psi = A sin(pi x1/Lx) sin(pi x3/Lz).

Peclet number  Pe = U lambda_f / delta,  U = max |u_h| (face velocities),
lambda_f = 2 pi / k_f the fastest-growing linear wavelength (Section 6).
Two order parameters, both averaged over t in [T - 50, T]:
  amplitude  a = std(c3) / c3*  (all spatial variability of zooplankton);
  Turing-band fraction  f = share of the variance of c3 carried by cosine
  modes whose |k| lies in the linearly unstable band (Section 6), which
  separates the spotted Turing pattern from flow-scale structures.
Crossover Pe_x: where f first falls below 1/2 (linear interpolation).

Every A is run on the 128 x 52 grid (dt = 0.01) and on the 256 x 104 grid
(dt = 0.005).
Writes results/peclet.json, results/peclet_fields.npz.
Usage: python3 run_peclet.py   (about 90 minutes)
"""
import json
import os
import numpy as np
from scipy.fft import dctn
import fv_solver as fv
import npz_model as m

LX, LZ, CHIZ, T, TAVG, SEED = 20.0, 8.0, 4.0, 250.0, 50.0, 3
A_LIST = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
OUT = os.path.join(os.path.dirname(__file__), "results")
cs = m.equilibrium()
k_f = float(np.linspace(0.01, 6, 60001)[np.argmax(m.growth_rate(np.linspace(0.01, 6, 60001), chiz=CHIZ))])
LAMBDA_F = 2 * np.pi / k_f
_kk = np.linspace(0.01, 6, 60001); _gr = m.growth_rate(_kk, chiz=CHIZ)
BAND = (float(_kk[_gr > 0].min()), float(_kk[_gr > 0].max()))


def band_fraction(c3, g):
    P = dctn(c3 - c3.mean(), type=2, norm="ortho") ** 2
    k = g.wavenumbers()
    return float(P[(k >= BAND[0]) & (k <= BAND[1])].sum() / P.sum())


def run(A, N, dt):
    g = fv.Grid((LX, LZ), N)
    flow = fv.Streamfunction2D(g, A)
    s = fv.Solver(g, dt, chiz=CHIZ, flow=flow)
    c = [cs[i] + fv.smooth_perturbation(g, 1e-2, SEED + i) for i in range(3)]
    amps, fracs, mins = [], [], []

    def cb(t, c):
        mins.append(min(ci.min() for ci in c))
        if t > T - TAVG - 1e-9:
            amps.append(c[2].std() / cs[2]); fracs.append(band_fraction(c[2], g))
    c = s.run(c, T, cb, every=int(round(1.0 / dt)))
    Pe = flow.Umax * LAMBDA_F / m.DELTA
    return c, dict(A=A, Pe=float(Pe), amp=float(np.mean(amps)), amp_sd=float(np.std(amps)),
                   frac=float(np.mean(fracs)), frac_sd=float(np.std(fracs)),
                   min_all_times=float(min(mins)), max=float(max(ci.max() for ci in c)))


def crossover(rows, key="frac", level=0.5):
    for r0, r1 in zip(rows[:-1], rows[1:]):
        if r0[key] >= level > r1[key]:
            w = (r0[key] - level) / (r0[key] - r1[key])
            return r0["Pe"] + w * (r1["Pe"] - r0["Pe"])
    return float("nan")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out, fields = {}, {}
    for tag, N, dt in (("coarse", (128, 52), 0.01), ("fine", (256, 104), 0.005)):
        rows = []
        for A in A_LIST:
            c, info = run(A, N, dt)
            rows.append(info); fields[f"{tag}_A{A}"] = c[2]
            print(tag, info, flush=True)
        out[tag] = rows
    res = dict(k_f=k_f, lambda_f=LAMBDA_F, band=BAND, **out,
               Pe_x_coarse=crossover(out["coarse"]), Pe_x_fine=crossover(out["fine"]))
    with open(os.path.join(OUT, "peclet.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "peclet_fields.npz"), **fields)
    print(json.dumps({k: res[k] for k in ("Pe_x_coarse", "Pe_x_fine", "lambda_f", "band")}, indent=2))
