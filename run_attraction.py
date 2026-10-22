"""
Section 7.2, attractive nutrient-taxis: chi_p = -8 (phytoplankton moves up the
nutrient gradient), chi_z = 0, delta_i = 0.1, on (0,60)^2.

Linear phase (1e-7 perturbation, fit on t in [100, 400]) against the discrete
amplification factor of the fastest cosine mode; nonlinear runs on 128^2 and
192^2 cells (dt = 0.02) up to T = 2000: amplitude std(c2)/c2*, dominant
wavenumber, min over all times, stationarity residual.
Writes results/attraction.json and results/attraction_fields.npz.
Usage: python3 run_attraction.py   (about 10 minutes)
"""
import json
import os
import numpy as np
from scipy.fft import dctn
import fv_solver as fv
import npz_model as m

L, CHIP, T, SEED, DT = 60.0, -8.0, 2000.0, 11, 0.02
OUT = os.path.join(os.path.dirname(__file__), "results")
cs = m.equilibrium()


def initial(g, rms):
    return [cs[i] + fv.smooth_perturbation(g, rms, SEED + i) for i in range(3)]


def linear_phase(N=128):
    g = fv.Grid((L, L), (N, N))
    s = fv.Solver(g, DT, chip=CHIP)
    J, *_ = m.jacobian_closed()
    Nn = np.zeros((3, 3)); Nn[1, 0] = CHIP * m.sigma(cs[0])
    kd2 = g.discrete_k2()
    rate = np.full(g.N, -np.inf)
    for idx in np.ndindex(30, 30):
        G = (np.eye(3) + DT * (J - kd2[idx] * Nn)) / (1.0 + DT * m.DELTA * kd2[idx])
        rate[idx] = np.log(np.max(np.abs(np.linalg.eigvals(G)))) / DT
    mode = np.unravel_index(np.argmax(rate), g.N)
    c = initial(g, 1e-7)
    ts, amp = [], []

    def cb(t, c):
        ts.append(t); amp.append(abs(dctn(c[1], type=2, norm="ortho")[mode]))
    s.run(c, 400.0, cb, every=50)
    ts, amp = np.array(ts), np.array(amp)
    sel = ts >= 100.0
    kk = np.linspace(0.01, 3, 30001)
    gr = m.growth_rate(kk, chip=CHIP)
    return dict(mode=[int(i) for i in mode], k_mode=float(g.wavenumbers()[mode]),
                measured_rate=float(np.polyfit(ts[sel], np.log(amp[sel]), 1)[0]),
                lambda_discrete_at_mode=float(rate[mode]),
                k_fastest=float(kk[gr.argmax()]), lambda_fastest=float(gr.max()),
                band=[float(kk[gr > 0].min()), float(kk[gr > 0].max())])


def nonlinear(N):
    g = fv.Grid((L, L), (N, N))
    s = fv.Solver(g, DT, chip=CHIP)
    c = initial(g, 1e-2)
    hist = []

    def cb(t, c):
        hist.append((t, min(ci.min() for ci in c), c[1].std() / cs[1]))
    c = s.run(c, T, cb, every=int(round(5.0 / DT)))
    hist = np.array(hist)
    resid = max(np.max(np.abs(a - b)) for a, b in zip(s.step(c), c)) / DT
    late = hist[:, 0] >= T - 500
    return c, dict(N=N, amp_rel=float(c[1].std() / cs[1]),
                   amp_drift_last500=float(np.ptp(hist[late, 2])),
                   k_peak=fv.dominant_wavenumber(c[1], g)[0],
                   min_all_times=float(hist[:, 1].min()), min_final=float(min(ci.min() for ci in c)),
                   max=float(max(ci.max() for ci in c)), stationarity=float(resid))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = dict(chi_p=CHIP, linear=linear_phase())
    print(res["linear"], flush=True)
    fields = {}
    for N in (128, 192):
        c, info = nonlinear(N)
        res[f"N{N}"] = info; fields[f"N{N}"] = np.array(c)
        print(info, flush=True)
    with open(os.path.join(OUT, "attraction.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "attraction_fields.npz"), **fields)
    print(json.dumps(res, indent=2))
