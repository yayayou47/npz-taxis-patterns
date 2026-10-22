"""
Section 7.2: prey-taxis patterns on (0,20)^2, chi_z = 4, chi_p = 0, delta_i = 0.1.

 1. linear phase: growth rate of the cosine mode (m, n) that is fastest for the
    discrete operator, fitted on t in [20, 60] from a 1e-7 perturbation, against
    (a) the continuous dispersion relation at its wavenumber and (b) the fully
    discrete IMEX amplification factor;
 2. nonlinear runs on N = 128, 192, 256 (dt = 0.01) and N = 128 with dt = 0.005,
    from the same smooth initial datum (first 20 cosine modes per axis), up to
    T = 500; statistics of the final state and, for 128/256 and dt/2,
    the relative L2 difference of the final fields;
 3. pattern wavelength (Neumann/DCT spectrum), stationarity residual
    max|c^{n+1}-c^n|/dt, min over all times and max of the solution
    (positivity; the truncation is inactive iff 0 <= c <= K).

Writes results/patterns2d.json and results/patterns2d_fields.npz.
Usage: python3 run_patterns2d.py   (about 20 minutes)
"""
import json
import os
import numpy as np
import fv_solver as fv
import npz_model as m

L, CHIZ, T, SEED = 20.0, 4.0, 500.0, 7
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
cs = m.equilibrium()


def initial(grid, rms):
    return [cs[i] + fv.smooth_perturbation(grid, rms, SEED + i) for i in range(3)]


def linear_phase(N=128, dt=0.01):
    from scipy.fft import dctn
    g = fv.Grid((L, L), (N, N))
    s = fv.Solver(g, dt, chiz=CHIZ)
    J, *_ = m.jacobian_closed()
    Nn = np.zeros((3, 3)); Nn[2, 1] = CHIZ * m.sigma(cs[1])
    D = m.DELTA * np.eye(3) + Nn
    # fully discrete amplification factor of each cosine mode (m, n <= 40)
    kd2 = g.discrete_k2()
    rate_disc = np.full(g.N, -np.inf)
    for idx in np.ndindex(41, 41):
        k2 = kd2[idx]
        G = (np.eye(3) + dt * (J - k2 * Nn)) / (1.0 + dt * m.DELTA * k2)
        rate_disc[idx] = np.log(np.max(np.abs(np.linalg.eigvals(G)))) / dt
    mode = np.unravel_index(np.argmax(rate_disc), g.N)
    kc = float(g.wavenumbers()[mode])
    lam_cont = float(np.max(np.linalg.eigvals(J - kc**2 * D).real))
    c = initial(g, 1e-7)
    ts, amp = [], []

    def cb(t, c):
        ts.append(t); amp.append(abs(dctn(c[2], type=2, norm="ortho")[mode]))
    s.run(c, 60.0, cb, every=10)
    ts, amp = np.array(ts), np.array(amp)
    sel = ts >= 20.0
    rate = float(np.polyfit(ts[sel], np.log(amp[sel]), 1)[0])
    kk = np.linspace(0.01, 6, 60001)
    gr = m.growth_rate(kk, chiz=CHIZ)
    return dict(mode=[int(i) for i in mode], k_mode=kc, measured_rate=rate,
                lambda_continuous_at_mode=lam_cont, lambda_discrete_at_mode=float(rate_disc[mode]),
                k_fastest=float(kk[gr.argmax()]), lambda_fastest=float(gr.max()),
                band=[float(kk[gr > 0].min()), float(kk[gr > 0].max())])


def nonlinear(N, dt):
    g = fv.Grid((L, L), (N, N))
    s = fv.Solver(g, dt, chiz=CHIZ)
    c = initial(g, 1e-2)
    hist = []

    def cb(t, c):
        hist.append((t, min(ci.min() for ci in c), c[2].std() / cs[2]))
    c = s.run(c, T, cb, every=int(round(1.0 / dt)))
    hist = np.array(hist)
    c_next = s.step(c)
    resid = max(np.max(np.abs(a - b)) for a, b in zip(c_next, c)) / dt
    kpk, kmean = fv.dominant_wavenumber(c[2], g)
    late = hist[:, 0] >= T - 100
    neg = hist[:, 1] < 0
    return g, c, dict(N=N, dt=dt, min_final=float(min(ci.min() for ci in c)),
                      min_all_times=float(hist[:, 1].min()),
                      t_at_min=float(hist[np.argmin(hist[:, 1]), 0]),
                      last_negative_time=float(hist[neg, 0].max()) if neg.any() else None,
                      max=float(max(ci.max() for ci in c)), max_c3_over_mean=float(c[2].max() / cs[2]),
                      stationarity=float(resid), k_peak=kpk, k_mean=kmean,
                      amp_rel=float(c[2].std() / cs[2]),
                      amp_drift_last100=float(np.ptp(hist[late, 2])))


def restrict(f, factor):
    n = f.shape[0] // factor
    return f.reshape(n, factor, n, factor).mean(axis=(1, 3))


if __name__ == "__main__":
    res = dict(linear=linear_phase())
    print(res["linear"], flush=True)
    runs, fields = {}, {}
    for N, dt in ((128, 0.01), (192, 0.01), (256, 0.01), (128, 0.005)):
        g, c, info = nonlinear(N, dt)
        key = f"N{N}_dt{dt}"
        runs[key] = info; fields[key] = np.array(c)
        print(key, info, flush=True)
    res["runs"] = runs
    rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
    f128, f256 = fields["N128_dt0.01"], fields["N256_dt0.01"]
    res["space"] = dict(diff_128_256=rel(np.array([restrict(x, 2) for x in f256]), f128))
    res["time"] = dict(diff_dt=rel(fields["N128_dt0.005"], fields["N128_dt0.01"]))
    with open(os.path.join(OUT, "patterns2d.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "patterns2d_fields.npz"), **fields)
    print(json.dumps(res, indent=2))
