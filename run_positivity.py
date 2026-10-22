"""
Illustration of Proposition 5.2 (loss of positivity) on (0,20), one dimension.

Initial data: nutrient bump c1 = n0 + 0.5 exp(-(x-10)^2/2), phytoplankton free
zone c2 = 0 on |x - 10| <= 2 joined C^1 to c2* on |x - 10| >= 3, c3 = c3*.
Nutrient-taxis chi_p = 0.05 lies INSIDE (HD) for delta_i = 0.1
(delta_1 - chi_p/2 = delta_2 - chi_p/2 = 0.075 > 0), chi_z = 0.

Reports min c2 on t in [0, 2] for chi_p = 0.05 and chi_p = 0, and the
initial slope chi_p * int Lap h(c1_0) phi of Proposition 5.2 for a bump phi.
Writes results/positivity.json.
"""
import json
import os
import numpy as np
import fv_solver as fv
import npz_model as m

L, N, DT, T, CHIP = 20.0, 800, 1e-3, 2.0, 0.05
OUT = os.path.join(os.path.dirname(__file__), "results")
cs = m.equilibrium()
g = fv.Grid((L,), (N,))
x = g.centres(0)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


c10 = m.PARAMS["n0"] + 0.5 * np.exp(-(x - 10) ** 2 / 2)
c20 = cs[1] * smoothstep(np.abs(x - 10) - 2.0)
c30 = np.full(N, cs[2])


def run(chip):
    s = fv.Solver(g, DT, chip=chip)
    c = [c10.copy(), c20.copy(), c30.copy()]
    hist = []

    def cb(t, c):
        hist.append((t, float(c[1].min())))
    s.run(c, T, cb, every=10)
    t, mn = np.array(hist).T
    i = int(np.argmin(mn))
    return dict(chi_p=chip, min_c2=float(mn[i]), t_min=float(t[i]),
                first_negative_time=float(t[np.argmax(mn < 0)]) if (mn < 0).any() else None)


# initial slope of (c2(t), phi) for phi = (1 - ((x-10)/1.5)^2)^3 on |x-10| < 1.5
h = lambda c: c / (1 + c)
phi = np.where(np.abs(x - 10) < 1.5, (1 - ((x - 10) / 1.5) ** 2) ** 3, 0.0)
lap_h = np.gradient(np.gradient(h(c10), x), x)
slope = CHIP * np.sum(lap_h * phi) * g.h[0]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = dict(initial_slope=float(slope), with_taxis=run(CHIP), without_taxis=run(0.0),
               HD_margin=float(min(m.DELTA - CHIP / 2, m.DELTA - CHIP / 2)))
    with open(os.path.join(OUT, "positivity.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
