"""
Non-regression tests of fv_solver.py (exit code 0/1):
 1. the DCT inversion solves (I - dt delta Lap_h) x = b exactly;
 2. discrete mass balance: sum(c1+c2+c3) changes only by S(n0 - Pi_K c1),
    with cross-diffusion and advection switched on (conservative fluxes);
 3. the advection field is discretely divergence-free with u.n = 0 on the boundary;
 4. a constant state is preserved by the cross-diffusive, advected scheme.
"""
import sys
import numpy as np
import fv_solver as fv
import npz_model as m

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


g = fv.Grid((20.0, 8.0), (40, 16))
rng = np.random.default_rng(0)

# 1. implicit solve
s = fv.Solver(g, dt=0.05)
b = rng.standard_normal(g.N)
x = s.implicit(b, 0)
lap_x = fv.div_flux(x, np.ones(g.N), g)
check("DCT implicit solve", np.max(np.abs(x - 0.05 * m.DELTA * lap_x - b)) < 1e-12)

# 2. mass balance with taxis and advection
flow = fv.Streamfunction2D(g, A=3.0)
s = fv.Solver(g, dt=0.01, chip=1.0, chiz=4.0, flow=flow)
cs = m.equilibrium()
c = [cs[i] + 0.3 * rng.standard_normal(g.N) for i in range(3)]
M0 = sum(ci.sum() for ci in c) * g.vol
F1, *_ = m.reaction(*c)
src = (m.PARAMS["S"] * (m.PARAMS["n0"] - np.clip(c[0], 0, m.K_TRUNC))).sum() * g.vol
c = s.step(c)
M1 = sum(ci.sum() for ci in c) * g.vol
check(f"discrete mass balance (defect {abs(M1 - M0 - s.dt * src):.1e})",
      abs(M1 - M0 - s.dt * src) < 1e-10 * abs(M0))

# 3. discrete divergence-free velocity, zero normal velocity on the boundary
div = (flow.ux[1:, :] - flow.ux[:-1, :]) / g.h[0] + (flow.uz[:, 1:] - flow.uz[:, :-1]) / g.h[1]
check("discrete div u = 0", np.max(np.abs(div)) < 1e-12)
check("u.n = 0 on boundary faces",
      max(np.abs(flow.ux[[0, -1], :]).max(), np.abs(flow.uz[:, [0, -1]]).max()) < 1e-12)

# 4. constant equilibrium preserved
c = [np.full(g.N, ci) for ci in cs]
c = s.run(c, 1.0)
check("equilibrium preserved", max(np.max(np.abs(ci - csi)) for ci, csi in zip(c, cs)) < 1e-12)

sys.exit(0 if ok else 1)
