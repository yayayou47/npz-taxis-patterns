"""
Non-regression test (exit code 0/1) of the closed forms of Section 3.3:
Jacobian at c*, Routh-Hurwitz coefficients, cofactor C32 = -mu3 (S + delta1 k^2),
and the coefficients a1, a2, a3 of det(lambda I - M(k^2)) for both taxis
directions. Symbolic check with SymPy at generic parameters, then numerical
check against eigenvalues.
"""
import sys
import numpy as np
import sympy as sp
import npz_model as m

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# ---- symbolic: generic parameters, generic (unequal) diffusivities --------
S, n0, beta, alpha, mu2, mu3 = sp.symbols("S n0 beta alpha mu2 mu3", positive=True)
d1, d2, d3, k2, chip, chiz, lam = sp.symbols("delta1 delta2 delta3 k2 chi_p chi_z lambda", positive=True)
c1, c2, c3 = sp.symbols("c1 c2 c3")
F = sp.Matrix([S * (n0 - c1) - beta * c1 * c2 / (1 + c1) + mu2 * c2 + mu3 * c3,
               beta * c1 * c2 / (1 + c1) - alpha * c2 * c3 / (1 + c2) - mu2 * c2,
               alpha * c2 * c3 / (1 + c2) - mu3 * c3])
cs = {c1: n0, c2: mu3 / (alpha - mu3),
      c3: (n0 * (beta - mu2) - mu2) / ((alpha - mu3) * (n0 + 1))}
check("F(c*) = 0", all(sp.simplify(Fi.subs(cs)) == 0 for Fi in F))
J = sp.simplify(F.jacobian([c1, c2, c3]).subs(cs))
b = beta * cs[c2] / (1 + n0) ** 2
g = beta * n0 / (1 + n0) - mu2
r = mu3 / alpha
Jc = sp.Matrix([[-S - b, -g, mu3], [b, g * r, -mu3], [0, g * (1 - r), 0]])
check("closed-form Jacobian (Lemma 3.7)", sp.simplify(J - Jc) == sp.zeros(3, 3))
A1 = S + b - g * r
A2 = g * (b * (1 - r) - r * S + mu3 * (1 - r))
A3 = g * mu3 * S * (1 - r)
cp = (lam * sp.eye(3) - Jc).det()
P = sp.Poly(sp.expand(cp), lam)
check("Hurwitz A1", sp.simplify(P.coeff_monomial(lam**2) - A1) == 0)
check("Hurwitz A2", sp.simplify(P.coeff_monomial(lam) - A2) == 0)
check("Hurwitz A3", sp.simplify(P.coeff_monomial(1) - A3) == 0)

sig1, sig2 = sp.symbols("sigma1 sigma2", positive=True)
D = sp.Matrix([[d1, 0, 0], [chip * sig1, d2, 0], [0, chiz * sig2, d3]])
M = Jc - k2 * D
C32 = (-1) ** (3 + 2) * M.minor_submatrix(2, 1).det()
check("cofactor C32 = -mu3 (S + (delta1 + chi_p sigma1) k^2), any delta_i",
      sp.simplify(C32 - (-mu3 * (S + (d1 + chip * sig1) * k2))) == 0)

# equal diffusivities, s = delta k^2
dl, s, tau = sp.symbols("delta s tau", positive=True)
for kind in ("prey", "nutrient"):
    if kind == "prey":
        Dk = sp.Matrix([[dl, 0, 0], [0, dl, 0], [0, tau * dl, dl]])
        a1e, a2e, a3e = 3 * s + A1, 3 * s**2 + (2 * A1 - tau * mu3) * s + A2, \
            s**3 + (A1 - tau * mu3) * s**2 + (A2 - tau * mu3 * S) * s + A3
    else:
        Dk = sp.Matrix([[dl, 0, 0], [tau * dl, dl, 0], [0, 0, dl]])
        a1e, a2e, a3e = 3 * s + A1, 3 * s**2 + (2 * A1 - tau * g) * s + A2, \
            s**3 + (A1 - tau * g) * s**2 + (A2 + tau * mu3 * g * (1 - r)) * s + A3
    Mk = (Jc - (s / dl) * Dk)
    Pk = sp.Poly(sp.expand((lam * sp.eye(3) - Mk).det()), lam)
    check(f"{kind}: a1(s)", sp.simplify(Pk.coeff_monomial(lam**2) - a1e) == 0)
    check(f"{kind}: a2(s)", sp.simplify(Pk.coeff_monomial(lam) - a2e) == 0)
    check(f"{kind}: a3(s)", sp.simplify(Pk.coeff_monomial(1) - a3e) == 0)

# ---- numerical: module vs finite differences and eigenvalues --------------
check("numeric Jacobian matches closed form",
      np.allclose(m.jacobian_numeric(), m.jacobian_closed()[0], atol=1e-7))
for kind, chi in (("prey", "chiz"), ("nutrient", "chip")):
    chi_c, k_c, tau_c = m.chi_threshold(kind)
    kk = np.linspace(0.01, 6, 3000)
    below = m.growth_rate(kk, **{chi: 0.999 * chi_c}).max()
    above = m.growth_rate(kk, **{chi: 1.001 * chi_c}).max()
    check(f"{kind}: stable just below chi_c ({below:.2e} < 0)", below < 0)
    check(f"{kind}: unstable just above chi_c ({above:.2e} > 0)", above > 0)

chi_a, k_a, _ = m.chi_threshold("nutrient_attr")
kk = np.linspace(0.01, 6, 3000)
check("attractive nutrient-taxis: stable just below |chi_p|^c",
      m.growth_rate(kk, chip=-0.999 * chi_a).max() < 0)
check("attractive nutrient-taxis: unstable just above |chi_p|^c",
      m.growth_rate(kk, chip=-1.001 * chi_a).max() > 0)
check("attractive prey-taxis stable for chi_z in [-100, 0] (b >= g r)",
      m.attraction_stable_prey() and
      max(m.growth_rate(kk, chiz=-c).max() for c in (0.1, 1, 10, 100)) < 0)

sys.exit(0 if ok else 1)
