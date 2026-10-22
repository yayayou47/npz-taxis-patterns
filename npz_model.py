"""
NPZ model with nutrient-taxis and prey-taxis: kinetics, coexistence
equilibrium, closed-form Jacobian and the taxis-instability thresholds of
Propositions 6.2-6.4 (revised manuscript).

Every number of Section 6 of the manuscript is computed here; the other
scripts import this module, so there is a single definition of the
parameters, of the equilibrium and of the thresholds.
"""
import numpy as np
from scipy.optimize import minimize_scalar

# ---- reference kinetic parameters (non-dimensional) -----------------------
PARAMS = dict(S=0.18, n0=1.2, beta=1.6, alpha=0.5, mu2=0.34, mu3=0.34)
DELTA = 0.1          # common self-diffusivity delta_1 = delta_2 = delta_3
K_TRUNC = 50.0       # truncation level K of Pi_K


def sigma(s):
    """Saturation sigma(s) = 1/(1+s)^2, evaluated on Pi_K(s) >= 0."""
    return 1.0 / (1.0 + np.clip(s, 0.0, K_TRUNC)) ** 2


def reaction(c1, c2, c3, p=PARAMS, K=K_TRUNC):
    """Truncated reaction F o Pi_K (arrays of any shape)."""
    q1, q2, q3 = (np.clip(c, 0.0, K) for c in (c1, c2, c3))
    H1 = q1 * q2 / (1 + q1)
    H2 = q2 * q3 / (1 + q2)
    F1 = p["S"] * (p["n0"] - q1) - p["beta"] * H1 + p["mu2"] * q2 + p["mu3"] * q3
    F2 = p["beta"] * H1 - p["alpha"] * H2 - p["mu2"] * q2
    F3 = p["alpha"] * H2 - p["mu3"] * q3
    return F1, F2, F3


def admissible(p=PARAMS):
    """Condition (HC): alpha > mu3 and beta*n0/(1+n0) > mu2."""
    return p["alpha"] > p["mu3"] and p["beta"] * p["n0"] / (1 + p["n0"]) > p["mu2"]


def equilibrium(p=PARAMS):
    """Coexistence equilibrium c* (eq. (6.1))."""
    S, n0, beta, alpha, mu2, mu3 = (p[k] for k in ("S", "n0", "beta", "alpha", "mu2", "mu3"))
    c2 = mu3 / (alpha - mu3)
    c3 = (n0 * (beta - mu2) - mu2) / ((alpha - mu3) * (n0 + 1))
    return np.array([n0, c2, c3])


def jacobian_closed(p=PARAMS):
    """Closed-form Jacobian at c* (Lemma 6.1): returns J and (b, g, r)."""
    S, n0, beta, alpha, mu3 = (p[k] for k in ("S", "n0", "beta", "alpha", "mu3"))
    c2 = equilibrium(p)[1]
    b = beta * c2 / (1 + n0) ** 2
    g = beta * n0 / (1 + n0) - p["mu2"]
    r = mu3 / alpha
    J = np.array([[-S - b, -g, mu3],
                  [b, g * r, -mu3],
                  [0.0, g * (1 - r), 0.0]])
    return J, b, g, r


def jacobian_numeric(p=PARAMS, h=1e-6):
    """Centred finite-difference Jacobian of F at c* (used by the tests)."""
    c = equilibrium(p)
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = h
        Fp = np.array(reaction(*(c + e), p=p)); Fm = np.array(reaction(*(c - e), p=p))
        J[:, j] = (Fp - Fm) / (2 * h)
    return J


def hurwitz(p=PARAMS):
    """Routh-Hurwitz coefficients A1, A2, A3 of J (closed form, Lemma 6.1)."""
    S, mu3 = p["S"], p["mu3"]
    _, b, g, r = jacobian_closed(p)
    A1 = S + b - g * r
    A2 = g * (b * (1 - r) - r * S + mu3 * (1 - r))
    A3 = g * mu3 * S * (1 - r)
    return A1, A2, A3


def kinetically_stable(p=PARAMS):
    A1, A2, A3 = hurwitz(p)
    return A1 > 0 and A3 > 0 and A1 * A2 > A3


# ---- dispersion polynomial in s = delta k^2 --------------------------------
def coeffs_prey(tau, p=PARAMS):
    """a1, a2, a3 as polynomials in s for prey-taxis, tau = chi_z sigma(c2*)/delta."""
    A1, A2, A3 = hurwitz(p)
    mu3, S = p["mu3"], p["S"]
    a1 = np.array([3.0, A1])                                   # 3 s + A1
    a2 = np.array([3.0, 2 * A1 - tau * mu3, A2])               # 3 s^2 + ...
    a3 = np.array([1.0, A1 - tau * mu3, A2 - tau * mu3 * S, A3])
    return a1, a2, a3


def coeffs_nutrient(tau, p=PARAMS):
    """Same for nutrient-taxis, tau = chi_p sigma(n0)/delta."""
    A1, A2, A3 = hurwitz(p)
    _, b, g, r = jacobian_closed(p)
    mu3 = p["mu3"]
    a1 = np.array([3.0, A1])
    a2 = _a2_nutrient(tau, p)
    a3 = np.array([1.0, A1 - tau * g, A2 + tau * mu3 * g * (1 - r), A3])
    return a1, a2, a3


def _a2_nutrient(tau, p=PARAMS):
    # M = J - s I - s*tau*E21; principal minor (1,2) = M11 M22 - M12 M21,
    # dM21 = -s tau, so the minor changes by +M12 * s * tau = -g s tau.
    A1, A2, _ = hurwitz(p)
    _, b, g, r = jacobian_closed(p)
    return np.array([3.0, 2 * A1 - tau * g, A2])


def Q_prey(s, p=PARAMS):
    """tau above which a3(s) < 0 (prey-taxis): Q(s) = q(s) / (mu3 s (s+S))."""
    A1, A2, A3 = hurwitz(p)
    return (s**3 + A1 * s**2 + A2 * s + A3) / (p["mu3"] * s * (s + p["S"]))


def Q_nutrient(s, p=PARAMS):
    """Nutrient-taxis analogue, defined for s > s0 = mu3 (1 - r)."""
    A1, A2, A3 = hurwitz(p)
    _, b, g, r = jacobian_closed(p)
    s0 = p["mu3"] * (1 - r)
    return (s**3 + A1 * s**2 + A2 * s + A3) / (g * s * (s - s0))


def Q_nutrient_attr(s, p=PARAMS):
    """Attractive nutrient-taxis (tau_p < 0): |tau_p| above which a3(s) < 0, 0 < s < s0."""
    A1, A2, A3 = hurwitz(p)
    _, b, g, r = jacobian_closed(p)
    s0 = p["mu3"] * (1 - r)
    return (s**3 + A1 * s**2 + A2 * s + A3) / (g * s * (s0 - s))


def threshold(kind="prey", p=PARAMS):
    """Return (tau_c, s_c): min of Q over its domain and the minimiser.

    kind = "prey" (repulsive prey cross-diffusion, tau > 0), "nutrient"
    (repulsive nutrient cross-diffusion, tau_p > 0) or "nutrient_attr"
    (attractive nutrient-taxis, returns |tau_p| at threshold)."""
    if kind == "nutrient_attr":
        _, _, g, r = jacobian_closed(p)
        s0 = p["mu3"] * (1 - r)
        xs = np.linspace(1e-9, 1 - 1e-9, 200001)
        vals = Q_nutrient_attr(s0 * xs, p)
        i = int(np.argmin(vals))
        res = minimize_scalar(lambda x: Q_nutrient_attr(s0 * x, p),
                              bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]),
                              method="bounded", options=dict(xatol=1e-14))
        return float(res.fun), float(s0 * res.x)
    if kind == "prey":
        f, lo = (lambda x: Q_prey(np.exp(x), p)), -12.0
    else:
        _, _, g, r = jacobian_closed(p)
        s0 = p["mu3"] * (1 - r)
        f, lo = (lambda x: Q_nutrient(s0 + np.exp(x), p)), -12.0
    xs = np.linspace(lo, 12.0, 6001)
    i = int(np.argmin(f(xs)))                     # f is vectorised
    res = minimize_scalar(f, bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]),
                          method="bounded", options=dict(xatol=1e-13))
    x = res.x if res.fun <= f(xs[i]) else xs[i]
    s_c = np.exp(x) if kind == "prey" else s0 + np.exp(x)
    return float(f(x)), float(s_c)


def chi_threshold(kind="prey", p=PARAMS, delta=DELTA):
    """Critical sensitivity chi^c = delta tau_c / sigma(c_signal*)."""
    tau_c, s_c = threshold(kind, p)
    c = equilibrium(p)
    sig = sigma(c[1]) if kind == "prey" else sigma(c[0])
    return delta * tau_c / sig, np.sqrt(s_c / delta), tau_c


def attraction_stable_prey(p=PARAMS):
    """Sufficient condition b >= g r for stability under attractive prey-taxis."""
    _, b, g, r = jacobian_closed(p)
    return b >= g * r


def oscillatory_threshold(kind="prey", p=PARAMS, tau_max=50.0, sign=1.0):
    """Smallest |tau| (tau = sign*|tau|) at which a1 a2 - a3 < 0 for some s >= 0."""
    cf = coeffs_prey if kind == "prey" else coeffs_nutrient
    s = np.concatenate([[0.0], np.logspace(-6, 3, 6000)])

    def worst(tau):
        a1, a2, a3 = cf(sign * tau, p)
        return np.min(np.polyval(a1, s) * np.polyval(a2, s) - np.polyval(a3, s))

    if worst(tau_max) > 0:
        return np.inf
    lo, hi = 0.0, tau_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if worst(mid) > 0 else (lo, mid)
    return hi


def growth_rate(k, chip=0.0, chiz=0.0, p=PARAMS, delta=DELTA):
    """max Re lambda of M(k^2) = J - k^2 D(c*) (eigenvalues, no shortcut)."""
    J, *_ = jacobian_closed(p)
    c = equilibrium(p)
    D = np.array([[delta, 0, 0], [chip * sigma(c[0]), delta, 0], [0, chiz * sigma(c[1]), delta]])
    k = np.atleast_1d(k)
    return np.array([np.max(np.linalg.eigvals(J - kk**2 * D).real) for kk in k])


if __name__ == "__main__":
    c = equilibrium()
    J, b, g, r = jacobian_closed()
    print("c* =", c, " (HC):", admissible(), " stable:", kinetically_stable())
    print("spec J =", np.linalg.eigvals(J))
    print("A1,A2,A3 =", hurwitz())
    for kind in ("prey", "nutrient"):
        chi_c, k_c, tau_c = chi_threshold(kind)
        print(f"{kind}: tau_c={tau_c:.6f}  chi_c={chi_c:.6f}  k_c={k_c:.6f}  "
              f"tau_osc={oscillatory_threshold(kind):.6f}")
    chi_a, k_a, tau_a = chi_threshold("nutrient_attr")
    print(f"nutrient attraction: |tau_p|_c={tau_a:.6f}  |chi_p|_c={chi_a:.6f}  k_c={k_a:.6f}  "
          f"tau_osc={oscillatory_threshold('nutrient', sign=-1.0):.6f}")
    print("attractive prey-taxis always stable (b >= g r):", attraction_stable_prey(),
          " tau_osc(prey, attraction) =", oscillatory_threshold("prey", sign=-1.0))
