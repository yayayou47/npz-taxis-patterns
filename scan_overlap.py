"""
Section 6 (after Proposition 6.4): can the prey-taxis instability occur inside the coercivity regime?

With chi_p = 0 and delta_i = delta, (HS) reads chi_z < 2 delta, i.e.
tau = chi_z sigma(c2*)/delta < 2 sigma(c2*) = 2 (1 - mu3/alpha)^2, so the
instability is compatible with (HS) iff tau_c < 2 (1 - r)^2.

1. Random scan: log-uniform kinetic parameters (ranges below, seed fixed),
   kept when (HC) and kinetic stability hold; fraction with overlap and the
   distribution of c3* among them.
2. Reference parameters with mu2 -> beta n0/(1+n0) (g -> 0): tau_c / sqrt(g)
   stays bounded (Proposition 6.4).
Writes results/overlap.json.
"""
import json
import os
import numpy as np
import npz_model as m

N_SAMPLES, SEED = 5000, 2026
RANGES = dict(S=(-2, 1), n0=(-1, 1.5), beta=(-1, 1), alpha=(-1, 1), mu2=(-2, 0.5), mu3=(-2, 0.5))

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    rows, tried = [], 0
    while len(rows) < N_SAMPLES:
        p = {k: 10 ** rng.uniform(*v) for k, v in RANGES.items()}
        tried += 1
        if not (m.admissible(p) and m.kinetically_stable(p)):
            continue
        tau_c, _ = m.threshold("prey", p)
        r = p["mu3"] / p["alpha"]
        rows.append((tau_c / (2 * (1 - r) ** 2), m.equilibrium(p)[2]))
    ratio, c3 = np.array(rows).T
    ov = ratio < 1
    ref = dict(m.PARAMS)
    gmax = ref["beta"] * ref["n0"] / (1 + ref["n0"])
    scaling = []
    for g in (1e-1, 1e-2, 1e-3, 1e-4):
        p = dict(ref, mu2=gmax - g)
        scaling.append(dict(g=g, stable=bool(m.kinetically_stable(p)),
                            tau_c=m.threshold("prey", p)[0],
                            tau_c_over_sqrt_g=m.threshold("prey", p)[0] / np.sqrt(g)))
    tau_ref = m.threshold("prey")[0]
    r_ref = ref["mu3"] / ref["alpha"]
    res = dict(n_samples=N_SAMPLES, n_tried=tried, seed=SEED, ranges_log10=RANGES,
               overlap_fraction=float(ov.mean()),
               c3_median_overlap=float(np.median(c3[ov])),
               c3_q90_overlap=float(np.quantile(c3[ov], 0.9)),
               c3_median_all=float(np.median(c3)),
               ref_tau_c=tau_ref, ref_bound=2 * (1 - r_ref) ** 2,
               ref_ratio=tau_ref / (2 * (1 - r_ref) ** 2), scaling=scaling)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "overlap.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
