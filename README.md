# npz-taxis-patterns

Code for *Weak solutions and taxis-driven patterns of a quasi-linear NPZ model with nutrient-taxis and prey-taxis cross-diffusion*, Y. Y. Yaya, M. Sy, D. Ngom (revised version submitted to *Communications in Mathematics and Applications*).

Requirements: Python 3 with `numpy`, `scipy`, `sympy`, `matplotlib` (`requirements.txt`). License: MIT.

Every figure and every number of Sections 5–7 of the revised manuscript is produced here. The numbers reach the manuscript only through `values.tex`, written by `make_values.py`, which the LaTeX source `\input`s. The simulation results used in the paper are included in `results/` (JSON + `.npz`), so figures and numbers can be regenerated without rerunning the simulations.

| Script | Produces | Time |
|---|---|---|
| `npz_model.py` | parameters, equilibrium, closed-form Jacobian and Hurwitz coefficients, thresholds (Props. 6.2–6.4) | s |
| `fv_solver.py` | finite-volume IMEX solver (DCT-implicit diffusion, conservative fluxes, streamfunction advection) | — |
| `test_closed_forms.py` | SymPy check of every closed form + eigenvalue check of every threshold (exit 0/1) | ~30 s |
| `test_solver.py` | DCT solve, discrete mass balance, div u = 0, u·n = 0, equilibrium preserved (exit 0/1) | s |
| `run_positivity.py` | 1D loss-of-positivity example → `results/positivity.json` | s |
| `scan_overlap.py` | random scan after Prop. 6.4 → `results/overlap.json` | ~3 min |
| `run_patterns2d.py` | linear validation + avoidance patterns, 3 grids, 2 time steps → `results/patterns2d.json` | ~20 min |
| `run_attraction.py` | nutrient-attraction case, 2 grids → `results/attraction.json` | ~15 min |
| `run_peclet.py` | overturning-cell sweep, 14 values × 2 grids → `results/peclet.json` | ~90 min |
| `make_figures.py` | figures 1–4 (PNG, 400 dpi) in `figures/` | s |
| `make_values.py` | `values.tex` | s |
| `check_refs.py` | Crossref audit of every `\doi{}` of the manuscript | ~20 s |

Reproduction, in this order:

```bash
python3 test_closed_forms.py && python3 test_solver.py
python3 run_positivity.py; python3 scan_overlap.py
python3 run_patterns2d.py; python3 run_attraction.py; python3 run_peclet.py
python3 make_figures.py; python3 make_values.py
python3 check_refs.py path/to/Manuscript_CMA.tex
```

Seeds are fixed in every script; results are stored as JSON + `.npz` (no pickle). Sign convention: the flux of species *i* is −Σⱼ D_ij ∇c_j, so χ < 0 is attraction (taxis up the gradient) and χ > 0 avoidance.

The scripts of the first submitted version are superseded: explicit Euler without truncation in 2D, periodic wrap-around (`np.roll`) in the upwind advection, horizontal-only taxis in 3D and in the Péclet study.
