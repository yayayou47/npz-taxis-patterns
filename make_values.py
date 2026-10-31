"""
Writes 01_revised_manuscript/values.tex: every number quoted in Sections 2.3 and 3.2-3.5
of the manuscript, as LaTeX macros, from npz_model.py and results/*.json.
The manuscript and the response letter read this file; never edit it by hand.
"""
import json
import os
import numpy as np
import npz_model as m

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
_PKG = os.path.join(HERE, "..", "01_revised_manuscript")
OUT = os.path.join(_PKG if os.path.isdir(_PKG) else HERE, "values.tex")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def f(x, nd=3):
    return f"{x:.{nd}f}"


def sci(x, nd=1):
    mant, ex = f"{x:.{nd}e}".split("e")
    return f"{mant}\\times10^{{{int(ex)}}}"


vals = {}
c = m.equilibrium()
J, *_ = m.jacobian_closed()
ev = np.linalg.eigvals(J)
real = ev[np.abs(ev.imag) < 1e-12].real[0]
cplx = ev[ev.imag > 1e-12][0]
vals.update(EqA=f(c[0], 2), EqB=f(c[1], 3), EqC=f(c[2], 3),
            SpecA=f(real), SpecB=f"{cplx.real:.3f}\\pm{cplx.imag:.3f}\\,i")
chi_c, k_c, tau_c = m.chi_threshold("prey")
chip_c, kp_c, taup_c = m.chi_threshold("nutrient")
vals.update(TauC=f(tau_c), ChizC=f(chi_c), KC=f(k_c), ChipC=f(chip_c), TauP=f(taup_c),
            TauH=f(m.oscillatory_threshold("prey")), TauHp=f(m.oscillatory_threshold("nutrient")))

lin = load("patterns2d.json")["linear"]
vals.update(BandLo=f(lin["band"][0], 2), BandHi=f(lin["band"][1], 2), KF=f(lin["k_fastest"], 2),
            LamF=f(lin["lambda_fastest"], 3), LamWave=f(2 * np.pi / lin["k_fastest"], 2),
            ModeM=str(lin["mode"][0]), ModeN=str(lin["mode"][1]), KMode=f(lin["k_mode"], 3),
            GrowthMeas=f(lin["measured_rate"], 5), GrowthDisc=f(lin["lambda_discrete_at_mode"], 5),
            GrowthCont=f(lin["lambda_continuous_at_mode"], 5))

p2 = load("patterns2d.json")
runs = p2["runs"]
r256 = runs["N256_dt0.01"]
vals.update(AmpRel=f(r256["amp_rel"]), KPeak=f(r256["k_peak"], 2),
            LambdaPeak=f(2 * np.pi / r256["k_peak"], 2),
            AmpA=f(runs["N128_dt0.01"]["amp_rel"]), AmpB=f(runs["N192_dt0.01"]["amp_rel"]),
            AmpC=f(r256["amp_rel"]),
            DiffTime=f"{100 * p2['time']['diff_dt']:.0f}\\,\\%",
            DiffSpace=f"{100 * p2['space']['diff_128_256']:.0f}\\,\\%",
            AmpDrift=f(max(r["amp_drift_last100"] for r in runs.values()), 4),
            Stationarity=f(r256["stationarity"], 3),
            MaxRatio=f(r256["max_c3_over_mean"], 1),
            MinAll=f(r256["min_all_times"], 2), TMin=f"{r256['t_at_min']:.0f}",
            MinFinal=f(r256["min_final"], 2),
            LastNeg=f"{r256['last_negative_time']:.0f}",
            MinAllA=f(runs["N128_dt0.01"]["min_all_times"], 2),
            MinAllB=f(runs["N192_dt0.01"]["min_all_times"], 2))

pos = load("positivity.json")
vals.update(PosFirstNeg=f(pos["with_taxis"]["first_negative_time"], 2),
            PosMin=sci(pos["with_taxis"]["min_c2"]),
            PosMinNoTaxis=sci(pos["without_taxis"]["min_c2"], 0),
            PosSlope=sci(pos["initial_slope"]))

ov = load("overlap.json")
vals.update(RefRatio=f(ov["ref_ratio"], 1), TauScaling=f(ov["scaling"][-1]["tau_c_over_sqrt_g"], 2),
            ScanN=f"{ov['n_samples']}", ScanFrac=f"{100 * ov['overlap_fraction']:.1f}",
            ScanCthreeOv=f(ov["c3_median_overlap"], 3), ScanCthreeAll=f(ov["c3_median_all"], 2))

_, b_, g_, r_ = m.jacobian_closed()
chia, ka, taua = m.chi_threshold("nutrient_attr")
s0 = m.PARAMS["mu3"] * (1 - r_)
vals.update(Bval=f(b_), GRval=f(g_ * r_), KCp=f(kp_c), ChipA=f(chia), KCa=f(ka),
            KZero=f(np.sqrt(s0 / m.DELTA), 2), AttrRatio=f(chia / (2 * m.DELTA), 1))

try:
    at = load("attraction.json")
    la = at["linear"]
    a1, a2 = at["N128"], at["N192"]
    vals.update(AttChi=f"{abs(at['chi_p']):.0f}", AttGrowthMeas=f(la["measured_rate"], 5),
                AttGrowthDisc=f(la["lambda_discrete_at_mode"], 5), AttKF=f(la["k_fastest"], 2),
                AttLambdaF=f(2 * np.pi / la["k_fastest"], 1), AttBandLo=f(la["band"][0], 2),
                AttBandHi=f(la["band"][1], 2), AttKPeak=f(a2["k_peak"], 2),
                AttLambdaPeak=f(2 * np.pi / a2["k_peak"], 1), AttAmpA=f(a1["amp_rel"]),
                AttAmpB=f(a2["amp_rel"]), AttMin=f(min(a1["min_all_times"], a2["min_all_times"]), 2),
                AttMinFinal=f(a2["min_final"], 2), AttStat=sci(a2["stationarity"]),
                AttMax=f(a2["max"], 1), AttKPeakA=f(a1["k_peak"], 2),
                AttDrift=f(max(a1["amp_drift_last500"], a2["amp_drift_last500"]), 2))
except FileNotFoundError:
    pass

pe = load("peclet.json")
if "fine" in pe and isinstance(pe["fine"], list) and pe["fine"] and "frac" in pe["fine"][0]:
    def cross(rows, key):
        lev = rows[0][key] / 2
        for r0, r1 in zip(rows[:-1], rows[1:]):
            if r0[key] >= lev > r1[key]:
                w = (r0[key] - lev) / (r0[key] - r1[key])
                return r0["Pe"] + w * (r1["Pe"] - r0["Pe"])
        return float("nan")
    co, fi = pe["coarse"], pe["fine"]
    vals.update(ScanA=str(len(co)), LambdaF=f(pe["lambda_f"], 2),
                PeXc=f"{cross(co, 'frac'):.0f}", PeXf=f"{cross(fi, 'frac'):.0f}",
                FracZero=f(fi[0]["frac"], 2), FracHigh=f(fi[-1]["frac"], 2),
                AmpZero=f(fi[0]["amp"], 2), AmpHigh=f(fi[-1]["amp"], 2), PeMax=f"{fi[-1]['Pe']:.0f}",
                AmpMinFine=f(min(r["amp"] for r in fi), 2),
                PeGridDiff=f(max(abs(x["amp"] - y["amp"]) for x, y in zip(co, fi)), 2))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("% generated by 07_code/make_values.py -- do not edit\n")
    for k, v in vals.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(vals)} macros to {os.path.normpath(OUT)}")
