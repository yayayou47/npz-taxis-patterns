"""Single style module: Wong colour-blind-safe palette, 300 dpi PNG output
(the journal accepts .jpg, .png or TikZ, at least 300 dpi), fonts sized to
stay legible after reduction to the text width."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")   # 2026-01-01, reproducible files

WONG = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
            yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")
DPI = 400
# inside the submission package figures go next to the manuscript; in the
# stand-alone repository they go to ./figures
_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_revised_manuscript")
FIGDIR = os.path.join(_PKG, "figures") if os.path.isdir(_PKG) else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "lines.linewidth": 1.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "image.cmap": "viridis", "mathtext.fontset": "cm",
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    print("saved", os.path.normpath(path))
