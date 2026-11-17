# %% Imports
import sys
from pathlib import Path
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from algorithms.selective_refinement import (
    selective_refinement,
    selective_refinement_vectorised
    )
from algorithms.mlmc import mlmc_sr
from algorithms.monte_carlo import MC_Pilot,MC

from models.corroded_beam import G,G_k, sample



# %% Plot-Stil mit echtem LaTeX
##AI Generated Plot Style
import os
import matplotlib as mpl

# Falls VS Code LaTeX nicht findet (typisch auf dem Mac mit MacTeX):
os.environ["PATH"] += ":" + os.path.expanduser("~/Library/TinyTeX/bin/universal-darwin")

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 11, "axes.titlesize": 12,
    "axes.grid": True, "grid.linestyle": "--", "grid.color": "#cccccc", "grid.linewidth": 0.8,
    "axes.linewidth": 1.0, "axes.edgecolor": "black",
    "legend.frameon": True, "legend.edgecolor": "#cccccc", "legend.framealpha": 1.0,
    "lines.linewidth": 1.8, "lines.markersize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE, RED = "#3b75af", "#e8312a"

#Params
p_ref = 0.010453573
y=4000
t=20
eta=2000
eps = 1e-3
gamma = 1/2
q=1
rng = np.random.default_rng(0)
l_max = 10

#MC
est,n,cost = MC(G_k,sample,t,y,eta,rng,eps,p_ref,l_max,gamma,q)

#MC SR


##MLMC
p_ref_4000_MC = 0.010453573
y=4000
t=20
eta=2000
eps = 1e-4
rng = np.random.default_rng(0)

est, n, y1, ym1, kosten = mlmc_sr(G_k,sample,t=t, y=y, eta=eta, q=1, eps=eps, N_start=1e3,N_0=1e3, rng=rng,L_max=10,debug=True)
print("Schätzer:", est)

print("Absoluter Fehler:", abs(est - p_ref_4000_MC))

x = np.arange(10 + 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, n, "o-", color=BLUE, label="N pro Level")

ax.set_xlabel("Level $j$")
ax.set_ylabel("N die Level $j$ erreicht")
ax.set_title(f"Level ($y = {y}$, $t = {t}$, $\\eta = {eta}$)")
ax.legend(loc="upper right")
