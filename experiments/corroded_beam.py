# %% Imports
from matplotlib.pylab import default_rng, number
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from algorithms.selective_refinement import (
    selective_refinement,
    selective_refinement_vectorised,
)
from models.corroded_beam import G, sample

#Formulas
def p_mean(runs):
    runs.sum()
    return runs.sum() / runs.size

def s(runs):

    m = runs.size
    p = p_mean(runs)
    return np.sqrt(((runs - p)**2).sum() / (m-1))

def rel_s(runs):
    return s(runs) / p_mean(runs)

def standard_error(runs):
    return s(runs) / np.sqrt(runs.size)


# %% Algorithm Parameter
t = 20

y=0
l_max = 10
eta=4000

z=10
real_val = np.zeros(z)
sr_val = np.zeros(z)
sr_val_vectorised = np.zeros(z)

time_mc=np.zeros(z)
time_sr=np.zeros(z)
time_sr_vectorised=np.zeros(z)

# %% Plain MC + scalar SR
for i in range(z):

    rng = np.random.default_rng(i)

    ## Plain Monte Carlo
    n_mc = 10**7
    t0 = time.perf_counter()
    block_number=10
    treffer = 0
    for j in range(block_number):
        g = G(*sample(n_mc, rng), t)
        treffer += (g <=y).sum()

    p = treffer / (n_mc*block_number)
    real_val[i] = p

    t1 = time.perf_counter()
    time_mc[i] = t1-t0


    '''
    ## Selective Refinment Algorithm
    n_sr = 10**5
    t0 = time.perf_counter()

    number_hits = 0

    list = np.zeros(l_max+1)

    for b in range(n_sr):
        x,j = selective_refinement(l_max, sample(1, rng),t, y,eta,rng)
        number_hits+=x
        list[j] += 1

    p = number_hits/n_sr
    #print(f"Probability of failure: {p} and number of hits: {number_hits}")

    sr_val[i] = p
    t1 = time.perf_counter()
    time_sr[i] = t1-t0
    '''

    ##Selective Refinement Algorithm Vectorized
    n_v = 10**7
    t0 = time.perf_counter()

    w = np.array(sample(n_v, rng))
    x,j_vals = selective_refinement_vectorised(l_max,w,t,y,eta,rng)

    t1 = time.perf_counter()

    sr_val_vectorised[i] = np.divide(x.sum(),n_v)

    #print(f"Time taken Vectorized SR: {t1-t0:.3f} s")
    time_sr_vectorised[i] = t1-t0




# %% Auswertung der Validierung (p_m, s, z-Wert)




pred_mc = np.sqrt(n_mc*block_number*p_mean(real_val))**-1
#pred_sr_scalar = np.sqrt(n_sr*p_mean(real_val))**-1
pred_sr_vectorised = np.sqrt(n_v*p_mean(real_val))**-1


##Varianz Check
print(f"Relative standard deviation vs predicted MC: {rel_s(real_val):.2%} vs {pred_mc:.2%}")
#print(f"Relative standard deviation vs predicted SR Scalar: {rel_s(sr_val):.2%} vs {pred_sr_scalar:.2%}")
print(f"Relative standard deviation vs predicted SR Vectorised: {rel_s(sr_val_vectorised):.2%} vs {pred_sr_vectorised:.2%}")

##Bias Check

#bias_sr_scalar = (p_mean(sr_val)-p_mean(real_val))/np.sqrt((s(sr_val)**2+s(real_val)**2)/z)
bias_sr_vectorised = (p_mean(sr_val_vectorised)-p_mean(real_val))/np.sqrt((s(sr_val_vectorised)**2+s(real_val)**2)/z)

#print(f"Bias SR Scalar: {bias_sr_scalar:.2f}")
print(f"Bias SR Vectorised: {bias_sr_vectorised:.2f}")

##Time Check
print(f"Time taken MC: {time_mc.mean():.3f} s")
#print(f"Time taken SR Scalar: {time_sr.mean():.3f} s")
print(f"Time taken SR Vectorised: {time_sr_vectorised.mean():.3f} s")


##Künstliches Kosten modell wie im Seminar Paper
j_vals_counts= np.bincount(j_vals)

gamma=1/2

for q in [1,1.5,2]:
    def cost(vals):
        j = 0
        for i in range(len(vals)):
            for k in range(i+1):
                j += vals[i]*gamma**(-k*q)
        return j

    cost_sr_vectorised = cost(j_vals_counts)

    ##Plain MC Artifical Cost
    cost_mc = n_v * gamma**(-l_max*q)
    print(f"q={q}")
    print(f"Cost MC: {cost_mc:.2f}")
    print(f"Cost SR Vectorised: {cost_sr_vectorised:.2f}")
    print(f"Cost Ratio MC/SR Vectorised: {cost_mc/cost_sr_vectorised:.2f}")

##Probabiltion of failure
print(f"Probability of failure MC: {p_mean(real_val):.4e} ± {standard_error(real_val):.4e}")
#print(f"Probability of failure SR Scalar: {p_mean(sr_val):.2e}")
print(f"Probability of failure SR Vectorised: {p_mean(sr_val_vectorised):.4e} ± {standard_error(sr_val_vectorised):.4e}")



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

# %% Levelverteilung
share = np.cumsum(j_vals_counts[::-1])[::-1] / n_v
x = np.arange(l_max + 1)
j0 = 5
C = share[j0] * 2.0**j0
xr = np.arange(3, l_max + 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, share, "o-", color=BLUE, label="Selective Refinement")
ax.plot(xr, C * 2.0**(-xr), "--", color=RED, label=r"$\propto 2^{-j}$ (Lemma 5.2)")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xlabel("Level $j$")
ax.set_ylabel("Anteil, der Level $j$ erreicht")
ax.set_title(f"Levelverteilung ($y = {y}$, $t = {t}$, $\\eta = {eta}$)")
ax.legend(loc="upper right")

fig.savefig(f"ergebnisse/levelverteilung_y{y}_t{t}.pdf")
plt.show()


# %% Probability of failure vs. time


##Selective Refinement Algorithm Vectorized

y=0
l_max = 10
eta=4000
n_v = 10**6
block_number = 100

timeline = np.arange(0, 22, 2)

prob_fail_vectorised = np.zeros(timeline.shape[0])
level_counts_vectorised = np.zeros((timeline.shape[0], l_max + 1), dtype=int)

prob_fail_mc = np.zeros(timeline.shape[0])


t0 = time.perf_counter()



for i in range(0,len(timeline)):
    rng = np.random.default_rng(42)
    treffer=0
    for j in range(block_number):
        g = G(*sample(n_v, rng), timeline[i])
        treffer += (g <=y).sum()
    prob_fail_mc[i] = np.divide(treffer,block_number*n_v)

t1 = time.perf_counter()

print(f"Time taken Vectorized MC: {t1-t0:.3f} s")


t0 = time.perf_counter()

for i in range(0,len(timeline)):
    rng = np.random.default_rng(41)
    x=0
    level_counts=np.zeros(l_max+1)
    for j in range(0,block_number):
        w = np.array(sample(n_v, rng))
        x_run,j_vals = selective_refinement_vectorised(l_max,w,timeline[i],y,eta,rng)
        x+=np.divide(x_run.sum(),n_v)
        level_counts+=np.bincount(j_vals,minlength=l_max+1)
    
    prob_fail_vectorised[i] = np.divide(x.sum(),block_number)
    level_counts_vectorised[i,:] = level_counts

t1 = time.perf_counter()

print(f"Time taken Vectorized SR: {t1-t0:.3f} s")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(timeline, prob_fail_vectorised, "o-", color=BLUE, label="SR")
ax.plot(timeline, prob_fail_mc, "o-", color=RED, label="MC")

#ax.set_yscale("log")
#ax.set_xticks(timeline)
ax.set_xlabel("Time $t$")
ax.set_ylabel("Probability of failure")
ax.set_title(f"($y = {y}$, $\\eta = {eta}$)")
ax.legend(loc="upper right")

#fig.savefig(f"ergebnisse/probability of failure vs time_y{y}_t{t}.pdf")
plt.show()


def standard_dev(p1,p2,n1,n2):
    se1=np.divide(p1*(1-p1),n1)
    se2=np.divide(p2*(1-p2),n2)
    se = np.sqrt(se1+se2)
    return np.divide(p2-p1,se)

for t in range(0,len(timeline)):

    print(standard_dev(prob_fail_mc[t],prob_fail_vectorised[t],n_v*block_number,n_v*block_number))

# %%

