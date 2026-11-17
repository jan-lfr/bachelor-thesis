
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from algorithms.selective_refinement import (
    selective_refinement,
    selective_refinement_vectorised
    )

from algorithms.monte_carlo import MC_Pilot,MC
from models.corroded_beam import G,G_k, sample

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
eta=2000

z=10
real_val = np.zeros(z)
sr_val = np.zeros(z)
sr_val_vectorised = np.zeros(z)

time_mc=np.zeros(z)
time_sr=np.zeros(z)
time_sr_vectorised=np.zeros(z)
rng = np.random.default_rng(0)

n_mc = 10**7
block_number=100
treffer = 0
for j in range(block_number):
    g = G(*sample(n_mc, rng), t)
    treffer += (g <=y).sum()

p = treffer / (n_mc*block_number)
print(p)




# %% Plain MC + scalar SR
for i in range(z):

    rng = np.random.default_rng(i)

    ## Plain Monte Carlo
    n_mc = 10**7
    t0 = time.perf_counter()
    block_number=100
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
'''
    #%% MLMC Selective Refinment
    gamma=1/2
    init_samples=100
    L=0
    
    x,x_grob,j_vals = selective_refinement_vectorised(l_max,w,t,y,eta,rng)
    print(x)
    print(x_grob)

    #prob_vals_1 = np.zeros()
    #prob_vals_1 = np.zeros()


    while True:
        Nl = init_samples*gamma**-L 
        
        w = np.array(sample(Nl, rng))

        #MC w SR
        x,j_vals = selective_refinement_vectorised(L,w,t,y,eta,rng)
        #prob_vals[] = np.divide(x.sum(),Nl) ##Logik vom appenden

        #Estimate V[Q_l - Q_l-1]
        #Compute optimal level allopcation
        for l in range(0,L):
            #Compute Y_Nl,l
            prob_vals_1
            #Estimatae E[Ql-Ql-1]
        if (7.7) < tolerance 
            break
        L +=1


'''

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

#fig.savefig(f"ergebnisse/levelverteilung_y{y}_t{t}.pdf")
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


'''

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
'''

#%%
from models.corroded_beam import G_k

def selective_refinement_vectorised_MLMC(l_max,w,t,y,eta,rng):

    n=w.shape[1]

    Uniform = rng.uniform(-1,1,n) ##Anzahl der Zufallszahlen = Anzahl der Samples

    j=0

    j_vals = np.zeros(n, dtype=int)

    g = G_k(*w,t,j,Uniform,eta)

    check = abs(g-y) < 2**(-j)*eta

    g_grob = g.copy()


    while check.sum() > 0 and j<l_max:
        j+=1
        j_vals[check] = j
        idx = np.nonzero(check)[0]   # Positionen aller True
        g[idx] = G_k(*w[:,idx],t,j,Uniform[idx],eta)
        
        if j<=l_max-1:
            g_grob = g.copy()
        check = abs(g-y) < 2**(-j)*eta
        #print(f"j={j}, g={g[idx]}, check={check},idx={idx}")

        
    x = (g-y)<=0
    x_grob = (g_grob-y)<=0

    return x.astype(float),x_grob.astype(float),j_vals

w = np.array(sample(10**7, rng))
x,x_grob,j_vals = selective_refinement_vectorised_MLMC(10,w,20,4000,4000,rng)
y_l = x-x_grob
print(y_l)
print((y_l == 1).sum())
print((y_l == 0).sum())
print((y_l == -1).sum())

# %%

def cost(vals,q):
        j = 0
        for i in range(len(vals)):
            for k in range(i+1):
                j += vals[i]*gamma**(-k*q)
        return j

def level_stats(l,n,t,y,eta,q,rng):
    w = np.array(sample(n, rng))
    x,x_grob,j_vals = selective_refinement_vectorised_MLMC(l,w,t,y,eta,rng)
    if l == 0:
        y_one =(x == 1).sum()
        y_zero = (x == 0).sum()
        y_nOne =(x == -1).sum()
    else:
        y_l = x-x_grob
        y_one =(y_l == 1).sum()
        y_zero = (y_l == 0).sum()
        y_nOne =(y_l == -1).sum()
    j_vals_counts= np.bincount(j_vals)
    cost_sr_vectorised = cost(j_vals_counts,q)
    return y_one,y_zero,y_nOne,cost_sr_vectorised

def p_tilde(x,n,k):
    return np.divide(x+k,n+k)

def var_est(y_one,y_nOne,n,k,l):
    if l == 0:
        p = np.divide(y_one,n)
        var = p*(1-p)
    else:
        var = p_tilde(y_one+y_nOne,n,k)
    return var

def mean_est(y_one,y_nOne,n,k,l):
    if l == 0:
        mean = np.divide(y_one,n)
    else:
        p1 = p_tilde(y_one,n,k)
        pn1 = p_tilde(y_nOne,n,k)
        mean = max(p1,pn1)
    return mean

def optimal_N(V,c,eps):
    N = np.ceil(np.sqrt(V/c)*2*eps**-2*np.sum(np.sqrt(V * c)))
    return N.astype(int)


V = np.array([0.035, 0.0027])
c = np.array([1.0, 1.9])
eps = 1e-3

optimal_N(V,c,eps)


def mlmc_sr(t, y, eta, q, eps, N_start=1000,N_0=10, k=1, gamma=0.5, L_max=10, rng=None):
    n, y1, ym1, kosten = [], [], [], []

    L = 0
    while True:
        n_new = N_start if L == 0 else int(N_0 * gamma**(-L))
        # Let NL = Nγ−L and compute YMC Nℓ,L using selective refinement (Algorithm 1)
        y_one,y_zero,y_nOne,cost_sr_vectorised =level_stats(L, n_new, t, y, eta, q, rng)
        n.append(n_new)
        y1.append(y_one)
        ym1.append(y_nOne)
        kosten.append(cost_sr_vectorised)
        Varianz = np.zeros(L+1)
        Cost = np.zeros(L+1)
        
        # Estimate V[Qℓ−Qℓ−1] using (7.6)
        for l in range(0,L+1):
            V_l = var_est(y1[l], ym1[l],  n[l], k, l)
            c_l = kosten[l] / n[l]
            Varianz[l] = V_l
            Cost[l] = c_l

        # Compute the optimal {Nℓ}L ℓ=0 using (4.2) and cost model (6.1)
        N_opt = optimal_N(Varianz, Cost, eps)

        #Compute YMC Nℓ,ℓ for all levels ℓ = 0,...,L using selective refinement (Algorithm 1)
        for l in range(0,L+1):
            if N_opt[l] > n[l]:
                y_one,y_zero,y_nOne,cost_sr_vectorised =level_stats(l, N_opt[l]-n[l], t, y, eta, q, rng)
                n[l] += N_opt[l]-n[l]
                y1[l] += y_one
                ym1[l] += y_nOne
                kosten[l] += cost_sr_vectorised

        # Estimate E[Qℓ−Qℓ−1] using (7.5)
        E_Y_L = mean_est(y1[L], ym1[L], n[L], k, L)
        if L >= 1:
            E_Y_Lm1 = mean_est(y1[L-1], ym1[L-1], n[L-1], k, L-1)

        if L == 0:
            left = E_Y_L
        else:
            left = max(gamma * E_Y_Lm1, E_Y_L)
        right = (1/gamma - 1) * eps / np.sqrt(2)

        #Terminate if converged by checking inequality (7.7)
        if left < right:
            break

        if L == L_max:
            print("L_max erreicht: Abbruchkriterium nicht erfüllt")
            break
        L += 1

    # The MLMC-SR estimator is QML {Nℓ},L=L ℓ=0 YMC
    y1 = np.array(y1)
    ym1 = np.array(ym1)
    n = np.array(n)
    estimator = np.sum((y1 - ym1) / n)
    return estimator, n, y1, ym1, kosten

'''rng = np.random.default_rng(0)
est, n, y1, ym1, kosten = mlmc_sr(t=20, y=4000, eta=4000, q=1, eps=1e-3, rng=rng)

print("Schätzer:", est)
print("Level L:", len(n) - 1)
print("n pro Level:", n)
print("y1:", y1)
print("ym1:", ym1)'''

#%% MLMC-SR mit verschiedenen N0-Werten

seeds = range(20)
ergebnisse_n0 = []
level_n0 = []

for s in seeds:
    rng = np.random.default_rng(s)
    est, n, y1, ym1, kosten = mlmc_sr(t=20, y=4000, eta=4000, q=1, eps=1e-3, N0=10, rng=rng)
    ergebnisse_n0.append(est)
    level_n0.append(len(n) - 1)

ergebnisse_n0 = np.array(ergebnisse_n0)
level_n0 = np.array(level_n0)       

print((ergebnisse_n0 == 0).sum())

ergebnisse_n10000 = []
level_n10000 = []

for s in seeds:
    rng = np.random.default_rng(s)
    est, n, y1, ym1, kosten = mlmc_sr(t=20, y=4000, eta=4000, q=1, eps=1e-3, N0=100, rng=rng)
    ergebnisse_n10000.append(est)
    level_n10000.append(len(n) - 1)

ergebnisse_n10000 = np.array(ergebnisse_n10000)
level_n10000 = np.array(level_n10000)   

print(ergebnisse_n10000.mean())
print(np.sqrt(((ergebnisse_n10000 - 0.0104)**2).mean()))

# %%
rng = np.random.default_rng(0)
est, n, y1, ym1, kosten = mlmc_sr(t=20, y=0, eta=4000, q=1, eps=3e-6, N0=1000, rng=rng)
print("Schätzer:", est)
print("Level L:", len(n) - 1)
print("n pro Level:", n)
print("y1:", y1)
print("ym1:", ym1)
print("Kosten pro Level:", kosten)


