import numpy as np

rng = np.random.default_rng(0)


b=0.1
gamma = 1/2
N = 10

#%% 
def QOI_k(QOI,j,b,gamma,rng):
    n = QOI.shape[0]
    UI = rng.uniform(0,1,n)
    return QOI + gamma**(j)*(2*UI-1+b)/(1+b)

q_list = [1,2,3]

#QOI_k(w,0,b,gamma,rng)

def selective_refinement_vectorised(l_max,w,y,eps,rng,gamma):

    n=w.shape[0]
    j=0

    j_vals = np.zeros(n, dtype=int)

    value = QOI_k(w,j,b,gamma,rng=rng)
    value_grob = value.copy()
    check = abs(value-y) < gamma**(j)

    while check.sum() > 0 and j<l_max:
        j+=1
        j_vals[check] = j

        idx = np.nonzero(check)[0]
        value[idx] = QOI_k(w[idx],j,b,gamma,rng=rng)

        if j<=l_max-1:
            value_grob = value.copy()

        check = abs(value-y) < gamma**(j)

    x = (value-y)<=0
    x_grob = (value_grob-y)<=0
    return x.astype(float),x_grob.astype(float),j_vals

# %% Bausteine fuer Algorithm 2
def cost(vals,q):
        j = 0
        for i in range(len(vals)):
            for k in range(i+1):
                j += vals[i]*gamma**(-k*q)
        return j

def level_stats(l,n,t,y,eps,q,rng,gamma):
    w = rng.normal(0,1,n)
    x,x_grob,j_vals = selective_refinement_vectorised(l,w,y,eps,rng=rng,gamma=gamma)
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

def var_est(y_one,y_nOne,n,k,l,switch=False):
    if l == 0:
        p = np.divide(y_one,n)
        var = p*(1-p)
        if var == 0 and switch == False:
            var = 1/n
    else:
        var = p_tilde(y_one+y_nOne,n,k)
    return var

def mean_est(y_one,y_nOne,n,k,l,switch=False):
    if l == 0:
        mean = np.divide(y_one,n)
        if mean == 0 and switch == False:
            mean = 1/n
    else:
        p1 = p_tilde(y_one,n,k)
        pn1 = p_tilde(y_nOne,n,k)
        mean = max(p1,pn1)
    return mean

def optimal_N(V,c,eps):
    N = np.ceil(np.sqrt(V/c)*2*eps**-2*np.sum(np.sqrt(V * c)))
    return N.astype(int)

def mlmc_sr(t, y, eta, q, eps, N_start=1000, N_0=10, k=1, gamma=0.5, L_max=10, rng=None,switch=False):
    n, y1, ym1, kosten = np.zeros(L_max+1), np.zeros(L_max+1), np.zeros(L_max+1), np.zeros(L_max+1)

    L = 0
    while True:

        # Level 0: N_start Samples (gross, wegen L=0-Problem); ab Level 1: Paper-Formel N_0 * gamma^-L
        n_new = N_start 

        #if L == 0 else int(N_0 * gamma**(-L))

        # Let NL = Nγ−L and compute YMC Nℓ,L using selective refinement (Algorithm 1)
        y_one,y_zero,y_nOne,cost_sr_vectorised =level_stats(L, int(N_0 * gamma**(-L)), t, y, eta, q, rng=rng, gamma=gamma)
        n.append(n_new)
        y1.append(y_one)
        ym1.append(y_nOne)
        kosten.append(cost_sr_vectorised)
        Varianz = np.zeros(L+1)
        Cost = np.zeros(L+1)
        
        # Estimate V[Qℓ−Qℓ−1] using (7.6)
        for l in range(0,L+1):
            V_l = var_est(y1[l], ym1[l],  n[l], k, l,switch=switch)
            c_l = kosten[l] / n[l]
            Varianz[l] = V_l
            Cost[l] = c_l

        # Compute the optimal {Nℓ}L ℓ=0 using (4.2) and cost model (6.1)
        N_opt = optimal_N(Varianz, Cost, eps)

       #print(f"n={n}, y1={y1}, ym1={ym1}, kosten={kosten}")
       #print(f"Varianz={Varianz}, Kosten={Cost}, N_opt={N_opt}")

        #Compute YMC Nℓ,ℓ for all levels ℓ = 0,...,L using selective refinement (Algorithm 1)
        for l in range(0,L+1):
            if N_opt[l] > n[l]:
                y_one,y_zero,y_nOne,cost_sr_vectorised =level_stats(l, int(N_opt[l]-n[l]), t, y, eta, q, rng=rng, gamma=gamma)
                n[l] += N_opt[l]-n[l]
                y1[l] += y_one
                ym1[l] += y_nOne
                kosten[l] += cost_sr_vectorised
                #print(f"Level {l}: n={n[l]}, y1={y1[l]}, ym1={ym1[l]}, kosten={kosten[l]}")


        # Estimate E[Qℓ−Qℓ−1] using (7.5)
        E_Y_L = mean_est(y1[L], ym1[L], n[L], k, L,switch=switch)
        if L >= 1:
            E_Y_Lm1 = mean_est(y1[L-1], ym1[L-1], n[L-1], k, L-1,switch=switch)

        if L == 0:
            left = E_Y_L
        else:
            left = max(gamma * E_Y_Lm1, E_Y_L)
        right = (1/gamma - 1) * eps / np.sqrt(2)

        #Terminate if converged by checking inequality (7.7)
        if left < right:
            #print("Abbruchkriterium erfüllt")
            break

        if L == L_max:
            print("L_max erreicht: Abbruchkriterium nicht erfüllt")
            break
        L += 1

        '''
        y1_check = np.array(y1)
        ym1_check = np.array(ym1)
        n_check = np.array(n)
        estimator = np.sum((y1_check - ym1_check) / n_check)
        print(f"Level {L} abgeschlossen. estimator={estimator}, left={left}, right={right}")
        '''


    # The MLMC-SR estimator is QML {Nℓ},L=L ℓ=0 YMC

    estimator = np.sum((y1[:L+1] - ym1[:L+1]) / n[:L+1])

    return estimator, n, y1, ym1, kosten

runs_test=100
array_est_test = np.zeros(runs_test)
for k in range(0,100):
            estimator, n, y1, ym1, kosten = mlmc_sr(0, 0.8, 0, 1, 1e-3, N_start=1000, N_0=10, k=1, gamma=0.5, L_max=10, rng = np.random.default_rng(k))           
            array_est_test[k] = estimator

fehler = np.abs(array_est_test - 0.78814)
print(array_est_test)
print(fehler)
(fehler > 0.1).sum()
array_est_test[fehler > 0.1]

#print("Schätzer:", estimator)
# %% Paper Replicaiton

q_list = [1,2,3]
rtols = np.logspace(-3, -1, 8)
runs=100

array_est = np.zeros((len(q_list),len(rtols),runs))
array_cost = np.zeros((len(q_list),len(rtols),runs))

import pandas as pd

for l,q in enumerate(q_list):
    print(f"q={q}")
    for i,eps in enumerate(rtols):
        for k in range(0,runs):
            estimator, n, y1, ym1, kosten = mlmc_sr(0, 0.8, 0, q, eps, N_start=1000, N_0=10, k=1, gamma=0.5, L_max=10, rng = np.random.default_rng(k))           
            array_est[l, i, k] = estimator
            array_cost[l, i, k] = np.sum(kosten)



# %% Plots wie Figure 2 und Figure 3 im Paper
##AI GENERATED: The following code is generated by AI to create plots similar to Figure 2 and Figure 3 in the paper. It visualizes the RMSE against tolerance and the average cost against tolerance for different values of q.
import matplotlib.pyplot as plt

P_REF = 0.78814                                   # Phi(0.8)
BLUE, RED, GREY = "#3b75af", "#e8312a", "#8a8f98"

fehler = np.abs(array_est - P_REF)                # Form (q, eps, Lauf)
rmse = np.sqrt((fehler**2).mean(axis=2))          # Form (q, eps)
kosten_mittel = array_cost.mean(axis=2)           # Form (q, eps)

# --- Figure 2: RMSE gegen Toleranz, ein Panel pro q ---
fig, axes = plt.subplots(1, len(q_list), figsize=(4.6 * len(q_list), 4.2), sharey=True)
for l, (ax, q) in enumerate(zip(axes, q_list)):
    for i, eps in enumerate(rtols):               # Einzelfehler aller Laeufe als blasse Punkte
        ax.loglog(np.full(runs, eps), fehler[l, i], ".", color=GREY, alpha=0.35, ms=4,
                  label="Einzelfehler $|\\hat{Q}_i - 0.78814|$" if i == 0 else None)
    ax.loglog(rtols, rmse[l], "s-", color=BLUE, label="RMSE")
    ax.loglog(rtols, rtols, "--", color=RED, label=r"Toleranz $\epsilon$")
    ax.set_title(f"q = {q}")
    ax.set_xlabel(r"Toleranz $\epsilon$")
    ax.grid(True, which="major", ls="--", color="#cccccc", lw=0.8)
axes[0].set_ylabel("RMSE")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("MLMC-SR, Beispiel 8.1: RMSE gegen Toleranz", y=1.0)
fig.tight_layout()
plt.show()

# --- Figure 3: mittlere Kosten gegen Toleranz, mit den Raten aus Theorem 6.1 ---
raten = {                                          # (6.2): q<2, q=2, q>2
    1: (lambda e: e**-2.0,                     r"$\epsilon^{-2}$"),
    2: (lambda e: e**-2.0 * np.log(1 / e)**2,  r"$\epsilon^{-2}(\log \epsilon^{-1})^2$"),
    3: (lambda e: e**-3.0,                     r"$\epsilon^{-3}$"),
}
farben = [BLUE, "#d39a1f", RED]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for l, q in enumerate(q_list):
    ax.loglog(rtols, kosten_mittel[l], "o-", color=farben[l], label=f"q = {q}")
    f, name = raten[q]
    ref = f(rtols) * kosten_mittel[l, 0] / f(rtols[0])   # Referenz am kleinsten eps angeheftet
    ax.loglog(rtols, ref, "--", color=farben[l], alpha=0.6, lw=1.2, label=name)
ax.set_xlabel(r"Toleranz $\epsilon$")
ax.set_ylabel("mittlere Gesamtkosten")
ax.set_title("MLMC-SR, Beispiel 8.1: Kosten gegen Toleranz")
ax.grid(True, which="major", ls="--", color="#cccccc", lw=0.8)
ax.legend(ncol=3, fontsize=9, loc="upper right")
plt.show()

# --- Zahlen zum Nachschauen ---
import pandas as pd
display(pd.DataFrame(rmse.T, index=rtols, columns=[f"RMSE q={q}" for q in q_list]))
display(pd.DataFrame(kosten_mittel.T, index=rtols, columns=[f"Kosten q={q}" for q in q_list]))



# %%
