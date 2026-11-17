import numpy as np
import pandas as pd
from IPython.display import display

##Corroded Beam MLMC
def selective_refinement_vectorised_MLMC(func,l_max,w,t,y,eta,rng,gamma):

    n=w.shape[1]

    #Uniform = rng.uniform(-1,1,n) ##Anzahl der Zufallszahlen = Anzahl der Samples

    j=0

    j_vals = np.zeros(n, dtype=int)

    g = func(*w,t,j,eta)

    check = abs(g-y) < gamma**(j)*eta

    g_grob = g.copy()


    while check.sum() > 0 and j<l_max:
        j+=1
        j_vals[check] = j
        idx = np.nonzero(check)[0]   # Positionen aller True
        g[idx] = func(*w[:,idx],t,j,eta)
        
        if j<=l_max-1:
            g_grob = g.copy()
        check = abs(g-y) < gamma**(j)*eta
        #print(f"j={j}, g={g[idx]}, check={check},idx={idx}")

        
    x = (g-y)<=0
    x_grob = (g_grob-y)<=0

    return x.astype(float),x_grob.astype(float),j_vals

def cost(vals,q,gamma):
        j = 0
        for i in range(len(vals)):
            for k in range(i+1):
                j += vals[i]*gamma**(-k*q)
        return j

def level_stats(func,sample,l,n,t,y,eta,q,rng,gamma):
    w = np.array(sample(n, rng))
    x,x_grob,j_vals = selective_refinement_vectorised_MLMC(func,l,w,t,y,eta,rng,gamma)
    if l == 0:
        y_one =(x == 1).sum()
        y_zero = (x == 0).sum()
        y_nOne =(x == -1).sum()
    else:
        y_l = x-x_grob
        y_one =(y_l == 1).sum()
        y_zero = (y_l == 0).sum()
        y_nOne =(y_l == -1).sum()
    j_vals_counts = np.bincount(j_vals, minlength=l+1)
    cost_sr_vectorised = cost(j_vals_counts,q,gamma)
    return y_one,y_zero,y_nOne,cost_sr_vectorised,j_vals_counts

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


def mlmc_sr(func,sample,t, y, eta, q, eps, N_start=1000, N_0=10, k=1, gamma=0.5, L_max=10, rng=None,debug=False):
    n, y1, ym1, kosten = np.zeros(L_max+1), np.zeros(L_max+1), np.zeros(L_max+1), np.zeros(L_max+1)
    L = 0
    est_alt = 0.0
    beitrag_alt = np.zeros(L_max+1)
    while True:
        
        calls = []
        
        # Level 0: N_start Samples (gross, wegen L=0-Problem); ab Level 1: Paper-Formel N_0 * gamma^-L
        n_new = N_start 

        if L == 0: 
            n_new = N_start 
        else:
            n_new = int(N_0 * gamma**(-L))

        # Let NL = Nγ−L and compute YMC Nℓ,L using selective refinement (Algorithm 1)
        y_one,y_zero,y_nOne,cost_sr_vectorised,j_counts =level_stats(func,sample, L , int(n_new), t, y, eta, q, rng,gamma)       
        n[L] = n_new
        y1[L] = y_one
        ym1[L] = y_nOne
        kosten[L] = cost_sr_vectorised
        Varianz = np.zeros(L+1)
        Cost = np.zeros(L+1)
        
        calls.append(("Start", L, int(n_new), y_one, y_zero, y_nOne, cost_sr_vectorised, j_counts))        
        # Estimate V[Qℓ−Qℓ−1] using (7.6)
        for l in range(0,L+1):
            V_l = var_est(y1[l], ym1[l],  n[l], k, l)
            c_l = kosten[l] / n[l]
            Varianz[l] = V_l
            Cost[l] = c_l

        # Compute the optimal {Nℓ}L ℓ=0 using (4.2) and cost model (6.1)
        N_opt = optimal_N(Varianz, Cost, eps)
        
        n_vorher = n[:L+1].astype(int)
        N_neu = np.maximum(N_opt - n[:L+1], 0).astype(int)

        #Compute YMC Nℓ,ℓ for all levels ℓ = 0,...,L using selective refinement (Algorithm 1)
        for l in range(0,L+1):
            if N_opt[l] > n[l]:
                y_one,y_zero,y_nOne,cost_sr_vectorised,j_counts =level_stats(func,sample, l, int(N_opt[l]-n[l]), t, y, eta, q, rng,gamma)
                calls.append(("Nachziehen", l, int(N_opt[l]-n[l]), y_one, y_zero, y_nOne, cost_sr_vectorised, j_counts))
                n[l] += N_opt[l]-n[l]
                y1[l] += y_one
                ym1[l] += y_nOne
                kosten[l] += cost_sr_vectorised
        beitrag = (y1[:L+1] - ym1[:L+1]) / n[:L+1]
        est_L = beitrag.sum()
        df = pd.DataFrame({
            "n_vorher": n_vorher,
            "N_neu": N_neu,
            "n_nachher": n[:L+1].astype(int),
            "N_neu": N_neu,
            "y1": y1[:L+1].astype(int),
            "ym1": ym1[:L+1].astype(int),
            "Varianz": Varianz,
            "Kosten": Cost,
            "N_opt": N_opt,
            "Beitrag": beitrag,
            "Änderung": beitrag - beitrag_alt[:L+1],
        })
        df.index.name = "Level"

        if debug == True:
            display(df.style.set_caption(
                f"Durchlauf L = {L} | Estimator = {est_L:.6e} | "
                f"Änderung zum vorherigen Durchlauf = {est_L - est_alt:+.3e}"
            ).format({
                "n_vorher": "{:,}", "n_nachher": "{:,}","N_neu": "{:,}", "y1": "{:,}", "ym1": "{:,}",
                "Varianz": "{:.3e}", "Kosten": "{:.3f}", "N_opt": "{:,}",
                "Beitrag": "{:+.3e}", "Änderung": "{:+.3e}",
            }))

            df_calls = pd.DataFrame(calls, columns=["Aufruf", "Level", "Samples", "y_one", "y_zero", "y_nOne", "Kosten", "j_counts"])
            df_calls["Kosten/Sample"] = df_calls["Kosten"] / df_calls["Samples"]
            df_j = pd.DataFrame(df_calls.pop("j_counts").tolist()).reindex(columns=range(L+1))
            df_j.columns = [f"j={j}" for j in range(L+1)]
            df_calls = pd.concat([df_calls, df_j], axis=1)
            display(df_calls.style.set_caption(f"level_stats-Aufrufe in Durchlauf L = {L}").format({
                "Samples": "{:,}", "y_one": "{:,}", "y_zero": "{:,}", "y_nOne": "{:,}",
                "Kosten": "{:,.0f}", "Kosten/Sample": "{:.3f}",
                **{c: "{:,.0f}" for c in df_j.columns},
            }, na_rep="").hide(axis="index"))

        est_alt = est_L
        beitrag_alt[:L+1] = beitrag

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
        #print(f"Level {L} abgeschlossen. E[Y_L]={E_Y_L}, E[Y_L-1]={E_Y_Lm1}, left={left}, right={right}")


    # The MLMC-SR estimator is QML {Nℓ},L=L ℓ=0 YMC
    estimator = np.sum((y1[:L+1] - ym1[:L+1]) / n[:L+1])
    return estimator, n, y1, ym1, kosten
