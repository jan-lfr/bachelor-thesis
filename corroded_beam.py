# %% Imports
from matplotlib.pylab import default_rng, number
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


# %% Modell:

# Parameter
L = 5
b_0 = 0.2
h_0 = 0.04
p_st = 78.5e3
p = b_0 * h_0 * p_st
k = 0.03e-3
Y = 100

# Random variables

def lognormal(mean, cov, size=1):
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu,sigma, size = size)


def sample(n, rng):
    fy = lognormal(240e6, 0.1, n)
    b0 = lognormal(0.2, 0.05, n)
    h0 = lognormal(0.04, 0.1, n)
    F = rng.normal(3500.0, 0.20 * 3500.0, n)
    return fy, b0, h0, F

# QOI

def G(fy, b0, h0, F, t):
    b_t = b0 - 2 * k * t
    h_t = h0 - 2 * k * t
    MU = b_t * h_t**2 * fy / 4
    M = F * L / 4 + p_st * b0 * h0 * L**2 / 8
    return MU - M


def G_k(fy, b0, h0, F, t,j,u,eta):
    return G(fy, b0, h0, F, t) + 2**(-j)*eta * u

#Algorithms
def selective_refinement(l_max,w,t,y,eta):
        Uniform = rng.uniform(-1,1)
        j=0
        g = G_k(*w,t,j,Uniform,eta)
        while j<l_max and abs(g-y) < 2**(-j)*eta:
            j+=1
            g = G_k(*w, t, j, Uniform,eta)
            #print(f"j={j}, g={g}")
        x = (g-y)<=0
        return x[0],j



def selective_refinement_vectorised(l_max,w,t,y,eta):

    n=w.shape[1]

    Uniform = rng.uniform(-1,1,n) ##Anzahl der Zufallszahlen = Anzahl der Samples

    j=0

    j_vals = np.zeros(n, dtype=int)

    g = G_k(*w,t,j,Uniform,eta)

    check = abs(g-y) < 2**(-j)*eta

    while check.sum() > 0 and j<l_max:
        j+=1
        j_vals[check] = j
        idx = np.nonzero(check)[0]   # Positionen aller True
        g[idx] = G_k(*w[:,idx],t,j,Uniform[idx],eta)
        check = abs(g-y) < 2**(-j)*eta
        #print(f"j={j}, g={g[idx]}, check={check},idx={idx}")

    x = (g-y)<=0.
    return x,j_vals



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
        x,j = selective_refinement(l_max, sample(1, rng),t, y,eta)
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
    x,j_vals = selective_refinement_vectorised(l_max,w,t,y,eta)

    t1 = time.perf_counter()

    sr_val_vectorised[i] = np.divide(x.sum(),n_v)

    #print(f"Time taken Vectorized SR: {t1-t0:.3f} s")
    time_sr_vectorised[i] = t1-t0




# %% Auswertung der Validierung (p_m, s, z-Wert)


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

