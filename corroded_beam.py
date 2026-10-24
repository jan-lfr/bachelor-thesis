from matplotlib.pylab import default_rng, number
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

z=10
real_val = np.zeros(z)
sr_val = np.zeros(z)

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



for i in range(z):

    rng = np.random.default_rng(i)

    ## Plotting

    """ G(*sample(1, rng),0.0)

    rows = []

    n=1

    for r in range(0, n, 1):
        s = sample(1, rng)
        for t in range(0, Y, 1):
            rows.append([r,t, *G(*s, t)])

    arr = np.array(rows)

    rows
    GM = arr[:,2].reshape(n,Y)

    for r in range(0, n, 1):
        plt.plot(np.arange(Y), GM[r], label = f"run {r}")
    plt.legend()
    plt.show()

    np.diff(GM[0]) """



    ## Plain Monte Carlo
    import time
    t0 = time.perf_counter()



    t = 20
    n = 10**5
    y=4000
    treffer = 0 

    for j in range(100):
        g = G(*sample(n, rng), t)
        treffer += (g <=y).sum()

    p = treffer / (n*100)
    #print(f"{time.perf_counter() - t0:.3f} s")
    print(f"Probability of failure: {p}")

    real_val[i] = p



    ## Selective Refinment Algorithm

    def G_k(fy, b0, h0, F, t,j,u,eta):
        return G(fy, b0, h0, F, t) + 2**(-j)*eta * u


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

    eta=4000
    number_hits = 0
    n = 10**4
    l_max = 10
    list = np.zeros(l_max+1)

    for b in range(n):
        x,j = selective_refinement(l_max, sample(1, rng),t, y,eta)
        number_hits+=x
        list[j] += 1

    p = number_hits/n
    #print(f"Probability of failure: {p} and number of hits: {number_hits}")
    list

    sr_val[i] = p

    ## Vectorized selective refinement algorithm for MC






def p_m(runs):
    runs.sum()
    return runs.sum() / runs.size

def s(runs):

    m = runs.size
    p = p_m(runs)
    return np.sqrt(((runs - p)**2).sum() / (m-1))


for i in range(z):
    print(f"Run {i}: Plain Monte Carlo: {real_val[i]}, Selective Refinement: {sr_val[i]}")

print(p_m(real_val))
print(p_m(sr_val))
print(s(real_val))
print(s(sr_val))



n=10**4
p = p_m(real_val)

1/np.sqrt(n*p)
s_val = s(sr_val)/p_m(sr_val)

(p_m(sr_val)-p)/(s(sr_val)/np.sqrt(10))




##Vecotrised selective refinement algorithm for MC
eta=4000
number_hits = 0
n = 10**7
l_max = 10
list = np.zeros(l_max+1)

w = np.array(sample(n, rng))

G(*w,0.0)




def G_k(fy, b0, h0, F, t,j,u,eta):
    return G(fy, b0, h0, F, t) + 2**(-j)*eta * u



Uniform = rng.uniform(-1,1,n) ##Anzahl der Zufallszahlen = Anzahl der Samples

j=0

g = G_k(*w,0,j,Uniform,4000)

check = abs(g-4000) < 2**(-j)*eta

while check.sum() > 0 and j<l_max:
    j+=1
    idx = np.nonzero(check)[0]   # Positionen aller True, z. B. array([0, 3, 7])
    g[idx] = G_k(*w[:,idx],0,j,Uniform[idx],4000) 
    check = abs(g-4000) < 2**(-j)*eta
    print(f"j={j}, g={g[idx]}, check={check},idx={idx}")

