import numpy as np

from models.corroded_beam import G_k


def selective_refinement(l_max,w,t,y,eta,rng):
        Uniform = rng.uniform(-1,1)
        j=0
        g = G_k(*w,t,j,Uniform,eta)
        while j<l_max and abs(g-y) < 2**(-j)*eta:
            j+=1
            g = G_k(*w, t, j, Uniform,eta)
            #print(f"j={j}, g={g}")
        x = (g-y)<=0
        return x[0],j



def selective_refinement_vectorised(l_max,w,t,y,eta,rng):

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
