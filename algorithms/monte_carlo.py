import numpy as np
import math

def MC_Pilot(func,sample,t,y,rng,eps,p_ref):
    n = int(math.ceil(np.divide(1-p_ref,(eps**2)*p_ref)))
    Block = 1
    treffer = 0

    if n > 10**7:
        Block = math.ceil(np.divide(n,10**7))
        n_block = int(10**7)
        for j in range(Block):
            g = func(*sample(n_block, rng), t)
            treffer += (g <=y).sum()
        est = treffer / (Block*n_block)

    else:
        g = func(*sample(n, rng), t)
        treffer += (g <=y).sum()
        est = treffer / n
    
    return est,n





def MC(func,sample,t,y,eta,rng,eps,p_ref,l,gamma,q):
    n = int(math.ceil(np.divide(1-p_ref,(eps**2)*p_ref)))
    Block = 1
    treffer = 0

    if n > 10**7:
        Block = math.ceil(np.divide(n,10**7))
        n_block = int(10**7)
        for j in range(Block):
            g = func(*sample(n_block, rng), t,l,eta)
            treffer += (g <=y).sum()
        est = treffer / (Block*n_block)
        cost = Block*n_block*gamma**(-l*q)

    else:
        g = func(*sample(n_block, rng), t,l,eta)
        treffer += (g <=y).sum()
        est = treffer / n
        cost = n*gamma**(-l*q)

    return est,n,cost