import numpy as np

# Parameter
L = 5
b_0 = 0.2
h_0 = 0.04
p_st = 78.5e3
p = b_0 * h_0 * p_st
k = 0.03e-3
Y = 100

# Random variables

def lognormal(mean, cov, rng, size=1):
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu,sigma, size = size)


def sample(n, rng):
    fy = lognormal(240e6, 0.1, rng, n)
    b0 = lognormal(0.2, 0.05, rng, n)
    h0 = lognormal(0.04, 0.1, rng, n)
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
