import numpy as np

rng = np.random.default_rng(0)

#Constants

L = 1.35 #meters
Ds = 0.247 #meters
HZero = 0.06 #meters

a = 7.9752e-6 
n = 0.4365
tau = 0.6528

amount_samples = 1

time_grid = np.linspace(7,10,num=100)

#RV


Pa0 = rng.normal(9.2e6,9.2e4,amount_samples) #Pa
#CStar = rng.normal(1.75, 31.5, amount_samples)
CStar = [1750]
rho = rng.normal(1690,33.8,amount_samples) #kg/m^3
Dt = rng.normal(0.064,0.0013,amount_samples) #meter
c = rng.normal(1, 0.05, amount_samples) #c(t) for instantanous 


# %%c(t) For Cum Probability
def p(t1,t2):
    return np.exp(-np.divide((t2-t1),0.01)**2)

C_Matrix = np.zeros((100,100))
sigma = 0.05
for i in range(C_Matrix.shape[0]):
    for j in range(C_Matrix.shape[1]):
        C_Matrix[i,j] = (sigma**2)*p(time_grid[i],time_grid[j])

mean_multivaraite = np.ones(100)
x = rng.multivariate_normal(mean_multivaraite, C_Matrix)

# %% Plot-Stil mit echtem LaTeX
##AI Generated Plot Style
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


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
'''
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_grid, x, "o-", color=BLUE)
plt.show()

C_Matrix[0,1]/sigma**2
'''

##Formulas

def At(Dt):
    return np.divide(np.pi*Dt**2,4)

def Af(Ds,Ht,L):
    return np.pi*(Ds - 2*Ht)*L

def V(Ds,Ht,L):
    return L*np.pi*((Ds/2)-Ht)**2

parameters = (CStar[0],Dt[0],rho[0])

def rhs(parameters,Ht,Pt,t):
    CStar= parameters[0]
    Dt = parameters[1]
    rho=parameters[2]


    return np.array(
        [
            -a*Pt**n
            ,
            np.divide(tau**2*CStar*At(Dt),V(Ds,Ht,L))*
            (
                np.divide(rho*Af(Ds,Ht,L),At(Dt))   
                *CStar*a*Pt**n-Pt
             )
    ]
    )


def euler(function,timeintervall,deltaT,Parameters,H0,P0):
    yk = np.array([H0,P0])
    k=0
    for i in np.linspace(timeintervall[0],timeintervall[1],int((timeintervall[1]-timeintervall[0])/deltaT),endpoint=False):
        yk = yk + deltaT*function(Parameters,*yk,i)
    return yk



euler(rhs,[0,10],0.01,parameters,0.06,2e6)
    
from scipy.integrate import solve_ivp

f = lambda t, y: rhs(parameters, y[0], y[1], t)

sol = solve_ivp(f, t_span=(0, 10), y0=[0.06, 2e6],
                rtol=1e-10, atol=1e-10)

sol.y[:, -1]   # Zustand (H, P) am Ende

