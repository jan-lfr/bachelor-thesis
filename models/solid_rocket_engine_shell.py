import numpy as np

rng = np.random.default_rng(0)

#Constants

L = 1.35 #meters
Ds = 0.247 #meters
HZero = 0.06 #meters

a_burn = 7.9752e-6   # Brennratenkonstante
n_burn = 0.4365      # Brennratenexponent
tau = 0.6528

amount_samples = 1

time_grid = np.linspace(7,10,num=100)

#RV


Pa0 = rng.normal(9.2e6,9.2e4,amount_samples) #Pa
#CStar = rng.normal(1.75, 31.5, amount_samples)
CStar = [1750]
rho = rng.normal(1690,33.8,amount_samples) #kg/m^3
Dt = rng.normal(0.064,0.0013,amount_samples) #meter
c_rv = rng.normal(1, 0.05, amount_samples) #c(t) for instantanous 


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
            -a_burn*Pt**n_burn
            ,
            np.divide(tau**2*CStar*At(Dt),V(Ds,Ht,L))*
            (
                np.divide(rho*Af(Ds,Ht,L),At(Dt))   
                *CStar*a_burn*Pt**n_burn-Pt
             )
    ]
    )

# %% PDE Solver

def euler(function,timeintervall,deltaT,Parameters,H0,P0):
    yk = np.array([H0,P0])
    for i in np.linspace(timeintervall[0],timeintervall[1],int((timeintervall[1]-timeintervall[0])/deltaT),endpoint=False):
        yk = yk + deltaT*function(Parameters,*yk,i)
    return yk

##

euler_v = euler(rhs,[0,10],0.01,parameters,0.06,2e6)


##Runge Kutta Heun

time_grid_rungkutta = np.linspace(0,10,1000+1)
y = np.array([0.06,2e6])
k=0
delta = time_grid_rungkutta[1]-time_grid_rungkutta[0]

for k in range(0,1000):
    k1 = rhs(parameters, *y, time_grid_rungkutta[k])
    yk = y + delta*k1
    k+=1
    k2 = rhs(parameters, *yk, time_grid_rungkutta[k])
    y = y +delta*(k1+k2)/2

heun = y

# Runge Kutta 4 Stufen

time_grid_rungkutta = np.linspace(0,10,1000+1)
delta = time_grid_rungkutta[1]-time_grid_rungkutta[0]
y = np.array([0.06,2e6])
k=np.zeros((4,2))

c=[0,1/2,1/2,1]
b=[1/6,1/3,1/3,1/6]
A = np.array([
    [0,   0,   0, 0],
    [1/2, 0,   0, 0],
    [0,   1/2, 0, 0],
    [0,   0,   1, 0],
])


for t in range(0,1000):
    for i in range(0,4):
        weight_k = np.zeros(2)
        for j in range(0,4):
            weight_k+=A[i,j]*k[j]
        k[i,:] = rhs(parameters, *y+delta*weight_k, time_grid_rungkutta[t]+c[i]*delta)
    weight_y = np.zeros(2)
    for j in range(0,4):
        weight_y+=b[j]*k[j]
    y+=delta*weight_y

rk4 = y




# Scipy

from scipy.integrate import solve_ivp

f = lambda t, y: rhs(parameters, y[0], y[1], t)

# Referenz: separater, deutlich strengerer Lauf (DOP853 = Ordnung 8)
ref = solve_ivp(f, t_span=(0, 10), y0=[0.06, 2e6],
                method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]

rtols = np.logspace(-3, -12, 10)
fehler_scipy = []
nfev_scipy = []

for rtol in rtols:
    sol = solve_ivp(f, t_span=(0, 10), y0=[0.06, 2e6],
                    rtol=rtol, atol=1e-12)
    fehler_scipy.append(abs((sol.y[1, -1] - ref[1]) / ref[1]))
    nfev_scipy.append(sol.nfev)

fehler_scipy = np.array(fehler_scipy)
nfev_scipy = np.array(nfev_scipy)

# Vergleich bei dt = 0.01: relativer Fehler in P und f-Auswertungen
print(f"{'Verfahren':<12} {'Fehler P':>10} {'nfev':>6}")
for name, y_end, nf_ in [("Euler", euler_v, 1000),
                          ("Heun",  heun,    2000),
                          ("RK4",   rk4,     4000)]:
    print(f"{name:<12} {abs((y_end[1] - ref[1]) / ref[1]):>10.2e} {nf_:>6d}")

print()
print(f"{'scipy rtol':<12} {'Fehler P':>10} {'nfev':>6}")
for r, e, nf in zip(rtols, fehler_scipy, nfev_scipy):
    print(f"{r:<12.0e} {e:>10.2e} {nf:>6d}")




# %% Allgemeine Formel

def rk(A,b,c,N):
    time_grid_rungkutta = np.linspace(0,10,N+1)
    delta = time_grid_rungkutta[1]-time_grid_rungkutta[0]
    y = np.array([0.06,2e6])
    k=np.zeros((len(c),2))
    for t in range(0,N):
        for i in range(0,len(c)):
            weight_k = np.zeros(2)
            for j in range(0,len(c)):
                weight_k+=A[i,j]*k[j]
            k[i,:] = rhs(parameters, *y+delta*weight_k, time_grid_rungkutta[t]+c[i]*delta)
        weight_y = np.zeros(2)
        for j in range(0,len(b)):
            weight_y+=b[j]*k[j]
        y+=delta*weight_y
    return y


#RKEuler
c=[0]
b=[1]
A = np.array([[0]])
y_Euler = rk(A,b,c,1000)



#RkHeun
c=[0,1]
b=[1/2,1/2]
A = np.array([
    [0, 0],
    [1, 0],
])

y_Heun = rk(A,b,c,1000)




#RK4
c=[0,1/2,1/2,1]
b=[1/6,1/3,1/3,1/6]
A = np.array([
    [0,   0,   0, 0],
    [1/2, 0,   0, 0],
    [0,   1/2, 0, 0],
    [0,   0,   1, 0],
])
y_RK4 = rk(A,b,c,1000)

# %%
tableaus = {
    "Euler": (np.array([[0]]), [1], [0]),
    "Heun":  (np.array([
    [0, 0],
    [1, 0],
    ]), [1/2, 1/2], [0, 1]),
    "RK4":   (np.array([
    [0,   0,   0, 0],
    [1/2, 0,   0, 0],
    [0,   1/2, 0, 0],
    [0,   0,   1, 0],
    ]), [1/6, 1/3, 1/3, 1/6], [0, 1/2, 1/2, 1]),
}

N_list=[500,1000,2000,4000,8000,16000,32000,64000]
fehler = np.zeros((len(N_list), len(tableaus)))
fehler_sign = np.zeros((len(N_list), len(tableaus)))
nfev = np.zeros((len(N_list), len(tableaus)))

for name, (A_, b_, c_) in tableaus.items():
    for i, N in enumerate(N_list):
        y = rk(A_, b_, c_, N)
        fehler[i, list(tableaus.keys()).index(name)] = abs((y[1] - ref[1]) / ref[1])
        fehler_sign[i, list(tableaus.keys()).index(name)] = (y[1] - ref[1]) / ref[1]
        nfev[i, list(tableaus.keys()).index(name)] = N * len(b_)
    print(f"{name:<12} {abs((y[1] - ref[1]) / ref[1]):>10.2e}")
# %%
import pandas as pd
display(pd.DataFrame(fehler, index=N_list, columns=tableaus.keys()))
display(pd.DataFrame(fehler_sign, index=N_list, columns=tableaus.keys()))
display(pd.DataFrame(nfev, index=N_list, columns=tableaus.keys()))
display(pd.DataFrame(np.log2(fehler[:-1]/fehler[1:]), index=N_list[1:], columns=tableaus.keys()))
# %%

reference_list = {
    "Euler": 1,
    "Heun": 2,
    "RK4": 4,
}

x = np.divide(10, N_list)
fig, ax = plt.subplots(figsize=(6, 4))
for m, name in enumerate(tableaus.keys()):
    ax.loglog(x, fehler[:, m], "o-", label=name)
    p=reference_list[name]
    ax.loglog(x, fehler[7, m] * (x / x[7]) ** p, "--", color="gray", alpha=0.5, label=rf"$\mathcal{{O}}(\Delta t^{p})$")
ax.set_xlabel(r"Schrittweite $\Delta t$")
ax.set_ylabel(r"relativer Fehler in $P(T)$")
ax.set_title("Konvergenz von Euler, Heun und RK4")
ax.legend()
ax.invert_xaxis()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
for m, name in enumerate(tableaus.keys()):
    ax.loglog(nfev[:, m], fehler[:, m], "o-", label=name)
ax.loglog(nfev_scipy, fehler_scipy, "--", label="scipy")
ax.set_xlabel(r"Anzahl $f$-Auswertungen")
ax.set_ylabel(r"relativer Fehler in $P(T)$")
ax.set_title("Kosten gegen Genauigkeit")
ax.legend()
#ax.invert_xaxis()
plt.show()

# %% Scipy Test wie wählen wir die Schrittweite



sol_grob =solve_ivp(f, t_span=(0, 10), y0=[0.06, 2e6],
                    rtol=1e-3, atol=1e-12)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t, sol_grob.y[1], "o-", label="$P(t)$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$P(t)$")
ax.set_title("Scipy")
ax.legend()
plt.show()
'''
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t, sol_grob.y[0], "o-", label="$H(t)$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$H(t)$")
ax.set_title("Scipy")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.legend()
plt.show()
'''
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t[:-1], np.diff(sol_grob.t), "o-", label=r"d$\Delta t$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$\Delta t$")
ax.set_title("Scipy")
ax.legend()
plt.show()
# %%


sol_grob =solve_ivp(f, t_span=(0, 10), y0=[0.06, 2e6],
                    rtol=1e-3, atol=[1e-6, 1e1])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t, sol_grob.y[1], "o-", label="$P(t)$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$P(t)$")
ax.set_title("Scipy")
ax.legend()
plt.show()
'''
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t, sol_grob.y[0], "o-", label="$H(t)$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$H(t)$")
ax.set_title("Scipy")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.legend()
plt.show()
'''
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sol_grob.t[:-1], np.diff(sol_grob.t), "o-", label=r"d$\Delta t$")
ax.set_xlabel(r" $t$")
ax.set_ylabel(r"$\Delta t$")
ax.set_title("Scipy")
ax.legend()
plt.show()
# %%
