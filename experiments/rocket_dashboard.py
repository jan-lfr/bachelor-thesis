# %% Rocket Dashboard
# Persoenliche Uebersicht: alles, was bisher am solid rocket engine shell gemacht wurde.
# Nicht fuer die Thesis. Laeuft eigenstaendig (importiert models/solid_rocket_engine_shell.py
# bewusst NICHT, weil das beim Import alle Experimente neu rechnet).
#
# Panels:
#   A  Loesung P(t), H(t)                       D  Empirische Ordnung
#   B  RK45: Schrittweite vs. Stabilitaetsgrenze E  Work-Precision (Fehler gegen f-Auswertungen)
#   C  Konvergenz Euler / Heun / RK4             F  Vorzeichen des Fehlers (Heun-Knick)
#
# Laufzeit: ca. 10 bis 20 Sekunden.

from pathlib import Path
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# %% Modell (gleiche Konstanten und gleiche Zufallsziehung wie in models/solid_rocket_engine_shell.py)
L_SHELL, DS = 1.35, 0.247          # Laenge, Innendurchmesser [m]
A_BURN, N_BURN = 7.9752e-6, 0.4365  # Brennrate a * P^n
GAMMA_FN = 0.6528                   # im Modell-File "tau"
H0, P0, T_END = 0.06, 2e6, 10.0

rng = np.random.default_rng(0)      # gleiche Reihenfolge wie im Modell-File -> gleiche Parameter
_Pa0 = rng.normal(9.2e6, 9.2e4, 1)
CSTAR = 1750.0
RHO = float(rng.normal(1690, 33.8, 1)[0])
DT_THROAT = float(rng.normal(0.064, 0.0013, 1)[0])

A_T = np.pi * DT_THROAT**2 / 4
K_RELAX = GAMMA_FN**2 * CSTAR * A_T  # Vorfaktor Gamma^2 C* A_t


def rhs(t, y):
    """Rechte Seite, mit Python-Floats (schnell in der Zeitschleife)."""
    H, P = y[0], y[1]
    Pn = P**N_BURN
    A_f = np.pi * (DS - 2 * H) * L_SHELL
    V = L_SHELL * np.pi * (DS / 2 - H) ** 2
    return np.array([-A_BURN * Pn, K_RELAX / V * (RHO * A_f / A_T * CSTAR * A_BURN * Pn - P)])


def lam_P(H, P):
    """Eigenwert der P-Gleichung, d(P')/dP. Bestimmt die Stabilitaetsgrenze."""
    A_f = np.pi * (DS - 2 * H) * L_SHELL
    V = L_SHELL * np.pi * (DS / 2 - H) ** 2
    return K_RELAX / V * (RHO * A_f / A_T * CSTAR * A_BURN * N_BURN * P ** (N_BURN - 1) - 1)


# %% Explizites Runge-Kutta ueber Butcher-Tableau (gleiche Idee wie rk(A, b, c, N))
TABLEAUS = {
    "Euler": (np.array([[0.0]]), np.array([1.0]), np.array([0.0])),
    "Heun": (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.5, 0.5]), np.array([0.0, 1.0])),
    "RK4": (
        np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.0, 0]]),
        np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        np.array([0, 0.5, 0.5, 1.0]),
    ),
}
ORDER = {"Euler": 1, "Heun": 2, "RK4": 4}


def rk(A, b, c, N):
    h = T_END / N
    y = np.array([H0, P0])
    s = len(b)
    k = np.zeros((s, 2))
    for step in range(N):
        t = step * h
        for i in range(s):
            k[i] = rhs(t + c[i] * h, y + h * (A[i, :i] @ k[:i]))
        y = y + h * (b @ k)
    return y


# %% Rechnen
t0 = time.time()
print("Referenz (DOP853, rtol 1e-13) ...")
ref_sol = solve_ivp(rhs, (0, T_END), [H0, P0], method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True)
ref = ref_sol.y[:, -1]
rel_err = lambda y: (y[1] - ref[1]) / ref[1]

N_LIST = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000])
DTS = T_END / N_LIST
err_signed = {}
nfev = {}
for name, (A, b, c) in TABLEAUS.items():
    print(f"{name} ueber {len(N_LIST)} Schrittweiten ...")
    err_signed[name] = np.array([rel_err(rk(A, b, c, int(N))) for N in N_LIST])
    nfev[name] = N_LIST * len(b)
err = {n: np.abs(e) for n, e in err_signed.items()}
p_emp = {n: np.log2(e[:-1] / e[1:]) for n, e in err.items()}

print("scipy RK45 ueber rtol ...")
RTOLS = np.logspace(-3, -12, 10)
sc_err, sc_nfev = [], []
for rtol in RTOLS:
    s_ = solve_ivp(rhs, (0, T_END), [H0, P0], rtol=rtol, atol=1e-12)
    sc_err.append(abs(rel_err(s_.y[:, -1])))
    sc_nfev.append(s_.nfev)
sc_err, sc_nfev = np.array(sc_err), np.array(sc_nfev)

sol_grob = solve_ivp(rhs, (0, T_END), [H0, P0], rtol=1e-3, atol=1e-12)
sol_fein = solve_ivp(rhs, (0, T_END), [H0, P0], rtol=1e-8, atol=1e-12)

tt = np.linspace(0, T_END, 1500)
H_ref, P_ref = ref_sol.sol(tt)
h_stab = 3.31 / np.abs(lam_P(H_ref, P_ref))  # DOPRI5: stabil fuer h*lambda > -3.31
burnout = tt[np.argmax(H_ref <= 0)] if np.any(H_ref <= 0) else None
print(f"fertig nach {time.time() - t0:.1f} s")

# %% Stil
INK, MUTED, GRID, PAPER, PANEL = "#1c2330", "#6b7686", "#e3e7ee", "#f6f7f9", "#ffffff"
COL = {"Euler": "#c2553a", "Heun": "#d39a1f", "RK4": "#1d6a8a", "scipy": "#5b4b9a"}
MARK = {"Euler": "o", "Heun": "s", "RK4": "^", "scipy": "D"}

mpl.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.labelcolor": MUTED, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.facecolor": PANEL, "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 1.9, "lines.markersize": 5.5,
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def tag(ax, letter):
    ax.text(-0.02, 1.085, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11.5, fontweight="bold", color=MUTED)


def note(ax, text, xy=(0.03, 0.05), **kw):
    ax.text(*xy, text, transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom", linespacing=1.45, **kw)


# %% Dashboard
fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.86, bottom=0.075, hspace=0.42, wspace=0.31)

fig.text(0.055, 0.945, "Solid Rocket Engine Shell", fontsize=21, fontweight="bold", color=INK)
fig.text(0.055, 0.905,
         r"$H' = -aP^{\,n}$,   $P' = \frac{\Gamma^2 C^* A_t}{V(H)}\left(\frac{\rho A_f(H)}{A_t}\,C^* a P^{\,n} - P\right)$"
         r"      $H(0)=0.06$,  $P(0)=2\cdot10^6$,  $T=10$",
         fontsize=11.5, color=MUTED)
summary = "   ·   ".join(
    [f"{n}: Ordnung {np.median(p_emp[n]):.2f}" for n in TABLEAUS]
    + [f"RK45 min. {sc_nfev.min()} f-Auswertungen"]
)
fig.text(0.985, 0.945, summary, fontsize=10.5, color=INK, ha="right", va="bottom")
fig.text(0.985, 0.912, f"Referenz DOP853, rtol 1e-13   ·   P(T) = {ref[1]:.6e} Pa", fontsize=9.5, color=MUTED, ha="right")

# A  Loesung
ax = fig.add_subplot(gs[0, 0]); tag(ax, "A")
ax.plot(tt, P_ref / 1e6, color=COL["RK4"])
ax.set_title("Lösung: Druck und Restdicke")
ax.set_xlabel("t  [s]"); ax.set_ylabel("P(t)  [MPa]", color=COL["RK4"])
ax.tick_params(axis="y", colors=COL["RK4"])
ax2 = ax.twinx()
ax2.plot(tt, H_ref * 1000, color=COL["Euler"], ls="--")
ax2.set_ylabel("H(t)  [mm]", color=COL["Euler"]); ax2.tick_params(axis="y", colors=COL["Euler"])
ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(GRID); ax2.grid(False)
ax2.axhline(0, color=COL["Euler"], lw=0.8, alpha=0.5)
if burnout is not None:
    ax.axvline(burnout, color=MUTED, lw=1, ls=":")
    ax.text(burnout - 0.12, 0.06, f"H = 0 bei t ≈ {burnout:.1f}", transform=ax.get_xaxis_transform(),
            ha="right", fontsize=8.8, color=MUTED)
note(ax, "P springt sofort aufs\nGleichgewicht und steigt\ndann, weil die Brenn-\nfläche wächst.", xy=(0.04, 0.40))

# B  Schrittweite RK45 vs Stabilitaet
ax = fig.add_subplot(gs[0, 1]); tag(ax, "B")
ax.fill_between(tt, h_stab, 10, color=COL["Euler"], alpha=0.08, lw=0)
ax.plot(tt, h_stab, color=COL["Euler"], lw=1.4, ls="--", label=r"Stabilitätsgrenze  $3.31/|\lambda(t)|$")
ax.plot(sol_grob.t[:-1], np.diff(sol_grob.t), "o-", color=COL["scipy"], ms=3.2, lw=1.2,
        label=f"RK45, rtol 1e-3  ({sol_grob.nfev} f-Ausw.)")
ax.plot(sol_fein.t[:-1], np.diff(sol_fein.t), "-", color=COL["RK4"], lw=1.2,
        label=f"RK45, rtol 1e-8  ({sol_fein.nfev} f-Ausw.)")
ax.set_yscale("log"); ax.set_ylim(np.diff(sol_fein.t).min() * 0.5, h_stab.max() * 3)
ax.set_title("RK45: Schrittweite gegen Stabilitätsgrenze")
ax.set_xlabel("t  [s]"); ax.set_ylabel(r"Schrittweite $h$")
ax.legend(loc="upper left")
note(ax, "Grobe Toleranz läuft an die Grenze:\nStabilität begrenzt h, nicht Genauigkeit.", xy=(0.40, 0.05))

# C  Konvergenz
ax = fig.add_subplot(gs[0, 2]); tag(ax, "C")
for name in TABLEAUS:
    p = ORDER[name]
    ax.loglog(DTS, err[name][-1] * (DTS / DTS[-1]) ** p, color=COL[name], lw=1, ls=":", alpha=0.8)
    ax.loglog(DTS, err[name], MARK[name] + "-", color=COL[name], label=rf"{name}   (gepunktet: $\Delta t^{{{p}}}$)")
ax.invert_xaxis()
ax.set_title("Konvergenz bei fester Schrittweite")
ax.set_xlabel(r"Schrittweite $\Delta t$   (feiner →)"); ax.set_ylabel("rel. Fehler in P(T)")
ax.legend(loc="lower left")

# D  Empirische Ordnung
ax = fig.add_subplot(gs[1, 0]); tag(ax, "D")
for name in TABLEAUS:
    ax.axhline(ORDER[name], color=COL[name], lw=1, ls=":", alpha=0.8)
    ax.semilogx(DTS[1:], p_emp[name], MARK[name] + "-", color=COL[name], label=name)
ax.invert_xaxis(); ax.set_ylim(0, 5.2)
ax.set_title(r"Empirische Ordnung  $p \approx \log_2\, e(\Delta t)/e(\Delta t/2)$")
ax.set_xlabel(r"Schrittweite $\Delta t$   (feiner →)"); ax.set_ylabel("p")
ax.legend(loc="upper right", ncol=3)
note(ax, "Heun springt beim Vorzeichen-\nwechsel (abgeschnitten), dann\nlangsam gegen 2. RK4 fällt am\nEnde ab: Referenz-Floor.", xy=(0.47, 0.42))

# E  Work-Precision
ax = fig.add_subplot(gs[1, 1]); tag(ax, "E")
for name in TABLEAUS:
    ax.loglog(nfev[name], err[name], MARK[name] + "-", color=COL[name], label=name)
ax.loglog(sc_nfev, sc_err, MARK["scipy"] + "--", color=COL["scipy"], label="scipy RK45 (rtol 1e-3 … 1e-12)")
ax.axvline(sc_nfev.min(), color=COL["scipy"], lw=1, ls=":", alpha=0.7)
ax.text(sc_nfev.min() * 0.93, 0.5, f"RK45 nie unter {sc_nfev.min()}", transform=ax.get_xaxis_transform(),
        ha="right", va="center", fontsize=8.8, color=COL["scipy"], rotation=90)
ax.set_title("Kosten gegen Genauigkeit")
ax.set_xlabel("Anzahl f-Auswertungen"); ax.set_ylabel("rel. Fehler in P(T)")
ax.legend(loc="lower left")
note(ax, "Weiter links unten = besser.", xy=(0.97, 0.90), ha="right")

# F  Vorzeichen des Fehlers
ax = fig.add_subplot(gs[1, 2]); tag(ax, "F")
lin = 1e-13
ax.axhline(0, color=INK, lw=0.8)
for name in TABLEAUS:
    ax.plot(DTS, err_signed[name], MARK[name] + "-", color=COL[name], label=name)
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=lin, linscale=0.6); ax.invert_xaxis()
ax.set_yticks([-1e-4, -1e-7, -1e-10, 0, 1e-10, 1e-7, 1e-4]); ax.set_ylim(-3e-3, 3e-3); ax.minorticks_off()
ax.set_title("Fehler mit Vorzeichen")
ax.set_xlabel(r"Schrittweite $\Delta t$   (feiner →)"); ax.set_ylabel(r"$(P_h(T) - P(T))\,/\,P(T)$")
ax.legend(loc="upper right")
flips = [n for n in TABLEAUS if np.any(np.sign(err_signed[n][:-1]) != np.sign(err_signed[n][1:]))]
if flips:
    note(ax, "Vorzeichenwechsel bei: " + ", ".join(flips) + ".\nDort heben sich Fehlerterme auf,\ndie Ordnung wirkt schief.",
         xy=(0.03, 0.80))

out = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "experiments").is_dir()), Path.cwd()) / "ergebnisse"
out.mkdir(exist_ok=True)
fig.savefig(out / "rocket_dashboard.png")
print("gespeichert:", out / "rocket_dashboard.png")
plt.show()

# %% Zahlen als Tabelle (zum Nachschauen)
print(f"\n{'dt':>10} | " + " | ".join(f"{n:>20}" for n in TABLEAUS))
for i, dt in enumerate(DTS):
    row = [f"{err[n][i]:>10.2e} {(p_emp[n][i-1] if i else float('nan')):>9.2f}" for n in TABLEAUS]
    print(f"{dt:>10.6f} | " + " | ".join(row))
print(f"\n{'rtol':>8} {'Fehler P':>10} {'nfev':>6}")
for r, e, n in zip(RTOLS, sc_err, sc_nfev):
    print(f"{r:>8.0e} {e:>10.2e} {n:>6d}")
