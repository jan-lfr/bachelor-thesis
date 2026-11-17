# %% Verteilung von G: linker Tail in gross
import numpy as np
import matplotlib.pyplot as plt

n = int(5e6)                                   # mehr Samples = weniger Zappeln im Tail
y_liste = [0, 4000]
eta = 4000

w = sample(n, np.random.default_rng(42))
g0, g20 = G(*w, 0), G(*w, 20)                  # dieselben Balken, t = 0 und t = 20

x_min, x_max = -3000, 12000                    # nur der linke Rand
bins = np.arange(x_min, x_max + 1, 250)
mitte = 0.5 * (bins[:-1] + bins[1:])

def dichte(g):                                 # Histogramm als Dichte (Flaeche = 1 ueber ALLE Samples)
    h, _ = np.histogram(g, bins=bins)
    return h / (len(g) * np.diff(bins))

def cdf(g, x):                                 # empirische Verteilungsfunktion P(G <= x)
    return np.searchsorted(np.sort(g), x, side="right") / len(g)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# --- links: Dichte ---
ax1.axvspan(-eta, eta, color="tab:red", alpha=0.07, label=rf"Band $y \pm \eta$ um $y=0$")
ax1.step(mitte, dichte(g0),  where="mid", color="tab:blue",   lw=1.2, alpha=0.7, label="t = 0")
ax1.step(mitte, dichte(g20), where="mid", color="tab:orange", lw=2.0, label="t = 20")
for y_s, farbe in zip(y_liste, ["tab:red", "tab:green"]):
    d = dichte(g20)[np.searchsorted(bins, y_s, side="right") - 1]
    ax1.axvline(y_s, color=farbe, ls="--", lw=1.3)
    ax1.plot(y_s + 125, d, "o", color=farbe)
    ax1.annotate(f"y = {y_s}\nDichte ≈ {d:.1e}", (y_s + 125, d), textcoords="offset points",
                 xytext=(10, -28), color=farbe, fontsize=10)
ax1.set_yscale("log"); ax1.set_xlim(x_min, x_max)
ax1.set_xticks(np.arange(-2000, x_max + 1, 2000))
ax1.grid(True, which="major", ls="--", color="#cccccc", lw=0.8)
ax1.grid(True, which="minor", axis="y", ls=":", color="#e5e5e5", lw=0.6)
ax1.set_xlabel("G"); ax1.set_ylabel("Dichte von G (log)")
ax1.set_title("Dichte im linken Tail")
ax1.legend(loc="lower right", fontsize=9)

# --- rechts: Verteilungsfunktion, daraus liest man P(G <= y) direkt ab ---
xs = np.linspace(x_min, x_max, 400)
ax2.axvspan(-eta, eta, color="tab:red", alpha=0.07)
ax2.plot(xs, cdf(g0, xs),  color="tab:blue",   lw=1.2, alpha=0.7, label="t = 0")
ax2.plot(xs, cdf(g20, xs), color="tab:orange", lw=2.0, label="t = 20")
for y_s, farbe in zip(y_liste, ["tab:red", "tab:green"]):
    p = cdf(g20, y_s)
    ax2.axvline(y_s, color=farbe, ls="--", lw=1.3)
    ax2.plot(y_s, p, "o", color=farbe)
    ax2.annotate(f"P(G ≤ {y_s}) ≈ {p:.2e}", (y_s, p), textcoords="offset points",
                 xytext=(10, -14), color=farbe, fontsize=10)
ax2.set_yscale("log"); ax2.set_xlim(x_min, x_max); ax2.set_ylim(1e-7, 1)
ax2.set_xticks(np.arange(-2000, x_max + 1, 2000))
ax2.grid(True, which="major", ls="--", color="#cccccc", lw=0.8)
ax2.grid(True, which="minor", axis="y", ls=":", color="#e5e5e5", lw=0.6)
ax2.set_xlabel("Schwelle x"); ax2.set_ylabel(r"$F_G(x) = P(G \leq x)$ (log)")
ax2.set_title("Verteilungsfunktion im linken Tail")
ax2.legend(loc="lower right", fontsize=9)

fig.suptitle("Corroded beam: Verteilung von G am linken Rand", y=1.0)
fig.tight_layout()
plt.show()

# --- Zahlen zum Ablesen ---
print(f"{'x':>7} {'F_G(x), t=20':>14} {'Dichte, t=20':>14}")
for x in [-4000, -2000, 0, 1000, 2000, 4000, 8000]:
    i = np.searchsorted(bins, x, side="right") - 1
    d = dichte(g20)[i] if 0 <= i < len(mitte) else float("nan")
    print(f"{x:>7} {cdf(g20, x):>14.3e} {d:>14.3e}")



#%%
# %% Verteilung von G: ungestoert gegen Level 0 (gestoert)
import numpy as np
import matplotlib.pyplot as plt

n = int(5e6)
t = 20
y_liste = [0, 4000]
eta_liste = [4000, 1000]                       # Stoerungsbreiten auf Level 0
farben_eta = {4000: "tab:red", 1000: "tab:purple"}

rng = np.random.default_rng(42)
w = sample(n, rng)
u = rng.uniform(-1, 1, n)                      # dieselbe Stoerung fuer alle eta

g = G(*w, t)                                               # ungestoert
g_lvl0 = {eta: G_k(*w, t, 0, u, eta) for eta in eta_liste}  # Level 0: G + eta * u

x_min, x_max = -5000, 12000
bins = np.arange(x_min, x_max + 1, 250)
mitte = 0.5 * (bins[:-1] + bins[1:])

def dichte(werte):
    h, _ = np.histogram(werte, bins=bins)
    return h / (len(werte) * np.diff(bins))

def cdf(werte, x):
    return np.searchsorted(np.sort(werte), x, side="right") / len(werte)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
xs = np.linspace(x_min, x_max, 400)

# --- links: Dichte ---
ax1.step(mitte, dichte(g), where="mid", color="black", lw=2.0, label="G ungestört")
for eta in eta_liste:
    ax1.step(mitte, dichte(g_lvl0[eta]), where="mid", color=farben_eta[eta], lw=1.5,
             label=rf"Level 0, $\eta$ = {eta}")
ax1.set_ylabel("Dichte (log)"); ax1.set_title("Dichte im linken Tail")

# --- rechts: Verteilungsfunktion mit den Werten bei y ---
ax2.plot(xs, cdf(g, xs), color="black", lw=2.0, label="G ungestört")
for eta in eta_liste:
    ax2.plot(xs, cdf(g_lvl0[eta], xs), color=farben_eta[eta], lw=1.5, label=rf"Level 0, $\eta$ = {eta}")
for y_s in y_liste:
    ax2.plot(y_s, cdf(g, y_s), "o", color="black", zorder=5)
    for eta in eta_liste:
        ax2.plot(y_s, cdf(g_lvl0[eta], y_s), "o", color=farben_eta[eta], zorder=5)
ax2.set_ylabel(r"$P(\cdot \leq x)$ (log)"); ax2.set_ylim(1e-7, 1)
ax2.set_title(r"Verteilungsfunktion: Punkte = $p$ (schwarz) und $p_0$ (farbig) bei $y$")

for ax in (ax1, ax2):
    for y_s in y_liste:
        ax.axvline(y_s, color="gray", ls="--", lw=1.2)
        ax.text(y_s, 0.03, f" y = {y_s}", transform=ax.get_xaxis_transform(), ha="left", fontsize=9, color="gray")
    ax.set_yscale("log"); ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(-4000, x_max + 1, 2000))
    ax.grid(True, which="major", ls="--", color="#cccccc", lw=0.8)
    ax.set_xlabel("x")
    ax.legend(loc="lower right", fontsize=9)

fig.suptitle(f"Corroded beam, t = {t}: Was die Störung auf Level 0 mit dem Tail macht", y=1.02)
fig.tight_layout()
plt.show()

# --- Tabelle: p gegen p_0 ---
print(f"{'y':>6} {'p = P(G<=y)':>14}" + "".join(f"{'p_0, eta=' + str(e):>16}{'p_0/p':>8}" for e in eta_liste))
for y_s in y_liste:
    p = cdf(g, y_s)
    zeile = f"{y_s:>6} {p:>14.3e}"
    for eta in eta_liste:
        p0 = cdf(g_lvl0[eta], y_s)
        zeile += f"{p0:>16.3e}{p0 / p:>8.1f}"
    print(zeile)