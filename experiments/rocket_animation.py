# %% Rocket Animation
# Persoenliche Visualisierung des solid rocket engine shell, nicht fuer die Thesis.
# Zeigt, was das Modell physikalisch beschreibt:
#   - Laengsschnitt: Huelle, Treibstoffschicht der Dicke H(t), Gasraum (Farbe = Druck), Duese, Abgasstrahl
#   - Querschnitt:   Ring aus Treibstoff, der von innen nach aussen abbrennt
#   - Zeitreihen:    P(t), H(t), Brennflaeche A_f(t), Gasvolumen V(t)
#
# Laeuft eigenstaendig. Speichert ein GIF nach ergebnisse/rocket_animation.gif
# und zeigt die Animation (in einem eigenen Fenster: `python experiments/rocket_animation.py`).
#
# Steuerung:
#   Leertaste        Play / Pause            Pfeil links/rechts   ein Frame zurueck / vor
#   Pfeil hoch/runter  schneller / langsamer   Shift + Pfeil        grosser Sprung
#   0 oder Pos1      zum Anfang              Regler unten         frei spulen (ziehen oder klicken)

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle, Polygon, Circle, Wedge
from scipy.integrate import solve_ivp

SAVE_GIF = False          # True: zusaetzlich GIF nach ergebnisse/ schreiben (dauert ca. 30 s)
N_FRAMES, FPS = 480, 25   # Frames zum Spulen; GIF nimmt jeden dritten

# %% Modell (gleiche Konstanten und gleiche Zufallsziehung wie in models/solid_rocket_engine_shell.py)
L_SHELL, DS = 1.35, 0.247
A_BURN, N_BURN = 7.9752e-6, 0.4365
GAMMA_FN = 0.6528
H0, P0 = 0.06, 2e6

rng = np.random.default_rng(0)
_Pa0 = rng.normal(9.2e6, 9.2e4, 1)
CSTAR = 1750.0
RHO = float(rng.normal(1690, 33.8, 1)[0])
DT_THROAT = float(rng.normal(0.064, 0.0013, 1)[0])
A_T = np.pi * DT_THROAT**2 / 4
R = DS / 2

A_f = lambda H: np.pi * (DS - 2 * H) * L_SHELL        # Brennflaeche (Innenmantel)
V_gas = lambda H: L_SHELL * np.pi * (R - H) ** 2      # freies Gasvolumen


def rhs(t, y):
    H, P = y
    Pn = P**N_BURN
    return [-A_BURN * Pn, GAMMA_FN**2 * CSTAR * A_T / V_gas(H) * (RHO * A_f(H) / A_T * CSTAR * A_BURN * Pn - P)]


burnout_event = lambda t, y: y[0]
burnout_event.terminal, burnout_event.direction = True, -1
sol = solve_ivp(rhs, (0, 12), [H0, P0], method="DOP853", rtol=1e-10, atol=1e-12,
                dense_output=True, events=burnout_event)
T_BURN = sol.t[-1]
t_all = np.linspace(0, T_BURN, 800)
H_all, P_all = sol.sol(t_all)
H_all = np.clip(H_all, 0, None)
t_fr = np.linspace(0, T_BURN, N_FRAMES)
H_fr, P_fr = sol.sol(t_fr)
H_fr = np.clip(H_fr, 0, None)
print(f"Ausbrand bei t = {T_BURN:.2f} s,  P_max = {P_all.max() / 1e6:.2f} MPa")

# %% Stil
BG, PANEL, INK, MUTED, GRID = "#0e1218", "#151b24", "#e8ecf2", "#8a95a6", "#263040"
STEEL, STEEL_EDGE = "#5d6978", "#9aa7b8"
FUEL, FUEL_EDGE = "#b98a3e", "#e2b866"
C_P, C_H, C_A, C_V = "#ff8a3d", "#e2b866", "#6fb5d3", "#b49cff"
CMAP = mpl.colormaps["inferno"]
P_NORM = mpl.colors.Normalize(vmin=0.0, vmax=P_all.max() * 1.08)

mpl.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 10,
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})

fig = plt.figure(figsize=(14, 7.6), dpi=80)
gs = fig.add_gridspec(2, 4, height_ratios=[1.45, 1], left=0.045, right=0.95, top=0.9, bottom=0.2,
                      hspace=0.32, wspace=0.42)
fig.text(0.045, 0.945, "Solid Rocket Engine Shell", fontsize=19, fontweight="bold")
fig.text(0.985, 0.95, r"$H' = -aP^{\,n}$      $P' = \frac{\Gamma^2 C^* A_t}{V(H)}\left(\frac{\rho A_f(H)}{A_t}C^* a P^{\,n} - P\right)$",
         fontsize=11.5, color=MUTED, ha="right")

# ---------- Laengsschnitt ----------
ax = fig.add_subplot(gs[0, :])
ax.set_aspect(2.0); ax.axis("off")   # radial 2x ueberhoeht, sonst ist die Schicht kaum zu sehen
WALL = 0.012
X_NOZ, X_THR, X_EXIT = L_SHELL, L_SHELL + 0.11, L_SHELL + 0.30
R_THR, R_EXIT = DT_THROAT / 2, 0.085
ax.set_xlim(-0.12, X_EXIT + 0.72); ax.set_ylim(-0.215, 0.215)

gas = Rectangle((0, -R), L_SHELL, 2 * R, fc=CMAP(0.3), ec="none", zorder=1)
noz_gas = Polygon([[X_NOZ, R], [X_THR, R_THR], [X_EXIT, R_EXIT], [X_EXIT, -R_EXIT], [X_THR, -R_THR], [X_NOZ, -R]],
                  closed=True, fc=CMAP(0.3), ec="none", zorder=1)
fuel_top = Rectangle((0, R - H0), L_SHELL, H0, fc=FUEL, ec="none", zorder=2, hatch="////", lw=0)
fuel_bot = Rectangle((0, -R), L_SHELL, H0, fc=FUEL, ec="none", zorder=2, hatch="////", lw=0)
mpl.rcParams["hatch.color"] = "#8f6a2c"; mpl.rcParams["hatch.linewidth"] = 0.7
burn_top, = ax.plot([0, L_SHELL], [R - H0] * 2, color="#ffd27a", lw=2.2, zorder=3, solid_capstyle="butt")
burn_bot, = ax.plot([0, L_SHELL], [-(R - H0)] * 2, color="#ffd27a", lw=2.2, zorder=3, solid_capstyle="butt")
for p_ in (gas, noz_gas, fuel_top, fuel_bot):
    ax.add_patch(p_)

steel = dict(fc=STEEL, ec=STEEL_EDGE, lw=1.0, zorder=4)
ax.add_patch(Rectangle((-WALL, R), L_SHELL + WALL, WALL, **steel))
ax.add_patch(Rectangle((-WALL, -R - WALL), L_SHELL + WALL, WALL, **steel))
ax.add_patch(Rectangle((-WALL, -R - WALL), WALL, 2 * R + 2 * WALL, **steel))
for sgn in (1, -1):
    ax.add_patch(Polygon([[X_NOZ, sgn * (R + WALL)], [X_NOZ, sgn * R], [X_THR, sgn * R_THR], [X_EXIT, sgn * R_EXIT],
                          [X_EXIT, sgn * (R_EXIT + WALL)], [X_THR, sgn * (R_THR + WALL * 1.6)]], closed=True, **steel))

plume_layers = []
for col, alpha, wid in [("#ff5a1f", 0.35, 1.0), ("#ff9a3d", 0.55, 0.72), ("#ffd27a", 0.8, 0.45), ("#fff6e0", 0.95, 0.2)]:
    pl = Polygon([[X_EXIT, 0], [X_EXIT, 0], [X_EXIT, 0]], closed=True, fc=col, ec="none", alpha=alpha, zorder=0)
    ax.add_patch(pl); plume_layers.append((pl, wid))

ann = dict(color=MUTED, fontsize=9)
ax.annotate("Hülle", (0.55, R + WALL), (0.55, R + 0.055), ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), **ann)
ax.annotate("Düsenhals  $A_t$", (X_THR, R_THR + 0.02), (X_THR + 0.02, R + 0.055), ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), **ann)
lab_fuel = ax.text(L_SHELL / 2, R - H0 / 2, "Treibstoff", ha="center", va="center", fontsize=9.5, color="#1b1408",
                   fontweight="bold", zorder=5, bbox=dict(fc=FUEL, ec="none", pad=1.5))
lab_gas = ax.text(L_SHELL / 2, 0, "", ha="center", va="center", fontsize=12, fontweight="bold", zorder=5)
lab_burn = ax.text(0.03, 0, "Brennfläche  $A_f$", fontsize=9, color="#ffd27a", va="bottom", zorder=5)
h_arrow = ax.annotate("", (L_SHELL * 0.82, R), (L_SHELL * 0.82, R - H0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1), zorder=6)
h_text = ax.text(L_SHELL * 0.835, R - H0 / 2, "", fontsize=9.5, va="center", zorder=6, color="#1b1408", fontweight="bold",
                 bbox=dict(fc=FUEL, ec="none", pad=1.5))
clock = ax.text(-0.11, 0.21, "", fontsize=15, fontweight="bold", va="top", family="monospace")
ax.plot([0, L_SHELL], [-R - 0.05] * 2, color=MUTED, lw=0.8, marker="|", ms=7)
ax.text(L_SHELL / 2, -R - 0.062, f"L = {L_SHELL} m      (radial 2x überhöht)", ha="center", va="top", **ann)

cax = fig.add_axes([0.775, 0.585, 0.17, 0.014])
cb = fig.colorbar(mpl.cm.ScalarMappable(P_NORM, CMAP), cax=cax, orientation="horizontal")
cb.outline.set_visible(False); cb.set_ticks([0, 2e6, 4e6, 6e6, 8e6]); cb.set_ticklabels(["0", "2", "4", "6", "8"])
cax.set_title("Druck im Gasraum  [MPa]", fontsize=8.5, color=MUTED, loc="left", fontweight="normal", pad=4)
cax.tick_params(labelsize=8, length=2)

# ---------- Querschnitt ----------
axq = fig.add_subplot(gs[1, 0]); axq.set_aspect("equal"); axq.axis("off")
axq.set_xlim(-0.16, 0.16); axq.set_ylim(-0.16, 0.16)
axq.set_title("Querschnitt", pad=2)
axq.add_patch(Circle((0, 0), R + WALL, **{**steel, "zorder": 1}))
q_fuel = Wedge((0, 0), R, 0, 360, width=H0, fc=FUEL, ec="none", hatch="////", lw=0, zorder=2)
q_gas = Circle((0, 0), R - H0, fc=CMAP(0.3), ec="#ffd27a", lw=2, zorder=3)
q_ghost = Circle((0, 0), R - H0, fc="none", ec=MUTED, lw=0.8, ls=":", zorder=4)
for p_ in (q_fuel, q_gas, q_ghost):
    axq.add_patch(p_)
q_text = axq.text(0, 0, "", ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)
axq.text(0, -0.155, "gepunktet: Anfangszustand", ha="center", fontsize=8, color=MUTED)

# ---------- Zeitreihen ----------
def series(ax_, y, color, title, unit):
    ax_.plot(t_all, y, color=color, lw=1.2, alpha=0.28)
    line, = ax_.plot([], [], color=color, lw=2.2)
    dot, = ax_.plot([], [], "o", color=color, ms=6, mec=BG, mew=1.2)
    ax_.set_title(title, pad=4); ax_.set_xlabel("t  [s]"); ax_.set_xlim(0, T_BURN * 1.02)
    pad = 0.08 * (y.max() - y.min()); ax_.set_ylim(y.min() - pad, y.max() + pad)
    val = ax_.text(0.97, 0.06 if y[-1] > y[0] else 0.9, "", transform=ax_.transAxes, ha="right", fontsize=11,
                   fontweight="bold", color=color, family="monospace")
    return line, dot, val, unit, y

S = [
    series(fig.add_subplot(gs[1, 1]), P_all / 1e6, C_P, "Druck  P(t)", "MPa"),
    series(fig.add_subplot(gs[1, 2]), H_all * 1e3, C_H, "Restdicke  H(t)", "mm"),
]
ax4 = fig.add_subplot(gs[1, 3])
S.append(series(ax4, A_f(H_all), C_A, "Brennfläche und Gasvolumen", "m²"))
ax4.set_ylabel("$A_f$  [m²]", color=C_A); ax4.tick_params(axis="y", colors=C_A)
ax4b = ax4.twinx(); ax4b.grid(False); ax4b.spines["right"].set_visible(True); ax4b.spines["right"].set_color(GRID)
ax4b.plot(t_all, V_gas(H_all) * 1e3, color=C_V, lw=1.2, alpha=0.28)
v_line, = ax4b.plot([], [], color=C_V, lw=2.2)
ax4b.set_ylabel("V  [Liter]", color=C_V); ax4b.tick_params(axis="y", colors=C_V)


# %% Animation
def update(i):
    t, H, P = t_fr[i], H_fr[i], P_fr[i]
    col = CMAP(P_NORM(P))
    gas.set_fc(col); noz_gas.set_fc(col); q_gas.set_fc(col)
    fuel_top.set_y(R - H); fuel_top.set_height(H); fuel_bot.set_height(H)
    for ln, s_ in ((burn_top, 1), (burn_bot, -1)):
        ln.set_ydata([s_ * (R - H)] * 2)
    lab_fuel.set_position((L_SHELL / 2, R - H / 2)); lab_fuel.set_visible(H > 0.012)
    lab_gas.set_text(f"{P / 1e6:.2f} MPa"); lab_gas.set_color("white" if P_NORM(P) < 0.75 else "#1b1408")
    lab_burn.set_position((0.03, -(R - H) + 0.006))
    h_arrow.xy = (L_SHELL * 0.82, R); h_arrow.set_position((L_SHELL * 0.82, R - H))
    h_arrow.set_visible(H > 0.004)
    h_text.set_position((L_SHELL * 0.835, R - H / 2 if H > 0.012 else R + 0.03)); h_text.set_text(f"H = {H * 1e3:.1f} mm")
    clock.set_text(f"t = {t:5.2f} s")

    flick = 1 + 0.05 * np.sin(i * 1.9) + 0.03 * np.sin(i * 3.7)
    length = 0.66 * P / P_all.max() * flick
    for pl, wid in plume_layers:
        r0 = R_EXIT * wid
        pl.set_xy([[X_EXIT, r0], [X_EXIT + length * (0.35 + 0.65 * wid) * 0.4, r0 * 1.25],
                   [X_EXIT + length * (0.35 + 0.65 * wid), 0],
                   [X_EXIT + length * (0.35 + 0.65 * wid) * 0.4, -r0 * 1.25], [X_EXIT, -r0]])

    q_fuel.set_width(max(H, 1e-6)); q_gas.set_radius(R - H)
    q_text.set_text(f"{100 * (1 - (R**2 - (R - H) ** 2) / (R**2 - (R - H0) ** 2)):.0f} %\nverbrannt")
    q_text.set_color("white" if P_NORM(P) < 0.75 else "#1b1408")

    m = t_all <= t
    for line, dot, val, unit, y in S:
        line.set_data(t_all[m], y[m]); dot.set_data([t], [np.interp(t, t_all, y)])
        val.set_text(f"{np.interp(t, t_all, y):.2f} {unit}")
    v_line.set_data(t_all[m], V_gas(H_all[m]) * 1e3)
    return []


# GIF zuerst (ohne Bedienelemente im Bild)
if SAVE_GIF:
    out = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "experiments").is_dir()), Path.cwd()) / "ergebnisse"
    out.mkdir(exist_ok=True)
    FuncAnimation(fig, update, frames=range(0, N_FRAMES, 3), blit=False).save(
        out / "rocket_animation.gif", writer=PillowWriter(fps=FPS))
    print("gespeichert:", out / "rocket_animation.gif")

# %% Player: Play/Pause, Spulen, Tempo
for key in ("back", "forward", "home", "pan", "save", "fullscreen", "yscale", "xscale", "grid"):
    mpl.rcParams[f"keymap.{key}"] = []          # Standard-Tastenkuerzel von matplotlib freiraeumen

SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
state = {"pos": 0.0, "playing": True, "speed": 3}   # speed = Index in SPEEDS (1.0 = Echtzeit)
FRAMES_PER_TICK = (N_FRAMES - 1) / (T_BURN * FPS)   # bei 1x laeuft die Animation in Echtzeit

ax_sl = fig.add_axes([0.30, 0.066, 0.50, 0.022])
slider = Slider(ax_sl, "", 0.0, T_BURN, valinit=0.0, color=C_P, track_color=GRID, handle_style=dict(facecolor=INK, edgecolor=BG, size=13))
slider.valtext.set_visible(False)
ax_sl.set_facecolor(BG)
for tk in np.arange(0, T_BURN, 1.0):
    ax_sl.text(tk, -0.9, f"{tk:.0f}", ha="center", va="top", fontsize=8, color=MUTED, clip_on=False)
ax_sl.text(T_BURN, -0.9, "s", ha="center", va="top", fontsize=8, color=MUTED, clip_on=False)


def make_button(x, w, label):
    b = Button(fig.add_axes([x, 0.045, w, 0.048]), label, color=PANEL, hovercolor=GRID)
    b.label.set_color(INK); b.label.set_fontsize(11); b.label.set_fontweight("bold")
    for sp in b.ax.spines.values():
        sp.set_color(GRID)
    return b


b_start, b_back, b_play, b_fwd = make_button(0.045, 0.04, "|<"), make_button(0.09, 0.04, "<"), make_button(0.135, 0.06, "Pause"), make_button(0.20, 0.04, ">")
b_slow, b_fast = make_button(0.835, 0.035, "-"), make_button(0.915, 0.035, "+")
speed_text = fig.text(0.8925, 0.069, "", ha="center", va="center", fontsize=10.5, fontweight="bold", family="monospace")
fig.text(0.045, 0.012, "Leertaste: Play/Pause    Pfeile links/rechts: Frame zurück/vor (Shift: Sprung)    Pfeile hoch/runter: Tempo    0: Anfang",
         fontsize=8.5, color=MUTED)


def show(pos):
    """Springt zu Frame-Position pos (float) und zeichnet neu."""
    state["pos"] = float(np.clip(pos, 0, N_FRAMES - 1))
    slider.eventson = False
    slider.set_val(t_fr[int(round(state["pos"]))])
    slider.eventson = True
    update(int(round(state["pos"])))
    fig.canvas.draw_idle()


def set_playing(flag):
    if flag and state["pos"] >= N_FRAMES - 1:    # am Ende: Play startet von vorn
        state["pos"] = 0.0
    state["playing"] = flag
    b_play.label.set_text("Pause" if flag else "Play")
    fig.canvas.draw_idle()


def set_speed(idx):
    state["speed"] = int(np.clip(idx, 0, len(SPEEDS) - 1))
    speed_text.set_text(f"{SPEEDS[state['speed']]:g}x")
    fig.canvas.draw_idle()


def step(n):
    set_playing(False)
    show(state["pos"] + n)


def tick():
    if not state["playing"]:
        return
    nxt = state["pos"] + SPEEDS[state["speed"]] * FRAMES_PER_TICK
    if nxt >= N_FRAMES - 1:
        show(N_FRAMES - 1)
        set_playing(False)
    else:
        show(nxt)


def on_key(event):
    k = event.key
    if k == " ":
        set_playing(not state["playing"])
    elif k in ("right", "left"):
        step(1 if k == "right" else -1)
    elif k in ("shift+right", "shift+left"):
        step(N_FRAMES // 20 * (1 if k == "shift+right" else -1))
    elif k in ("up", "down"):
        set_speed(state["speed"] + (1 if k == "up" else -1))
    elif k in ("0", "home"):
        show(0)


def on_slider(val):
    state["pos"] = float(np.searchsorted(t_fr, val).clip(0, N_FRAMES - 1))
    update(int(state["pos"]))
    fig.canvas.draw_idle()


slider.on_changed(on_slider)
b_start.on_clicked(lambda e: show(0))
b_back.on_clicked(lambda e: step(-1))
b_fwd.on_clicked(lambda e: step(1))
b_play.on_clicked(lambda e: set_playing(not state["playing"]))
b_slow.on_clicked(lambda e: set_speed(state["speed"] - 1))
b_fast.on_clicked(lambda e: set_speed(state["speed"] + 1))
fig.canvas.mpl_connect("key_press_event", on_key)

set_speed(state["speed"])
show(0)
timer = fig.canvas.new_timer(interval=int(1000 / FPS))
timer.add_callback(tick)
timer.start()

plt.show()
