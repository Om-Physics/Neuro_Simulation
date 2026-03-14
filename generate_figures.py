"""
generate_figures.py
===================
Self-contained script that runs every biophysical simulation and
produces 10 publication-quality figures + 1 summary dashboard.

Figures
-------
  01  HH Action Potential — waveform, gating variables, ion currents
  02  HH Phase Plane      — V–n nullclines, limit cycle, fixed points
  03  f-I Curves          — HH vs LIF vs AdEx-RS vs AdEx-FS
  04  AdEx Firing Patterns— RS, IB, CH, FS, LTS with w(t)
  05  Synapse Kinetics    — AMPA / NMDA / GABA-A / GABA-B conductances
  06  NMDA Mg²⁺ Block     — B(V) + I(V) at multiple conductance levels
  07  STDP                — LTP/LTD window + weight convergence distribution
  08  E/I Network         — Raster, population rate, LFP proxy
  09  ISI Analysis        — ISI histogram, Poincaré plot, CV-ISI population
  10  LFP Power Spectrum  — Welch PSD with labelled frequency bands
  11  Summary Dashboard   — 12-panel overview of all major results

Run:   python generate_figures.py
Output: ./figures/fig_01_*.png … fig_11_*.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy import signal as sp_signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ── Project modules ───────────────────────────────────────────────────────
from neurons.hodgkin_huxley  import HodgkinHuxley
from neurons.integrate_fire  import (LeakyIntegrateAndFire,
                                     AdaptiveExponentialIF)
from synapses.synapse        import (AMPASynapse, NMDASynapse,
                                     GABAASynapse, GABABSynapse)
from synapses.plasticity     import STDPRule

# ── Output directory ──────────────────────────────────────────────────────
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── Global matplotlib style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "savefig.bbox":       "tight",
    "savefig.dpi":        200,
})

PAL = dict(V="#2c7bb6", m="#d7191c", h="#1a9641", n="#ff7f00",
           INa="#d7191c", IK="#2c7bb6", IL="#888888",
           E="#e41a1c", I="#377eb8", LFP="#4d4d4d", rate="#ff7f00",
           AMPA="#1b7837", NMDA="#762a83", GABAA="#2166ac", GABAB="#d6604d",
           w="#6a3d9a")

def _label(ax, txt, x=-0.16, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="right")

def _save(name):
    p = OUTDIR / name
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close("all")
    kb = p.stat().st_size // 1024
    print(f"  ✓  {name}  ({kb} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 1 — Hodgkin-Huxley full run
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/7] Hodgkin-Huxley …")
hh = HodgkinHuxley()
dt_hh, T_hh = 0.025, 120.0
n_hh = int(T_hh / dt_hh)
I_hh = np.zeros(n_hh)
I_hh[int(10/dt_hh):] = 10.0   # 10 µA/cm² from t=10 ms
hh_r = hh.simulate_detailed(I_hh, dt=dt_hh)
print(f"    HH: {len(hh_r['spikes'])} spikes, {hh_r['firing_rate_hz']:.1f} Hz")

# ── Fig 1 — HH Action Potential ─────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)

ax_V   = fig.add_subplot(gs[0, :])
ax_m   = fig.add_subplot(gs[1, 0])
ax_h   = fig.add_subplot(gs[1, 1])
ax_n   = fig.add_subplot(gs[2, 0])
ax_cur = fig.add_subplot(gs[2, 1])
ax_I   = fig.add_subplot(gs[3, :])

t = hh_r["t"]; V = hh_r["V"]

# Membrane voltage
ax_V.plot(t, V, color=PAL["V"], lw=1.8)
for sp in hh_r["spikes"]:
    ax_V.axvline(sp, color=PAL["V"], alpha=0.25, lw=0.8, ls="--")
ax_V.axhline(hh.E_Na, color=PAL["INa"], ls=":", lw=0.8, alpha=0.6, label=f"$E_{{Na}}$ = {hh.E_Na} mV")
ax_V.axhline(hh.E_K,  color=PAL["IK"],  ls=":", lw=0.8, alpha=0.6, label=f"$E_{{K}}$ = {hh.E_K} mV")
ax_V.axhline(hh.E_L,  color=PAL["IL"],  ls=":", lw=0.8, alpha=0.6, label=f"$E_{{L}}$ = {hh.E_L} mV")
ax_V.set_ylabel("V (mV)"); ax_V.set_title("Membrane Potential")
ax_V.legend(loc="upper right", frameon=False, fontsize=8)
_label(ax_V, "A")

# Gating variables
ax_m.plot(t, hh_r["m"], color=PAL["m"])
ax_m.set_ylabel("m (Na act.)"); ax_m.set_title("Na⁺ Activation Gate  m(t)")
_label(ax_m, "B")

ax_h.plot(t, hh_r["h"], color=PAL["h"])
ax_h.set_ylabel("h (Na inact.)"); ax_h.set_title("Na⁺ Inactivation Gate  h(t)")
_label(ax_h, "C")

ax_n.plot(t, hh_r["n"], color=PAL["n"])
ax_n.set_ylabel("n (K act.)"); ax_n.set_title("K⁺ Activation Gate  n(t)")
ax_n.set_xlabel("Time (ms)")
_label(ax_n, "D")

# Ion currents
ax_cur.plot(t, hh_r["I_Na"], color=PAL["INa"], label="$I_{Na}$", lw=1.4)
ax_cur.plot(t, hh_r["I_K"],  color=PAL["IK"],  label="$I_K$",   lw=1.4)
ax_cur.plot(t, hh_r["I_L"],  color=PAL["IL"],  label="$I_L$",   lw=1.0, ls="--")
ax_cur.axhline(0, color="k", lw=0.6, ls=":")
ax_cur.set_ylabel("Current (µA/cm²)"); ax_cur.set_title("Ion Currents")
ax_cur.legend(frameon=False); ax_cur.set_xlabel("Time (ms)")
_label(ax_cur, "E")

# Stimulus
ax_I.plot(t, I_hh, color="#333333", lw=1.2)
ax_I.set_ylabel("$I_{ext}$ (µA/cm²)"); ax_I.set_xlabel("Time (ms)")
ax_I.set_title("Injected Current")
ax_I.set_ylim(-1, 13)
_label(ax_I, "F")

fig.suptitle("Figure 1 — Hodgkin-Huxley Action Potential Dynamics",
             fontsize=13, fontweight="bold", y=1.01)
_save("fig_01_hh_action_potential.png")

# ── Fig 2 — Phase Plane ──────────────────────────────────────────────────
print("[2/7] HH Phase Plane …")
V_range = np.linspace(-80, 50, 300)
nullc = hh.nullclines(V_range)

# For the V-nullcline in (V, n) subspace: solve dV/dt=0 → n = f(V)
# I_Na=g_Na*m_inf³*h_inf*(V-E_Na), I_K=g_K*n⁴*(V-E_K), I_L=g_L*(V-E_L)
# dV/dt=0: I_ext = I_Na + I_K + I_L → solve n analytically
I_stim = 10.0
n_nullcline_V = np.zeros_like(V_range)
for i, Vv in enumerate(V_range):
    m_inf = nullc["m_inf"][i]; h_inf = nullc["h_inf"][i]
    I_Na_leak = hh.g_Na * (m_inf**3) * h_inf * (Vv - hh.E_Na)
    I_L_leak  = hh.g_L * (Vv - hh.E_L)
    num = (I_stim - I_Na_leak - I_L_leak) / (hh.g_K * (Vv - hh.E_K + 1e-9))
    n4 = max(0, num)
    n_nullcline_V[i] = n4**0.25

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# V–n nullclines + limit cycle
ax = axes[0]
ax.plot(V_range, n_nullcline_V,  color=PAL["V"],  lw=2.0, label="V-nullcline ($\\dot{V}=0$)")
ax.plot(V_range, nullc["n_inf"], color=PAL["n"],  lw=2.0, label="n-nullcline ($\\dot{n}=0$)")
# Limit cycle (V vs n during spikes)
mask = hh_r["t"] > 15
ax.plot(hh_r["V"][mask], hh_r["n"][mask], color="gray", lw=0.8, alpha=0.7, label="Trajectory")
ax.set_xlabel("Membrane voltage V (mV)"); ax.set_ylabel("K⁺ activation n")
ax.set_title("Phase Plane: V–n nullclines + Limit Cycle")
ax.legend(frameon=False, fontsize=8)
ax.set_xlim(-85, 55); ax.set_ylim(-0.05, 0.85)
_label(ax, "A")

# Steady-state gating curves vs V
ax2 = axes[1]
ax2.plot(V_range, nullc["m_inf"], color=PAL["m"],  lw=2, label="$m_\\infty(V)$")
ax2.plot(V_range, nullc["h_inf"], color=PAL["h"],  lw=2, label="$h_\\infty(V)$")
ax2.plot(V_range, nullc["n_inf"], color=PAL["n"],  lw=2, label="$n_\\infty(V)$")
ax2.axvline(-65, color="gray", ls="--", lw=1, label="$V_{rest}$")
ax2.set_xlabel("Membrane voltage V (mV)")
ax2.set_ylabel("Steady-state activation")
ax2.set_title("Gating Variable Steady-States $x_\\infty(V)$")
ax2.legend(frameon=False); ax2.set_xlim(-85, 55)
_label(ax2, "B")

fig.suptitle("Figure 2 — HH Phase Plane Analysis", fontsize=13, fontweight="bold")
_save("fig_02_phase_plane.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 2 — f-I curves
# ═══════════════════════════════════════════════════════════════════════════
print("[3/7] f-I Curves …")
T_fi = 800.0; dt_fi = 0.1; n_fi = int(T_fi / dt_fi)

I_hh_range = np.linspace(0, 40, 40)
f_hh = []
for I_val in I_hh_range:
    nn = HodgkinHuxley()
    I_arr = np.zeros(int(500/0.025)); I_arr[int(50/0.025):] = I_val
    r = nn.simulate(I_arr, dt=0.025)
    f_hh.append(r["firing_rate_hz"])

I_lif_range = np.linspace(0, 700, 40)
f_lif, f_adex_rs, f_adex_fs = [], [], []
for I_val in I_lif_range:
    I_arr = np.full(n_fi, I_val)
    nn = LeakyIntegrateAndFire(); r = nn.simulate(I_arr, dt=dt_fi)
    f_lif.append(r["firing_rate_hz"])
    nn2 = AdaptiveExponentialIF.from_preset("RS"); r2 = nn2.simulate(I_arr, dt=dt_fi)
    f_adex_rs.append(r2["firing_rate_hz"])
    nn3 = AdaptiveExponentialIF.from_preset("FS"); r3 = nn3.simulate(I_arr, dt=dt_fi)
    f_adex_fs.append(r3["firing_rate_hz"])

# ── Fig 3 — f-I Curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(I_hh_range, f_hh, color=PAL["V"], lw=2, marker="o", ms=4, label="HH (µA/cm²)")
rheo_idx = next((i for i,f in enumerate(f_hh) if f>1), None)
if rheo_idx:
    ax.axvline(I_hh_range[rheo_idx], color=PAL["V"], ls="--", lw=1, alpha=0.6,
               label=f"Rheobase ≈ {I_hh_range[rheo_idx]:.1f} µA/cm²")
ax.set_xlabel("$I_{ext}$ (µA/cm²)"); ax.set_ylabel("Firing rate (Hz)")
ax.set_title("Hodgkin-Huxley f-I Curve"); ax.legend(frameon=False)
_label(ax, "A")

ax2 = axes[1]
ax2.plot(I_lif_range, f_lif,      color="#1b7837", lw=2, label="LIF")
ax2.plot(I_lif_range, f_adex_rs,  color="#762a83", lw=2, label="AdEx-RS (adapted)")
ax2.plot(I_lif_range, f_adex_fs,  color="#d6604d", lw=2, label="AdEx-FS (non-adapted)")
ax2.set_xlabel("$I_{ext}$ (pA)"); ax2.set_ylabel("Firing rate (Hz)")
ax2.set_title("LIF vs AdEx f-I Curves"); ax2.legend(frameon=False)
_label(ax2, "B")

fig.suptitle("Figure 3 — Frequency–Current (f-I) Curves: Model Comparison",
             fontsize=13, fontweight="bold")
_save("fig_03_fI_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 3 — AdEx firing patterns
# ═══════════════════════════════════════════════════════════════════════════
print("[4/7] AdEx Firing Patterns …")
T_adex = 1000.0; dt_adex = 0.1; n_adex = int(T_adex / dt_adex)
# Step current starts at t=100 ms; tuned to physiological rates
preset_cfg = {
    "RS":  ("Regular Spiking",         250.0,  "#2c7bb6"),
    "IB":  ("Intrinsic Bursting",       600.0,  "#d7191c"),
    "CH":  ("Chattering",               500.0,  "#1a9641"),
    "FS":  ("Fast Spiking",             250.0,  "#ff7f00"),
    "LTS": ("Low-Threshold Spiking",    300.0,  "#984ea3"),
}
pat_results = {}
for preset, (label, I_val, col) in preset_cfg.items():
    neuron = AdaptiveExponentialIF.from_preset(preset)
    I_arr = np.zeros(n_adex); I_arr[int(100/dt_adex):] = I_val
    res = neuron.simulate_detailed(I_arr, dt=dt_adex)
    pat_results[preset] = {"res": res, "label": label, "I": I_val, "col": col}
    n_sp = len(res["spikes"]); rate = res["firing_rate_hz"]
    cv_str = f"{res['cv_isi']:.3f}" if not np.isnan(res["cv_isi"]) else "N/A"
    print(f"    {preset}: {n_sp} spikes  {rate:.1f} Hz  CV-ISI={cv_str}")

# ── Fig 4 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(14, 14),
                          gridspec_kw={"width_ratios": [3, 1]})
fig.subplots_adjust(hspace=0.45, wspace=0.30)
panel_letters = "ABCDEFGHIJ"
t_adex = np.arange(n_adex) * dt_adex

for row, (preset, (label, I_val, col)) in enumerate(preset_cfg.items()):
    res = pat_results[preset]["res"]
    ax_v = axes[row, 0]; ax_isi = axes[row, 1]

    # V(t) + w(t)
    ax_v.plot(t_adex, res["V"], color=col, lw=1.4, label="V(t)")
    ax_w = ax_v.twinx()
    ax_w.plot(t_adex, res["w"], color=PAL["w"], lw=1.0, ls="--", alpha=0.7, label="w(t)")
    ax_w.set_ylabel("w (pA)", color=PAL["w"], fontsize=8)
    ax_w.tick_params(axis="y", labelcolor=PAL["w"], labelsize=8)
    ax_v.spines["right"].set_visible(True)

    for sp in res["spikes"][:80]:
        ax_v.axvline(sp, color=col, alpha=0.15, lw=0.7)
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title(f"{preset} — {label}  ($I$={I_val} pA)", fontsize=10)
    ax_v.set_xlim(0, 1000)
    if row == 4: ax_v.set_xlabel("Time (ms)")
    _label(ax_v, panel_letters[row*2])

    # ISI histogram
    spk = np.array(res["spikes"])
    if len(spk) > 2:
        isi = np.diff(spk)
        ax_isi.hist(isi, bins=30, color=col, alpha=0.8, edgecolor="white", lw=0.4)
        ax_isi.set_xlabel("ISI (ms)"); ax_isi.set_ylabel("Count")
        cv_val = np.std(isi)/np.mean(isi) if np.mean(isi)>0 else 0
        ax_isi.set_title(f"ISI  CV={cv_val:.2f}", fontsize=9)
    else:
        ax_isi.text(0.5, 0.5, "< 3 spikes", ha="center", va="center",
                    transform=ax_isi.transAxes, color="gray")
        ax_isi.set_title("ISI", fontsize=9)
    _label(ax_isi, panel_letters[row*2+1])

fig.suptitle("Figure 4 — AdEx Firing Pattern Gallery\n"
             "Left: V(t) (solid) + adaptation w(t) (dashed) | Right: ISI distribution",
             fontsize=12, fontweight="bold")
_save("fig_04_adex_patterns.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 4 — Synapse kinetics
# ═══════════════════════════════════════════════════════════════════════════
print("[5/7] Synapse Kinetics …")
T_syn = 400.0; dt_syn = 0.1; n_syn = int(T_syn / dt_syn)
t_syn = np.arange(n_syn) * dt_syn

syn_cfg = {
    "AMPA":   (AMPASynapse(g_max=1.0),   PAL["AMPA"],  0.0,   [10.0]),
    "NMDA":   (NMDASynapse(g_max=1.0),   PAL["NMDA"],  0.0,   [10.0]),
    "GABA-A": (GABAASynapse(g_max=1.0),  PAL["GABAA"], -65.0, [10.0]),
    "GABA-B": (GABABSynapse(g_max=1.0),  PAL["GABAB"], -65.0,
               [10.0, 20.0, 30.0, 40.0]),  # needs burst
}
syn_g = {}
for name, (syn_obj, col, V_post, spike_times) in syn_cfg.items():
    syn_obj.reset()
    g_arr = np.zeros(n_syn)
    spike_set = set(int(s/dt_syn) for s in spike_times)
    for i in range(n_syn):
        spk = (i in spike_set)
        syn_obj.step(dt_syn, V_post, spike=spk)
        g_arr[i] = syn_obj.g
    syn_g[name] = g_arr
    print(f"    {name}: peak g = {g_arr.max()*1000:.1f} pS at t={t_syn[g_arr.argmax()]:.1f} ms")

# ── Fig 5 — Synapse Kinetics ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.45, wspace=0.32)

for ax, (name, (_, col, _, spk_times)) in zip(axes.flat, syn_cfg.items()):
    g = syn_g[name]
    ax.plot(t_syn, g * 1000, color=col, lw=2)   # nS → pS
    for sp in spk_times:
        ax.axvline(sp, color="k", lw=1.2, ls="--", alpha=0.5, label="Pre spike" if sp==spk_times[0] else "")
    peak_g = g.max() * 1000
    t_peak = t_syn[g.argmax()]
    # decay time constant (time to reach 1/e of peak)
    post_peak = g[g.argmax():]
    thresh = g.max() * np.exp(-1)
    decay_idx = np.where(post_peak <= thresh)[0]
    tau_decay = (decay_idx[0] * dt_syn) if len(decay_idx) else float("nan")
    ax.set_title(f"{name} Synapse\nPeak = {peak_g:.1f} pS,  τ_decay ≈ {tau_decay:.1f} ms")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Conductance g (pS)")
    ax.legend(frameon=False, fontsize=8)

for i, ax in enumerate(axes.flat):
    _label(ax, "ABCD"[i])

fig.suptitle("Figure 5 — Synaptic Conductance Kinetics\n"
             "Single-spike responses (AMPA/NMDA/GABA-A) and burst response (GABA-B)",
             fontsize=12, fontweight="bold")
_save("fig_05_synapse_kinetics.png")


# ── Fig 6 — NMDA Mg²⁺ Block ─────────────────────────────────────────────
print("[5b/7] NMDA Mg²⁺ block …")
nmda_ref = NMDASynapse()
V_sweep = np.linspace(-90, 60, 300)
Mg_concentrations = [0.0, 0.5, 1.0, 2.0]
colors_mg = ["#4dac26", "#b8e186", "#d01c8b", "#7b3294"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for Mg, col in zip(Mg_concentrations, colors_mg):
    nmda_ref.Mg_conc = Mg
    B = np.array([nmda_ref.Mg_block(v) for v in V_sweep])
    ax.plot(V_sweep, B, color=col, lw=2, label=f"[Mg²⁺] = {Mg} mM")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("Membrane voltage V (mV)")
ax.set_ylabel("Mg²⁺ unblock factor B(V)")
ax.set_title("NMDA Voltage-Dependent Mg²⁺ Block")
ax.legend(frameon=False); ax.set_xlim(-90, 60)
_label(ax, "A")

ax2 = axes[1]
nmda_ref.Mg_conc = 1.0
g_open = 1.0   # normalised max conductance
E_rev_NMDA = 0.0
for Mg, col in zip(Mg_concentrations, colors_mg):
    nmda_ref.Mg_conc = Mg
    B = np.array([nmda_ref.Mg_block(v) for v in V_sweep])
    I_NMDA = B * g_open * (V_sweep - E_rev_NMDA)
    ax2.plot(V_sweep, I_NMDA, color=col, lw=2, label=f"[Mg²⁺] = {Mg} mM")
ax2.axhline(0, color="k", lw=0.8, ls=":")
ax2.axvline(E_rev_NMDA, color="gray", ls=":", lw=1)
ax2.set_xlabel("Membrane voltage V (mV)"); ax2.set_ylabel("$I_{NMDA}$ (norm.)")
ax2.set_title("NMDA I-V Relationship")
ax2.legend(frameon=False)
_label(ax2, "B")

fig.suptitle("Figure 6 — NMDA Mg²⁺ Block and Voltage Dependence",
             fontsize=13, fontweight="bold")
_save("fig_06_nmda_mg_block.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 5 — STDP
# ═══════════════════════════════════════════════════════════════════════════
print("[6/7] STDP …")
rng = np.random.default_rng(42)

# --- STDP window: theoretical LTP/LTD curve ---
A_plus = 0.010; A_minus = 0.0105
tau_plus = 20.0; tau_minus = 20.0
delta_t = np.linspace(-80, 80, 500)
dw_theory = np.where(delta_t >= 0,
                      A_plus  * np.exp(-delta_t / tau_plus),
                     -A_minus * np.exp( delta_t / tau_minus))

# --- Weight evolution under Poisson pre+post (20 Hz each) ---
stdp = STDPRule(A_plus=A_plus, A_minus=A_minus,
                tau_plus=tau_plus, tau_minus=tau_minus)
dt_s = 0.1; T_s = 30000; w_val = 0.5
pre_rate = 20e-3; post_rate = 20e-3   # spikes/ms
for _ in range(int(T_s/dt_s)):
    pre_sp  = rng.random() < pre_rate  * dt_s
    post_sp = rng.random() < post_rate * dt_s
    w_val = stdp.update(w_val, dt_s, bool(pre_sp), bool(post_sp))
w_hist = stdp.weight_history

# --- 200-synapse steady-state distribution ---
w_final_dist = []
for _ in range(200):
    stdp2 = STDPRule(A_plus=A_plus, A_minus=A_minus)
    w2 = rng.random()
    for _ in range(int(T_s/dt_s)):
        stdp2.update(w2, dt_s, bool(rng.random()<pre_rate*dt_s),
                               bool(rng.random()<post_rate*dt_s))
    if len(stdp2.weight_history):
        w_final_dist.append(stdp2.weight_history[-1])

# ── Fig 7 — STDP ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: LTP/LTD window
ax = axes[0]
ax.fill_between(delta_t[delta_t>=0], 0, dw_theory[delta_t>=0],
                color="#d7191c", alpha=0.35, label="LTP")
ax.fill_between(delta_t[delta_t<0],  0, dw_theory[delta_t<0],
                color="#2c7bb6", alpha=0.35, label="LTD")
ax.plot(delta_t, dw_theory, "k", lw=1.8)
ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel("$\\Delta t = t_{post} - t_{pre}$ (ms)")
ax.set_ylabel("$\\Delta w$")
ax.set_title("STDP Learning Window\n$A_+ = 0.010$,  $A_- = 0.0105$")
ax.legend(frameon=False)
_label(ax, "A")

# Panel B: Weight trajectory
ax2 = axes[1]
t_w = np.arange(len(w_hist)) * dt_s / 1000   # → seconds
ax2.plot(t_w, w_hist, color=PAL["w"], lw=0.8, alpha=0.7)
ax2.axhline(w_hist[-1], color="k", ls="--", lw=1,
            label=f"Final w = {w_hist[-1]:.3f}")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Synaptic weight w")
ax2.set_title(f"Weight Evolution\n(single synapse, 20 Hz Poisson)")
ax2.legend(frameon=False)
_label(ax2, "B")

# Panel C: Steady-state distribution
ax3 = axes[2]
ax3.hist(w_final_dist, bins=30, color=PAL["w"], alpha=0.85,
         edgecolor="white", lw=0.4)
ax3.axvline(np.median(w_final_dist), color="k", ls="--", lw=1.5,
            label=f"Median = {np.median(w_final_dist):.3f}")
ax3.set_xlabel("Steady-state weight"); ax3.set_ylabel("Count")
ax3.set_title("Weight Distribution\n(200 synapses × 30 s)")
ax3.legend(frameon=False)
_label(ax3, "C")

fig.suptitle("Figure 7 — Spike-Timing Dependent Plasticity (STDP)",
             fontsize=13, fontweight="bold")
_save("fig_07_stdp.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 6 — E/I Recurrent Network (fast, 300 ms)
# ═══════════════════════════════════════════════════════════════════════════
print("[7/7] E/I Network (300 neurons, 500 ms) …")
from networks.network import SpikingNetwork

N_E, N_I = 240, 60
net = SpikingNetwork(
    N_E=N_E, N_I=N_I, p_conn=0.12,
    dt=0.2, neuron_type="LIF",
    g_EE=0.30, g_EI=0.30, g_IE=1.80, g_II=1.80,
    nu_ext=12.0, g_ext=0.9, seed=99,
)
n_steps_net = int(500.0 / net.dt)
I_dc = np.full((N_E + N_I, n_steps_net), 200.0)
net_r = net.run(T_ms=500.0, I_ext=I_dc,
                progress_cb=lambda t: print(f"    t={t:.0f} ms …", end="\r"))
print()

t_net = net_r["t"]
all_spk = net_r["spikes"]
rates_E = np.array([len(all_spk[i])/0.5 for i in range(N_E)])
rates_I = np.array([len(all_spk[N_E+i])/0.5 for i in range(N_I)])
all_cv = []
for i in range(N_E + N_I):
    sp = np.array(all_spk[i])
    if len(sp) > 2:
        isi = np.diff(sp)
        all_cv.append(float(np.std(isi)/np.mean(isi)))
print(f"    E rate: {rates_E.mean():.1f} ± {rates_E.std():.1f} Hz")
print(f"    I rate: {rates_I.mean():.1f} ± {rates_I.std():.1f} Hz")
cv_mean = np.mean(all_cv) if all_cv else float("nan")
state = "AI" if cv_mean > 0.7 else "SR"
print(f"    CV-ISI: {cv_mean:.3f} → {state} state")

# Smoothed population rates (10 ms Gaussian)
from scipy.ndimage import gaussian_filter1d
win_sigma = int(10 / net.dt)   # 10 ms in samples
pop_E_smooth = gaussian_filter1d(net_r["pop_E_rate"], sigma=win_sigma)
pop_I_smooth = gaussian_filter1d(net_r["pop_I_rate"], sigma=win_sigma)

# ── Fig 8 — Network Dynamics ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                         height_ratios=[2.5, 1, 1, 1])

ax_rast = fig.add_subplot(gs[0])
ax_rate = fig.add_subplot(gs[1], sharex=ax_rast)
ax_lfp  = fig.add_subplot(gs[2], sharex=ax_rast)
ax_sync = fig.add_subplot(gs[3], sharex=ax_rast)

# Raster
for nid in range(N_E + N_I):
    sp = np.array(all_spk[nid])
    if len(sp) == 0: continue
    col = PAL["E"] if nid < N_E else PAL["I"]
    ax_rast.scatter(sp, np.full_like(sp, nid), s=0.8, c=col, linewidths=0)
ax_rast.axhline(N_E, color="k", lw=0.8, ls="--", alpha=0.5)
ax_rast.set_ylabel("Neuron index")
ax_rast.set_title(f"E/I Network Raster  ({N_E}E + {N_I}I, "
                  f"E rate={rates_E.mean():.1f} Hz, I rate={rates_I.mean():.1f} Hz, "
                  f"CV-ISI={cv_mean:.2f})")
# legend patches
from matplotlib.patches import Patch
ax_rast.legend(handles=[Patch(color=PAL["E"], label="Excitatory"),
                         Patch(color=PAL["I"], label="Inhibitory")],
               frameon=False, fontsize=8, loc="upper right")
_label(ax_rast, "A")

# Population firing rate
ax_rate.plot(t_net, pop_E_smooth, color=PAL["E"], lw=1.4, label="E")
ax_rate.plot(t_net, pop_I_smooth, color=PAL["I"], lw=1.4, label="I")
ax_rate.set_ylabel("Rate (Hz)"); ax_rate.legend(frameon=False, fontsize=8)
ax_rate.set_title("Population Firing Rate (10 ms Gaussian smoothing)")
_label(ax_rate, "B")

# LFP proxy (mean E voltage)
lfp_smooth = gaussian_filter1d(net_r["LFP"], sigma=win_sigma)
ax_lfp.plot(t_net, lfp_smooth, color=PAL["LFP"], lw=1.2)
ax_lfp.set_ylabel("mV"); ax_lfp.set_title("LFP Proxy: Mean E-cell Voltage")
_label(ax_lfp, "C")

# E–I balance
ax_sync.plot(t_net, pop_E_smooth, color=PAL["E"], lw=1.2, alpha=0.7)
ax_sync.plot(t_net, pop_I_smooth, color=PAL["I"], lw=1.2, alpha=0.7)
ax_sync.fill_between(t_net, pop_E_smooth, pop_I_smooth,
                     where=pop_E_smooth>pop_I_smooth,
                     alpha=0.2, color=PAL["E"], label="E dominant")
ax_sync.fill_between(t_net, pop_E_smooth, pop_I_smooth,
                     where=pop_I_smooth>pop_E_smooth,
                     alpha=0.2, color=PAL["I"], label="I dominant")
ax_sync.set_xlabel("Time (ms)"); ax_sync.set_ylabel("Rate (Hz)")
ax_sync.set_title("E–I Balance"); ax_sync.legend(frameon=False, fontsize=8)
_label(ax_sync, "D")

fig.suptitle("Figure 8 — Recurrent E/I Spiking Network Dynamics",
             fontsize=13, fontweight="bold")
_save("fig_08_network_dynamics.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 7 — ISI Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("[8/10] ISI Analysis …")
E_trains = [np.array(all_spk[i]) for i in range(N_E) if len(all_spk[i]) > 3]
all_ISI  = np.concatenate([np.diff(sp) for sp in E_trains]) if E_trains else np.array([5.0])
all_CV   = np.array(all_cv)

# Poincaré return map
long_trains = [sp for sp in E_trains if len(sp) > 4]
if long_trains:
    ref_train = max(long_trains, key=len)
    isi_seq = np.diff(ref_train)
    poincare_x = isi_seq[:-1]; poincare_y = isi_seq[1:]
else:
    poincare_x = poincare_y = np.array([])

# ── Fig 9 — ISI Analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ISI histogram
ax = axes[0]
bins = np.logspace(np.log10(max(0.5, all_ISI.min())),
                   np.log10(all_ISI.max() + 1), 50)
ax.hist(all_ISI, bins=bins, color=PAL["E"], alpha=0.8, edgecolor="white", lw=0.3)
ax.set_xscale("log")
ax.set_xlabel("ISI (ms, log scale)"); ax.set_ylabel("Count")
cv_all = float(np.std(all_ISI)/np.mean(all_ISI)) if len(all_ISI)>1 else 0
ax.set_title(f"ISI Distribution\nAll E-cells pooled  (CV={cv_all:.3f})")
_label(ax, "A")

# Poincaré return map
ax2 = axes[1]
if len(poincare_x) > 0:
    ax2.scatter(poincare_x, poincare_y, s=4, color=PAL["E"], alpha=0.5)
    lim = max(poincare_x.max(), poincare_y.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="ISI$_n$ = ISI$_{n+1}$")
    ax2.set_xlabel("ISI$_n$ (ms)"); ax2.set_ylabel("ISI$_{n+1}$ (ms)")
    ax2.set_title(f"Poincaré Return Map\n(busiest E-neuron, {len(poincare_x)} pairs)")
    ax2.legend(frameon=False, fontsize=8)
else:
    ax2.text(0.5, 0.5, "Insufficient spikes", ha="center", va="center",
             transform=ax2.transAxes)
_label(ax2, "B")

# CV-ISI histogram
ax3 = axes[2]
if len(all_CV) > 0:
    ax3.hist(all_CV, bins=25, color="#762a83", alpha=0.85, edgecolor="white", lw=0.3)
    ax3.axvline(1.0, color="k", ls="--", lw=1.5, label="CV=1 (Poisson)")
    ax3.axvline(np.median(all_CV), color="#ff7f00", ls="--", lw=1.5,
                label=f"Median CV={np.median(all_CV):.2f}")
    ax3.set_xlabel("CV-ISI"); ax3.set_ylabel("Number of neurons")
    ax3.set_title(f"CV-ISI Distribution\n({len(all_CV)} E-neurons, state: {state})")
    ax3.legend(frameon=False, fontsize=8)
_label(ax3, "C")

fig.suptitle("Figure 9 — ISI Statistics and Firing Irregularity",
             fontsize=13, fontweight="bold")
_save("fig_09_isi_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION 8 — LFP Power Spectrum
# ═══════════════════════════════════════════════════════════════════════════
print("[9/10] Power Spectrum …")
LFP = net_r["LFP"]
fs_hz = 1000.0 / net.dt
nperseg = min(len(LFP), 2048)
freqs, psd = sp_signal.welch(LFP, fs=fs_hz, nperseg=nperseg, scaling="density")

# Band definitions (Hz)
bands = {"δ (0.5–4)": (0.5, 4),   "θ (4–8)":   (4, 8),
         "α (8–13)":  (8, 13),    "β (13–30)": (13, 30),
         "γ (30–100)":(30, 100)}
band_colors = {"δ (0.5–4)":"#1b7837","θ (4–8)":"#762a83",
               "α (8–13)":"#d6604d","β (13–30)":"#2166ac","γ (30–100)":"#e08214"}

# ── Fig 10 — PSD ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(freqs[freqs <= 200], psd[freqs <= 200], color=PAL["LFP"], lw=1.5)
for bname, (flo, fhi) in bands.items():
    mask = (freqs >= flo) & (freqs <= fhi)
    if mask.any():
        ax.fill_between(freqs[mask], 1e-10, psd[mask],
                        alpha=0.35, color=band_colors[bname], label=bname)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (mV²/Hz)")
ax.set_title("LFP Power Spectral Density\n(Welch method)")
ax.legend(frameon=False, fontsize=8); ax.set_xlim(0, 200)
_label(ax, "A")

ax2 = axes[1]
band_powers = {}
for bname, (flo, fhi) in bands.items():
    mask = (freqs >= flo) & (freqs <= fhi)
    bp = np.trapz(psd[mask], freqs[mask]) if mask.sum() > 1 else 0.0
    band_powers[bname] = max(bp, 1e-15)

bnames = list(band_powers.keys())
bvals  = list(band_powers.values())
cols   = [band_colors[b] for b in bnames]
bars = ax2.bar(range(len(bnames)), bvals, color=cols, alpha=0.85, edgecolor="white")
ax2.set_yscale("log")
ax2.set_xticks(range(len(bnames))); ax2.set_xticklabels(bnames, rotation=25, ha="right", fontsize=8)
ax2.set_ylabel("Band Power (mV²)")
ax2.set_title("Power in Frequency Bands")
for bar, val in zip(bars, bvals):
    ax2.text(bar.get_x() + bar.get_width()/2, val*1.3,
             f"{val:.2e}", ha="center", va="bottom", fontsize=7)
_label(ax2, "B")

fig.suptitle("Figure 10 — LFP Power Spectrum Analysis",
             fontsize=13, fontweight="bold")
_save("fig_10_lfp_psd.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 11 — Summary Dashboard (12-panel overview)
# ═══════════════════════════════════════════════════════════════════════════
print("[10/10] Summary Dashboard …")
fig = plt.figure(figsize=(18, 14))
gs_main = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40)

# ── Row 0: HH basics ──────────────────────────────────────────────────────
# (0,0) HH voltage trace
ax00 = fig.add_subplot(gs_main[0, 0])
ax00.plot(hh_r["t"], hh_r["V"], color=PAL["V"], lw=1.2)
ax00.set_title("HH: Action Potential", fontsize=9)
ax00.set_xlabel("Time (ms)"); ax00.set_ylabel("V (mV)")
_label(ax00, "A")

# (0,1) Gating variables
ax01 = fig.add_subplot(gs_main[0, 1])
ax01.plot(hh_r["t"], hh_r["m"], color=PAL["m"],  lw=1.2, label="m")
ax01.plot(hh_r["t"], hh_r["h"], color=PAL["h"],  lw=1.2, label="h")
ax01.plot(hh_r["t"], hh_r["n"], color=PAL["n"],  lw=1.2, label="n")
ax01.set_title("HH: Gating Variables", fontsize=9)
ax01.set_xlabel("Time (ms)"); ax01.set_ylabel("Gate value")
ax01.legend(frameon=False, fontsize=7)
_label(ax01, "B")

# (0,2) Phase plane V–n
ax02 = fig.add_subplot(gs_main[0, 2])
mask = hh_r["t"] > 15
ax02.plot(hh_r["V"][mask], hh_r["n"][mask], color="gray", lw=0.7)
ax02.plot(V_range, nullc["n_inf"], color=PAL["n"],  lw=1.5, label="n-nullcline")
ax02.set_xlabel("V (mV)"); ax02.set_ylabel("n")
ax02.set_title("Phase Plane V–n", fontsize=9)
ax02.legend(frameon=False, fontsize=7); ax02.set_xlim(-80, 50)
_label(ax02, "C")

# (0,3) f-I curve HH
ax03 = fig.add_subplot(gs_main[0, 3])
ax03.plot(I_hh_range, f_hh, color=PAL["V"], lw=1.5, marker="o", ms=3)
ax03.set_title("HH f-I Curve", fontsize=9)
ax03.set_xlabel("$I_{ext}$ (µA/cm²)"); ax03.set_ylabel("Rate (Hz)")
_label(ax03, "D")

# ── Row 1: Synapses + STDP ───────────────────────────────────────────────
# (1,0) Synaptic conductances (all 4)
ax10 = fig.add_subplot(gs_main[1, 0])
for sname, col in [("AMPA", PAL["AMPA"]), ("NMDA", PAL["NMDA"]),
                    ("GABA-A", PAL["GABAA"]), ("GABA-B", PAL["GABAB"])]:
    scale = 1000 if sname != "GABA-B" else 1e6
    unit  = "pS" if sname != "GABA-B" else "fS"
    ax10.plot(t_syn[:int(200/dt_syn)],
              syn_g[sname][:int(200/dt_syn)] * scale,
              lw=1.2, label=f"{sname}")
ax10.set_xlabel("Time (ms)"); ax10.set_ylabel("g (pS / fS)")
ax10.set_title("Synaptic Conductances", fontsize=9)
ax10.legend(frameon=False, fontsize=7)
_label(ax10, "E")

# (1,1) NMDA Mg block
ax11 = fig.add_subplot(gs_main[1, 1])
for Mg, col in zip([0.0, 1.0, 2.0], ["#4dac26", "#d01c8b", "#7b3294"]):
    nmda_ref.Mg_conc = Mg
    B = [nmda_ref.Mg_block(v) for v in V_sweep]
    ax11.plot(V_sweep, B, color=col, lw=1.4, label=f"{Mg} mM")
ax11.set_xlabel("V (mV)"); ax11.set_ylabel("B(V)")
ax11.set_title("NMDA Mg²⁺ Block", fontsize=9)
ax11.legend(frameon=False, fontsize=7)
_label(ax11, "F")

# (1,2) STDP window
ax12 = fig.add_subplot(gs_main[1, 2])
ax12.fill_between(delta_t[delta_t>=0], 0, dw_theory[delta_t>=0], color="#d7191c", alpha=0.4)
ax12.fill_between(delta_t[delta_t<0],  0, dw_theory[delta_t<0],  color="#2c7bb6", alpha=0.4)
ax12.plot(delta_t, dw_theory, "k", lw=1.4)
ax12.axhline(0, color="k", lw=0.6); ax12.axvline(0, color="k", lw=0.6, ls=":")
ax12.set_xlabel("Δt (ms)"); ax12.set_ylabel("Δw")
ax12.set_title("STDP Window", fontsize=9)
_label(ax12, "G")

# (1,3) STDP weight evolution
ax13 = fig.add_subplot(gs_main[1, 3])
ax13.plot(np.arange(len(w_hist))*dt_s/1000, w_hist, color=PAL["w"], lw=0.8, alpha=0.8)
ax13.set_xlabel("Time (s)"); ax13.set_ylabel("Weight w")
ax13.set_title("STDP Weight Convergence", fontsize=9)
_label(ax13, "H")

# ── Row 2: Network ───────────────────────────────────────────────────────
# (2,0:2) Raster (wide)
ax20 = fig.add_subplot(gs_main[2, :2])
for nid in range(N_E + N_I):
    sp = np.array(all_spk[nid])
    if len(sp):
        c = PAL["E"] if nid < N_E else PAL["I"]
        ax20.scatter(sp, np.full_like(sp, nid), s=0.5, c=c, linewidths=0)
ax20.axhline(N_E, color="k", lw=0.7, ls="--")
ax20.set_xlabel("Time (ms)"); ax20.set_ylabel("Neuron index")
ax20.set_title(f"E/I Network Raster  (E={rates_E.mean():.0f} Hz, I={rates_I.mean():.0f} Hz)", fontsize=9)
_label(ax20, "I")

# (2,2) Population rate
ax21 = fig.add_subplot(gs_main[2, 2])
ax21.plot(t_net, pop_E_smooth, color=PAL["E"], lw=1.2, label="E")
ax21.plot(t_net, pop_I_smooth, color=PAL["I"], lw=1.2, label="I")
ax21.set_xlabel("Time (ms)"); ax21.set_ylabel("Rate (Hz)")
ax21.set_title("Population Rates", fontsize=9)
ax21.legend(frameon=False, fontsize=7)
_label(ax21, "J")

# (2,3) ISI CV distribution
ax22 = fig.add_subplot(gs_main[2, 3])
if len(all_CV) > 0:
    ax22.hist(all_CV, bins=20, color="#762a83", alpha=0.85, edgecolor="white", lw=0.3)
    ax22.axvline(1.0, color="k", ls="--", lw=1.2, label="Poisson")
    ax22.axvline(np.median(all_CV), color="#ff7f00", ls="--", lw=1.2,
                 label=f"Median={np.median(all_CV):.2f}")
ax22.set_xlabel("CV-ISI"); ax22.set_ylabel("Count")
ax22.set_title("ISI Irregularity", fontsize=9)
ax22.legend(frameon=False, fontsize=7)
_label(ax22, "K")

# global annotation
txt = (
    f"Models: HH · LIF · AdEx (RS/IB/CH/FS/LTS)  |  "
    f"Synapses: AMPA · NMDA · GABA-A · GABA-B  |  "
    f"Plasticity: STDP  |  Network: {N_E}E + {N_I}I LIF"
)
fig.text(0.5, -0.01, txt, ha="center", fontsize=8, color="#555555")

fig.suptitle("Figure 11 — Computational Neuroscience: Summary Dashboard",
             fontsize=14, fontweight="bold", y=1.02)
_save("fig_11_summary_dashboard.png")


# ── Final report ──────────────────────────────────────────────────────────
figs = sorted(OUTDIR.glob("fig_*.png"))
print("\n" + "=" * 62)
print("  ALL SIMULATIONS COMPLETE")
print("=" * 62)
print(f"  {'Figure':<45} {'Size':>6}")
print("  " + "-" * 55)
for f in figs:
    kb = f.stat().st_size // 1024
    print(f"  {f.name:<45} {kb:>4} KB")
print(f"\n  Total figures: {len(figs)}")
print("=" * 62)
