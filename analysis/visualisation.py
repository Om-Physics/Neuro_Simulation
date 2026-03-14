"""
analysis/visualisation.py
==========================
Publication-quality figures for computational neuroscience simulations.

All figures follow journal conventions:
  - Font size ≥ 10 pt; axis labels with units; scale bars where appropriate
  - Colormaps chosen for colour-blind accessibility (viridis, RdBu, tab10)
  - All panel letters automated via enumerate

Figures produced
----------------
  Fig 1 — HH single-neuron: AP waveform + gating variables + ion currents
  Fig 2 — HH phase plane (V–n nullclines) + limit cycle
  Fig 3 — f-I curves: HH vs LIF vs AdEx comparison
  Fig 4 — AdEx firing patterns (RS, IB, CH, FS, LTS)
  Fig 5 — Synaptic conductances: AMPA, NMDA, GABA-A, GABA-B
  Fig 6 — NMDA Mg²⁺ block voltage-dependence
  Fig 7 — STDP weight dynamics: LTP/LTD windows
  Fig 8 — Network raster + population rate + LFP + synchrony
  Fig 9 — ISI distribution + CV-ISI histogram (population)
  Fig 10— Power spectral density of network LFP
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
})

OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

COLORS = {
    "V":    "#2c7bb6",
    "m":    "#d7191c",
    "h":    "#1a9641",
    "n":    "#ff7f00",
    "I_Na": "#d7191c",
    "I_K":  "#2c7bb6",
    "I_L":  "#999999",
    "E":    "#e41a1c",
    "I":    "#377eb8",
    "LFP":  "#4d4d4d",
    "rate": "#ff7f00",
}


def _panel_label(ax, label: str, x=-0.18, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="right")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — HH Action Potential: waveform, gating, currents
# ═══════════════════════════════════════════════════════════════════════════
def fig1_hh_action_potential(hh_result: dict, save: bool = True) -> plt.Figure:
    """Full HH action potential anatomy across 4 panels."""
    t, V = hh_result["t"], hh_result["V"]
    m, h, n = hh_result["m"], hh_result["h"], hh_result["n"]
    I_Na, I_K, I_L = hh_result["I_Na"], hh_result["I_K"], hh_result["I_L"]
    spikes = hh_result["spikes"]

    # zoom to first 3 spikes
    if len(spikes) >= 2:
        t_end = min(spikes[min(2, len(spikes)-1)] + 20, t[-1])
    else:
        t_end = t[-1]
    mask = t <= t_end

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle("Hodgkin-Huxley Neuron — Action Potential Anatomy",
                 fontsize=13, fontweight="bold", y=0.98)

    # (A) Membrane voltage
    ax = axes[0]
    ax.plot(t[mask], V[mask], color=COLORS["V"], lw=1.8, label="V(t)")
    ax.axhline(-65, color="gray", lw=0.8, ls="--", alpha=0.6, label="V_rest")
    ax.axhline(-55, color="salmon", lw=0.8, ls=":", alpha=0.8, label="Threshold")
    for sp in spikes:
        if sp <= t_end:
            ax.axvline(sp, color="#aaaaaa", lw=0.6, alpha=0.5)
    ax.set_ylabel("V (mV)")
    ax.set_ylim(-80, 50)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    _panel_label(ax, "A")

    # (B) Gating variables
    ax = axes[1]
    ax.plot(t[mask], m[mask], color=COLORS["m"], label="m (Na act.)")
    ax.plot(t[mask], h[mask], color=COLORS["h"], label="h (Na inact.)")
    ax.plot(t[mask], n[mask], color=COLORS["n"], label="n (K act.)")
    ax.set_ylabel("Gate probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    _panel_label(ax, "B")

    # (C) Na channel conductance = m³h
    ax = axes[2]
    g_Na_trace = m[mask]**3 * h[mask]
    g_K_trace  = n[mask]**4
    ax.plot(t[mask], g_Na_trace, color=COLORS["m"], label=r"$m^3h$ (Na)")
    ax.plot(t[mask], g_K_trace,  color=COLORS["n"], label=r"$n^4$ (K)")
    ax.set_ylabel("Norm. conductance")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    _panel_label(ax, "C")

    # (D) Ion currents
    ax = axes[3]
    ax.plot(t[mask], I_Na[mask], color=COLORS["I_Na"], label=r"$I_{Na}$")
    ax.plot(t[mask], I_K[mask],  color=COLORS["I_K"],  label=r"$I_{K}$")
    ax.plot(t[mask], I_L[mask],  color=COLORS["I_L"],  ls="--", label=r"$I_L$")
    ax.set_ylabel(r"Current (µA/cm²)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.axhline(0, color="k", lw=0.5)
    _panel_label(ax, "D")

    fig.align_ylabels(axes)
    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig1_hh_action_potential.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — Phase Plane Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig2_phase_plane(hh_neuron, hh_result: dict, save: bool = True) -> plt.Figure:
    """HH phase plane: V vs n with nullclines and limit cycle trajectory."""
    V_range = np.linspace(-80, 50, 500)
    nc = hh_neuron.nullclines(V_range)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Phase Plane Analysis — Hodgkin-Huxley",
                 fontsize=13, fontweight="bold")

    # Left: V–n phase plane
    ax = axes[0]
    ax.plot(V_range, nc["n_inf"], "b-", lw=2, label=r"$n_\infty(V)$ nullcline")
    ax.plot(hh_result["V"], hh_result["n"], color="#333333",
            lw=0.7, alpha=0.7, label="Limit cycle")
    ax.set_xlabel("Membrane potential V (mV)")
    ax.set_ylabel("K⁺ gate variable n")
    ax.set_xlim(-80, 50)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("V–n Phase Plane")
    _panel_label(ax, "A")

    # Right: steady-state gating curves
    ax = axes[1]
    ax.plot(V_range, nc["m_inf"], color=COLORS["m"], lw=2,
            label=r"$m_\infty(V)$")
    ax.plot(V_range, nc["h_inf"], color=COLORS["h"], lw=2,
            label=r"$h_\infty(V)$")
    ax.plot(V_range, nc["n_inf"], color=COLORS["n"], lw=2,
            label=r"$n_\infty(V)$")
    ax.set_xlabel("Membrane potential V (mV)")
    ax.set_ylabel("Steady-state value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Gating Variable Steady-States")
    _panel_label(ax, "B")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig2_phase_plane.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — f-I Curves
# ═══════════════════════════════════════════════════════════════════════════
def fig3_fi_curves(fi_results: dict, save: bool = True) -> plt.Figure:
    """
    Compare f-I curves for HH, LIF, and AdEx models.
    fi_results: dict mapping model_name → {'I': arr, 'f_hz': arr}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Frequency-Current (f-I) Curves — Model Comparison",
                 fontsize=13, fontweight="bold")

    model_colors = {"HH": "#d7191c", "LIF": "#2c7bb6",
                    "AdEx-RS": "#1a9641", "AdEx-FS": "#ff7f00",
                    "AdEx-IB": "#984ea3"}
    model_ls = {"HH": "-", "LIF": "--", "AdEx-RS": "-.",
                "AdEx-FS": ":", "AdEx-IB": "-"}

    ax = axes[0]
    for name, data in fi_results.items():
        color = model_colors.get(name, "gray")
        ls = model_ls.get(name, "-")
        ax.plot(data["I"], data["f_hz"], color=color, ls=ls,
                lw=2, label=name)
    ax.set_xlabel(r"Input current ($\mu$A/cm² or pA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("f-I Gain Comparison")
    ax.legend(frameon=False)
    _panel_label(ax, "A")

    # Right: log-log to show power-law vs threshold relationship
    ax = axes[1]
    for name, data in fi_results.items():
        f = np.array(data["f_hz"])
        I = np.array(data["I"])
        mask = f > 1.0
        if mask.sum() > 2:
            color = model_colors.get(name, "gray")
            ls = model_ls.get(name, "-")
            ax.semilogy(I[mask], f[mask], color=color, ls=ls, lw=2, label=name)
    ax.set_xlabel(r"Input current")
    ax.set_ylabel("Firing rate (Hz, log scale)")
    ax.set_title("Supra-threshold Gain (semi-log)")
    ax.legend(frameon=False)
    _panel_label(ax, "B")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig3_fi_curves.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — AdEx Firing Patterns
# ═══════════════════════════════════════════════════════════════════════════
def fig4_adex_patterns(pattern_results: dict, save: bool = True) -> plt.Figure:
    """
    AdEx firing patterns: RS, IB, CH, FS, LTS in a 2×3 grid.
    pattern_results: dict mapping preset_name → simulate_detailed() output
    """
    presets = list(pattern_results.keys())
    n_plots = len(presets)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows),
                             sharex=False, sharey=False)
    axes = axes.flatten()

    fig.suptitle("AdEx Model — Biologically Realistic Firing Patterns",
                 fontsize=13, fontweight="bold")

    full_names = {
        "RS": "Regular Spiking (RS)",
        "IB": "Intrinsic Bursting (IB)",
        "CH": "Chattering (CH)",
        "FS": "Fast Spiking (FS)",
        "LTS": "Low-Threshold Spiking (LTS)",
        "TC": "Thalamocortical (TC)",
    }

    for idx, preset in enumerate(presets):
        res = pattern_results[preset]
        ax = axes[idx]
        ax.plot(res["t"], res["V"], color="#2c7bb6", lw=1.2)
        for sp in res["spikes"]:
            ax.axvline(sp, color="#bbbbbb", lw=0.5, alpha=0.5)
        title = full_names.get(preset, preset)
        f = res["firing_rate_hz"]
        ax.set_title(f"{title}\n({f:.1f} Hz, CV={res['cv_isi']:.2f})"
                     if not np.isnan(res['cv_isi']) else f"{title}\n({f:.1f} Hz)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("V (mV)")
        ax.set_ylim(-90, 60)
        _panel_label(ax, chr(65 + idx))

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig4_adex_patterns.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — Synaptic Conductances
# ═══════════════════════════════════════════════════════════════════════════
def fig5_synaptic_conductances(syn_results: dict, save: bool = True) -> plt.Figure:
    """
    Plot g(t) for each synapse type after a single presynaptic spike.
    syn_results: dict mapping synapse_name → {'t': array, 'g': array}
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    fig.suptitle("Synaptic Conductance Kinetics (single presynaptic spike at t=10 ms)",
                 fontsize=13, fontweight="bold")

    syn_names   = list(syn_results.keys())
    syn_colors  = {"AMPA": "#d7191c", "NMDA": "#2c7bb6",
                   "GABA-A": "#1a9641", "GABA-B": "#ff7f00"}
    descriptions = {
        "AMPA":   "Fast excitatory\n(glutamate, τ_peak ≈ 2 ms)",
        "NMDA":   "Slow excitatory\n(glutamate + Mg²⁺ block, τ_decay ≈ 150 ms)",
        "GABA-A": "Fast inhibitory\n(Cl⁻, τ_decay ≈ 5 ms)",
        "GABA-B": "Slow inhibitory\n(K⁺, G-protein, τ_decay ≈ 200 ms)",
    }

    for idx, (name, data) in enumerate(syn_results.items()):
        ax = axes[idx]
        color = syn_colors.get(name, "gray")
        ax.plot(data["t"], data["g"] * 1000, color=color, lw=2)  # nS→pS for display
        ax.axvline(10, color="k", lw=0.8, ls=":", alpha=0.6, label="Pre-spike")
        ax.set_title(f"{name}\n{descriptions.get(name, '')}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Conductance (pS)")
        peak_g = float(np.max(data["g"])) * 1000
        t_peak = float(data["t"][np.argmax(data["g"])])
        ax.annotate(f"Peak: {peak_g:.0f} pS\nt={t_peak:.1f} ms",
                    xy=(t_peak, peak_g), xytext=(t_peak + 15, peak_g * 0.7),
                    arrowprops=dict(arrowstyle="->", lw=0.8),
                    fontsize=8)
        _panel_label(ax, chr(65 + idx))

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig5_synaptic_conductances.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — NMDA Mg²⁺ Block
# ═══════════════════════════════════════════════════════════════════════════
def fig6_nmda_mg_block(save: bool = True) -> plt.Figure:
    """NMDA Mg²⁺ block curve and current-voltage relationship."""
    V_range = np.linspace(-90, 50, 500)
    Mg_concs = [0.0, 0.5, 1.0, 2.0]  # mM

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("NMDA Receptor Voltage-Dependent Mg²⁺ Block",
                 fontsize=13, fontweight="bold")

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(Mg_concs) - 1)) for i in range(len(Mg_concs))]

    ax = axes[0]
    for Mg, color in zip(Mg_concs, colors):
        B = 1.0 / (1.0 + np.exp(-0.062 * V_range) * Mg / 3.57)
        label = f"[Mg²⁺] = {Mg} mM" if Mg > 0 else "No Mg²⁺ block"
        ax.plot(V_range, B, color=color, lw=2, label=label)
    ax.set_xlabel("Membrane potential (mV)")
    ax.set_ylabel("Unblock factor B(V)")
    ax.set_title(r"Mg²⁺ unblock: $B(V) = \frac{1}{1 + e^{-0.062V}[Mg]/3.57}$")
    ax.legend(frameon=False)
    ax.axvline(-65, color="gray", ls="--", lw=0.8, alpha=0.6, label="Rest")
    _panel_label(ax, "A")

    # I-V curve for NMDA at different [Mg²⁺]
    ax = axes[1]
    E_rev = 0.0
    g_max = 1.0
    for Mg, color in zip(Mg_concs, colors):
        B = 1.0 / (1.0 + np.exp(-0.062 * V_range) * Mg / 3.57)
        I_NMDA = g_max * B * (V_range - E_rev)
        ax.plot(V_range, I_NMDA, color=color, lw=2,
                label=f"[Mg²⁺]={Mg} mM")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Membrane potential (mV)")
    ax.set_ylabel("I_NMDA (norm.)")
    ax.set_title("NMDA I-V Relationship")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "B")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig6_nmda_mg_block.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7 — STDP Weight Dynamics
# ═══════════════════════════════════════════════════════════════════════════
def fig7_stdp(stdp_result: dict, save: bool = True) -> plt.Figure:
    """
    STDP learning window and weight evolution.
    stdp_result: {'delta_t': arr, 'dw': arr, 'w_history': arr, 't': arr}
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Spike-Timing Dependent Plasticity (STDP)",
                 fontsize=13, fontweight="bold")

    # (A) STDP learning window
    ax = axes[0]
    dt_range = np.linspace(-100, 100, 500)
    tau_plus, tau_minus = 20.0, 20.0
    A_plus, A_minus = 0.01, 0.0105
    dw_ltp = A_plus  * np.exp(-dt_range / tau_plus)  * (dt_range > 0)
    dw_ltd = -A_minus * np.exp( dt_range / tau_minus) * (dt_range < 0)
    dw_total = dw_ltp + dw_ltd

    ax.fill_between(dt_range, dw_total, 0,
                    where=(dw_total > 0), alpha=0.3, color="#d7191c", label="LTP")
    ax.fill_between(dt_range, dw_total, 0,
                    where=(dw_total < 0), alpha=0.3, color="#2c7bb6", label="LTD")
    ax.plot(dt_range, dw_total, "k-", lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(r"$\Delta t = t_{post} - t_{pre}$ (ms)")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title("STDP Learning Window\n(Bi & Poo 1998)")
    ax.legend(frameon=False)
    _panel_label(ax, "A")

    # (B) Weight evolution under natural spiking
    ax = axes[1]
    if "w_history" in stdp_result:
        w_h = stdp_result["w_history"]
        t_w = np.arange(len(w_h))
        ax.plot(t_w, w_h, color="#1a9641", lw=1.2)
        ax.set_xlabel("Spike pair index")
        ax.set_ylabel("Synaptic weight w")
        ax.set_title("Weight Evolution\n(Poisson pre and post, 20 Hz)")
        ax.set_ylim(0, 1)
        _panel_label(ax, "B")

    # (C) Weight distribution at steady state
    ax = axes[2]
    if "w_final_dist" in stdp_result:
        ax.hist(stdp_result["w_final_dist"], bins=30, color="#ff7f00",
                edgecolor="white", alpha=0.8, density=True)
        ax.set_xlabel("Steady-state weight")
        ax.set_ylabel("Probability density")
        ax.set_title("Weight Distribution\n(100 synapses, t=100 s)")
    else:
        # Analytical prediction: bimodal distribution (Song et al. 2000)
        w_range = np.linspace(0, 1, 200)
        mu1, s1 = 0.1, 0.07
        mu2, s2 = 0.9, 0.07
        p_bimodal = (0.5 * np.exp(-0.5*((w_range-mu1)/s1)**2) / (s1*np.sqrt(2*np.pi)) +
                     0.5 * np.exp(-0.5*((w_range-mu2)/s2)**2) / (s2*np.sqrt(2*np.pi)))
        ax.plot(w_range, p_bimodal, color="#ff7f00", lw=2)
        ax.fill_between(w_range, p_bimodal, alpha=0.3, color="#ff7f00")
        ax.set_xlabel("Synaptic weight")
        ax.set_ylabel("Probability density")
        ax.set_title("Predicted Bimodal Distribution\n(Song et al. 2000)")
    _panel_label(ax, "C")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig7_stdp.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8 — Network Raster + Population Rate + LFP
# ═══════════════════════════════════════════════════════════════════════════
def fig8_network(net_result: dict, N_E: int, N_I: int,
                 save: bool = True) -> plt.Figure:
    """
    4-panel network figure: raster, E/I rates, LFP, E/I balance.
    """
    t = net_result["t"]
    spikes = net_result["spikes"]
    pop_E  = net_result["pop_E_rate"]
    pop_I  = net_result["pop_I_rate"]
    LFP    = net_result["LFP"]
    dt     = float(t[1] - t[0])

    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)
    fig.suptitle("E/I Recurrent Spiking Network — Population Dynamics",
                 fontsize=13, fontweight="bold")

    # ── (A) Raster plot ──────────────────────────────────────────────────
    ax_raster = fig.add_subplot(gs[0, :])
    max_shown = min(len(spikes), N_E + N_I)
    for nid in range(max_shown):
        st = np.array(spikes[nid])
        color = COLORS["E"] if nid < N_E else COLORS["I"]
        if len(st):
            ax_raster.scatter(st, np.full(len(st), nid),
                              s=0.8, c=color, alpha=0.6, linewidths=0)
    ax_raster.axhline(N_E - 0.5, color="k", lw=0.8, ls="--", alpha=0.4)
    ax_raster.set_ylabel("Neuron index")
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_title("Spike Raster (red=E, blue=I)")
    ax_raster.set_xlim(t[0], t[-1])
    _panel_label(ax_raster, "A")

    # ── (B) E/I population rates (smoothed) ──────────────────────────────
    ax_rate = fig.add_subplot(gs[1, :])
    from scipy.ndimage import uniform_filter1d
    smooth_win = max(1, int(5.0 / dt))   # 5 ms smoothing
    rate_E_sm = uniform_filter1d(pop_E, smooth_win)
    rate_I_sm = uniform_filter1d(pop_I, smooth_win)
    ax_rate.plot(t, rate_E_sm, color=COLORS["E"], lw=1.5, label="E neurons")
    ax_rate.plot(t, rate_I_sm, color=COLORS["I"], lw=1.5,
                 ls="--", label="I neurons")
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.set_xlabel("Time (ms)")
    ax_rate.legend(frameon=False, ncol=2)
    ax_rate.set_title("Population Firing Rates (5 ms smooth)")
    ax_rate.set_xlim(t[0], t[-1])
    _panel_label(ax_rate, "B")

    # ── (C) LFP proxy ────────────────────────────────────────────────────
    ax_lfp = fig.add_subplot(gs[2, :])
    ax_lfp.plot(t, LFP, color=COLORS["LFP"], lw=0.8, alpha=0.8)
    ax_lfp.set_ylabel("LFP proxy (mV)")
    ax_lfp.set_xlabel("Time (ms)")
    ax_lfp.set_title("Local Field Potential Proxy (Mean E-cell Voltage)")
    ax_lfp.set_xlim(t[0], t[-1])
    _panel_label(ax_lfp, "C")

    # ── (D) Firing rate histogram (E population) ──────────────────────────
    ax_hist = fig.add_subplot(gs[3, 0])
    T_s = (t[-1] - t[0]) * 1e-3
    rates_E = np.array([len(spikes[i]) / T_s for i in range(N_E)])
    rates_I = np.array([len(spikes[N_E + i]) / T_s for i in range(N_I)])
    ax_hist.hist(rates_E, bins=20, color=COLORS["E"], alpha=0.7,
                 label=f"E: μ={rates_E.mean():.1f} Hz", density=True)
    ax_hist.hist(rates_I, bins=20, color=COLORS["I"], alpha=0.7,
                 label=f"I: μ={rates_I.mean():.1f} Hz", density=True)
    ax_hist.set_xlabel("Firing rate (Hz)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Single-neuron Rate Distribution")
    ax_hist.legend(frameon=False)
    _panel_label(ax_hist, "D")

    # ── (E) E/I rate cross-correlogram ────────────────────────────────────
    ax_cc = fig.add_subplot(gs[3, 1])
    from scipy.signal import correlate
    lag_max = min(len(rate_E_sm) - 1, int(200 / dt))
    cc = correlate(rate_E_sm - rate_E_sm.mean(),
                   rate_I_sm - rate_I_sm.mean(),
                   mode="full")
    lags = np.arange(-lag_max, lag_max + 1) * dt
    center = len(cc) // 2
    cc_window = cc[center - lag_max: center + lag_max + 1]
    norm = np.std(rate_E_sm) * np.std(rate_I_sm) * len(rate_E_sm)
    ax_cc.plot(lags, cc_window / (norm + 1e-12), color="#984ea3", lw=1.5)
    ax_cc.axvline(0, color="k", lw=0.5, ls="--")
    ax_cc.axhline(0, color="k", lw=0.5)
    ax_cc.set_xlabel("Lag (ms)")
    ax_cc.set_ylabel("Normalised cross-correlation")
    ax_cc.set_title("E–I Rate Cross-Correlogram")
    _panel_label(ax_cc, "E")

    if save:
        fig.savefig(OUTDIR / "fig8_network_dynamics.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9 — ISI & CV-ISI Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig9_isi_analysis(isi_results: dict, pop_cv: np.ndarray,
                       save: bool = True) -> plt.Figure:
    """ISI distribution + fitted curves + population CV histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Inter-Spike Interval Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    isi = isi_results["isi"]
    if len(isi):
        ax.hist(isi, bins=isi_results["bin_edges"], density=True,
                color="#2c7bb6", alpha=0.7, label="Observed ISI")
        # overlay best fits
        isi_plot = np.linspace(0, isi.max(), 300)
        from scipy import stats
        fits = isi_results["fits"]
        best = isi_results.get("best_fit", "gamma")
        # gamma fit
        a  = fits["gamma"]["a"]
        sc = fits["gamma"]["scale"]
        ax.plot(isi_plot, stats.gamma.pdf(isi_plot, a, scale=sc),
                "r-", lw=2, label=f"Gamma (a={a:.2f})")
        # exponential
        ax.plot(isi_plot, stats.expon.pdf(isi_plot, scale=fits["exponential"]["scale"]),
                "g--", lw=1.5, label="Exponential (Poisson)")
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("Probability density")
        ax.set_title(f"ISI Distribution\n(best fit: {best})")
        ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "A")

    # Survival function (complementary CDF)
    ax = axes[1]
    if len(isi):
        isi_sorted = np.sort(isi)
        survival = 1.0 - np.arange(len(isi_sorted)) / len(isi_sorted)
        ax.semilogy(isi_sorted, survival, color="#2c7bb6", lw=2, label="Empirical")
        ax.semilogy(isi_plot,
                    1 - stats.gamma.cdf(isi_plot, a, scale=sc),
                    "r--", lw=1.5, label="Gamma fit")
        ax.semilogy(isi_plot,
                    1 - stats.expon.cdf(isi_plot,
                                        scale=fits["exponential"]["scale"]),
                    "g:", lw=1.5, label="Exponential")
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("P(ISI > x)")
        ax.set_title("ISI Survival Function")
        ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "B")

    # CV-ISI population distribution
    ax = axes[2]
    cv_clean = pop_cv[np.isfinite(pop_cv)]
    if len(cv_clean):
        ax.hist(cv_clean, bins=20, color="#ff7f00", alpha=0.8,
                edgecolor="white", density=True)
        ax.axvline(1.0, color="r", ls="--", lw=1.5, label="CV=1 (Poisson)")
        ax.axvline(float(np.mean(cv_clean)), color="k", ls="-", lw=1.5,
                   label=f"Mean CV={np.mean(cv_clean):.2f}")
        ax.set_xlabel("CV-ISI")
        ax.set_ylabel("Density")
        ax.set_title("Population CV-ISI\n(AI state: CV ≈ 1)")
        ax.legend(frameon=False)
    _panel_label(ax, "C")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig9_isi_analysis.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10 — Power Spectral Density
# ═══════════════════════════════════════════════════════════════════════════
def fig10_psd(psd_result: dict, save: bool = True) -> plt.Figure:
    """LFP power spectrum + band-power bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Local Field Potential Power Spectral Density",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    freqs = psd_result["freqs"]
    psd   = psd_result["psd"]
    mask  = (freqs >= 0.5) & (freqs <= 200)
    ax.semilogy(freqs[mask], psd[mask], color=COLORS["LFP"], lw=1.5)
    ax.axvline(psd_result["dominant_freq_hz"], color="r", ls="--", lw=1,
               label=f"Dominant: {psd_result['dominant_freq_hz']:.1f} Hz")

    band_colors = {"delta": "#4575b4", "theta": "#74add1",
                   "alpha": "#abd9e9", "beta": "#f46d43", "gamma": "#d73027"}
    band_ranges = {"delta": (0.5, 4), "theta": (4, 8),
                   "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 100)}
    for bname, (lo, hi) in band_ranges.items():
        bm = (freqs >= lo) & (freqs <= hi)
        if bm.any():
            ax.fill_between(freqs[bm], psd[bm], alpha=0.25,
                            color=band_colors[bname], label=bname.capitalize())

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (mV² / Hz)")
    ax.set_title("LFP Power Spectrum (Welch)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_xlim(0.5, 200)
    _panel_label(ax, "A")

    # Band powers bar chart
    ax = axes[1]
    bands = psd_result["band_powers"]
    names = list(bands.keys())
    powers = [bands[n] for n in names]
    colors = [band_colors[n] for n in names]
    bars = ax.bar(names, powers, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Band power (mV²)")
    ax.set_title("Frequency Band Powers")
    ax.set_xlabel("EEG Band")
    for bar, val in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2e}", ha="center", va="bottom", fontsize=8)
    _panel_label(ax, "B")

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig10_psd.png")
    return fig
