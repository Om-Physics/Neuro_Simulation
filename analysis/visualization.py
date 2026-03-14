"""
visualization.py
================
Publication-quality figure generation for computational neuroscience simulations.

Figure catalogue
----------------
  plot_action_potential()        — Single AP with ion-channel overlays
  plot_gating_variables()        — m(t), h(t), n(t) dynamics
  plot_fI_curve()                — Frequency-current relationship
  plot_nullclines()              — Phase-plane with vector field
  plot_isi_analysis()            — ISI histogram + CV/LV stats
  plot_adex_patterns()           — Cell-type firing patterns (RS/IB/CH/FS)
  plot_synapse_kinetics()        — AMPA/NMDA/GABA conductance traces
  plot_stdp_window()             — STDP learning window Δw vs Δt
  plot_weight_evolution()        — Synaptic weight trajectory
  plot_network_raster()          — Spike raster + PSTH + LFP
  plot_power_spectrum()          — PSD with band annotations
  plot_pairwise_correlation()    — Correlation histogram
  plot_summary_dashboard()       — 6-panel overview figure
  plot_bifurcation_fI()          — Multi-model f-I comparison
  plot_nmda_mg_block()           — NMDA Mg2+ unblock vs voltage

All functions accept an optional `ax` argument for embedding into larger
figure layouts; if None, a new figure is created and saved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

# ── Shared style ──────────────────────────────────────────────────────────
PALETTE = {
    "E":     "#E63946",   # excitatory — red
    "I":     "#457B9D",   # inhibitory — blue
    "Na":    "#E07A5F",   # sodium — orange-red
    "K":     "#3D405B",   # potassium — dark slate
    "L":     "#81B29A",   # leak — green
    "AMPA":  "#F2CC8F",   # AMPA — warm yellow
    "NMDA":  "#F4845F",   # NMDA — orange
    "GABAA": "#457B9D",   # GABA-A — blue
    "GABAB": "#9B5DE5",   # GABA-B — purple
    "LTP":   "#06D6A0",   # LTP — teal
    "LTD":   "#EF476F",   # LTD — pink
    "text":  "#2D2D2D",
    "bg":    "#FAFAFA",
}

def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.5,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
    })

_style()
OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def _save(fig, name: str) -> str:
    path = str(OUT / f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    return path


# ── 1. Action potential with ion-channel currents ─────────────────────────
def plot_action_potential(result: dict, title: str = "Hodgkin-Huxley Action Potential") -> str:
    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.12, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    t   = result["t"]
    V   = result["V"]

    # spike markers
    sp = result.get("spikes", np.array([]))
    sp_idx = [np.argmin(np.abs(t - s)) for s in sp]

    ax1.plot(t, V, color=PALETTE["E"], lw=1.8, label="V(t)")
    ax1.scatter(t[sp_idx], V[sp_idx], color="gold", s=40, zorder=5,
                label=f"Spikes (n={len(sp)})")
    ax1.axhline(-55, ls="--", lw=0.8, color="gray", label="V_thresh")
    ax1.set_ylabel("Membrane\npotential (mV)")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")

    if "I_Na" in result:
        ax2.plot(t, -result["I_Na"], color=PALETTE["Na"],  lw=1.5, label="−I_Na (inward)")
        ax2.plot(t,  result["I_K"],  color=PALETTE["K"],   lw=1.5, label="I_K (outward)")
        ax2.plot(t,  result["I_L"],  color=PALETTE["L"],   lw=1.0, ls="--", label="I_L")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_ylabel("Ion current\n(µA/cm²)")
        ax2.legend(fontsize=8, loc="upper right")

    if "m" in result:
        ax3.plot(t, result["m"], color=PALETTE["Na"],  lw=1.5, label="m (Na act.)")
        ax3.plot(t, result["h"], color=PALETTE["NMDA"],lw=1.5, label="h (Na inact.)")
        ax3.plot(t, result["n"], color=PALETTE["K"],   lw=1.5, label="n (K act.)")
        ax3.set_ylabel("Gating\nvariable")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=8, loc="upper right")

    ax3.set_xlabel("Time (ms)")
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.25, lw=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    return _save(fig, "01_action_potential")


# ── 2. f-I curve ──────────────────────────────────────────────────────────
def plot_fI_curve(fI_data: dict, title: str = "Frequency–Current (f-I) Curve") -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    I   = fI_data["I"]
    f   = fI_data["f_hz"]
    ax.plot(I, f, "o-", color=PALETTE["E"], lw=2, ms=5)
    ax.fill_between(I, 0, f, alpha=0.12, color=PALETTE["E"])

    # Rheobase annotation
    above = np.where(f > 0.5)[0]
    if len(above) > 0:
        rheobase = I[above[0]]
        ax.axvline(rheobase, ls="--", color="gray", lw=1.0)
        ax.text(rheobase + 0.02 * (I.max() - I.min()),
                0.05 * f.max(), f"Rheobase\n{rheobase:.2f} µA/cm²",
                fontsize=8, color="gray")

    ax.set_xlabel("Injected current $I$ (µA/cm²)")
    ax.set_ylabel("Firing rate $f$ (Hz)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    return _save(fig, "02_fI_curve")


# ── 3. Nullclines and phase plane ─────────────────────────────────────────
def plot_nullclines(nullcline_data: dict, trajectory: Optional[dict] = None,
                    title: str = "HH Phase Plane (V, n)") -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    V  = nullcline_data["V"]
    nV = nullcline_data.get("V_nc")    # dV/dt = 0
    nN = nullcline_data.get("n_nc")    # dn/dt = 0

    if nV is not None:
        ax.plot(V, nV, color=PALETTE["E"], lw=2.0, label="V-nullcline (dV/dt=0)")
    if nN is not None:
        ax.plot(V, nN, color=PALETTE["I"], lw=2.0, label="n-nullcline (dn/dt=0)")

    # vector field (if provided)
    if "VV" in nullcline_data and "dV" in nullcline_data:
        VV = nullcline_data["VV"]; NN = nullcline_data["NN"]
        dV = nullcline_data["dV"]; dn = nullcline_data["dn"]
        # subsample
        step = max(1, VV.shape[0] // 12)
        ax.quiver(VV[::step, ::step], NN[::step, ::step],
                  dV[::step, ::step], dn[::step, ::step],
                  alpha=0.35, color="gray", scale=None, width=0.003)

    if trajectory is not None:
        ax.plot(trajectory["V"], trajectory["n"],
                color="gold", lw=1.2, alpha=0.85, label="Trajectory")
        ax.plot(trajectory["V"][0], trajectory["n"][0],
                "go", ms=7, label="Start")

    ax.set_xlabel("Membrane potential V (mV)")
    ax.set_ylabel("K⁺ activation n")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.5)
    return _save(fig, "03_nullclines")


# ── 4. ISI analysis ───────────────────────────────────────────────────────
def plot_isi_analysis(sa, title: str = "ISI Analysis") -> str:
    """sa: SpikeAnalysis instance."""
    isi_hist = sa.isi_histogram(n_bins=40)
    inst     = sa.instantaneous_rate(sigma_ms=15.0)
    stats    = sa.summary()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # ISI histogram
    ax1.bar(isi_hist["bins"], isi_hist["pdf"], width=np.diff(isi_hist["edges"]),
            color=PALETTE["E"], alpha=0.8, edgecolor="white", lw=0.5)
    # fit exponential (Poisson prediction)
    if len(sa.isi) > 5 and np.mean(sa.isi) > 0:
        lam = 1 / np.mean(sa.isi)
        x   = np.linspace(0, sa.isi.max(), 200)
        ax1.plot(x, lam * np.exp(-lam * x), "k--", lw=1.5, label="Poisson fit")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("ISI (ms)"); ax1.set_ylabel("Probability density")
    ax1.set_title("ISI Distribution")

    # ISI return map
    isi = sa.isi
    if len(isi) > 2:
        ax2.scatter(isi[:-1], isi[1:], s=8, alpha=0.5, color=PALETTE["I"])
        lim = [0, isi.max() * 1.1]
        ax2.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("ISI_i (ms)"); ax2.set_ylabel("ISI_{i+1} (ms)")
    ax2.set_title("ISI Return Map")

    # Statistics bar chart
    metric_names  = ["CV ISI", "LV", "IR", "Burstiness"]
    metric_values = [stats["cv_isi"], stats["lv"], stats["ir"], stats["burstiness"]]
    colors = [PALETTE["E"] if not np.isnan(v) else "lightgray" for v in metric_values]
    metric_values = [0 if np.isnan(v) else v for v in metric_values]
    ax3.barh(metric_names, metric_values, color=colors, edgecolor="white")
    ax3.axvline(1.0, ls="--", color="gray", lw=1.0, label="Poisson (CV=1)")
    ax3.set_xlabel("Value"); ax3.set_title("Spike Statistics")
    ax3.legend(fontsize=8)

    # Instantaneous firing rate
    ax4.plot(inst["t"], inst["rate"], color=PALETTE["Na"], lw=1.5)
    ax4.fill_between(inst["t"], 0, inst["rate"], alpha=0.2, color=PALETTE["Na"])
    # spike ticks
    ax4b = ax4.twinx()
    if len(sa.spikes) > 0:
        ax4b.eventplot(sa.spikes, colors="gray", lineoffsets=0.5,
                       linelengths=0.7, linewidths=0.8, alpha=0.5)
    ax4b.set_ylim(0, 5); ax4b.set_yticks([])
    ax4.set_xlabel("Time (ms)"); ax4.set_ylabel("Firing rate (Hz)")
    ax4.set_title(f"Instantaneous Rate (σ={inst['sigma_ms']} ms kernel) | "
                  f"Mean: {stats['mean_rate_hz']:.1f} Hz | CV: {stats['cv_isi']:.3f}")
    ax4.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle(title, fontweight="bold", y=1.01)
    return _save(fig, "04_isi_analysis")


# ── 5. AdEx firing patterns ───────────────────────────────────────────────
def plot_adex_patterns(results_dict: dict) -> str:
    """
    results_dict: {preset_name: simulate_detailed result dict}
    """
    presets = list(results_dict.keys())
    n = len(presets)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n))
    for ax, (name, res), col in zip(axes, results_dict.items(), colors):
        t = res["t"]; V = res["V"]
        ax.plot(t, V, color=col, lw=1.4)
        sp = res.get("spikes", np.array([]))
        sp_idx = [np.argmin(np.abs(t - s)) for s in sp if s <= t[-1]]
        ax.scatter(t[sp_idx] if sp_idx else [], V[sp_idx] if sp_idx else [],
                   color="gold", s=25, zorder=5)
        ax.set_ylabel("V (mV)", fontsize=9)
        fr = res.get("firing_rate_hz", 0)
        cv = res.get("cv_isi", float("nan"))
        label = f"{name}  |  {fr:.1f} Hz  |  CV={cv:.2f}" if not np.isnan(cv) \
                else f"{name}  |  {fr:.1f} Hz"
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_ylim(-90, 50)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("AdEx Firing Patterns by Cell Type", fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "05_adex_patterns")


# ── 6. Synapse conductance kinetics ──────────────────────────────────────
def plot_synapse_kinetics(results_dict: dict) -> str:
    """
    results_dict: {syn_name: {'t': ..., 'g': ..., 'I': ...}}
    """
    n   = len(results_dict)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)
    col_map   = {"AMPA": PALETTE["AMPA"], "NMDA": PALETTE["NMDA"],
                 "GABA-A": PALETTE["GABAA"], "GABA-B": PALETTE["GABAB"],
                 "Alpha": "#A8DADC"}

    for col_i, (name, res) in enumerate(results_dict.items()):
        color = col_map.get(name, "#888")
        t = res["t"]; g = res["g"]; I = res["I"]
        # conductance
        ax = axes[0, col_i]
        ax.plot(t, g, color=color, lw=2.0)
        ax.fill_between(t, 0, g, alpha=0.2, color=color)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Conductance (nS)" if col_i == 0 else "")
        ax.grid(True, alpha=0.25, lw=0.5)
        # current
        ax2 = axes[1, col_i]
        ax2.plot(t, I, color=color, lw=2.0)
        ax2.fill_between(t, 0, I, alpha=0.2, color=color)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("I_syn (pA)" if col_i == 0 else "")
        ax2.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle("Synaptic Receptor Kinetics", fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "06_synapse_kinetics")


# ── 7. STDP window ────────────────────────────────────────────────────────
def plot_stdp_window(dt_range: np.ndarray, dw_ltp: np.ndarray,
                     dw_ltd: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dt_range[dt_range >= 0], dw_ltp[dt_range >= 0],
            color=PALETTE["LTP"], lw=2.5, label="LTP (Δt > 0, post after pre)")
    ax.plot(dt_range[dt_range <= 0], dw_ltd[dt_range <= 0],
            color=PALETTE["LTD"], lw=2.5, label="LTD (Δt < 0, pre after post)")
    ax.axhline(0, color="gray", lw=1.0)
    ax.axvline(0, color="gray", lw=1.0)
    ax.fill_between(dt_range[dt_range >= 0], 0, dw_ltp[dt_range >= 0],
                    alpha=0.15, color=PALETTE["LTP"])
    ax.fill_between(dt_range[dt_range <= 0], 0, dw_ltd[dt_range <= 0],
                    alpha=0.15, color=PALETTE["LTD"])
    ax.set_xlabel("Δt = t_post − t_pre (ms)")
    ax.set_ylabel("Synaptic weight change Δw")
    ax.set_title("STDP Learning Window (Bi & Poo 1998)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, lw=0.5)
    return _save(fig, "07_stdp_window")


# ── 8. Synaptic weight evolution ──────────────────────────────────────────
def plot_weight_evolution(t: np.ndarray, w: np.ndarray,
                          pre_spikes: np.ndarray = np.array([]),
                          post_spikes: np.ndarray = np.array([])) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={"height_ratios": [2, 1]},
                                    sharex=True)
    ax1.plot(t, w, color=PALETTE["AMPA"], lw=1.5)
    ax1.fill_between(t, w.min(), w, alpha=0.15, color=PALETTE["AMPA"])
    ax1.set_ylabel("Synaptic weight w")
    ax1.set_title("STDP Weight Evolution", fontweight="bold")
    ax1.grid(True, alpha=0.25, lw=0.5)

    if len(pre_spikes):
        ax2.eventplot(pre_spikes,  colors=PALETTE["I"], lineoffsets=1.5,
                      linelengths=0.8, linewidths=0.8, label="Pre-syn spikes")
    if len(post_spikes):
        ax2.eventplot(post_spikes, colors=PALETTE["E"], lineoffsets=0.5,
                      linelengths=0.8, linewidths=0.8, label="Post-syn spikes")
    ax2.set_yticks([0.5, 1.5]); ax2.set_yticklabels(["Post", "Pre"])
    ax2.set_xlabel("Time (ms)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2, lw=0.5)
    plt.tight_layout()
    return _save(fig, "08_weight_evolution")


# ── 9. Network raster + PSTH + LFP ───────────────────────────────────────
def plot_network_raster(net_result: dict, N_E: int, title: str = "Network Activity") -> str:
    t       = net_result["t"]
    spikes  = net_result["spikes"]
    N       = len(spikes)
    LFP     = net_result.get("LFP", np.zeros_like(t))
    pop_E   = net_result.get("pop_E_rate", np.zeros_like(t))
    pop_I   = net_result.get("pop_I_rate", np.zeros_like(t))

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08,
                            height_ratios=[3, 1.2, 1.2, 1.2])
    ax_raster = fig.add_subplot(gs[0])
    ax_psth   = fig.add_subplot(gs[1], sharex=ax_raster)
    ax_lfp    = fig.add_subplot(gs[2], sharex=ax_raster)
    ax_rate   = fig.add_subplot(gs[3], sharex=ax_raster)

    # Raster
    for nid, sp in enumerate(spikes):
        color = PALETTE["E"] if nid < N_E else PALETTE["I"]
        if len(sp) > 0:
            ax_raster.scatter(sp, [nid] * len(sp), s=0.8,
                              color=color, alpha=0.6, linewidths=0)
    ax_raster.axhline(N_E, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_raster.set_ylabel("Neuron index")
    ax_raster.set_title(title, fontweight="bold")

    # E/I legend
    e_patch = mpatches.Patch(color=PALETTE["E"], label=f"Excitatory ({N_E})")
    i_patch = mpatches.Patch(color=PALETTE["I"], label=f"Inhibitory ({N - N_E})")
    ax_raster.legend(handles=[e_patch, i_patch], fontsize=8, loc="upper right")

    # PSTH (smoothed with 10 ms window)
    from scipy.ndimage import uniform_filter1d
    smooth_E = uniform_filter1d(pop_E, size=max(1, int(10 / (t[1]-t[0]))))
    smooth_I = uniform_filter1d(pop_I, size=max(1, int(10 / (t[1]-t[0]))))
    ax_psth.plot(t, smooth_E, color=PALETTE["E"], lw=1.4, label="E pop. rate")
    ax_psth.plot(t, smooth_I, color=PALETTE["I"], lw=1.4, label="I pop. rate")
    ax_psth.set_ylabel("Rate (Hz)")
    ax_psth.legend(fontsize=8, loc="upper right")

    # LFP proxy
    ax_lfp.plot(t, LFP, color="#555", lw=0.8, alpha=0.85)
    ax_lfp.set_ylabel("LFP proxy\n(mean V, mV)")

    # E/I balance
    ratio = np.where(smooth_I > 0.1, smooth_E / smooth_I, np.nan)
    ax_rate.plot(t, ratio, color=PALETTE["NMDA"], lw=1.2)
    ax_rate.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_rate.set_ylabel("E/I ratio")
    ax_rate.set_xlabel("Time (ms)")

    for ax in [ax_raster, ax_psth, ax_lfp]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.grid(True, alpha=0.15, lw=0.5)
    ax_rate.grid(True, alpha=0.15, lw=0.5)
    return _save(fig, "09_network_raster")


# ── 10. Power spectrum ────────────────────────────────────────────────────
def plot_power_spectrum(psd_data: dict, band_data: Optional[dict] = None,
                        title: str = "LFP Power Spectrum") -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    f = psd_data["f"]; Pxx_dB = psd_data["Pxx_dB"]; Pxx = psd_data["Pxx"]
    bands = {"delta":(0.5,4,"#A8DADC"), "theta":(4,8,"#457B9D"),
             "alpha":(8,13,"#E07A5F"), "beta":(13,30,"#F2CC8F"),
             "gamma":(30,100,"#E63946")}

    ax1.semilogy(f, Pxx, color=PALETTE["I"], lw=1.5)
    for name, (lo, hi, col) in bands.items():
        mask = (f >= lo) & (f <= hi)
        ax1.fill_between(f[mask], Pxx[mask], alpha=0.35, color=col,
                         label=f"{name.capitalize()} ({lo}–{hi} Hz)")
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Power (V²/Hz)")
    ax1.set_title("PSD (log-log)")
    ax1.set_xlim(0, min(150, f[-1])); ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.2, lw=0.5)

    ax2.plot(f, Pxx_dB, color=PALETTE["I"], lw=1.5)
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Power (dB)")
    ax2.set_title("PSD (dB)")
    ax2.set_xlim(0, min(150, f[-1]))
    ax2.grid(True, alpha=0.2, lw=0.5)

    if band_data:
        names = list(band_data.keys())
        vals  = [band_data[k] * 100 for k in names]
        inset = ax2.inset_axes([0.55, 0.55, 0.42, 0.38])
        xs = np.arange(len(names))
        inset.bar(xs, vals, color=[bands[n][2] for n in names], edgecolor="white")
        inset.set_xticks(xs); inset.set_xticklabels(names, rotation=45, ha="right")
        inset.set_ylabel("Rel. power (%)", fontsize=7)
        inset.set_title("Band power", fontsize=7)
        inset.tick_params(labelsize=6)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "10_power_spectrum")


# ── 11. NMDA Mg²⁺ block ───────────────────────────────────────────────────
def plot_nmda_mg_block(Mg_concs: list = [0.0, 0.5, 1.0, 2.0]) -> str:
    V_range = np.linspace(-90, 60, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(Mg_concs)))
    for Mg, col in zip(Mg_concs, colors):
        B = 1.0 / (1.0 + np.exp(-0.062 * V_range) * Mg / 3.57)
        ax.plot(V_range, B, color=col, lw=2,
                label=f"[Mg²⁺]={Mg} mM")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="50% unblock")
    ax.set_xlabel("Membrane potential V (mV)")
    ax.set_ylabel("Mg²⁺ unblock factor B(V)")
    ax.set_title("NMDA Receptor Mg²⁺ Voltage Block (Jahr & Stevens 1990)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)
    return _save(fig, "11_nmda_mg_block")


# ── 12. Summary dashboard ─────────────────────────────────────────────────
def plot_summary_dashboard(hh_result: dict, fI_data: dict,
                            sa, nullclines: dict) -> str:
    """6-panel overview figure for the HH model."""
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])   # voltage trace
    ax2 = fig.add_subplot(gs[0, 1])   # ion currents
    ax3 = fig.add_subplot(gs[0, 2])   # gating variables
    ax4 = fig.add_subplot(gs[1, 0])   # f-I curve
    ax5 = fig.add_subplot(gs[1, 1])   # phase plane
    ax6 = fig.add_subplot(gs[1, 2])   # ISI histogram

    t = hh_result["t"]; V = hh_result["V"]
    sp = hh_result.get("spikes", np.array([]))
    sp_idx = [np.argmin(np.abs(t - s)) for s in sp]

    # Panel 1 — V(t)
    ax1.plot(t, V, color=PALETTE["E"], lw=1.5)
    ax1.scatter(t[sp_idx] if sp_idx else [],
                V[sp_idx] if sp_idx else [],
                color="gold", s=20, zorder=5)
    ax1.set(xlabel="Time (ms)", ylabel="V (mV)", title="Membrane Potential")
    ax1.grid(True, alpha=0.2, lw=0.5)

    # Panel 2 — Ion currents
    if "I_Na" in hh_result:
        ax2.plot(t, -hh_result["I_Na"], color=PALETTE["Na"],  lw=1.3, label="-I_Na")
        ax2.plot(t,  hh_result["I_K"],  color=PALETTE["K"],   lw=1.3, label="I_K")
        ax2.plot(t,  hh_result["I_L"],  color=PALETTE["L"],   lw=1.0, ls="--", label="I_L")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2, lw=0.5)
    ax2.set(xlabel="Time (ms)", ylabel="Current (µA/cm²)", title="Ion Currents")

    # Panel 3 — Gating variables
    if "m" in hh_result:
        ax3.plot(t, hh_result["m"], color=PALETTE["Na"],   lw=1.3, label="m")
        ax3.plot(t, hh_result["h"], color=PALETTE["NMDA"], lw=1.3, label="h")
        ax3.plot(t, hh_result["n"], color=PALETTE["K"],    lw=1.3, label="n")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2, lw=0.5)
    ax3.set(xlabel="Time (ms)", ylabel="Gating variable", title="HH Gate Dynamics")

    # Panel 4 — f-I
    ax4.plot(fI_data["I"], fI_data["f_hz"], "o-", color=PALETTE["E"], ms=4, lw=1.8)
    ax4.fill_between(fI_data["I"], 0, fI_data["f_hz"], alpha=0.1, color=PALETTE["E"])
    ax4.set(xlabel="I (µA/cm²)", ylabel="f (Hz)", title="f-I Curve")
    ax4.grid(True, alpha=0.2, lw=0.5)

    # Panel 5 — Phase plane
    V_nc = nullclines.get("V_nc"); n_nc = nullclines.get("n_nc")
    Vx   = nullclines.get("V")
    if V_nc is not None and Vx is not None:
        ax5.plot(Vx, V_nc, color=PALETTE["E"], lw=2, label="V-null")
        ax5.plot(Vx, n_nc, color=PALETTE["I"], lw=2, label="n-null")
        ax5.set_ylim(0, 1)
    ax5.set(xlabel="V (mV)", ylabel="n", title="Phase Plane"); ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.2, lw=0.5)

    # Panel 6 — ISI histogram
    isi_h = sa.isi_histogram(n_bins=30)
    if len(isi_h["bins"]) > 0:
        ax6.bar(isi_h["bins"], isi_h["pdf"], width=np.diff(isi_h["edges"]),
                color=PALETTE["Na"], alpha=0.8, edgecolor="white")
    stats = sa.summary()
    ax6.set_title(f"ISI | CV={stats['cv_isi']:.3f} | {stats['mean_rate_hz']:.1f} Hz")
    ax6.set(xlabel="ISI (ms)", ylabel="PDF")
    ax6.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle("Hodgkin-Huxley Neuron — Summary Dashboard",
                 fontweight="bold", fontsize=14, y=1.01)
    return _save(fig, "12_summary_dashboard")


# ── 13. Multi-model f-I comparison ───────────────────────────────────────
def plot_bifurcation_fI(models_fI: dict) -> str:
    """models_fI: {model_name: fI_data_dict}"""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Set1(np.linspace(0, 0.85, len(models_fI)))
    for (name, fI), col in zip(models_fI.items(), colors):
        ax.plot(fI["I"], fI["f_hz"], "o-", color=col, lw=2, ms=4, label=name)
    ax.set_xlabel("Injected current (pA or µA/cm²)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("f-I Curve Comparison: HH vs LIF vs AdEx", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)
    return _save(fig, "13_model_fI_comparison")


# ── 14. Pairwise correlation histogram ───────────────────────────────────
def plot_pairwise_correlation(corr_data: dict) -> str:
    r = corr_data["r_values"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r[~np.isnan(r)], bins=40, color=PALETTE["I"], edgecolor="white", alpha=0.85)
    ax.axvline(corr_data["mean_r"], color=PALETTE["E"], lw=2,
               label=f'Mean r = {corr_data["mean_r"]:.3f}')
    ax.axvline(0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Pearson correlation r"); ax.set_ylabel("Count")
    ax.set_title("Pairwise Spike-Count Correlations", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)
    return _save(fig, "14_pairwise_correlation")


# ── 15. Time constants vs voltage ─────────────────────────────────────────
def plot_time_constants(neuron) -> str:
    V_range = np.linspace(-100, 50, 300)
    tau_m = []; tau_h = []; tau_n = []
    m_inf = []; h_inf = []; n_inf = []
    for V in V_range:
        tc = neuron.time_constants(V)
        tau_m.append(tc["tau_m"]); tau_h.append(tc["tau_h"]); tau_n.append(tc["tau_n"])
        ss = neuron._steady_state(V)
        m_inf.append(ss[0]); h_inf.append(ss[1]); n_inf.append(ss[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(V_range, tau_m, color=PALETTE["Na"],   lw=2, label="τ_m (Na act.)")
    ax1.plot(V_range, tau_h, color=PALETTE["NMDA"], lw=2, label="τ_h (Na inact.)")
    ax1.plot(V_range, tau_n, color=PALETTE["K"],    lw=2, label="τ_n (K act.)")
    ax1.set_xlabel("V (mV)"); ax1.set_ylabel("Time constant (ms)")
    ax1.set_title("HH Gating Time Constants τ(V)"); ax1.legend(); ax1.grid(True, alpha=0.2)

    ax2.plot(V_range, m_inf, color=PALETTE["Na"],   lw=2, label="m∞(V)")
    ax2.plot(V_range, h_inf, color=PALETTE["NMDA"], lw=2, label="h∞(V)")
    ax2.plot(V_range, n_inf, color=PALETTE["K"],    lw=2, label="n∞(V)")
    ax2.set_xlabel("V (mV)"); ax2.set_ylabel("Steady-state activation")
    ax2.set_title("HH Steady-State Gating x∞(V)"); ax2.legend(); ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return _save(fig, "15_time_constants")
