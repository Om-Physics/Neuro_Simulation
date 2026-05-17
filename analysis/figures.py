"""
figures.py
==========
Publication-quality figure generation for all eleven simulation results.
Each function is self-contained: it runs the required simulation internally,
produces a matplotlib figure, and saves it to the specified output directory.

Figures produced:
  01  Hodgkin-Huxley action potential anatomy
  02  Phase-plane analysis (V-n nullclines and limit cycle)
  03  Frequency-current (f-I) curves for HH, LIF, AdEx-RS, AdEx-FS
  04  AdEx firing-pattern gallery (RS, IB, CH, FS, LTS)
  05  Synaptic conductance kinetics (AMPA, NMDA, GABA-A, GABA-B)
  06  NMDA Mg2+ block: B(V) and N-shaped I-V curve
  07  STDP learning window, weight convergence, and steady-state distribution
  08  E/I network raster, population rates, LFP proxy
  09  ISI statistics: histogram, Poincare map, CV distribution
  10  LFP power spectral density with frequency-band decomposition
  11  Summary dashboard (all major results in one figure)

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter1d

from neurons.hodgkin_huxley import HodgkinHuxley
from neurons.integrate_fire import LeakyIntegrateAndFire, AdaptiveExponentialIF, ADEX_PRESETS
from synapses.synapse import AMPASynapse, NMDASynapse, GABAAsynapse, GABABsynapse
from synapses.plasticity import STDPRule
from networks.network import SpikingNetwork
from analysis.spike_analysis import (
    isi_statistics, fano_factor, power_spectrum, autocorrelogram
)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
}

COLORS = {
    "V":  "#2563EB",
    "m":  "#DC2626",
    "h":  "#16A34A",
    "n":  "#D97706",
    "Na": "#DC2626",
    "K":  "#D97706",
    "L":  "#6B7280",
    "E":  "#EF4444",
    "I":  "#3B82F6",
    "A":  "#16A34A",
    "N":  "#7C3AED",
    "GA": "#2563EB",
    "GB": "#EA580C",
    "W":  "#B45309",
    "LTP":"#DC2626",
    "LTD":"#2563EB",
}


def _save(fig: plt.Figure, path: str, name: str) -> str:
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, name)
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved {name}")
    return fpath


def fig01_action_potential(out_dir: str = "figures") -> str:
    """Figure 01: HH action potential anatomy."""
    with plt.rc_context(STYLE):
        hh = HodgkinHuxley()
        data = hh.simulate_detailed(T=120.0, dt=0.025, I_ext=10.0)

        fig = plt.figure(figsize=(11, 8))
        gs = gridspec.GridSpec(4, 1, hspace=0.45)
        t = data["t"]

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t, data["V"], color=COLORS["V"], lw=1.5, label="V(t)")
        ax1.axhline(55,  ls="--", color=COLORS["Na"], lw=0.8, alpha=0.6, label="E_Na")
        ax1.axhline(-77, ls="--", color=COLORS["K"],  lw=0.8, alpha=0.6, label="E_K")
        ax1.axhline(-54.4, ls="--", color=COLORS["L"], lw=0.8, alpha=0.6, label="E_L")
        ax1.set_ylabel("V (mV)"); ax1.set_title("Hodgkin-Huxley Action Potential")
        ax1.legend(loc="upper right", ncol=4)
        ax1.axvspan(10, 120, alpha=0.04, color="blue")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t, data["m"], color=COLORS["m"], lw=1.2, label="m (Na act.)")
        ax2.plot(t, data["h"], color=COLORS["h"], lw=1.2, label="h (Na inact.)")
        ax2.plot(t, data["n"], color=COLORS["n"], lw=1.2, label="n (K act.)")
        ax2.set_ylabel("Gate (0-1)"); ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="upper right", ncol=3)

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(t, -data["I_Na"], color=COLORS["Na"], lw=1.2, label="I_Na (inward)")
        ax3.plot(t,  data["I_K"],  color=COLORS["K"],  lw=1.2, label="I_K (outward)")
        ax3.plot(t,  data["I_L"],  color=COLORS["L"],  lw=1.0, label="I_L (leak)")
        ax3.set_ylabel("Current (µA/cm²)")
        ax3.legend(loc="upper right", ncol=3)

        ax4 = fig.add_subplot(gs[3])
        ax4.step(t, data["I_ext"], color="#374151", lw=1.2, where="post")
        ax4.set_ylabel("I_ext (µA/cm²)")
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylim(-1, 13)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, 120)

        n_spikes = len(data["spikes"])
        rate = round(n_spikes / 0.11)
        fig.suptitle(
            f"HH model · {n_spikes} spikes · {rate} Hz · squid axon 6.3°C · RK4 dt=0.025 ms",
            fontsize=9, color="#374151"
        )
        return _save(fig, out_dir, "fig_01_hh_action_potential.png")


def fig02_phase_plane(out_dir: str = "figures") -> str:
    """Figure 02: V-n phase plane with nullclines and limit cycle."""
    with plt.rc_context(STYLE):
        hh = HodgkinHuxley()
        nc  = hh.nullclines(I_ext=10.0)
        rec = hh.simulate(T=200.0, dt=0.025, I_ext=10.0, t_start=10.0)
        tc  = hh.time_constants()

        m   = rec.metadata["m"]
        n   = rec.metadata["n"]
        h   = rec.metadata["h"]
        V   = rec.voltages

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        ax1 = axes[0]
        ax1.plot(V, n, color=COLORS["V"], lw=0.7, alpha=0.85, label="Limit cycle")
        ax1.plot(nc["V"], np.clip(nc["V_null"], 0, 1), color=COLORS["Na"],
                 lw=2, label="V-nullcline")
        ax1.plot(nc["V"], nc["n_null"], color=COLORS["n"],
                 lw=2, label="n-nullcline")
        ax1.set_xlabel("Membrane voltage V (mV)")
        ax1.set_ylabel("K+ activation n")
        ax1.set_xlim(-85, 50)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title("V-n Phase Plane")
        ax1.legend(loc="upper left")

        ax2 = axes[1]
        Vr = tc["V"]
        ax2.plot(Vr, tc["tau_m"], color=COLORS["m"], lw=1.8, label="tau_m (Na act.)")
        ax2.plot(Vr, tc["tau_h"], color=COLORS["h"], lw=1.8, label="tau_h (Na inact.)")
        ax2.plot(Vr, tc["tau_n"], color=COLORS["n"], lw=1.8, label="tau_n (K act.)")
        Vr2 = np.linspace(-85, 55, 300)
        m_inf = np.array([hh._alpha_m(v)/(hh._alpha_m(v)+hh._beta_m(v)) for v in Vr2])
        h_inf = np.array([hh._alpha_h(v)/(hh._alpha_h(v)+hh._beta_h(v)) for v in Vr2])
        n_inf = nc["n_inf"]
        m_inf_r = np.array([hh._alpha_m(v)/(hh._alpha_m(v)+hh._beta_m(v)) for v in Vr])
        h_inf_r = np.array([hh._alpha_h(v)/(hh._alpha_h(v)+hh._beta_h(v)) for v in Vr])
        n_inf_r = np.array([hh._alpha_n(v)/(hh._alpha_n(v)+hh._beta_n(v)) for v in Vr])
        ax2b = ax2.twinx()
        ax2b.plot(Vr, m_inf_r, color=COLORS["m"], lw=1, ls="--", alpha=0.6, label="m_inf")
        ax2b.plot(Vr, h_inf_r, color=COLORS["h"], lw=1, ls="--", alpha=0.6, label="h_inf")
        ax2b.plot(Vr, n_inf_r, color=COLORS["n"], lw=1, ls="--", alpha=0.6, label="n_inf")
        ax2b.set_ylabel("Steady-state gate (0-1)")
        ax2.set_xlabel("Voltage (mV)")
        ax2.set_ylabel("Time constant (ms)")
        ax2.set_title("Gating Kinetics and Steady States")
        ax2.legend(loc="upper right")

        fig.suptitle("Phase-Plane Analysis of Hodgkin-Huxley Model", fontsize=11)
        return _save(fig, out_dir, "fig_02_phase_plane.png")


def fig03_fi_curves(out_dir: str = "figures") -> str:
    """Figure 03: Frequency-current (f-I) curves."""
    with plt.rc_context(STYLE):
        I_hh = np.arange(0, 25, 0.5)
        I_lif = np.arange(0, 700, 10.0)

        hh = HodgkinHuxley()
        _, rates_hh = hh.fI_curve(I_hh, T=400, dt=0.025, warmup=100)

        lif = LeakyIntegrateAndFire()
        _, rates_lif = lif.fI_curve(I_lif, T=400, dt=0.1, warmup=100)
        rates_lif_a = np.array([lif.analytical_rate(I) for I in I_lif])

        rs  = AdaptiveExponentialIF.from_preset("RS")
        fs  = AdaptiveExponentialIF.from_preset("FS")
        _, rates_rs = rs.fI_curve(I_lif, T=400, dt=0.1, warmup=100)
        _, rates_fs = fs.fI_curve(I_lif, T=400, dt=0.1, warmup=100)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        ax1 = axes[0]
        ax1.plot(I_hh, rates_hh, color=COLORS["V"], lw=2, label="HH (simulated)")
        ax1.set_xlabel("Injected current (µA/cm²)")
        ax1.set_ylabel("Firing rate (Hz)")
        ax1.set_title("HH f-I Curve")
        ax1.legend()

        ax2 = axes[1]
        ax2.plot(I_lif, rates_lif,   color=COLORS["K"],  lw=2, label="LIF (simulated)")
        ax2.plot(I_lif, rates_lif_a, color=COLORS["K"],  lw=1.5, ls="--", alpha=0.6,
                 label="LIF (analytical)")
        ax2.plot(I_lif, rates_rs,    color=COLORS["Na"], lw=2, label="AdEx-RS (adapting)")
        ax2.plot(I_lif, rates_fs,    color=COLORS["A"],  lw=2, label="AdEx-FS (non-adapting)")
        ax2.set_xlabel("Injected current (pA)")
        ax2.set_title("LIF vs AdEx f-I Comparison")
        ax2.legend()

        fig.suptitle("Frequency-Current (f-I) Relationships", fontsize=11)
        return _save(fig, out_dir, "fig_03_fi_curves.png")


def fig04_adex_patterns(out_dir: str = "figures") -> str:
    """Figure 04: AdEx firing-pattern gallery."""
    with plt.rc_context(STYLE):
        presets = ["RS", "IB", "CH", "FS", "LTS"]
        fig, axes = plt.subplots(len(presets), 2, figsize=(12, 10))
        fig.suptitle("AdEx Cortical Cell-Type Firing Patterns (Naud et al. 2008)", fontsize=11)

        for row, preset in enumerate(presets):
            p = ADEX_PRESETS[preset]
            neuron = AdaptiveExponentialIF.from_preset(preset)
            rec = neuron.simulate(T=1000.0, dt=0.1, I_ext=p["I_default"], t_start=100.0)
            t  = rec.time_axis
            V  = rec.voltages
            w  = rec.metadata["w"]

            ax_v = axes[row, 0]
            ax_v.plot(t, V, color=COLORS["V"], lw=1.0)
            ax_v2 = ax_v.twinx()
            ax_v2.plot(t, w, color=COLORS["W"], lw=0.8, ls="--", alpha=0.7)
            ax_v2.set_ylabel("w (pA)", fontsize=7, color=COLORS["W"])
            ax_v.set_ylabel("V (mV)")
            ax_v.set_title(f"{preset} — {p['label']}  ({len(rec.times)} spikes)", fontsize=9)
            ax_v.set_xlim(0, 1000)

            ax_isi = axes[row, 1]
            isi_stats = isi_statistics(rec.times)
            isi = isi_stats["isi"]
            if len(isi) > 1:
                ax_isi.hist(isi, bins=30, color=COLORS["V"], edgecolor="white",
                            alpha=0.8, density=True)
                ax_isi.set_title(
                    f"ISI histogram  CV={isi_stats['cv']:.3f}", fontsize=9
                )
            else:
                ax_isi.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                            transform=ax_isi.transAxes)
            ax_isi.set_xlabel("ISI (ms)")
            ax_isi.set_ylabel("Density")

            if row < len(presets) - 1:
                ax_v.set_xticklabels([])

        axes[-1, 0].set_xlabel("Time (ms)")
        fig.tight_layout()
        return _save(fig, out_dir, "fig_04_adex_patterns.png")


def fig05_synapse_kinetics(out_dir: str = "figures") -> str:
    """Figure 05: Synaptic conductance kinetics."""
    with plt.rc_context(STYLE):
        ampa  = AMPASynapse(g_max=0.5)
        nmda  = NMDASynapse(g_max=0.5, Mg=1.0)
        gabaa = GABAAsynapse(g_max=0.8)
        gabab = GABABsynapse(g_max=0.5)

        sp = [10.0]
        sp4 = [10.0, 30.0, 50.0, 70.0]

        d_a  = ampa.simulate(T=150, dt=0.1, spike_times=sp)
        d_n  = nmda.simulate(T=300, dt=0.1, spike_times=sp, V_post=-65)
        d_ga = gabaa.simulate(T=150, dt=0.1, spike_times=sp)
        d_gb = gabab.simulate(T=500, dt=0.1, spike_times=sp4)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle("Synaptic Receptor Kinetics (Destexhe et al. 1994)", fontsize=11)

        axes[0,0].plot(d_a["t"],  d_a["g"]*1000,  color=COLORS["A"],  lw=1.8)
        axes[0,0].set_title("AMPA  (E_rev = 0 mV, tau ~5 ms)")
        axes[0,0].set_ylabel("g (pS)")

        axes[0,1].plot(d_n["t"],  d_n["g"]*1000,  color=COLORS["N"],  lw=1.8)
        axes[0,1].set_title("NMDA  (E_rev = 0 mV, tau ~150 ms, Mg block)")
        axes[0,1].set_ylabel("g · B(V) (pS)")

        axes[1,0].plot(d_ga["t"], d_ga["g"]*1000, color=COLORS["GA"], lw=1.8)
        axes[1,0].set_title("GABA-A  (E_rev = -70 mV, tau ~5 ms)")
        axes[1,0].set_ylabel("g (pS)")
        axes[1,0].set_xlabel("Time (ms)")

        axes[1,1].plot(d_gb["t"], d_gb["g"]*1e6,  color=COLORS["GB"], lw=1.8)
        axes[1,1].set_title("GABA-B  (E_rev = -95 mV, G-protein cascade, 4 spikes)")
        axes[1,1].set_ylabel("g (fS)")
        axes[1,1].set_xlabel("Time (ms)")

        for ax in axes.flat:
            for st in sp4:
                ax.axvline(st, color="#9CA3AF", lw=0.6, ls="--", alpha=0.5)

        fig.tight_layout()
        return _save(fig, out_dir, "fig_05_synapse_kinetics.png")


def fig06_nmda_mg_block(out_dir: str = "figures") -> str:
    """Figure 06: NMDA Mg2+ block and N-shaped I-V."""
    with plt.rc_context(STYLE):
        V_range = np.linspace(-90, 60, 400)
        Mg_vals = [0.0, 0.5, 1.0, 2.0]
        colors  = ["#9CA3AF", "#16A34A", "#7C3AED", "#DC2626"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("NMDA Receptor Mg2+ Block (Jahr and Stevens 1990)", fontsize=11)

        ax1 = axes[0]
        for Mg, col in zip(Mg_vals, colors):
            syn = NMDASynapse(Mg=Mg)
            B   = np.array([syn.B(V) for V in V_range])
            ax1.plot(V_range, B, color=col, lw=1.8,
                     label=f"[Mg2+] = {Mg} mM")
        ax1.axvline(-65, ls="--", color="#6B7280", lw=0.8, label="V_rest")
        ax1.set_xlabel("Membrane voltage (mV)")
        ax1.set_ylabel("Unblock factor B(V)")
        ax1.set_title("Mg2+ Voltage-Dependent Block")
        ax1.legend()

        ax2 = axes[1]
        for Mg, col in zip(Mg_vals, colors):
            syn = NMDASynapse(g_max=1.0, Mg=Mg)
            iv  = syn.iv_curve(V_range)
            ax2.plot(V_range, -iv["I"], color=col, lw=1.8,
                     label=f"[Mg2+] = {Mg} mM")
        ax2.axhline(0, color="#6B7280", lw=0.6)
        ax2.axvline(-65, ls="--", color="#6B7280", lw=0.8)
        ax2.set_xlabel("Membrane voltage (mV)")
        ax2.set_ylabel("Inward current (-I, a.u.)")
        ax2.set_title("N-Shaped I-V Curve (physiological Mg2+)")
        ax2.legend()

        fig.tight_layout()
        return _save(fig, out_dir, "fig_06_nmda_mg_block.png")


def fig07_stdp(out_dir: str = "figures") -> str:
    """Figure 07: STDP learning window, weight evolution, and distribution."""
    with plt.rc_context(STYLE):
        stdp = STDPRule(A_plus=0.010, A_minus=0.0105, tau_plus=20, tau_minus=20)

        dt_range = np.linspace(-100, 100, 500)
        dw_theory = stdp.learning_window(dt_range)

        result = stdp.run(
            T=15000, dt=0.1, rate_pre=20, rate_post=20,
            w_init=0.5, n_synapses=50, seed=42
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Spike-Timing Dependent Plasticity (Bi and Poo 1998)", fontsize=11)

        ax1 = axes[0]
        ax1.fill_between(dt_range[dt_range > 0], 0, dw_theory[dt_range > 0],
                         color=COLORS["LTP"], alpha=0.25, label="LTP region")
        ax1.fill_between(dt_range[dt_range < 0], 0, dw_theory[dt_range < 0],
                         color=COLORS["LTD"], alpha=0.25, label="LTD region")
        ax1.plot(dt_range, dw_theory, color="#1F2937", lw=2)
        ax1.axhline(0, color="#6B7280", lw=0.6)
        ax1.axvline(0, color="#6B7280", lw=0.6, ls="--")
        ax1.set_xlabel("Delta t = t_post - t_pre (ms)")
        ax1.set_ylabel("Weight change dw")
        ax1.set_title("STDP Learning Window")
        ax1.legend()

        ax2 = axes[1]
        t_s = result["t"] * 1e-3
        ax2.plot(t_s, result["w_history"][:, 0], color=COLORS["W"], lw=1.5,
                 label="Single synapse")
        ax2.plot(t_s, result["w_history"].mean(axis=1), color="#1F2937",
                 lw=1.2, ls="--", label="Population mean")
        ax2.axhline(0.5, color="#6B7280", lw=0.6, ls=":")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Synaptic weight w")
        ax2.set_title("Weight Evolution (20 Hz Poisson)")
        ax2.set_ylim(0, 1)
        ax2.legend()

        ax3 = axes[2]
        w_final = result["w_final"]
        ax3.hist(w_final, bins=20, color=COLORS["V"], edgecolor="white",
                 alpha=0.85, density=True)
        ax3.axvline(np.median(w_final), color=COLORS["Na"], lw=1.5, ls="--",
                    label=f"Median = {np.median(w_final):.3f}")
        ax3.set_xlabel("Final weight w")
        ax3.set_ylabel("Density")
        ax3.set_title(f"Steady-State Distribution (n=50 synapses)")
        ax3.legend()

        fig.tight_layout()
        return _save(fig, out_dir, "fig_07_stdp.png")


def fig08_network_dynamics(out_dir: str = "figures") -> str:
    """Figure 08: E/I network raster, population rates, LFP proxy."""
    with plt.rc_context(STYLE):
        net = SpikingNetwork(N_E=100, N_I=25, dt=0.2, seed=42)
        res = net.run(T=400.0)

        fig = plt.figure(figsize=(12, 9))
        gs  = gridspec.GridSpec(4, 1, hspace=0.45)
        t   = res["t"]

        ax1 = fig.add_subplot(gs[0])
        mask_E = res["raster_id"] < net.N_E
        mask_I = ~mask_E
        ax1.scatter(res["raster_t"][mask_E], res["raster_id"][mask_E],
                    s=1.0, c=COLORS["E"], alpha=0.6, label="Excitatory")
        ax1.scatter(res["raster_t"][mask_I], res["raster_id"][mask_I],
                    s=1.5, c=COLORS["I"], alpha=0.8, label="Inhibitory")
        ax1.axhline(net.N_E, color="#9CA3AF", lw=0.5, ls="--")
        ax1.set_ylabel("Neuron index"); ax1.set_title("Raster Plot")
        ax1.legend(loc="upper right", markerscale=4)

        ax2 = fig.add_subplot(gs[1])
        win_ms = 10.0
        sigma  = win_ms / (net.dt)
        rE_sm  = gaussian_filter1d(res["pop_rate_E"], sigma=sigma)
        rI_sm  = gaussian_filter1d(res["pop_rate_I"], sigma=sigma)
        ax2.plot(t, rE_sm, color=COLORS["E"], lw=1.5, label=f"E: {res['mean_rate_E']:.1f} Hz")
        ax2.plot(t, rI_sm, color=COLORS["I"], lw=1.5, label=f"I: {res['mean_rate_I']:.1f} Hz")
        ax2.set_ylabel("Pop. rate (Hz)"); ax2.set_title("Population Firing Rates")
        ax2.legend()

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(t, res["LFP"], color="#374151", lw=0.8, alpha=0.8)
        ax3.set_ylabel("LFP proxy (mV)"); ax3.set_title("Mean E-Cell Voltage (LFP Proxy)")

        ax4 = fig.add_subplot(gs[3])
        ax4.fill_between(t, 0, rE_sm, color=COLORS["E"], alpha=0.4, label="E drive")
        ax4.fill_between(t, 0, -rI_sm, color=COLORS["I"], alpha=0.4, label="I drive")
        ax4.axhline(0, color="#6B7280", lw=0.6)
        ax4.set_ylabel("Rate (Hz)"); ax4.set_xlabel("Time (ms)")
        ax4.set_title("E/I Balance")
        ax4.legend()

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, 400)

        fig.suptitle(
            f"Brunel (2000) Sparse Random E/I Network  "
            f"N_E={net.N_E}  N_I={net.N_I}  p={net.p_conn}",
            fontsize=10
        )
        return _save(fig, out_dir, "fig_08_network_dynamics.png")


def fig09_isi_analysis(out_dir: str = "figures") -> str:
    """Figure 09: ISI statistics across E-cell population."""
    with plt.rc_context(STYLE):
        net = SpikingNetwork(N_E=100, N_I=25, dt=0.2, seed=42)
        res = net.run(T=500.0)

        all_isi, cv_vals = [], []
        for nid in range(net.N_E):
            st = res["spike_trains"].get(nid, np.array([]))
            if len(st) > 2:
                isi = np.diff(st)
                all_isi.extend(isi.tolist())
                cv_vals.append(float(np.std(isi) / np.mean(isi)))

        all_isi = np.array(all_isi)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("ISI Statistics: E-Cell Population", fontsize=11)

        ax1 = axes[0]
        if len(all_isi) > 10:
            bins = np.logspace(np.log10(max(1, all_isi.min())),
                               np.log10(all_isi.max()), 40)
            ax1.hist(all_isi, bins=bins, color=COLORS["V"], edgecolor="white",
                     alpha=0.8, density=True)
            ax1.set_xscale("log")
        ax1.set_xlabel("ISI (ms, log scale)"); ax1.set_ylabel("Density")
        ax1.set_title("ISI Histogram (log scale)")

        ax2 = axes[1]
        best_nid = max(
            (nid for nid in range(net.N_E) if len(res["spike_trains"].get(nid, [])) > 10),
            key=lambda x: len(res["spike_trains"].get(x, [])),
            default=0
        )
        st_best = res["spike_trains"].get(best_nid, np.array([]))
        if len(st_best) > 3:
            isi_b = np.diff(st_best)
            ax2.scatter(isi_b[:-1], isi_b[1:], s=15, color=COLORS["V"],
                        alpha=0.6, edgecolors="none")
        ax2.set_xlabel("ISI_n (ms)"); ax2.set_ylabel("ISI_{n+1} (ms)")
        ax2.set_title("Poincare Return Map (most active E-cell)")

        ax3 = axes[2]
        if cv_vals:
            ax3.hist(cv_vals, bins=20, color=COLORS["A"], edgecolor="white",
                     alpha=0.85, density=True)
            ax3.axvline(np.median(cv_vals), color=COLORS["Na"], lw=1.5, ls="--",
                        label=f"Median CV = {np.median(cv_vals):.2f}")
            ax3.axvline(1.0, color="#6B7280", lw=1, ls=":", label="Poisson (CV=1)")
        ax3.set_xlabel("CV-ISI"); ax3.set_ylabel("Density")
        ax3.set_title("CV-ISI Distribution (E cells)")
        ax3.legend()

        fig.tight_layout()
        return _save(fig, out_dir, "fig_09_isi_analysis.png")


def fig10_lfp_psd(out_dir: str = "figures") -> str:
    """Figure 10: LFP power spectral density with band decomposition."""
    with plt.rc_context(STYLE):
        net = SpikingNetwork(N_E=100, N_I=25, dt=0.2, seed=42)
        res = net.run(T=600.0)

        lfp = res["LFP"]
        fs  = 1000.0 / net.dt
        psd_data = power_spectrum(lfp, fs=fs, nperseg=2048)

        band_colors = {
            "delta": "#16A34A", "theta": "#7C3AED",
            "alpha": "#DC2626", "beta":  "#2563EB", "gamma": "#EA580C"
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("LFP Power Spectral Density (Welch's method)", fontsize=11)

        ax1 = axes[0]
        freqs, psd = psd_data["freqs"], psd_data["psd"]
        mask = (freqs >= 0.5) & (freqs <= 120)
        ax1.semilogy(freqs[mask], psd[mask], color="#1F2937", lw=1.5)

        bands = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13),
                 "beta":(13,30), "gamma":(30,100)}
        for name, (lo, hi) in bands.items():
            bm = (freqs >= lo) & (freqs <= hi)
            ax1.fill_between(freqs[bm], psd[bm], alpha=0.35,
                             color=band_colors[name], label=name.capitalize())
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("PSD (V2/Hz)")
        ax1.set_title("Power Spectrum")
        ax1.legend(ncol=2)

        ax2 = axes[1]
        bp = psd_data["band_powers"]
        names  = list(bp.keys())
        values = list(bp.values())
        colors = [band_colors[n] for n in names]
        bars = ax2.bar(names, values, color=colors, edgecolor="white", alpha=0.85)
        ax2.set_xlabel("Frequency band")
        ax2.set_ylabel("Band power (V2)")
        ax2.set_title("Integrated Band Powers")
        ax2.set_yscale("log")
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                     f"{val:.2e}", ha="center", fontsize=7, color="#374151")

        fig.tight_layout()
        return _save(fig, out_dir, "fig_10_lfp_psd.png")


def fig11_summary_dashboard(out_dir: str = "figures") -> str:
    """Figure 11: Summary dashboard combining key results from all simulations."""
    with plt.rc_context(STYLE):
        hh = HodgkinHuxley()
        hh_data = hh.simulate_detailed(T=80.0, dt=0.025, I_ext=10.0)

        rs = AdaptiveExponentialIF.from_preset("RS")
        rs_rec = rs.simulate(T=500.0, dt=0.1, I_ext=250.0, t_start=50.0)

        ampa  = AMPASynapse(g_max=0.5)
        nmda  = NMDASynapse(g_max=0.5, Mg=1.0)
        d_a   = ampa.simulate(T=100, dt=0.1)
        d_n   = nmda.simulate(T=300, dt=0.1)

        V_range = np.linspace(-90, 60, 300)
        syn_nm  = NMDASynapse(Mg=1.0)
        B_vals  = np.array([syn_nm.B(V) for V in V_range])

        stdp  = STDPRule()
        dt_r  = np.linspace(-80, 80, 400)
        dw_w  = stdp.learning_window(dt_r)
        res_s = stdp.run(T=8000, dt=0.1, n_synapses=30, seed=0)

        net = SpikingNetwork(N_E=80, N_I=20, dt=0.2, seed=7)
        net_res = net.run(T=300.0)

        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(3, 4, hspace=0.55, wspace=0.4)

        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(hh_data["t"], hh_data["V"], color=COLORS["V"], lw=1.2)
        ax00.set_title("HH Action Potential"); ax00.set_ylabel("V (mV)")

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(hh_data["t"], hh_data["m"], color=COLORS["m"], lw=1, label="m")
        ax01.plot(hh_data["t"], hh_data["h"], color=COLORS["h"], lw=1, label="h")
        ax01.plot(hh_data["t"], hh_data["n"], color=COLORS["n"], lw=1, label="n")
        ax01.set_title("Gating Variables"); ax01.legend(fontsize=7)

        ax02 = fig.add_subplot(gs[0, 2])
        ax02.plot(rs_rec.time_axis, rs_rec.voltages, color=COLORS["V"], lw=1.0)
        ax02.plot(rs_rec.time_axis, rs_rec.metadata["w"]/5, color=COLORS["W"],
                  lw=0.8, ls="--", alpha=0.7)
        ax02.set_title("AdEx Regular Spiking")

        ax03 = fig.add_subplot(gs[0, 3])
        ax03.plot(d_a["t"], d_a["g"]*1000, color=COLORS["A"], lw=1.5, label="AMPA")
        ax03.plot(d_n["t"], d_n["g"]*1000, color=COLORS["N"], lw=1.5, label="NMDA")
        ax03.set_title("Synapse Kinetics"); ax03.legend()

        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(V_range, B_vals, color=COLORS["N"], lw=1.8)
        ax10.axvline(-65, color="#9CA3AF", lw=0.6, ls="--")
        ax10.set_title("NMDA Mg2+ Block B(V)"); ax10.set_xlabel("V (mV)")

        ax11 = fig.add_subplot(gs[1, 1])
        ax11.fill_between(dt_r[dt_r > 0], 0, dw_w[dt_r > 0],
                          color=COLORS["LTP"], alpha=0.3)
        ax11.fill_between(dt_r[dt_r < 0], 0, dw_w[dt_r < 0],
                          color=COLORS["LTD"], alpha=0.3)
        ax11.plot(dt_r, dw_w, color="#1F2937", lw=1.5)
        ax11.set_title("STDP Window"); ax11.set_xlabel("Delta t (ms)")

        ax12 = fig.add_subplot(gs[1, 2])
        ax12.plot(res_s["t"] * 1e-3, res_s["w_history"].mean(axis=1),
                  color=COLORS["W"], lw=1.5)
        ax12.set_ylim(0, 1); ax12.set_title("STDP Weight Evolution")
        ax12.set_xlabel("Time (s)")

        ax13 = fig.add_subplot(gs[1, 3])
        w_fin = res_s["w_final"]
        ax13.hist(w_fin, bins=15, color=COLORS["V"], edgecolor="white", density=True)
        ax13.axvline(np.median(w_fin), color=COLORS["Na"], lw=1.2, ls="--")
        ax13.set_title("Weight Distribution"); ax13.set_xlabel("w")

        ax20 = fig.add_subplot(gs[2, :2])
        mask_E = net_res["raster_id"] < net.N_E
        ax20.scatter(net_res["raster_t"][mask_E], net_res["raster_id"][mask_E],
                     s=0.8, c=COLORS["E"], alpha=0.5)
        ax20.scatter(net_res["raster_t"][~mask_E], net_res["raster_id"][~mask_E],
                     s=1.2, c=COLORS["I"], alpha=0.8)
        ax20.set_title(f"E/I Network Raster  E:{net_res['mean_rate_E']:.1f} Hz  "
                       f"I:{net_res['mean_rate_I']:.1f} Hz")
        ax20.set_xlabel("Time (ms)"); ax20.set_ylabel("Neuron index")

        ax21 = fig.add_subplot(gs[2, 2:])
        t = net_res["t"]
        sigma = 10.0 / net.dt
        rE_sm = gaussian_filter1d(net_res["pop_rate_E"], sigma=sigma)
        rI_sm = gaussian_filter1d(net_res["pop_rate_I"], sigma=sigma)
        ax21.plot(t, rE_sm, color=COLORS["E"], lw=1.5, label="E population")
        ax21.plot(t, rI_sm, color=COLORS["I"], lw=1.5, label="I population")
        ax21.set_title("Population Firing Rates"); ax21.set_xlabel("Time (ms)")
        ax21.set_ylabel("Rate (Hz)"); ax21.legend()

        fig.suptitle(
            "Neuro_Simulation — Summary Dashboard\n"
            "Hodgkin-Huxley | LIF | AdEx | AMPA/NMDA/GABA | STDP | E/I Network",
            fontsize=12, fontweight="bold"
        )
        return _save(fig, out_dir, "fig_11_summary_dashboard.png")


def generate_all(out_dir: str = "figures") -> list[str]:
    """
    Run all eleven figure generators in sequence.

    Parameters
    ----------
    out_dir : Output directory for PNG files. Default 'figures'.

    Returns
    -------
    List of saved file paths.
    """
    generators = [
        fig01_action_potential,
        fig02_phase_plane,
        fig03_fi_curves,
        fig04_adex_patterns,
        fig05_synapse_kinetics,
        fig06_nmda_mg_block,
        fig07_stdp,
        fig08_network_dynamics,
        fig09_isi_analysis,
        fig10_lfp_psd,
        fig11_summary_dashboard,
    ]
    paths = []
    for i, fn in enumerate(generators, 1):
        print(f"[{i:02d}/11] {fn.__name__} ...")
        try:
            p = fn(out_dir)
            paths.append(p)
        except Exception as exc:
            print(f"  ERROR in {fn.__name__}: {exc}")
    print(f"\nAll figures saved to: {os.path.abspath(out_dir)}/")
    return paths


if __name__ == "__main__":
    generate_all()
