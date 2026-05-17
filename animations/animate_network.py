"""
animate_network.py
==================
Animated visualization of the sparse random excitatory-inhibitory (E/I)
recurrent spiking network simulation, illustrating in real time how collective
neural activity emerges from local synaptic interactions.

The animation contains four synchronized panels that update as the network
simulation advances frame by frame:

  Panel 1 (top, wide)    Scrolling spike raster: each row is a neuron,
                         each dot is a spike. Red = excitatory, blue = inhibitory.
                         A sliding 300 ms window reveals the temporal structure.

  Panel 2 (middle-left)  Population firing rates for the E and I populations,
                         smoothed with a 10 ms Gaussian kernel.

  Panel 3 (middle-right) LFP proxy (mean E-cell membrane voltage) showing
                         subthreshold fluctuations and emergent oscillations.

  Panel 4 (bottom)       E/I balance indicator: filled area plot contrasting
                         excitatory drive (upward) versus inhibitory drive
                         (downward), revealing the dynamic push-pull competition.

This animation is designed to build intuition for the concept of inhibitory
stabilisation: strong feedback inhibition prevents runaway excitation and
constrains the network to a stable asynchronous firing state.

Output: animations/network_dynamics.gif

Usage:
    python animations/animate_network.py

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.network import SpikingNetwork

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DARK = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#0D1117",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#8B949E",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "grid.color":       "#21262D",
    "grid.alpha":       1.0,
    "axes.grid":        True,
    "text.color":       "#C9D1D9",
    "font.family":      "monospace",
    "font.size":        8,
}
C_E = "#F78166"
C_I = "#58A6FF"
C_L = "#C9D1D9"


def precompute_network(T: float = 500.0, dt: float = 0.2, seed: int = 42) -> dict:
    """Run the full network simulation and return all data for animation."""
    net = SpikingNetwork(N_E=80, N_I=20, dt=dt, seed=seed, I_dc=220.0)
    res = net.run(T=T)
    sigma = 10.0 / dt
    res["rE_smooth"] = gaussian_filter1d(res["pop_rate_E"], sigma=sigma)
    res["rI_smooth"] = gaussian_filter1d(res["pop_rate_I"], sigma=sigma)
    res["dt"]  = dt
    res["N_E"] = net.N_E
    res["N_I"] = net.N_I
    return res


def make_animation(
    fps: int = 20,
    speed_factor: int = 10,
    T: float = 500.0,
    save_gif: bool = True,
) -> str:

    print("  Pre-computing network simulation ...")
    data  = precompute_network(T=T)
    dt    = data["dt"]
    t_ax  = data["t"]
    n_pts = len(t_ax)
    step  = max(1, speed_factor)
    N_E   = data["N_E"]
    N_I   = data["N_I"]
    N     = N_E + N_I
    RWIN  = 300.0

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(14, 10), facecolor="#0D1117")
        gs  = gridspec.GridSpec(3, 2, height_ratios=[1.8, 1, 1],
                                hspace=0.50, wspace=0.35,
                                left=0.09, right=0.97, top=0.92, bottom=0.07)
        fig.suptitle(
            f"E/I Recurrent Network  |  Brunel (2000)  |  "
            f"N_E={N_E}  N_I={N_I}  p=0.15  |  AMPA+GABA-A",
            color="#C9D1D9", fontsize=10
        )

        # Raster (spans both columns on top row)
        ax_raster = fig.add_subplot(gs[0, :])
        ax_raster.set_ylim(-1, N + 1)
        ax_raster.set_ylabel("Neuron index")
        ax_raster.set_title("Spike Raster  (red = E, blue = I)", color="#C9D1D9")
        ax_raster.axhline(N_E, color="#30363D", lw=0.7, ls="--")
        ax_raster.text(2, N_E + 1, "Inhibitory", color=C_I, fontsize=7)
        ax_raster.text(2, N_E - 3, "Excitatory", color=C_E, fontsize=7)

        scat_E = ax_raster.scatter([], [], s=1.5, c=C_E, alpha=0.7, edgecolors="none")
        scat_I = ax_raster.scatter([], [], s=2.0, c=C_I, alpha=0.9, edgecolors="none")
        txt_t  = ax_raster.text(0.01, 0.93, "t = 0 ms",
                                transform=ax_raster.transAxes,
                                color="#C9D1D9", fontsize=9, fontfamily="monospace")
        txt_r  = ax_raster.text(0.75, 0.93, "E: 0 Hz  I: 0 Hz",
                                transform=ax_raster.transAxes,
                                color="#C9D1D9", fontsize=8, fontfamily="monospace")

        # Population rates
        ax_rate = fig.add_subplot(gs[1, 0])
        ax_rate.set_ylim(0, max(data["rE_smooth"].max(), data["rI_smooth"].max()) * 1.3 + 5)
        ax_rate.set_ylabel("Rate (Hz)")
        ax_rate.set_title("Population Firing Rates (10 ms smoothed)", color="#C9D1D9")
        line_rE, = ax_rate.plot([], [], color=C_E, lw=1.5, label=f"E  mean")
        line_rI, = ax_rate.plot([], [], color=C_I, lw=1.5, label=f"I  mean")
        ax_rate.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                       labelcolor="#C9D1D9")

        # LFP proxy
        ax_lfp = fig.add_subplot(gs[1, 1])
        ax_lfp.set_ylim(data["LFP"].min() - 2, data["LFP"].max() + 2)
        ax_lfp.set_ylabel("Voltage (mV)")
        ax_lfp.set_title("LFP Proxy (mean E-cell voltage)", color="#C9D1D9")
        line_lfp, = ax_lfp.plot([], [], color=C_L, lw=0.9, alpha=0.85)

        # E/I balance
        ax_ei = fig.add_subplot(gs[2, :])
        ax_ei.set_ylabel("Rate (Hz)")
        ax_ei.set_xlabel("Time (ms)")
        ax_ei.set_title("E/I Balance  (E upward, I downward)", color="#C9D1D9")
        ax_ei.axhline(0, color="#30363D", lw=0.6)
        fill_E = ax_ei.fill_between([], [], alpha=0.45, color=C_E, label="E drive")
        fill_I = ax_ei.fill_between([], [], alpha=0.45, color=C_I, label="I drive")
        ax_ei.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                     labelcolor="#C9D1D9")
        max_r = max(data["rE_smooth"].max(), data["rI_smooth"].max())
        ax_ei.set_ylim(-max_r * 1.5, max_r * 1.5)

    def init():
        scat_E.set_offsets(np.empty((0, 2)))
        scat_I.set_offsets(np.empty((0, 2)))
        line_rE.set_data([], [])
        line_rI.set_data([], [])
        line_lfp.set_data([], [])
        return (scat_E, scat_I, line_rE, line_rI, line_lfp)

    def update(frame: int):
        i     = min((frame + 1) * step, n_pts)
        cur_t = t_ax[i - 1] if i > 0 else 0.0

        # Raster window
        lo   = cur_t - RWIN
        rt   = data["raster_t"]
        rid  = data["raster_id"]
        mask = (rt >= lo) & (rt <= cur_t)
        rt_w = rt[mask]; rid_w = rid[mask]

        ax_raster.set_xlim(lo, cur_t)
        mE = rid_w < N_E
        mI = ~mE
        scat_E.set_offsets(
            np.column_stack([rt_w[mE], rid_w[mE]]) if mE.any() else np.empty((0, 2))
        )
        scat_I.set_offsets(
            np.column_stack([rt_w[mI], rid_w[mI]]) if mI.any() else np.empty((0, 2))
        )

        txt_t.set_text(f"t = {cur_t:.0f} ms")
        rE_now = float(np.mean(data["rE_smooth"][max(0, i-50):i])) if i > 0 else 0.0
        rI_now = float(np.mean(data["rI_smooth"][max(0, i-50):i])) if i > 0 else 0.0
        txt_r.set_text(f"E: {rE_now:.1f} Hz  I: {rI_now:.1f} Hz")

        # Rate traces (full time up to cursor)
        tsl = t_ax[:i]
        line_rE.set_data(tsl, data["rE_smooth"][:i])
        line_rI.set_data(tsl, data["rI_smooth"][:i])
        for ax in [ax_rate, ax_lfp]:
            ax.set_xlim(0, max(cur_t, 50))

        # LFP
        line_lfp.set_data(tsl, data["LFP"][:i])

        # E/I balance
        ax_ei.set_xlim(0, max(cur_t, 50))
        for coll in list(ax_ei.collections):
            coll.remove()
        if i > 1:
            ax_ei.fill_between(tsl,  0,  data["rE_smooth"][:i],
                               alpha=0.45, color=C_E)
            ax_ei.fill_between(tsl, 0, -data["rI_smooth"][:i],
                               alpha=0.45, color=C_I)

        return (scat_E, scat_I, line_rE, line_rI, line_lfp, txt_t, txt_r)

    n_frames = n_pts // step
    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False
    )

    gif_path = os.path.join(OUT_DIR, "network_dynamics.gif")
    if save_gif:
        print(f"  Saving GIF ({n_frames} frames) -> {gif_path}")
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"  Saved: {gif_path}  ({os.path.getsize(gif_path)//1024} KB)")

    plt.close(fig)
    return gif_path


if __name__ == "__main__":
    make_animation(fps=20, speed_factor=10, T=500.0, save_gif=True)
