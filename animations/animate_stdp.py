"""
animate_stdp.py
===============
Animated visualization of spike-timing-dependent plasticity (STDP) learning,
showing in real time how synaptic weights evolve under competitive Hebbian
learning driven by Poisson pre- and postsynaptic spike trains at 20 Hz.

Layout (2 rows, 2 columns):
  Top-left    STDP learning window K(delta_t) with animated event dots
              showing where each recent spike pair falls on the curve
  Top-right   Real-time synaptic weight trace w(t) for a single synapse,
              together with the population mean across 50 synapses
  Bottom-left Presynaptic and postsynaptic spike raster (last 500 ms)
  Bottom-right Running histogram of all 50 final weights, updating each frame

The animation demonstrates three key STDP phenomena:
  1. LTP when postsynaptic spikes follow presynaptic spikes (delta_t > 0)
  2. LTD when the order is reversed (delta_t < 0)
  3. Convergence to a stable unimodal weight distribution under balanced input

Output: animations/stdp_learning.gif

Usage:
    python animations/animate_stdp.py

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapses.plasticity import STDPRule

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
C_LTP    = "#F78166"
C_LTD    = "#58A6FF"
C_WEIGHT = "#E3B341"
C_MEAN   = "#C9D1D9"
C_PRE    = "#56D364"
C_POST   = "#BC8CFF"


def precompute_stdp(
    T: float = 12000.0,
    dt: float = 0.5,
    n_synapses: int = 50,
    seed: int = 42,
) -> dict:
    """
    Run the full STDP simulation and record all spike events
    and weight trajectories for later animation.
    """
    rng        = np.random.default_rng(seed)
    n_steps    = int(T / dt)
    t_axis     = np.arange(n_steps) * dt
    p_pre      = 20.0 * dt * 1e-3
    p_post     = 20.0 * dt * 1e-3

    stdp    = STDPRule(A_plus=0.010, A_minus=0.0105)
    weights = np.full(n_synapses, 0.5)
    x_pre   = np.zeros(n_synapses)
    x_post  = np.zeros(n_synapses)
    dp      = np.exp(-dt / 20.0)

    w_hist    = np.empty((n_steps, n_synapses))
    pre_spikes : list[float] = []
    post_spikes: list[float] = []
    events     : list[tuple[float, float, str]] = []

    for i in range(n_steps):
        x_pre  *= dp
        x_post *= dp
        sp_pre  = rng.random(n_synapses) < p_pre
        sp_post = rng.random(n_synapses) < p_post

        if sp_pre[0]:
            pre_spikes.append(t_axis[i])
            for pt in post_spikes[-5:]:
                dt_val = t_axis[i] - pt
                if abs(dt_val) < 80:
                    events.append((t_axis[i], dt_val, "LTD"))

        if sp_post[0]:
            post_spikes.append(t_axis[i])
            for pt in pre_spikes[-5:]:
                dt_val = t_axis[i] - pt
                if abs(dt_val) < 80:
                    events.append((t_axis[i], dt_val, "LTP"))

        dw = np.zeros(n_synapses)
        dw[sp_pre]  -= stdp.A_minus * weights[sp_pre]  * x_post[sp_pre]
        x_pre[sp_pre]   += 1.0
        dw[sp_post] += stdp.A_plus * (1 - weights[sp_post]) * x_pre[sp_post]
        x_post[sp_post] += 1.0

        weights = np.clip(weights + dw, 0.0, 1.0)
        w_hist[i] = weights

    return {
        "t":          t_axis,
        "w_hist":     w_hist,
        "pre_spikes": np.array(pre_spikes),
        "post_spikes":np.array(post_spikes),
        "events":     events,
        "n_synapses": n_synapses,
    }


def make_animation(
    fps: int = 20,
    speed_factor: int = 30,
    T: float = 12000.0,
    save_gif: bool = True,
) -> str:

    print("  Pre-computing STDP simulation ...")
    data    = precompute_stdp(T=T, dt=0.5, n_synapses=50)
    t_axis  = data["t"]
    n_pts   = len(t_axis)
    step    = max(1, speed_factor)
    dt_win  = np.linspace(-100, 100, 400)
    stdp    = STDPRule()
    dw_win  = stdp.learning_window(dt_win)

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(13, 9), facecolor="#0D1117")
        gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.38,
                                left=0.09, right=0.97, top=0.91, bottom=0.08)
        fig.suptitle(
            "STDP Learning Dynamics  |  Bi and Poo (1998)  |  "
            "20 Hz Poisson input  |  50 synapses",
            color="#C9D1D9", fontsize=10
        )

        # Panel 1: STDP window + live event dots
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.fill_between(dt_win[dt_win > 0], 0, dw_win[dt_win > 0],
                         color=C_LTP, alpha=0.2, label="LTP region")
        ax1.fill_between(dt_win[dt_win < 0], 0, dw_win[dt_win < 0],
                         color=C_LTD, alpha=0.2, label="LTD region")
        ax1.plot(dt_win, dw_win, color="#C9D1D9", lw=1.8)
        ax1.axhline(0, color="#30363D", lw=0.6)
        ax1.axvline(0, color="#30363D", lw=0.6, ls="--")
        ax1.set_xlabel("Δt = t_post - t_pre (ms)")
        ax1.set_ylabel("Weight change Δw")
        ax1.set_title("STDP Learning Window", color="#C9D1D9")
        ax1.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="#C9D1D9")
        scat_ltp = ax1.scatter([], [], s=30, c=C_LTP, alpha=0.8,
                               edgecolors="none", zorder=5, label="LTP event")
        scat_ltd = ax1.scatter([], [], s=30, c=C_LTD, alpha=0.8,
                               edgecolors="none", zorder=5, label="LTD event")
        txt_ev = ax1.text(0.05, 0.06, "Events: 0",
                          transform=ax1.transAxes, color="#C9D1D9",
                          fontsize=8, fontfamily="monospace")

        # Panel 2: Weight trace
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(0, T * 1e-3)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Synaptic weight w")
        ax2.set_title("Weight Evolution (synapse 0 + population mean)", color="#C9D1D9")
        ax2.axhline(0.5, color="#30363D", lw=0.6, ls=":")
        line_w0,   = ax2.plot([], [], color=C_WEIGHT, lw=1.5, label="Synapse 0")
        line_wmean,= ax2.plot([], [], color=C_MEAN,   lw=1.0, ls="--", alpha=0.7,
                               label="Population mean")
        txt_w   = ax2.text(0.05, 0.88, "w = 0.500",
                           transform=ax2.transAxes, color=C_WEIGHT,
                           fontsize=9, fontfamily="monospace")
        ax2.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="#C9D1D9")

        # Panel 3: Spike raster (last 500 ms window)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["Pre", "Post"], fontsize=8)
        ax3.set_xlabel("Time (ms)")
        ax3.set_title("Spike Raster (sliding 500 ms window)", color="#C9D1D9")
        RWIN = 500.0
        scat_pre_r  = ax3.scatter([], [], s=20, c=C_PRE,  marker="|",
                                   linewidths=1.2, alpha=0.9, label="Pre")
        scat_post_r = ax3.scatter([], [], s=20, c=C_POST, marker="|",
                                   linewidths=1.2, alpha=0.9, label="Post")
        ax3.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="#C9D1D9", loc="upper right")

        # Panel 4: Weight histogram
        ax4 = fig.add_subplot(gs[1, 1])
        W_BINS = np.linspace(0, 1, 26)
        W_CENTERS = 0.5 * (W_BINS[:-1] + W_BINS[1:])
        bar_hist = ax4.bar(W_CENTERS, np.zeros(len(W_CENTERS)),
                           width=W_BINS[1] - W_BINS[0],
                           color=C_WEIGHT, alpha=0.8, edgecolor="none")
        ax4.set_xlim(0, 1)
        ax4.set_xlabel("Synaptic weight w")
        ax4.set_ylabel("Count")
        ax4.set_title("Weight Distribution (50 synapses)", color="#C9D1D9")
        vline_med = ax4.axvline(0.5, color=C_LTP, lw=1.5, ls="--")
        txt_med = ax4.text(0.6, 0.88, "Median: 0.500",
                           transform=ax4.transAxes, color=C_LTP,
                           fontsize=8, fontfamily="monospace")

    def init():
        line_w0.set_data([], [])
        line_wmean.set_data([], [])
        scat_ltp.set_offsets(np.empty((0, 2)))
        scat_ltd.set_offsets(np.empty((0, 2)))
        scat_pre_r.set_offsets(np.empty((0, 2)))
        scat_post_r.set_offsets(np.empty((0, 2)))
        for bar in bar_hist:
            bar.set_height(0)
        return (line_w0, line_wmean, scat_ltp, scat_ltd,
                scat_pre_r, scat_post_r, *bar_hist)

    def update(frame: int):
        i = min((frame + 1) * step, n_pts)
        cur_t = t_axis[i - 1] if i > 0 else 0.0
        t_s   = t_axis[:i] * 1e-3

        line_w0.set_data(t_s,   data["w_hist"][:i, 0])
        line_wmean.set_data(t_s, data["w_hist"][:i].mean(axis=1))
        txt_w.set_text(f"w = {data['w_hist'][i-1, 0]:.3f}")

        # LTP / LTD event dots on the STDP window (last 30 events)
        recent = [(ev[1], ev[2]) for ev in data["events"] if ev[0] <= cur_t][-30:]
        ltp_pts = np.array([[d, stdp.A_plus*0.5*np.exp(-abs(d)/20)]
                             for d, k in recent if k == "LTP"]) if recent else np.empty((0, 2))
        ltd_pts = np.array([[d, -stdp.A_minus*0.5*np.exp(-abs(d)/20)]
                             for d, k in recent if k == "LTD"]) if recent else np.empty((0, 2))
        scat_ltp.set_offsets(ltp_pts if len(ltp_pts) else np.empty((0, 2)))
        scat_ltd.set_offsets(ltd_pts if len(ltd_pts) else np.empty((0, 2)))
        txt_ev.set_text(f"Events: {len(data['events'])}")

        # Raster: spikes in last RWIN ms
        lo = cur_t - RWIN
        pre_in  = data["pre_spikes"][(data["pre_spikes"] >= lo) &
                                      (data["pre_spikes"] <= cur_t)]
        post_in = data["post_spikes"][(data["post_spikes"] >= lo) &
                                       (data["post_spikes"] <= cur_t)]
        ax3.set_xlim(lo, cur_t)
        scat_pre_r.set_offsets(
            np.column_stack([pre_in, np.zeros(len(pre_in))]) if len(pre_in) else np.empty((0, 2))
        )
        scat_post_r.set_offsets(
            np.column_stack([post_in, np.ones(len(post_in))]) if len(post_in) else np.empty((0, 2))
        )

        # Weight histogram
        w_now = data["w_hist"][i - 1]
        counts, _ = np.histogram(w_now, bins=W_BINS)
        max_c = max(counts.max(), 1)
        ax4.set_ylim(0, max_c * 1.3)
        for bar, cnt in zip(bar_hist, counts):
            bar.set_height(cnt)
        med = float(np.median(w_now))
        vline_med.set_xdata([med, med])
        txt_med.set_text(f"Median: {med:.3f}")

        return (line_w0, line_wmean, scat_ltp, scat_ltd,
                scat_pre_r, scat_post_r, *bar_hist,
                txt_w, txt_ev, vline_med, txt_med)

    n_frames = n_pts // step
    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False
    )

    gif_path = os.path.join(OUT_DIR, "stdp_learning.gif")
    if save_gif:
        print(f"  Saving GIF ({n_frames} frames) -> {gif_path}")
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"  Saved: {gif_path}  ({os.path.getsize(gif_path)//1024} KB)")

    plt.close(fig)
    return gif_path


if __name__ == "__main__":
    make_animation(fps=20, speed_factor=30, T=12000.0, save_gif=True)
