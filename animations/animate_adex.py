"""
animate_adex.py
===============
Animated side-by-side comparison of five cortical cell-type firing patterns
produced by the Adaptive Exponential Integrate-and-Fire (AdEx) model,
demonstrating how a single two-dimensional model reproduces the full
repertoire of cortical spiking behaviours through parameter variation.

Layout (5 rows, 2 columns):
  Left column   Membrane voltage V(t) with adaptation current w(t) overlaid
  Right column  Real-time ISI histogram building as spikes accumulate

Cell types animated simultaneously:
  RS   Regular Spiking       (spike-frequency adaptation, large b)
  IB   Intrinsic Bursting    (burst-then-quiescence, short tau_w)
  CH   Chattering            (high-frequency rhythmic bursts)
  FS   Fast Spiking          (no adaptation, a=0, b=0)
  LTS  Low-Threshold Spiking (strong subthreshold coupling)

Output: animations/adex_patterns.gif

Usage:
    python animations/animate_adex.py

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
from neurons.integrate_fire import AdaptiveExponentialIF, ADEX_PRESETS

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

ROW_COLORS = {
    "RS":  "#58A6FF",
    "IB":  "#F78166",
    "CH":  "#56D364",
    "FS":  "#E3B341",
    "LTS": "#BC8CFF",
}
W_COLOR = "#FF7B72"
PRESETS = ["RS", "IB", "CH", "FS", "LTS"]


def precompute(T: float = 1000.0, dt: float = 0.1) -> dict:
    """Simulate all five cell types and return their full traces."""
    results = {}
    for preset in PRESETS:
        p = ADEX_PRESETS[preset]
        neuron = AdaptiveExponentialIF.from_preset(preset)
        rec = neuron.simulate(T=T, dt=dt, I_ext=p["I_default"], t_start=100.0)
        results[preset] = {
            "t": rec.time_axis,
            "V": rec.voltages,
            "w": rec.metadata["w"],
            "spikes": rec.times,
            "label": p["label"],
            "I": p["I_default"],
        }
    return results


def make_animation(
    fps: int = 20,
    speed_factor: int = 20,
    T: float = 1000.0,
    save_gif: bool = True,
) -> str:
    """Build and save the AdEx firing-patterns animation."""
    print("  Pre-computing AdEx simulations ...")
    data = precompute(T=T, dt=0.1)
    t_full = data["RS"]["t"]
    n_pts  = len(t_full)
    step   = max(1, speed_factor)

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(13, 11), facecolor="#0D1117")
        gs  = gridspec.GridSpec(
            len(PRESETS), 2, hspace=0.55, wspace=0.35,
            left=0.09, right=0.97, top=0.92, bottom=0.06
        )
        fig.suptitle(
            "AdEx Cortical Firing Patterns  |  Naud et al. (2008)  |  "
            "Blue = V(t)   Dashed red = w(t)",
            color="#C9D1D9", fontsize=10
        )

        ax_V   = {}
        ax_ISI = {}
        lines_V, lines_w = {}, {}
        bars_ISI = {}
        texts    = {}
        cursors  = {}

        ISI_BIN_EDGES  = np.linspace(0, 200, 41)
        ISI_BIN_CENTERS = 0.5 * (ISI_BIN_EDGES[:-1] + ISI_BIN_EDGES[1:])

        for row, preset in enumerate(PRESETS):
            col = ROW_COLORS[preset]
            p_data = data[preset]

            axV = fig.add_subplot(gs[row, 0])
            axV.set_ylim(-90, 50)
            axV.set_xlim(0, T)
            axV.set_ylabel("V (mV)", fontsize=7)
            axV.set_title(
                f"{preset}  {p_data['label']}  I={p_data['I']:.0f} pA",
                color=col, fontsize=8
            )
            if row < len(PRESETS) - 1:
                axV.set_xticklabels([])
            else:
                axV.set_xlabel("Time (ms)")

            axV2 = axV.twinx()
            axV2.set_ylim(-200, 1200)
            axV2.set_ylabel("w (pA)", color=W_COLOR, fontsize=6)
            axV2.tick_params(axis="y", colors=W_COLOR, labelsize=6)
            axV2.spines["right"].set_color(W_COLOR)

            lV,  = axV.plot([],  [], color=col,     lw=1.3, zorder=3)
            lw,  = axV2.plot([], [], color=W_COLOR, lw=0.8, ls="--", alpha=0.75)
            cur  = axV.axvline(0, color="#FF7B72", lw=0.8, alpha=0.6)

            txt = axV.text(
                0.60, 0.87, "0 spikes | 0 Hz",
                transform=axV.transAxes, color="#C9D1D9",
                fontsize=7, fontfamily="monospace"
            )

            axISI = fig.add_subplot(gs[row, 1])
            axISI.set_xlim(0, 200)
            axISI.set_ylim(0, 1)
            axISI.set_xlabel("ISI (ms)", fontsize=7)
            axISI.set_ylabel("Count", fontsize=7)
            axISI.set_title(f"ISI distribution  CV = —", color=col, fontsize=8)

            bar_objs = axISI.bar(
                ISI_BIN_CENTERS, np.zeros(len(ISI_BIN_CENTERS)),
                width=(ISI_BIN_EDGES[1] - ISI_BIN_EDGES[0]) * 0.85,
                color=col, alpha=0.75, edgecolor="none"
            )

            ax_V[preset]   = (axV, axV2)
            ax_ISI[preset] = axISI
            lines_V[preset] = lV
            lines_w[preset] = lw
            bars_ISI[preset] = (bar_objs, ISI_BIN_EDGES, ISI_BIN_CENTERS)
            texts[preset]    = txt
            cursors[preset]  = cur

    def init():
        artists = []
        for preset in PRESETS:
            lines_V[preset].set_data([], [])
            lines_w[preset].set_data([], [])
            for bar in bars_ISI[preset][0]:
                bar.set_height(0)
            artists += [lines_V[preset], lines_w[preset],
                        *bars_ISI[preset][0], cursors[preset]]
        return artists

    def update(frame: int):
        i = min((frame + 1) * step, n_pts)
        artists = []

        for preset in PRESETS:
            d    = data[preset]
            tsl  = d["t"][:i]
            cur_t = d["t"][i-1] if i > 0 else 0.0

            lines_V[preset].set_data(tsl, d["V"][:i])
            lines_w[preset].set_data(tsl, d["w"][:i])
            cursors[preset].set_xdata([cur_t, cur_t])

            sp_so_far = d["spikes"][d["spikes"] <= cur_t]
            n_sp   = len(sp_so_far)
            rate   = round(n_sp / max(cur_t, 1.0) * 1000)
            texts[preset].set_text(f"{n_sp:3d} spikes | {rate:3d} Hz")

            bar_objs, edges, centers = bars_ISI[preset]
            if n_sp > 1:
                isi = np.diff(sp_so_far)
                counts, _ = np.histogram(isi, bins=edges)
                max_c = max(counts.max(), 1)
                axISI = ax_ISI[preset]
                axISI.set_ylim(0, max_c * 1.25)
                for bar, cnt in zip(bar_objs, counts):
                    bar.set_height(cnt)

                cv = float(np.std(isi) / np.mean(isi)) if len(isi) > 1 else 0.0
                axISI.set_title(
                    f"ISI distribution  CV = {cv:.3f}",
                    color=ROW_COLORS[preset], fontsize=8
                )

            artists += [lines_V[preset], lines_w[preset],
                        *bar_objs, cursors[preset], texts[preset]]

        return artists

    n_frames = n_pts // step
    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False
    )

    gif_path = os.path.join(OUT_DIR, "adex_patterns.gif")
    if save_gif:
        print(f"  Saving GIF ({n_frames} frames) -> {gif_path}")
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"  Saved: {gif_path}  ({os.path.getsize(gif_path)//1024} KB)")

    plt.close(fig)
    return gif_path


if __name__ == "__main__":
    make_animation(fps=20, speed_factor=20, T=1000.0, save_gif=True)
