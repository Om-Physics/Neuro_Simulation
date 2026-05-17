"""
animate_hh.py
=============
Animated visualization of the Hodgkin-Huxley action potential, showing
the coupled evolution of membrane voltage, gating variables (m, h, n),
and ionic currents in real time as a step current is injected.

The animation contains three synchronized panels:
  Panel 1  Membrane voltage V(t) with Nernst potential reference lines
  Panel 2  Gating variables m(t), h(t), n(t) and the products m^3*h, n^4
  Panel 3  Ionic currents I_Na(t), I_K(t), I_L(t)

A vertical cursor sweeps across all panels synchronously, illustrating
the precise temporal relationships between voltage, channel states,
and the resulting ionic currents that underlie the action potential.

Output: animations/hh_action_potential.gif   (Pillow backend)
        animations/hh_action_potential.mp4   (FFmpeg backend, if available)

Usage:
    python animations/animate_hh.py

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
from neurons.hodgkin_huxley import HodgkinHuxley

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE = {
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
    "font.size":        9,
}

COLORS = {
    "V":   "#58A6FF",
    "m":   "#F78166",
    "h":   "#56D364",
    "n":   "#E3B341",
    "Na":  "#F78166",
    "K":   "#E3B341",
    "L":   "#8B949E",
    "m3h": "#F7816680",
    "n4":  "#E3B34180",
    "cursor": "#FF7B72",
    "ref": "#30363D",
}


def build_simulation_data(T: float = 100.0, dt: float = 0.025,
                           I_ext: float = 10.0) -> dict:
    """Run the HH simulation and return all traces needed for animation."""
    hh = HodgkinHuxley()
    data = hh.simulate_detailed(T=T, dt=dt, I_ext=I_ext)
    data["I_Na_plot"] = -data["I_Na"]   # sign flip: show inward as positive
    data["m3h"] = data["m"]**3 * data["h"]
    data["n4"]  = data["n"]**4
    return data


def make_animation(
    fps: int = 30,
    speed_factor: int = 8,
    T: float = 100.0,
    save_gif: bool = True,
    save_mp4: bool = False,
) -> str:
    """
    Build and save the HH action potential animation.

    Parameters
    ----------
    fps          : Frames per second in the output file.
    speed_factor : How many simulation ms to advance per animation frame.
    T            : Total simulation duration to animate (ms).
    save_gif     : Whether to save a GIF file.
    save_mp4     : Whether to also try saving an MP4 (requires FFmpeg).

    Returns
    -------
    str : Path to the saved GIF file.
    """
    data = build_simulation_data(T=T, dt=0.025, I_ext=10.0)
    t    = data["t"]
    n_pts = len(t)
    step  = max(1, speed_factor)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 8), facecolor="#0D1117")
        gs  = gridspec.GridSpec(3, 1, hspace=0.45, left=0.10, right=0.95,
                                top=0.91, bottom=0.08)

        axes = [fig.add_subplot(gs[i]) for i in range(3)]

        # --- static reference lines
        hlines_V = [
            (55.0,   "#F78166", "E_Na = +55 mV"),
            (-77.0,  "#E3B341", "E_K  = -77 mV"),
            (-54.4,  "#8B949E", "E_L  = -54.4 mV"),
            (-65.0,  "#30363D", "V_rest = -65 mV"),
        ]
        for yval, col, lbl in hlines_V:
            axes[0].axhline(yval, color=col, lw=0.6, ls="--", alpha=0.5, label=lbl)

        axes[1].axhline(0, color=COLORS["ref"], lw=0.5)
        axes[2].axhline(0, color=COLORS["ref"], lw=0.5)

        # --- axis limits and labels
        axes[0].set_ylim(-85, 65)
        axes[0].set_ylabel("V (mV)", color="#8B949E")
        axes[0].set_title("Membrane Voltage", color="#C9D1D9", fontsize=10)

        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_ylabel("Gate value (0-1)", color="#8B949E")
        axes[1].set_title("Gating Variables", color="#C9D1D9", fontsize=10)

        axes[2].set_ylim(-12, 25)
        axes[2].set_ylabel("Current (µA/cm²)", color="#8B949E")
        axes[2].set_title("Ionic Currents", color="#C9D1D9", fontsize=10)
        axes[2].set_xlabel("Time (ms)", color="#8B949E")

        for ax in axes:
            ax.set_xlim(0, T)

        fig.suptitle(
            "Hodgkin-Huxley Model  |  I_ext = 10 µA/cm²  |  squid axon 6.3°C  |  RK4 dt = 0.025 ms",
            color="#C9D1D9", fontsize=10, fontfamily="monospace"
        )

        # --- live line objects
        line_V,   = axes[0].plot([], [], color=COLORS["V"],  lw=1.8, label="V(t)", zorder=3)
        line_m,   = axes[1].plot([], [], color=COLORS["m"],  lw=1.2, label="m (Na act.)")
        line_h,   = axes[1].plot([], [], color=COLORS["h"],  lw=1.2, label="h (Na inact.)")
        line_n,   = axes[1].plot([], [], color=COLORS["n"],  lw=1.2, label="n (K act.)")
        line_m3h, = axes[1].plot([], [], color=COLORS["m"],  lw=0.8, ls="--", alpha=0.5,
                                 label="m³h (Na open)")
        line_n4,  = axes[1].plot([], [], color=COLORS["n"],  lw=0.8, ls="--", alpha=0.5,
                                 label="n⁴  (K open)")
        line_Na,  = axes[2].plot([], [], color=COLORS["Na"], lw=1.5, label="-I_Na (inward)")
        line_K,   = axes[2].plot([], [], color=COLORS["K"],  lw=1.5, label="I_K  (outward)")
        line_L,   = axes[2].plot([], [], color=COLORS["L"],  lw=1.0, label="I_L  (leak)")

        for ax in axes:
            ax.legend(loc="upper right", fontsize=7,
                      facecolor="#161B22", edgecolor="#30363D",
                      labelcolor="#C9D1D9", ncol=3)

        # --- cursor lines (vertical marker at current time)
        cursors = [ax.axvline(0, color=COLORS["cursor"], lw=1.0, alpha=0.7, ls="-")
                   for ax in axes]

        # --- text annotations
        txt_t   = axes[0].text(0.02, 0.94, "t = 0.00 ms",
                               transform=axes[0].transAxes,
                               color=COLORS["cursor"], fontsize=9, fontfamily="monospace")
        txt_V   = axes[0].text(0.02, 0.82, "V = -65.0 mV",
                               transform=axes[0].transAxes,
                               color=COLORS["V"], fontsize=9, fontfamily="monospace")
        txt_ph  = axes[0].text(0.75, 0.94, "Phase: resting",
                               transform=axes[0].transAxes,
                               color="#C9D1D9", fontsize=9, fontfamily="monospace")
        txt_fr  = axes[2].text(0.02, 0.88, "Spikes: 0   Rate: 0 Hz",
                               transform=axes[2].transAxes,
                               color="#C9D1D9", fontsize=8, fontfamily="monospace")

    def _phase(V: float, m: float, h: float) -> str:
        if V < -60:
            return "resting"
        elif V < 0 and m > 0.3:
            return "depolarising"
        elif V >= 0:
            return "peak"
        elif h < 0.3:
            return "repolarising"
        else:
            return "hyperpolarised"

    def init():
        for line in [line_V, line_m, line_h, line_n, line_m3h, line_n4,
                     line_Na, line_K, line_L]:
            line.set_data([], [])
        return (line_V, line_m, line_h, line_n, line_m3h, line_n4,
                line_Na, line_K, line_L, *cursors, txt_t, txt_V, txt_ph, txt_fr)

    def update(frame: int):
        i = min((frame + 1) * step, n_pts)
        tslice = t[:i]

        line_V.set_data(tslice,   data["V"][:i])
        line_m.set_data(tslice,   data["m"][:i])
        line_h.set_data(tslice,   data["h"][:i])
        line_n.set_data(tslice,   data["n"][:i])
        line_m3h.set_data(tslice, data["m3h"][:i])
        line_n4.set_data(tslice,  data["n4"][:i])
        line_Na.set_data(tslice,  data["I_Na_plot"][:i])
        line_K.set_data(tslice,   data["I_K"][:i])
        line_L.set_data(tslice,   data["I_L"][:i])

        cur_t = t[i - 1] if i > 0 else 0.0
        cur_V = data["V"][i-1] if i > 0 else -65.0
        cur_m = data["m"][i-1] if i > 0 else 0.05
        cur_h = data["h"][i-1] if i > 0 else 0.60

        for cursor in cursors:
            cursor.set_xdata([cur_t, cur_t])

        txt_t.set_text(f"t = {cur_t:6.2f} ms")
        txt_V.set_text(f"V = {cur_V:+6.1f} mV")
        txt_ph.set_text(f"Phase: {_phase(cur_V, cur_m, cur_h)}")

        n_spikes = len(data["spikes"][data["spikes"] <= cur_t])
        rate     = round(n_spikes / max(cur_t, 1.0) * 1000)
        txt_fr.set_text(f"Spikes: {n_spikes:2d}   Rate: {rate:3d} Hz")

        return (line_V, line_m, line_h, line_n, line_m3h, line_n4,
                line_Na, line_K, line_L, *cursors, txt_t, txt_V, txt_ph, txt_fr)

    n_frames = n_pts // step
    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=True
    )

    gif_path = os.path.join(OUT_DIR, "hh_action_potential.gif")
    if save_gif:
        print(f"  Saving GIF ({n_frames} frames) -> {gif_path}")
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"  Saved: {gif_path}  ({os.path.getsize(gif_path)//1024} KB)")

    if save_mp4:
        try:
            from matplotlib.animation import FFMpegWriter
            mp4_path = os.path.join(OUT_DIR, "hh_action_potential.mp4")
            ani.save(mp4_path, writer=FFMpegWriter(fps=fps, bitrate=1200))
            print(f"  Saved: {mp4_path}")
        except Exception as exc:
            print(f"  MP4 not saved (FFmpeg unavailable): {exc}")

    plt.close(fig)
    return gif_path


if __name__ == "__main__":
    make_animation(fps=25, speed_factor=6, T=100.0, save_gif=True, save_mp4=False)
