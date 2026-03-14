"""
integrate_fire.py
=================
Point-neuron integrate-and-fire family:

  1. LeakyIntegrateAndFire  — classical LIF (Lapicque, 1907)
  2. ExponentialIntegrateAndFire — EIF (Fourcaud-Trocmé et al., 2003)
  3. AdaptiveExponentialIF  — AdEx (Brette & Gerstner, 2005)

These models sacrifice biophysical detail for computational tractability
while preserving essential subthreshold and firing-rate properties.

Canonical equations
-------------------
LIF:
    C_m dV/dt = -g_L(V - E_L) + I_ext
    If V ≥ V_thresh → spike, then V ← V_reset, refractory period t_ref

AdEx (superset):
    C_m dV/dt = -g_L(V-E_L) + g_L·ΔT·exp((V-V_T)/ΔT) - w + I_ext
    τ_w dw/dt = a(V-E_L) - w
    On spike: V ← V_reset, w ← w + b

References
----------
Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique
    des nerfs traitée comme une polarisation.
    J Physiol Pathol Gen, 9, 620-635.
Brette, R. & Gerstner, W. (2005). Adaptive exponential integrate-and-fire
    model as an effective description of neuronal activity.
    J Neurophysiol, 94, 3637-3642.
Naud, R. et al. (2008). Firing patterns in the adaptive exponential
    integrate-and-fire model. Biol Cybern, 99, 335-347.
"""

import numpy as np
from .base_neuron import BaseNeuron


# ── 1. Leaky Integrate-and-Fire ───────────────────────────────────────────
class LeakyIntegrateAndFire(BaseNeuron):
    """
    Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    C_m       : µF   membrane capacitance            (default 200 pF = 0.2 nF)
    g_L       : nS   leak conductance                (default 10 nS)
    E_L       : mV   leak / resting potential        (default -70 mV)
    V_thresh  : mV   spike threshold                 (default -55 mV)
    V_reset   : mV   reset potential after spike     (default -70 mV)
    V_peak    : mV   voltage clipped to at spike     (default +30 mV)
    t_ref     : ms   absolute refractory period      (default 2 ms)

    Note: currents in pA; C_m in pF; g_L in nS for standard cortical params.
    """

    def __init__(
        self,
        C_m:       float = 200.0,   # pF
        g_L:       float = 10.0,    # nS
        E_L:       float = -70.0,   # mV
        V_thresh:  float = -55.0,   # mV
        V_reset:   float = -70.0,   # mV
        V_peak:    float =  30.0,   # mV
        t_ref:     float =   2.0,   # ms
        neuron_id: int   = 0,
    ):
        self._V_rest   = E_L
        self._V_thresh = V_thresh
        super().__init__(neuron_id=neuron_id)

        self.C_m      = C_m
        self.g_L      = g_L
        self.E_L      = E_L
        self.V_reset  = V_reset
        self.V_peak   = V_peak
        self.t_ref    = t_ref

        self._ref_remaining = 0.0   # ms of refractory period left
        self._V = E_L

    @property
    def V_rest(self) -> float:
        return self._V_rest

    @property
    def V_thresh(self) -> float:
        return self._V_thresh

    def reset(self) -> None:
        self._V = self.E_L
        self._ref_remaining = 0.0
        self.t = 0.0
        from .base_neuron import SpikeRecord
        self.spikes = SpikeRecord(neuron_id=self.neuron_id)

    def step(self, I_ext: float, dt: float) -> float:
        """Euler integration with refractory clamp."""
        if self._ref_remaining > 0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            return self._V

        dV = (-(self._V - self.E_L) * self.g_L + I_ext) / self.C_m
        self._V += dV * dt

        if self._V >= self._V_thresh:
            self._V = self.V_peak
            self.spikes.add_spike(self.t)
            self._ref_remaining = self.t_ref
            # will be reset to V_reset on next refractory step
        return self._V

    def membrane_time_constant(self) -> float:
        """τ_m = C_m / g_L  (ms)."""
        return self.C_m / self.g_L   # pF / nS = ms  ✓


# ── 2. Exponential Integrate-and-Fire ─────────────────────────────────────
class ExponentialIntegrateAndFire(LeakyIntegrateAndFire):
    """
    Exponential Integrate-and-Fire (EIF) neuron.

    Adds an exponential spike-initiation current:
        C_m dV/dt = -g_L(V-E_L) + g_L·ΔT·exp((V-V_T)/ΔT) + I_ext

    ΔT (delta_T) is the 'sharpness' of spike initiation (mV).
    V_T is the effective threshold (softer than hard threshold).

    Reference: Fourcaud-Trocmé et al. (2003). J Neurosci, 23, 11628-11640.
    """

    def __init__(
        self,
        C_m:       float = 200.0,
        g_L:       float = 10.0,
        E_L:       float = -70.0,
        V_T:       float = -55.0,   # soft threshold
        V_thresh:  float = -30.0,   # hard cutoff (spike detected here)
        V_reset:   float = -70.0,
        V_peak:    float =  30.0,
        delta_T:   float =   2.0,   # mV — spike sharpness
        t_ref:     float =   2.0,
        neuron_id: int   = 0,
    ):
        super().__init__(
            C_m=C_m, g_L=g_L, E_L=E_L,
            V_thresh=V_thresh, V_reset=V_reset,
            V_peak=V_peak, t_ref=t_ref, neuron_id=neuron_id,
        )
        self.V_T     = V_T
        self.delta_T = delta_T

    def step(self, I_ext: float, dt: float) -> float:
        if self._ref_remaining > 0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            return self._V

        exp_term = self.g_L * self.delta_T * np.exp(
            (self._V - self.V_T) / self.delta_T
        )
        dV = (-(self._V - self.E_L) * self.g_L + exp_term + I_ext) / self.C_m
        self._V += dV * dt

        if self._V >= self._V_thresh:
            self._V = self.V_peak
            self.spikes.add_spike(self.t)
            self._ref_remaining = self.t_ref
        return self._V


# ── 3. Adaptive Exponential Integrate-and-Fire (AdEx) ────────────────────
class AdaptiveExponentialIF(ExponentialIntegrateAndFire):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx / Brette-Gerstner) model.

    Adds a subthreshold adaptation current w:
        C_m dV/dt = -g_L(V-E_L) + g_L·ΔT·exp((V-V_T)/ΔT) - w + I_ext
        τ_w dw/dt = a(V-E_L) - w
        On spike: V ← V_reset; w ← w + b

    Parameters
    ----------
    a    : nS    subthreshold adaptation coupling
    b    : pA    spike-triggered adaptation increment
    tau_w: ms    adaptation time constant
    """

    # Canonical AdEx parameter sets for different cell types
    PRESETS = {
        # (a_nS, b_pA, tau_w_ms)  — Naud et al. 2008
        "RS":    (4.0,    80.5,  144.0),   # regular spiking
        "IB":    (4.0,    80.5,   16.0),   # intrinsic bursting
        "CH":    (4.0,    80.5,    5.0),   # chattering
        "FS":    (0.0,     0.0,   40.0),   # fast spiking
        "LTS":   (8.0,   200.0,  200.0),   # low-threshold spiking
        "TC":    (4.0,     4.0,   40.0),   # thalamocortical
    }

    def __init__(
        self,
        C_m:       float = 200.0,
        g_L:       float = 10.0,
        E_L:       float = -70.0,
        V_T:       float = -55.0,
        V_thresh:  float = -30.0,
        V_reset:   float = -58.0,
        V_peak:    float =  30.0,
        delta_T:   float =   2.0,
        t_ref:     float =   0.0,
        a:         float =   4.0,    # nS
        b:         float =  80.5,    # pA
        tau_w:     float = 144.0,    # ms
        neuron_id: int   = 0,
        preset:    str   = None,
    ):
        super().__init__(
            C_m=C_m, g_L=g_L, E_L=E_L,
            V_T=V_T, V_thresh=V_thresh, V_reset=V_reset,
            V_peak=V_peak, delta_T=delta_T, t_ref=t_ref,
            neuron_id=neuron_id,
        )
        if preset and preset in self.PRESETS:
            a, b, tau_w = self.PRESETS[preset]

        self.a     = a
        self.b     = b
        self.tau_w = tau_w
        self.w     = 0.0          # adaptation variable (pA)
        self._preset = preset

    def reset(self) -> None:
        super().reset()
        self.w = 0.0

    def step(self, I_ext: float, dt: float) -> float:
        if self._ref_remaining > 0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            return self._V

        # exponential membrane equation
        exp_term = self.g_L * self.delta_T * np.exp(
            (self._V - self.V_T) / self.delta_T
        )
        dV = (-(self._V - self.E_L) * self.g_L
              + exp_term - self.w + I_ext) / self.C_m
        # adaptation current
        dw = (self.a * (self._V - self.E_L) - self.w) / self.tau_w

        self._V += dV * dt
        self.w  += dw * dt

        if self._V >= self._V_thresh:
            self._V = self.V_reset    # immediately reset (avoids exp overflow)
            self.spikes.add_spike(self.t)
            self.w += self.b          # spike-triggered adaptation jump
            self._ref_remaining = self.t_ref
        return self._V

    def simulate_detailed(
        self,
        I_ext: np.ndarray,
        dt: float = 0.1,
    ) -> dict:
        """Simulate and record w(t) along with V(t)."""
        self.reset()
        n = len(I_ext)
        t_arr = np.arange(n) * dt
        V_arr = np.zeros(n)
        w_arr = np.zeros(n)

        for i, (t_i, I_i) in enumerate(zip(t_arr, I_ext)):
            self.t = t_i
            self.step(I_i, dt)
            V_arr[i] = self._V
            w_arr[i] = self.w

        return {
            "t": t_arr, "V": V_arr, "w": w_arr,
            "spikes": np.array(self.spikes.times),
            "firing_rate_hz": self.spikes.firing_rate(t_arr[-1]),
            "cv_isi": self.spikes.cv_isi(),
            "preset": self._preset,
        }

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "AdaptiveExponentialIF":
        """Convenience constructor using named cell-type parameter sets."""
        return cls(preset=preset, **kwargs)
