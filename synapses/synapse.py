"""
synapse.py
==========
Kinetic synaptic receptor models for the four major receptor classes in
cortical circuits. All fast ionotropic receptors use the two-state scheme
of Destexhe et al. (1994). GABA-B uses the four-state G-protein cascade.

Receptor kinetics (two-state):
  dr/dt = alpha * [T](t) * (1 - r) - beta * r
  I_syn = g_max * r(t) * (V_post - E_rev)

NMDA Mg2+ block (Jahr and Stevens 1990):
  B(V) = 1 / (1 + exp(-0.062 * V) * [Mg] / 3.57)
  I_NMDA = g_max * r(t) * B(V) * (V_post - E_rev)

GABA-B G-protein cascade (Destexhe et al. 1994):
  dR/dt = K1 * [T] * (1 - R) - K2 * R
  dG/dt = K3 * R - K4 * G
  g_GABA_B = g_max * G^n / (G^n + Kd)

References:
  Destexhe, A., Mainen, Z. F., and Sejnowski, T. J. (1994).
      Synthesis of models for excitable membranes, synaptic transmission
      and neuromodulation using a common kinetic formalism.
      J. Comput. Neurosci. 1, 195-230.
  Jahr, C. E. and Stevens, C. F. (1990). Voltage dependence of NMDA-activated
      macroscopic conductances predicted by single-channel kinetics.
      J. Neurosci. 10(9), 3178-3182.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np


class AMPASynapse:
    """
    AMPA receptor synapse (fast glutamatergic excitation).

    Decay time constant ~ 5 ms. Excitatory, E_rev = 0 mV.
    """

    def __init__(
        self,
        g_max: float = 0.5,
        E_rev: float = 0.0,
        alpha: float = 1.1,
        beta: float = 0.19,
        t_peak: float = 1.0,
    ) -> None:
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.t_peak = t_peak

        self.r = 0.0
        self._T = 0.0
        self._spike_t = -1e9
        self.conductance_history: list[float] = []
        self.time_history: list[float] = []
        self._t = 0.0

    def receive_spike(self) -> None:
        """Register a presynaptic spike at the current simulation time."""
        self._spike_t = self._t

    def step(self, dt: float, V_post: float = -65.0) -> float:
        """
        Advance receptor state by dt ms.

        Returns the synaptic current in pA.
        """
        self._t += dt
        T = 1.0 if (self._t - self._spike_t) <= self.t_peak else 0.0
        self.r = np.clip(
            self.r + (self.alpha * T * (1.0 - self.r) - self.beta * self.r) * dt,
            0.0, 1.0
        )
        I_syn = self.g_max * self.r * (V_post - self.E_rev)
        self.conductance_history.append(self.g_max * self.r)
        self.time_history.append(self._t)
        return I_syn

    def simulate(
        self,
        T: float = 100.0,
        dt: float = 0.1,
        spike_times: list[float] | None = None,
        V_post: float = -65.0,
    ) -> dict:
        self.reset()
        if spike_times is None:
            spike_times = [10.0]
        n = int(T / dt)
        t_ax = np.linspace(0, T, n)
        g_trace = np.empty(n)
        I_trace = np.empty(n)
        sp_set = set(np.searchsorted(t_ax, spike_times))

        for i, t in enumerate(t_ax):
            if i in sp_set:
                self.receive_spike()
            g_trace[i] = self.g_max * self.r
            I_trace[i] = self.step(dt, V_post)

        return {"t": t_ax, "g": g_trace, "I": I_trace}

    def reset(self) -> None:
        self.r = 0.0
        self._t = 0.0
        self._spike_t = -1e9
        self.conductance_history.clear()
        self.time_history.clear()


class NMDASynapse:
    """
    NMDA receptor synapse with voltage-dependent Mg2+ block.

    Properties: slow kinetics (tau_decay ~ 150 ms), Ca2+ permeable,
    and N-shaped I-V curve due to Mg2+ block at hyperpolarised potentials.
    The Mg2+ block implements coincidence detection: simultaneous pre- and
    postsynaptic activity is required for significant conductance.
    """

    def __init__(
        self,
        g_max: float = 0.5,
        E_rev: float = 0.0,
        alpha: float = 0.072,
        beta: float = 0.0066,
        t_peak: float = 1.0,
        Mg: float = 1.0,
    ) -> None:
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.t_peak = t_peak
        self.Mg = Mg

        self.r = 0.0
        self._t = 0.0
        self._spike_t = -1e9

    def B(self, V: float) -> float:
        """
        Magnesium block factor B(V) from Jahr and Stevens (1990).

        B(V) = 0 at resting potential (channel blocked)
        B(V) = 1 at depolarised potential (channel open)
        """
        return 1.0 / (1.0 + np.exp(-0.062 * V) * self.Mg / 3.57)

    def receive_spike(self) -> None:
        self._spike_t = self._t

    def step(self, dt: float, V_post: float = -65.0) -> float:
        self._t += dt
        T = 1.0 if (self._t - self._spike_t) <= self.t_peak else 0.0
        self.r = np.clip(
            self.r + (self.alpha * T * (1.0 - self.r) - self.beta * self.r) * dt,
            0.0, 1.0
        )
        return self.g_max * self.r * self.B(V_post) * (V_post - self.E_rev)

    def effective_conductance(self, V: float) -> float:
        return self.g_max * self.r * self.B(V)

    def simulate(
        self,
        T: float = 300.0,
        dt: float = 0.1,
        spike_times: list[float] | None = None,
        V_post: float = -65.0,
    ) -> dict:
        self.reset()
        if spike_times is None:
            spike_times = [10.0]
        n = int(T / dt)
        t_ax = np.linspace(0, T, n)
        g_trace = np.empty(n)
        sp_set = set(np.searchsorted(t_ax, spike_times))

        for i in range(n):
            if i in sp_set:
                self.receive_spike()
            self.step(dt, V_post)
            g_trace[i] = self.effective_conductance(V_post)

        return {"t": t_ax, "g": g_trace, "B": self.B(V_post)}

    def iv_curve(self, V_range: np.ndarray | None = None) -> dict:
        """Return the N-shaped I-V relationship due to Mg2+ block."""
        if V_range is None:
            V_range = np.linspace(-90, 60, 300)
        r_ss = 0.8
        I = np.array([
            self.g_max * r_ss * self.B(V) * (V - self.E_rev)
            for V in V_range
        ])
        B_vals = np.array([self.B(V) for V in V_range])
        return {"V": V_range, "I": I, "B": B_vals}

    def reset(self) -> None:
        self.r = 0.0
        self._t = 0.0
        self._spike_t = -1e9


class GABAAsynapse:
    """
    GABA-A receptor synapse (fast chloride-mediated inhibition).

    Decay time constant ~ 5-10 ms. Inhibitory, E_rev = -70 mV (Cl-).
    """

    def __init__(
        self,
        g_max: float = 0.8,
        E_rev: float = -70.0,
        alpha: float = 5.0,
        beta: float = 0.18,
        t_peak: float = 1.0,
    ) -> None:
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.t_peak = t_peak
        self.r = 0.0
        self._t = 0.0
        self._spike_t = -1e9

    def receive_spike(self) -> None:
        self._spike_t = self._t

    def step(self, dt: float, V_post: float = -65.0) -> float:
        self._t += dt
        T = 1.0 if (self._t - self._spike_t) <= self.t_peak else 0.0
        self.r = np.clip(
            self.r + (self.alpha * T * (1.0 - self.r) - self.beta * self.r) * dt,
            0.0, 1.0
        )
        return self.g_max * self.r * (V_post - self.E_rev)

    def simulate(
        self,
        T: float = 100.0,
        dt: float = 0.1,
        spike_times: list[float] | None = None,
        V_post: float = -65.0,
    ) -> dict:
        self.reset()
        if spike_times is None:
            spike_times = [10.0]
        n = int(T / dt)
        t_ax = np.linspace(0, T, n)
        g_trace = np.empty(n)
        sp_set = set(np.searchsorted(t_ax, spike_times))
        for i in range(n):
            if i in sp_set:
                self.receive_spike()
            self.step(dt, V_post)
            g_trace[i] = self.g_max * self.r
        return {"t": t_ax, "g": g_trace}

    def reset(self) -> None:
        self.r = 0.0
        self._t = 0.0
        self._spike_t = -1e9


class GABABsynapse:
    """
    GABA-B receptor synapse using the G-protein cascade model.

    Metabotropic receptor coupled to inwardly rectifying K+ channels
    via a second-messenger G-protein pathway. Slow kinetics: peak ~ 120 ms,
    decay ~ 200 ms. E_rev = -95 mV (K+ Nernst potential).

    State variables:
      R : receptor activation (fraction bound)
      G : G-protein activation
      g_GABA_B = g_max * G^n / (G^n + Kd)  [Hill equation, n=4]
    """

    def __init__(
        self,
        g_max: float = 0.5,
        E_rev: float = -95.0,
        K1: float = 0.52,
        K2: float = 0.0013,
        K3: float = 0.098,
        K4: float = 0.033,
        Kd: float = 100.0,
        n: float = 4.0,
        t_peak: float = 0.3,
    ) -> None:
        self.g_max = g_max
        self.E_rev = E_rev
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.Kd = Kd
        self.n = n
        self.t_peak = t_peak
        self.R = 0.0
        self.G = 0.0
        self._t = 0.0
        self._spike_t = -1e9

    def receive_spike(self) -> None:
        self._spike_t = self._t

    @property
    def g(self) -> float:
        """Effective conductance via Hill equation."""
        Gn = self.G ** self.n
        return self.g_max * Gn / (Gn + self.Kd)

    def step(self, dt: float, V_post: float = -65.0) -> float:
        self._t += dt
        T = 0.5 if (self._t - self._spike_t) <= self.t_peak else 0.0
        self.R = np.clip(
            self.R + (self.K1 * T * (1.0 - self.R) - self.K2 * self.R) * dt,
            0.0, 1.0
        )
        self.G = max(0.0, self.G + (self.K3 * self.R - self.K4 * self.G) * dt)
        return self.g * (V_post - self.E_rev)

    def simulate(
        self,
        T: float = 500.0,
        dt: float = 0.1,
        spike_times: list[float] | None = None,
        V_post: float = -65.0,
    ) -> dict:
        self.reset()
        if spike_times is None:
            spike_times = [10.0, 30.0, 50.0, 70.0]
        n = int(T / dt)
        t_ax = np.linspace(0, T, n)
        g_trace = np.empty(n)
        sp_set = set(np.searchsorted(t_ax, spike_times))
        for i in range(n):
            if i in sp_set:
                self.receive_spike()
            self.step(dt, V_post)
            g_trace[i] = self.g
        return {"t": t_ax, "g": g_trace}

    def reset(self) -> None:
        self.R = 0.0
        self.G = 0.0
        self._t = 0.0
        self._spike_t = -1e9
