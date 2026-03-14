"""
synapse.py
==========
Biophysical synapse models based on kinetic receptor schemes.

Model hierarchy
---------------
  BaseSynapse          — abstract; defines the g(t) → I_syn pipeline
  ├── AlphaSynapse     — alpha-function phenomenological model
  ├── AMPASynapse      — fast glutamatergic AMPA (2-state)
  ├── NMDASynapse      — slow glutamatergic NMDA with Mg²⁺ block
  ├── GABAASynapse     — fast inhibitory GABA-A (Cl⁻)
  └── GABABSynapse     — slow inhibitory GABA-B (K⁺, G-protein coupled)

Conductance-based formulation:
    I_syn = g_syn(t) · (V - E_syn)

where g_syn evolves according to kinetic equations driven by
presynaptic spike events.

Units
-----
Conductance : nS
Current     : pA
Voltage     : mV
Time        : ms

References
----------
Destexhe, A., Mainen, Z.F. & Sejnowski, T.J. (1994).
    Synthesis of models for excitable membranes, synaptic transmission
    and neuromodulation. J Comput Neurosci, 1, 195-230.
Jahr, C.E. & Stevens, C.F. (1990). Voltage dependence of NMDA-activated
    macroscopic conductances. J Neurosci, 10, 3178-3182.
Mainen, Z.F. & Sejnowski, T.J. (1996). Influence of dendritic structure
    on firing pattern in model neocortical neurons. Nature, 382, 363-366.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSynapse(ABC):
    """
    Abstract conductance-based synapse.

    Parameters
    ----------
    g_max : nS   peak conductance
    E_rev : mV   reversal (Nernst) potential
    """

    def __init__(self, g_max: float, E_rev: float, synapse_id: int = 0):
        self.g_max     = g_max
        self.E_rev     = E_rev
        self.synapse_id = synapse_id
        self.g         = 0.0   # current conductance (nS)
        self._history: list[float] = []   # pre-synaptic spike times

    @abstractmethod
    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        """
        Advance synapse state by dt ms.

        Parameters
        ----------
        dt      : ms   timestep
        V_post  : mV   postsynaptic membrane voltage
        spike   : bool did a presynaptic spike arrive this timestep?

        Returns
        -------
        float : synaptic current I_syn (pA)
        """

    def current(self, V_post: float) -> float:
        """I_syn = g · (V - E_rev)  [pA]."""
        return self.g * (V_post - self.E_rev)

    def reset(self) -> None:
        self.g = 0.0
        self._history.clear()


# ── 1. Alpha-function synapse (phenomenological) ──────────────────────────
class AlphaSynapse(BaseSynapse):
    """
    Alpha-function synapse:
        g(t) = g_max · (t/τ) · exp(1 - t/τ)  per spike

    Simple, widely used for qualitative network simulations.

    Parameters
    ----------
    tau_s : ms  time constant (rise ≈ peak time)
    """

    def __init__(
        self,
        g_max: float = 1.0,
        E_rev: float = 0.0,
        tau_s: float = 2.0,
        synapse_id: int = 0,
    ):
        super().__init__(g_max, E_rev, synapse_id)
        self.tau_s = tau_s
        self._t_since_spike: list[float] = []

    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        if spike:
            self._t_since_spike.append(0.0)

        self.g = 0.0
        surviving = []
        for t_s in self._t_since_spike:
            t_s += dt
            g_contribution = (
                self.g_max * (t_s / self.tau_s) * np.exp(1 - t_s / self.tau_s)
            )
            self.g += max(0.0, g_contribution)
            if t_s < 10 * self.tau_s:
                surviving.append(t_s)
        self._t_since_spike = surviving

        return self.current(V_post)

    def reset(self) -> None:
        super().reset()
        self._t_since_spike.clear()


# ── 2. AMPA synapse (fast excitatory, 2-state kinetics) ──────────────────
class AMPASynapse(BaseSynapse):
    """
    AMPA receptor synapse — fast glutamatergic excitation.

    Two-state kinetic model (Destexhe et al. 1994):
        dr/dt = α · [T](1-r) - β · r
    where [T] is transmitter concentration (square pulse after spike),
    r is fraction of open channels.
        g_AMPA = g_max · r

    Default parameters: cortical AMPA (Destexhe 1994 Table 1)

    Parameters
    ----------
    alpha : 1/(mM·ms)   binding rate
    beta  : 1/ms        unbinding rate
    T_max : mM          max transmitter concentration
    t_pulse : ms        transmitter pulse duration
    """

    def __init__(
        self,
        g_max:   float = 0.5,    # nS
        E_rev:   float = 0.0,    # mV  (cation channel, ~0 mV)
        alpha:   float = 1.1,    # 1/(mM·ms)
        beta:    float = 0.19,   # 1/ms
        T_max:   float = 1.0,    # mM
        t_pulse: float = 1.0,    # ms
        synapse_id: int = 0,
    ):
        super().__init__(g_max, E_rev, synapse_id)
        self.alpha   = alpha
        self.beta    = beta
        self.T_max   = T_max
        self.t_pulse = t_pulse
        self.r       = 0.0       # open-channel fraction
        self._t_trans = 0.0      # time since last spike (for pulse)

    @property
    def _T(self) -> float:
        """Transmitter concentration (square pulse model)."""
        return self.T_max if self._t_trans < self.t_pulse else 0.0

    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        if spike:
            self._t_trans = 0.0
        else:
            self._t_trans += dt

        T = self._T
        dr = self.alpha * T * (1 - self.r) - self.beta * self.r
        self.r = np.clip(self.r + dr * dt, 0.0, 1.0)
        self.g = self.g_max * self.r
        return self.current(V_post)

    def reset(self) -> None:
        super().reset()
        self.r = 0.0
        self._t_trans = 1e9


# ── 3. NMDA synapse (slow excitatory, Mg²⁺ block) ────────────────────────
class NMDASynapse(BaseSynapse):
    """
    NMDA receptor synapse — slow glutamatergic excitation with
    voltage-dependent Mg²⁺ block.

    Mg²⁺ block (Jahr & Stevens 1990):
        B(V) = 1 / (1 + exp(-0.062·V) · [Mg²⁺]_o / 3.57)

    Kinetics (slower than AMPA):
        dr/dt = α · [T](1-r) - β · r

    The NMDA conductance:
        g_NMDA(t,V) = g_max · r(t) · B(V)

    This introduces a nonlinear V-dependence critical for coincidence
    detection and Hebbian plasticity.

    Parameters
    ----------
    Mg_conc : mM   extracellular Mg²⁺ concentration (default 1 mM)
    """

    def __init__(
        self,
        g_max:   float = 0.5,    # nS
        E_rev:   float = 0.0,    # mV
        alpha:   float = 0.072,  # 1/(mM·ms)
        beta:    float = 0.0066, # 1/ms   (τ_decay ≈ 150 ms)
        T_max:   float = 1.0,    # mM
        t_pulse: float = 1.0,    # ms
        Mg_conc: float = 1.0,    # mM
        synapse_id: int = 0,
    ):
        super().__init__(g_max, E_rev, synapse_id)
        self.alpha   = alpha
        self.beta    = beta
        self.T_max   = T_max
        self.t_pulse = t_pulse
        self.Mg_conc = Mg_conc
        self.r       = 0.0
        self._t_trans = 1e9

    def Mg_block(self, V: float) -> float:
        """
        Voltage-dependent Mg²⁺ unblock factor B(V).
        B → 0 at hyperpolarized V (blocked), B → 1 at depolarized V.
        """
        return 1.0 / (1.0 + np.exp(-0.062 * V) * self.Mg_conc / 3.57)

    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        if spike:
            self._t_trans = 0.0
        else:
            self._t_trans += dt

        T = self.T_max if self._t_trans < self.t_pulse else 0.0
        dr = self.alpha * T * (1 - self.r) - self.beta * self.r
        self.r = np.clip(self.r + dr * dt, 0.0, 1.0)

        B = self.Mg_block(V_post)
        self.g = self.g_max * self.r * B
        return self.current(V_post)

    def reset(self) -> None:
        super().reset()
        self.r = 0.0
        self._t_trans = 1e9


# ── 4. GABA-A synapse (fast inhibitory, Cl⁻ channel) ─────────────────────
class GABAASynapse(BaseSynapse):
    """
    GABA-A receptor synapse — fast chloride-mediated inhibition.

    Same 2-state kinetics as AMPA but with chloride reversal potential.

    Parameters
    ----------
    E_rev : mV  Cl⁻ reversal (Nernst) potential, typically −70 to −80 mV
                Use −60 mV for shunting/depolarising inhibition in immature.
    """

    def __init__(
        self,
        g_max:   float = 0.5,    # nS
        E_rev:   float = -70.0,  # mV  (Cl⁻, hyperpolarising)
        alpha:   float = 5.0,    # 1/(mM·ms)
        beta:    float = 0.18,   # 1/ms
        T_max:   float = 1.0,    # mM
        t_pulse: float = 1.0,    # ms
        synapse_id: int = 0,
    ):
        super().__init__(g_max, E_rev, synapse_id)
        self.alpha   = alpha
        self.beta    = beta
        self.T_max   = T_max
        self.t_pulse = t_pulse
        self.r       = 0.0
        self._t_trans = 1e9

    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        if spike:
            self._t_trans = 0.0
        else:
            self._t_trans += dt

        T = self.T_max if self._t_trans < self.t_pulse else 0.0
        dr = self.alpha * T * (1 - self.r) - self.beta * self.r
        self.r = np.clip(self.r + dr * dt, 0.0, 1.0)
        self.g = self.g_max * self.r
        return self.current(V_post)

    def reset(self) -> None:
        super().reset()
        self.r = 0.0
        self._t_trans = 1e9


# ── 5. GABA-B synapse (slow inhibitory, G-protein, K⁺ channel) ───────────
class GABABSynapse(BaseSynapse):
    """
    GABA-B receptor synapse — metabotropic, K⁺-mediated slow inhibition.

    Four-state G-protein kinetic model (Destexhe et al. 1994):
        dR/dt  = K1 · [T](1-R) - K2 · R         (receptor activation)
        dG/dt  = K3 · R - K4 · G                (G-protein activation)
        g_GABAB = g_max · G^n / (G^n + Kd)      (Hill equation)

    Parameters (all from Destexhe 1994 Table 2)
    """

    def __init__(
        self,
        g_max: float = 0.5,     # nS
        E_rev: float = -95.0,   # mV  (K⁺ reversal via Nernst eq.)
        K1:    float = 0.52,    # 1/(mM·ms)
        K2:    float = 0.0013,  # 1/ms
        K3:    float = 0.098,   # 1/ms
        K4:    float = 0.033,   # 1/ms
        Kd:    float = 100.0,   # µM (G-protein affinity)
        n:     float = 4.0,     # Hill coefficient
        T_max: float = 0.5,     # mM
        t_pulse: float = 0.3,   # ms
        synapse_id: int = 0,
    ):
        super().__init__(g_max, E_rev, synapse_id)
        self.K1 = K1; self.K2 = K2
        self.K3 = K3; self.K4 = K4
        self.Kd = Kd; self.n = n
        self.T_max = T_max; self.t_pulse = t_pulse
        self.R = 0.0    # receptor state
        self.G = 0.0    # G-protein state
        self._t_trans = 1e9

    def step(self, dt: float, V_post: float, spike: bool = False) -> float:
        if spike:
            self._t_trans = 0.0
        else:
            self._t_trans += dt

        T = self.T_max if self._t_trans < self.t_pulse else 0.0
        dR = self.K1 * T * (1 - self.R) - self.K2 * self.R
        dG = self.K3 * self.R - self.K4 * self.G
        self.R = np.clip(self.R + dR * dt, 0.0, 1.0)
        self.G = max(0.0, self.G + dG * dt)

        self.g = self.g_max * (self.G**self.n) / (self.G**self.n + self.Kd)
        return self.current(V_post)

    def reset(self) -> None:
        super().reset()
        self.R = 0.0
        self.G = 0.0
        self._t_trans = 1e9
