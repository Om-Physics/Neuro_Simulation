"""
network.py
==========
Sparse random excitatory-inhibitory (E/I) recurrent spiking neural network
based on the Brunel (2000) framework. Implements a population of LIF neurons
with random Erdos-Renyi connectivity, AMPA-like excitatory synapses, and
GABA-A-like inhibitory synapses, driven by an external Poisson process.

Network composition:
  N_E excitatory LIF neurons (default 100)
  N_I inhibitory LIF neurons (default 25, ratio 4:1)
  Synaptic connections drawn with probability p = 0.15 (no autapses)

Synaptic conductances:
  E -> E, E -> I : AMPA-like (fast excitation, E_rev = 0 mV)
  I -> E, I -> I : GABA-A-like (fast inhibition, E_rev = -70 mV)
  External Poisson : AMPA-like onto all neurons

Reference:
  Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory
  and inhibitory spiking neurons. J. Comput. Neurosci. 8(3), 183-208.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np


class SpikingNetwork:
    """
    Sparse random E/I LIF network simulation.

    Parameters
    ----------
    N_E     : Number of excitatory neurons. Default 100.
    N_I     : Number of inhibitory neurons. Default 25.
    dt      : Integration time step (ms). Default 0.2.
    p_conn  : Connection probability. Default 0.15.
    I_dc    : DC baseline current per neuron (pA). Default 220.
    g_EE    : E->E AMPA peak conductance (nS). Default 0.35.
    g_EI    : E->I AMPA peak conductance (nS). Default 0.35.
    g_IE    : I->E GABA-A peak conductance (nS). Default 2.0.
    g_II    : I->I GABA-A peak conductance (nS). Default 2.0.
    g_ext   : External Poisson AMPA conductance (nS). Default 0.9.
    nu_ext  : External Poisson rate (kHz). Default 15.
    Cm      : Membrane capacitance (pF). Default 200.
    gL      : Leak conductance (nS). Default 10.
    EL      : Leak reversal potential (mV). Default -70.
    V_thresh: Threshold (mV). Default -55.
    V_reset : Reset voltage (mV). Default -70.
    t_ref   : Refractory period (ms). Default 2.
    tau_E   : Excitatory synaptic decay (ms). Default 5.
    tau_I   : Inhibitory synaptic decay (ms). Default 10.
    seed    : Random seed for connectivity and Poisson input. Default 42.
    """

    def __init__(
        self,
        N_E: int = 100,
        N_I: int = 25,
        dt: float = 0.2,
        p_conn: float = 0.15,
        I_dc: float = 220.0,
        g_EE: float = 0.35,
        g_EI: float = 0.35,
        g_IE: float = 2.0,
        g_II: float = 2.0,
        g_ext: float = 0.9,
        nu_ext: float = 15.0,
        Cm: float = 200.0,
        gL: float = 10.0,
        EL: float = -70.0,
        V_thresh: float = -55.0,
        V_reset: float = -70.0,
        t_ref: float = 2.0,
        tau_E: float = 5.0,
        tau_I: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.N_E = N_E
        self.N_I = N_I
        self.N   = N_E + N_I
        self.dt  = dt
        self.p_conn = p_conn
        self.I_dc   = I_dc
        self.g_EE   = g_EE
        self.g_EI   = g_EI
        self.g_IE   = g_IE
        self.g_II   = g_II
        self.g_ext  = g_ext
        self.nu_ext = nu_ext
        self.Cm     = Cm
        self.gL     = gL
        self.EL     = EL
        self.V_thresh = V_thresh
        self.V_reset  = V_reset
        self.t_ref    = t_ref
        self.tau_E    = tau_E
        self.tau_I    = tau_I
        self.seed     = seed
        self.rng      = np.random.default_rng(seed)

        self._build_connectivity()
        self._init_state()

    def _build_connectivity(self) -> None:
        """Draw Erdos-Renyi connectivity matrix (no autapses)."""
        N = self.N
        M = self.rng.random((N, N))
        np.fill_diagonal(M, 1.0)          # no autapses
        self.conn = M < self.p_conn       # boolean adjacency matrix

    def _init_state(self) -> None:
        """Initialise membrane voltages and synaptic conductance arrays."""
        N = self.N
        self.V   = self.rng.uniform(self.EL, self.V_thresh, N)
        self.ref = np.zeros(N)
        self.gE  = np.zeros(N)            # total excitatory conductance
        self.gI  = np.zeros(N)            # total inhibitory conductance
        self.t   = 0.0
        self.raster: list[tuple[float, int]] = []
        self.pop_rate_E: list[float] = []
        self.pop_rate_I: list[float] = []
        self.V_mean_E: list[float] = []

    def step(self) -> np.ndarray:
        """
        Advance the network by one time step dt.

        Returns
        -------
        fired : boolean array, True for neurons that fired this step.
        """
        N, dt = self.N, self.dt
        decay_E = np.exp(-dt / self.tau_E)
        decay_I = np.exp(-dt / self.tau_I)

        # Synaptic decay
        self.gE *= decay_E
        self.gI *= decay_I

        # External Poisson drive
        p_ext = self.nu_ext * dt       # probability of ext spike per step per neuron
        ext_spikes = self.rng.random(N) < p_ext
        self.gE[ext_spikes] += self.g_ext

        # Membrane integration and spike detection
        fired = np.zeros(N, dtype=bool)
        for i in range(N):
            if self.ref[i] > 0:
                self.ref[i] -= dt
                self.V[i] = self.V_reset
                continue
            I_E = self.gE[i] * (0.0 - self.V[i])
            I_I = self.gI[i] * (-70.0 - self.V[i])
            dV  = (-self.gL * (self.V[i] - self.EL) + I_E + I_I + self.I_dc) / self.Cm
            self.V[i] += dV * dt
            if self.V[i] >= self.V_thresh:
                fired[i] = True
                self.raster.append((self.t, i))

        # Post-spike reset and synaptic propagation
        for i in np.where(fired)[0]:
            self.V[i] = self.V_reset
            self.ref[i] = self.t_ref
            targets = np.where(self.conn[i])[0]
            if i < self.N_E:
                self.gE[targets[targets < self.N_E]] += self.g_EE
                self.gE[targets[targets >= self.N_E]] += self.g_EI
            else:
                self.gI[targets[targets < self.N_E]] += self.g_IE
                self.gI[targets[targets >= self.N_E]] += self.g_II

        # Population statistics
        n_E_fired = np.sum(fired[:self.N_E])
        n_I_fired = np.sum(fired[self.N_E:])
        dur_s = dt * 1e-3
        self.pop_rate_E.append(n_E_fired / (self.N_E * dur_s))
        self.pop_rate_I.append(n_I_fired / (self.N_I * dur_s))
        self.V_mean_E.append(float(np.mean(self.V[:self.N_E])))

        self.t += dt
        return fired

    def run(self, T: float = 400.0) -> dict:
        """
        Run the network for T ms.

        Parameters
        ----------
        T : Total simulation duration (ms).

        Returns
        -------
        dict containing raster, population rates, LFP proxy, spike trains.
        """
        self._init_state()
        n_steps = int(T / self.dt)
        for _ in range(n_steps):
            self.step()

        t_axis = np.arange(len(self.pop_rate_E)) * self.dt
        raster_t = np.array([r[0] for r in self.raster])
        raster_id = np.array([r[1] for r in self.raster], dtype=int)

        # Build per-neuron spike trains
        spike_trains: dict[int, np.ndarray] = {}
        for n_id in range(self.N):
            mask = raster_id == n_id
            spike_trains[n_id] = raster_t[mask]

        # Mean rates
        win = min(200, len(self.pop_rate_E))
        mean_E = float(np.mean(self.pop_rate_E[-win:])) if self.pop_rate_E else 0.0
        mean_I = float(np.mean(self.pop_rate_I[-win:])) if self.pop_rate_I else 0.0

        return {
            "raster_t":    raster_t,
            "raster_id":   raster_id,
            "pop_rate_E":  np.array(self.pop_rate_E),
            "pop_rate_I":  np.array(self.pop_rate_I),
            "LFP":         np.array(self.V_mean_E),
            "t":           t_axis,
            "spike_trains": spike_trains,
            "mean_rate_E": mean_E,
            "mean_rate_I": mean_I,
            "T": T, "N_E": self.N_E, "N_I": self.N_I,
        }
