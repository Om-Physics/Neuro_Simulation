"""
network.py
==========
Recurrent excitatory/inhibitory (E/I) spiking neural network.

Architecture
------------
The canonical cortical microcircuit (Brunel 2000) with:
  • N_E excitatory LIF/AdEx neurons
  • N_I inhibitory LIF/AdEx neurons (N_I = N_E / 4  by default)
  • Random sparse connectivity (connection probability p)
  • AMPA synapses from E→E and E→I
  • GABA-A synapses from I→E and I→I
  • Optional NMDA component on E→E recurrent connections
  • External Poisson drive to all neurons

Network states (Brunel 2000 classification)
--------------------------------------------
  SR  : synchronous regular
  AI  : asynchronous irregular  ← biologically realistic
  SIf : synchronous irregular fast
  SIs : synchronous irregular slow

References
----------
Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory
    and inhibitory spiking neurons. J Comput Neurosci, 8, 183-208.
Amit, D.J. & Brunel, N. (1997). Model of global spontaneous activity and
    local structured activity during delay periods in cerebral cortex.
    Cereb Cortex, 7, 237-252.
"""

import numpy as np
from typing import Optional, Callable
from neurons.integrate_fire import LeakyIntegrateAndFire, AdaptiveExponentialIF
from synapses.synapse import AMPASynapse, GABAASynapse, NMDASynapse


class SpikingNetwork:
    """
    Random sparse recurrent E/I network.

    Parameters
    ----------
    N_E    : number of excitatory neurons
    N_I    : number of inhibitory neurons (default N_E//4)
    p_conn : connection probability (Erdős-Rényi)
    dt     : simulation timestep (ms)
    neuron_type : 'LIF' or 'AdEx'
    use_NMDA    : include NMDA on E→E synapses
    seed        : random seed for reproducibility
    """

    def __init__(
        self,
        N_E:          int   = 400,
        N_I:          Optional[int] = None,
        p_conn:       float = 0.1,
        dt:           float = 0.1,          # ms
        neuron_type:  str   = "LIF",
        use_NMDA:     bool  = False,
        seed:         int   = 42,
        # Synaptic weights (nS)
        g_EE:         float = 0.3,           # E→E AMPA
        g_EI:         float = 0.3,           # E→I AMPA
        g_IE:         float = 3.0,           # I→E GABA-A
        g_II:         float = 3.0,           # I→I GABA-A
        g_EE_NMDA:    float = 0.1,           # E→E NMDA (if enabled)
        # External Poisson drive
        nu_ext:       float = 2.0,           # kHz  (external rate)
        g_ext:        float = 0.3,           # nS   (external AMPA)
    ):
        self.rng = np.random.default_rng(seed)

        self.N_E = N_E
        self.N_I = N_I if N_I is not None else N_E // 4
        self.N   = self.N_E + self.N_I
        self.p_conn = p_conn
        self.dt     = dt
        self.neuron_type = neuron_type
        self.use_NMDA = use_NMDA

        self.g_EE = g_EE; self.g_EI = g_EI
        self.g_IE = g_IE; self.g_II = g_II
        self.g_EE_NMDA = g_EE_NMDA
        self.nu_ext = nu_ext * 1e-3  # convert kHz → spikes/ms
        self.g_ext  = g_ext

        self._build_neurons()
        self._build_connectivity()
        self._build_synapses()

        # State tracking
        self.t = 0.0
        self._spike_matrix: list[list[float]] = [[] for _ in range(self.N)]

    # ── Construction ──────────────────────────────────────────────────────
    def _build_neurons(self) -> None:
        """Instantiate E and I neuron populations."""
        def make_lif(nid):
            return LeakyIntegrateAndFire(
                C_m=200.0, g_L=10.0, E_L=-70.0,
                V_thresh=-50.0, V_reset=-60.0, t_ref=2.0,
                neuron_id=nid,
            )

        def make_adex_e(nid):
            return AdaptiveExponentialIF.from_preset("RS", neuron_id=nid)

        def make_adex_i(nid):
            return AdaptiveExponentialIF.from_preset("FS", neuron_id=nid)

        if self.neuron_type == "AdEx":
            self.neurons = (
                [make_adex_e(i) for i in range(self.N_E)] +
                [make_adex_i(i + self.N_E) for i in range(self.N_I)]
            )
        else:
            self.neurons = [make_lif(i) for i in range(self.N)]

    def _build_connectivity(self) -> None:
        """
        Generate random sparse connectivity matrix.
        conn[i][j] = True means neuron j sends a spike to neuron i.
        """
        self.conn = self.rng.random((self.N, self.N)) < self.p_conn
        np.fill_diagonal(self.conn, False)    # no autapses

    def _build_synapses(self) -> None:
        """Instantiate synapses for connected pairs."""
        # syn[i] = list of (source_idx, Synapse) that project TO neuron i
        self.syn: list[list] = [[] for _ in range(self.N)]

        for post in range(self.N):
            for pre in range(self.N):
                if not self.conn[post, pre]:
                    continue
                if pre < self.N_E:      # excitatory presynaptic
                    syn = AMPASynapse(g_max=self.g_EE if post < self.N_E
                                      else self.g_EI)
                    self.syn[post].append((pre, syn))
                    if self.use_NMDA and post < self.N_E:
                        nmda = NMDASynapse(g_max=self.g_EE_NMDA)
                        self.syn[post].append((pre, nmda))
                else:                   # inhibitory presynaptic
                    g = self.g_II if post >= self.N_E else self.g_IE
                    syn = GABAASynapse(g_max=g)
                    self.syn[post].append((pre, syn))

    # ── Simulation ────────────────────────────────────────────────────────
    def run(
        self,
        T_ms: float,
        I_ext: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """
        Simulate network for T_ms milliseconds.

        Parameters
        ----------
        T_ms       : total simulation duration (ms)
        I_ext      : (N, n_steps) array of external currents; if None uses
                     Poisson drive only
        progress_cb: optional callback f(t) for progress monitoring

        Returns
        -------
        dict with keys:
            't'         : time array (ms)
            'V'         : (N, n_steps) voltage array (mV)
            'spikes'    : list of length N, each element a list of spike times
            'pop_E_rate': (n_steps,) E population firing rate (Hz)
            'pop_I_rate': (n_steps,) I population firing rate (Hz)
            'LFP'       : (n_steps,) proxy local field potential
        """
        n_steps = int(T_ms / self.dt)
        t_arr   = np.arange(n_steps) * self.dt
        V_arr   = np.zeros((self.N, n_steps))
        spikes  = [[] for _ in range(self.N)]

        # rolling firing rate windows (50 ms)
        win_E = np.zeros(self.N_E, dtype=int)
        win_I = np.zeros(self.N_I, dtype=int)
        pop_E_rate = np.zeros(n_steps)
        pop_I_rate = np.zeros(n_steps)
        LFP_arr    = np.zeros(n_steps)

        prev_fired = np.zeros(self.N, dtype=bool)

        for step_i in range(n_steps):
            t = t_arr[step_i]
            fired = np.zeros(self.N, dtype=bool)

            for nid, neuron in enumerate(self.neurons):
                neuron.t = t
                # External Poisson input as equivalent current
                if self.rng.random() < self.nu_ext * self.dt:
                    I_poisson = self.g_ext * (neuron.V - 0.0) * (-1)
                    # current injection: Poisson event → delta of conductance
                    I_poisson = self.g_ext * 40.0  # approximation (pA)
                else:
                    I_poisson = 0.0

                I_user = I_ext[nid, step_i] if I_ext is not None else 0.0

                # Synaptic current
                I_syn = 0.0
                for (pre_id, syn) in self.syn[nid]:
                    pre_spiked = bool(prev_fired[pre_id])
                    I_syn += syn.step(self.dt, neuron.V, spike=pre_spiked)

                V_new = neuron.step(I_user + I_poisson - I_syn, self.dt)
                V_arr[nid, step_i] = V_new

                if len(neuron.spikes.times) > len(spikes[nid]):
                    fired[nid] = True
                    spikes[nid].append(t)

            prev_fired = fired

            # Population firing rate (Hz) via instantaneous spike count
            n_fired_E = np.sum(fired[:self.N_E])
            n_fired_I = np.sum(fired[self.N_E:])
            pop_E_rate[step_i] = (n_fired_E / self.N_E) / (self.dt * 1e-3)
            pop_I_rate[step_i] = (n_fired_I / self.N_I) / (self.dt * 1e-3)

            # LFP proxy: mean synaptic current to E cells
            LFP_arr[step_i] = np.mean(V_arr[:self.N_E, step_i])

            if progress_cb and step_i % 1000 == 0:
                progress_cb(t)

        return {
            "t":          t_arr,
            "V":          V_arr,
            "spikes":     spikes,
            "pop_E_rate": pop_E_rate,
            "pop_I_rate": pop_I_rate,
            "LFP":        LFP_arr,
        }

    def reset(self) -> None:
        """Reset all neurons and synapses to initial state."""
        for neuron in self.neurons:
            neuron.reset()
        for post_list in self.syn:
            for (_, syn) in post_list:
                syn.reset()
        self.t = 0.0
        self._spike_matrix = [[] for _ in range(self.N)]

    @property
    def excitatory_neurons(self):
        return self.neurons[:self.N_E]

    @property
    def inhibitory_neurons(self):
        return self.neurons[self.N_E:]

    def summary(self) -> str:
        lines = [
            f"SpikingNetwork Summary",
            f"  Neurons : {self.N_E} E + {self.N_I} I = {self.N} total",
            f"  Model   : {self.neuron_type}",
            f"  p_conn  : {self.p_conn}",
            f"  E→E (AMPA) weight: {self.g_EE} nS",
            f"  I→E (GABA-A) weight: {self.g_IE} nS",
            f"  NMDA    : {self.use_NMDA}",
            f"  Ext. drive: {self.nu_ext*1e3:.2f} kHz @ {self.g_ext} nS",
        ]
        return "\n".join(lines)
