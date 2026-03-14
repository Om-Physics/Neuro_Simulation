"""
hodgkin_huxley.py
=================
Full Hodgkin-Huxley (1952) conductance-based neuron model.

Biophysical basis
-----------------
The HH model describes the membrane as a capacitor in parallel with
voltage-gated ion channels.  The four coupled ODEs are:

    C_m dV/dt  = I_ext - I_Na - I_K - I_L
    dm/dt      = α_m(V)(1-m) - β_m(V)·m
    dh/dt      = α_h(V)(1-h) - β_h(V)·h
    dn/dt      = α_n(V)(1-n) - β_n(V)·n

where:
    I_Na = g̅_Na · m³h · (V - E_Na)   fast inward sodium current
    I_K  = g̅_K  · n⁴  · (V - E_K)   delayed-rectifier potassium current
    I_L  = g̅_L        · (V - E_L)   leak (passive) current

Units
-----
Voltage : mV
Time    : ms
Current : µA / cm²
Conductance: mS / cm²
Capacitance: µF / cm²

Original parameters from squid giant axon at 6.3 °C.

References
----------
Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of
    membrane current and its application to conduction and excitation in
    nerve. J Physiol, 117(4), 500-544.
Pospischil, M. et al. (2008). Minimal Hodgkin-Huxley type models for
    different classes of cortical and thalamic neurons.
    Biol Cybern, 99, 427-441.
"""

import numpy as np
from .base_neuron import BaseNeuron


class HodgkinHuxley(BaseNeuron):
    """
    Hodgkin-Huxley conductance-based point neuron.

    Parameters
    ----------
    C_m   : µF/cm²   membrane capacitance          (default 1.0)
    g_Na  : mS/cm²   max Na⁺ conductance           (default 120.0)
    g_K   : mS/cm²   max K⁺  conductance           (default 36.0)
    g_L   : mS/cm²   leak conductance              (default 0.3)
    E_Na  : mV       Na⁺ reversal potential        (default +55.0)
    E_K   : mV       K⁺  reversal potential        (default -77.0)
    E_L   : mV       leak reversal potential       (default -54.387)
    V_rest: mV       initial membrane potential    (default -65.0)
    """

    # ── HH canonical parameters (squid axon, 6.3 °C) ─────────────────────
    C_m_default  = 1.0      # µF/cm²
    g_Na_default = 120.0    # mS/cm²
    g_K_default  = 36.0     # mS/cm²
    g_L_default  = 0.3      # mS/cm²
    E_Na_default = 55.0     # mV
    E_K_default  = -77.0    # mV
    E_L_default  = -54.387  # mV
    V_rest_default = -65.0  # mV

    def __init__(
        self,
        C_m:   float = C_m_default,
        g_Na:  float = g_Na_default,
        g_K:   float = g_K_default,
        g_L:   float = g_L_default,
        E_Na:  float = E_Na_default,
        E_K:   float = E_K_default,
        E_L:   float = E_L_default,
        V_init: float = V_rest_default,
        neuron_id: int = 0,
    ):
        self._V_rest = V_init
        super().__init__(neuron_id=neuron_id)

        self.C_m  = C_m
        self.g_Na = g_Na
        self.g_K  = g_K
        self.g_L  = g_L
        self.E_Na = E_Na
        self.E_K  = E_K
        self.E_L  = E_L

        # gating variables initialised at steady-state
        self.m, self.h, self.n = self._steady_state(V_init)
        self._V = V_init

        # diagnostics
        self._spike_rising = False

    # ── BaseNeuron interface ──────────────────────────────────────────────
    @property
    def V_rest(self) -> float:
        return self._V_rest

    @property
    def V_thresh(self) -> float:
        return -55.0   # approximate peak of dV/dt crossover

    def reset(self) -> None:
        self._V = self._V_rest
        self.m, self.h, self.n = self._steady_state(self._V_rest)
        self.t = 0.0
        self.spikes = type(self.spikes)(neuron_id=self.neuron_id)
        self._spike_rising = False

    def step(self, I_ext: float, dt: float) -> float:
        """
        4th-order Runge-Kutta integration of HH equations.

        Parameters
        ----------
        I_ext : µA/cm²
        dt    : ms

        Returns
        -------
        float : V(t+dt) in mV
        """
        state = np.array([self._V, self.m, self.h, self.n])
        k1 = self._deriv(state, I_ext)
        k2 = self._deriv(state + 0.5 * dt * k1, I_ext)
        k3 = self._deriv(state + 0.5 * dt * k2, I_ext)
        k4 = self._deriv(state +       dt * k3, I_ext)
        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self._V, self.m, self.h, self.n = state_new
        # clamp gating variables to [0,1]
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)

        # spike detection via upward threshold crossing
        if self._V >= self.V_thresh and not self._spike_rising:
            self.spikes.add_spike(self.t)
            self._spike_rising = True
        elif self._V < self.V_thresh:
            self._spike_rising = False

        return self._V

    # ── full simulation returning ion-channel details ─────────────────────
    def simulate_detailed(
        self,
        I_ext: np.ndarray,
        dt: float = 0.025,
        t_start: float = 0.0,
    ) -> dict:
        """
        Simulate and record full ion-channel state trajectory.

        Returns dict with: t, V, m, h, n, I_Na, I_K, I_L, spikes.
        """
        self.reset()
        n_steps = len(I_ext)
        t_arr = t_start + np.arange(n_steps) * dt
        V_arr = np.zeros(n_steps)
        m_arr = np.zeros(n_steps)
        h_arr = np.zeros(n_steps)
        n_arr = np.zeros(n_steps)

        for i, (t_i, I_i) in enumerate(zip(t_arr, I_ext)):
            self.t = t_i
            self.step(I_i, dt)
            V_arr[i] = self._V
            m_arr[i] = self.m
            h_arr[i] = self.h
            n_arr[i] = self.n

        I_Na = self.g_Na * (m_arr**3) * h_arr * (V_arr - self.E_Na)
        I_K  = self.g_K  * (n_arr**4)          * (V_arr - self.E_K)
        I_L  = self.g_L                         * (V_arr - self.E_L)

        return {
            "t": t_arr,
            "V": V_arr,
            "m": m_arr,
            "h": h_arr,
            "n": n_arr,
            "I_Na": I_Na,
            "I_K":  I_K,
            "I_L":  I_L,
            "spikes": np.array(self.spikes.times),
            "firing_rate_hz": self.spikes.firing_rate(t_arr[-1] - t_arr[0]),
            "cv_isi": self.spikes.cv_isi(),
        }

    # ── internal helpers ──────────────────────────────────────────────────
    def _deriv(self, state: np.ndarray, I_ext: float) -> np.ndarray:
        """Compute d/dt [V, m, h, n]."""
        V, m, h, n = state
        I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
        I_K  = self.g_K  * (n**4)     * (V - self.E_K)
        I_L  = self.g_L               * (V - self.E_L)

        dVdt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dmdt = self._alpha_m(V) * (1 - m) - self._beta_m(V) * m
        dhdt = self._alpha_h(V) * (1 - h) - self._beta_h(V) * h
        dndt = self._alpha_n(V) * (1 - n) - self._beta_n(V) * n
        return np.array([dVdt, dmdt, dhdt, dndt])

    # ── Hodgkin-Huxley rate functions (original 1952 formulation) ─────────
    @staticmethod
    def _alpha_m(V: float) -> float:
        """Na activation forward rate (ms⁻¹)."""
        dV = V + 40.0
        if abs(dV) < 1e-7:
            return 1.0
        return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))

    @staticmethod
    def _beta_m(V: float) -> float:
        """Na activation backward rate (ms⁻¹)."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V: float) -> float:
        """Na inactivation forward rate (ms⁻¹)."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V: float) -> float:
        """Na inactivation backward rate (ms⁻¹)."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(V: float) -> float:
        """K activation forward rate (ms⁻¹)."""
        dV = V + 55.0
        if abs(dV) < 1e-7:
            return 0.1
        return 0.01 * dV / (1.0 - np.exp(-dV / 10.0))

    @staticmethod
    def _beta_n(V: float) -> float:
        """K activation backward rate (ms⁻¹)."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def _steady_state(self, V: float):
        """Gating variables at steady state for given V."""
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        an, bn = self._alpha_n(V), self._beta_n(V)
        m_inf = am / (am + bm)
        h_inf = ah / (ah + bh)
        n_inf = an / (an + bn)
        return m_inf, h_inf, n_inf

    def time_constants(self, V: float) -> dict:
        """
        Return gating time constants τ_x = 1/(α_x + β_x) at voltage V.
        Useful for biophysical interpretation.
        """
        tau_m = 1.0 / (self._alpha_m(V) + self._beta_m(V))
        tau_h = 1.0 / (self._alpha_h(V) + self._beta_h(V))
        tau_n = 1.0 / (self._alpha_n(V) + self._beta_n(V))
        return {"tau_m": tau_m, "tau_h": tau_h, "tau_n": tau_n}

    def nullclines(self, V_range: np.ndarray) -> dict:
        """
        Compute m∞(V), h∞(V), n∞(V) for phase-plane analysis.
        """
        m_inf = np.array([
            self._alpha_m(v) / (self._alpha_m(v) + self._beta_m(v))
            for v in V_range
        ])
        h_inf = np.array([
            self._alpha_h(v) / (self._alpha_h(v) + self._beta_h(v))
            for v in V_range
        ])
        n_inf = np.array([
            self._alpha_n(v) / (self._alpha_n(v) + self._beta_n(v))
            for v in V_range
        ])
        return {"V": V_range, "m_inf": m_inf, "h_inf": h_inf, "n_inf": n_inf}

    def fI_curve(
        self,
        I_range: np.ndarray,
        T_ms: float = 1000.0,
        dt: float = 0.025,
    ) -> dict:
        """
        Compute the frequency–current (f-I) curve.

        Parameters
        ----------
        I_range : µA/cm² values to test
        T_ms    : simulation duration per current step
        dt      : timestep (ms)

        Returns
        -------
        dict with 'I' and 'f_hz' arrays
        """
        n_steps = int(T_ms / dt)
        f_out = []
        for I_val in I_range:
            I_arr = np.full(n_steps, I_val)
            result = self.simulate(I_arr, dt=dt)
            f_out.append(result["firing_rate_hz"])
        return {"I": I_range, "f_hz": np.array(f_out)}
