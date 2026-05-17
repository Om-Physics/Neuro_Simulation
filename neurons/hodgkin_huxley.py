"""
hodgkin_huxley.py
=================
Full conductance-based Hodgkin-Huxley (1952) neuron model for the squid
giant axon at 6.3 degrees Celsius, integrated with fourth-order Runge-Kutta.

State vector: [V, m, h, n]
  V : membrane voltage (mV)
  m : Na+ activation gate  (dimensionless, [0,1])
  h : Na+ inactivation gate (dimensionless, [0,1])
  n : K+  activation gate  (dimensionless, [0,1])

Membrane equation:
  Cm * dV/dt = I_ext - I_Na - I_K - I_L
  I_Na = gNa * m^3 * h * (V - ENa)
  I_K  = gK  * n^4      * (V - EK)
  I_L  = gL             * (V - EL)

Reference:
  Hodgkin, A. L. and Huxley, A. F. (1952). A quantitative description of
  membrane current and its application to conduction and excitation in nerve.
  Journal of Physiology, 117(4), 500-544.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np
from neurons.base_neuron import BaseNeuron, SpikeRecord


class HodgkinHuxley(BaseNeuron):
    """
    Hodgkin-Huxley conductance-based neuron model.

    Parameters
    ----------
    Cm   : float  Membrane capacitance (uF/cm^2). Default 1.0.
    gNa  : float  Maximum Na+ conductance (mS/cm^2). Default 120.0.
    gK   : float  Maximum K+  conductance (mS/cm^2). Default 36.0.
    gL   : float  Leak conductance (mS/cm^2). Default 0.3.
    ENa  : float  Na+ reversal potential (mV). Default +55.0.
    EK   : float  K+  reversal potential (mV). Default -77.0.
    EL   : float  Leak reversal potential (mV). Default -54.387.
    """

    def __init__(
        self,
        Cm: float = 1.0,
        gNa: float = 120.0,
        gK: float = 36.0,
        gL: float = 0.3,
        ENa: float = 55.0,
        EK: float = -77.0,
        EL: float = -54.387,
    ) -> None:
        super().__init__(name="HodgkinHuxley")
        self.Cm  = Cm
        self.gNa = gNa
        self.gK  = gK
        self.gL  = gL
        self.ENa = ENa
        self.EK  = EK
        self.EL  = EL

        self.V_init = -65.0
        self._V = self.V_init
        self._m, self._h, self._n = self._steady_state(self.V_init)
        self._in_spike = False

    def _steady_state(self, V: float) -> tuple[float, float, float]:
        """Return (m_inf, h_inf, n_inf) at voltage V."""
        m = self._alpha_m(V) / (self._alpha_m(V) + self._beta_m(V))
        h = self._alpha_h(V) / (self._alpha_h(V) + self._beta_h(V))
        n = self._alpha_n(V) / (self._alpha_n(V) + self._beta_n(V))
        return m, h, n

    def _alpha_m(self, V: float) -> float:
        dv = V + 40.0
        if abs(dv) < 1e-7:
            return 1.0
        return 0.1 * dv / (1.0 - np.exp(-dv / 10.0))

    def _beta_m(self, V: float) -> float:
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V: float) -> float:
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V: float) -> float:
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V: float) -> float:
        dv = V + 55.0
        if abs(dv) < 1e-7:
            return 0.1
        return 0.01 * dv / (1.0 - np.exp(-dv / 10.0))

    def _beta_n(self, V: float) -> float:
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def _clip_gate(self, x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _derivatives(
        self, V: float, m: float, h: float, n: float, I: float
    ) -> tuple[float, float, float, float]:
        """Evaluate the four coupled ODEs at the given state."""
        I_Na = self.gNa * m**3 * h * (V - self.ENa)
        I_K  = self.gK  * n**4     * (V - self.EK)
        I_L  = self.gL             * (V - self.EL)

        dV = (I - I_Na - I_K - I_L) / self.Cm
        dm = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dh = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dn = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n
        return dV, dm, dh, dn

    def step(self, dt: float, I_ext: float = 0.0) -> float:
        """Advance by dt ms using RK4. Returns new membrane voltage."""
        V, m, h, n = self._V, self._m, self._h, self._n

        k1 = self._derivatives(V, m, h, n, I_ext)
        k2 = self._derivatives(
            V + 0.5*dt*k1[0],
            self._clip_gate(m + 0.5*dt*k1[1]),
            self._clip_gate(h + 0.5*dt*k1[2]),
            self._clip_gate(n + 0.5*dt*k1[3]), I_ext)
        k3 = self._derivatives(
            V + 0.5*dt*k2[0],
            self._clip_gate(m + 0.5*dt*k2[1]),
            self._clip_gate(h + 0.5*dt*k2[2]),
            self._clip_gate(n + 0.5*dt*k2[3]), I_ext)
        k4 = self._derivatives(
            V + dt*k3[0],
            self._clip_gate(m + dt*k3[1]),
            self._clip_gate(h + dt*k3[2]),
            self._clip_gate(n + dt*k3[3]), I_ext)

        self._V = V + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self._m = self._clip_gate(m + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]))
        self._h = self._clip_gate(h + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))
        self._n = self._clip_gate(n + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]))
        self._t += dt
        return self._V

    def simulate(
        self,
        T: float = 120.0,
        dt: float = 0.025,
        I_ext: float | np.ndarray = 10.0,
        t_start: float = 10.0,
    ) -> SpikeRecord:
        """
        Run a complete simulation.

        Parameters
        ----------
        T       : total duration (ms)
        dt      : RK4 step size (ms), should be <= 0.025 for HH
        I_ext   : scalar or array of injected current (uA/cm^2)
        t_start : onset time of the step current (ms)

        Returns
        -------
        SpikeRecord with voltage trace, spike times, and gating variables.
        """
        self.reset()
        n_steps = int(T / dt)
        t_axis = np.linspace(0, T, n_steps)

        if np.ndim(I_ext) == 0:
            I_array = np.where(t_axis >= t_start, float(I_ext), 0.0)
        else:
            I_array = np.asarray(I_ext, dtype=float)

        V_trace = np.empty(n_steps)
        m_trace = np.empty(n_steps)
        h_trace = np.empty(n_steps)
        n_trace = np.empty(n_steps)
        spike_times: list[float] = []
        in_spike = False

        for i, (t, I) in enumerate(zip(t_axis, I_array)):
            V = self.step(dt, I)
            V_trace[i] = V
            m_trace[i] = self._m
            h_trace[i] = self._h
            n_trace[i] = self._n

            if V >= -20.0 and not in_spike:
                spike_times.append(t)
                in_spike = True
            elif V < -40.0:
                in_spike = False

        rec = SpikeRecord(
            times=np.array(spike_times),
            voltages=V_trace,
            time_axis=t_axis,
            metadata={
                "m": m_trace, "h": h_trace, "n": n_trace,
                "model": "HodgkinHuxley",
                "I_array": I_array,
                "dt": dt, "T": T,
            },
        )
        return rec

    def simulate_detailed(
        self, T: float = 120.0, dt: float = 0.025, I_ext: float = 10.0
    ) -> dict:
        """Return full state traces including ionic currents."""
        rec = self.simulate(T, dt, I_ext)
        m = rec.metadata["m"]
        h = rec.metadata["h"]
        n = rec.metadata["n"]
        I_Na = self.gNa * m**3 * h * (rec.voltages - self.ENa)
        I_K  = self.gK  * n**4     * (rec.voltages - self.EK)
        I_L  = self.gL             * (rec.voltages - self.EL)
        return {
            "t": rec.time_axis, "V": rec.voltages,
            "m": m, "h": h, "n": n,
            "I_Na": I_Na, "I_K": I_K, "I_L": I_L,
            "I_ext": rec.metadata["I_array"],
            "spikes": rec.times,
        }

    def nullclines(
        self, V_range: np.ndarray | None = None, I_ext: float = 10.0
    ) -> dict:
        """
        Compute V-nullcline and n-nullcline for phase-plane analysis.

        Returns a dict with keys 'V', 'V_null', 'n_null', 'n_inf'.
        """
        if V_range is None:
            V_range = np.linspace(-80, 50, 400)

        V_null = np.array([
            (I_ext
             - self.gNa * self._alpha_m(v)**3 / (self._alpha_m(v)+self._beta_m(v))**3
               * self._alpha_h(v) / (self._alpha_h(v)+self._beta_h(v)) * (v - self.ENa)
             - self.gL * (v - self.EL))
            / (self.gK * (v - self.EK) + 1e-12)
            for v in V_range
        ])
        V_null = np.clip(V_null**0.25, 0, 1)

        n_inf = np.array([
            self._alpha_n(v) / (self._alpha_n(v) + self._beta_n(v))
            for v in V_range
        ])

        return {"V": V_range, "V_null": V_null, "n_null": n_inf, "n_inf": n_inf}

    def time_constants(self, V_range: np.ndarray | None = None) -> dict:
        """Return voltage-dependent time constants for m, h, n (ms)."""
        if V_range is None:
            V_range = np.linspace(-80, 50, 400)
        tau_m = np.array([1.0/(self._alpha_m(v)+self._beta_m(v)) for v in V_range])
        tau_h = np.array([1.0/(self._alpha_h(v)+self._beta_h(v)) for v in V_range])
        tau_n = np.array([1.0/(self._alpha_n(v)+self._beta_n(v)) for v in V_range])
        return {"V": V_range, "tau_m": tau_m, "tau_h": tau_h, "tau_n": tau_n}

    def reset(self) -> None:
        super().reset()
        self._V = self.V_init
        self._m, self._h, self._n = self._steady_state(self.V_init)
        self._in_spike = False
