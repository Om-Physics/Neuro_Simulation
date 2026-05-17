"""
integrate_fire.py
=================
Three reduced spiking neuron models of increasing biophysical detail:

1. LeakyIntegrateAndFire (LIF)
   Classical RC membrane with hard threshold and voltage reset.

2. ExponentialIntegrateAndFire (EIF)
   LIF augmented with an exponential spike-initiation term that
   reproduces the sharp voltage upstroke observed in cortical neurons.

3. AdaptiveExponentialIntegrateAndFire (AdEx)
   EIF further augmented with a subthreshold adaptation current w(t)
   capable of reproducing the full repertoire of cortical firing patterns
   (RS, IB, CH, FS, LTS, TC) through parameter variation alone.

Membrane equations (AdEx, Brette and Gerstner 2005):
  Cm * dV/dt = -gL*(V - EL) + gL*dT*exp((V-VT)/dT) - w + I_ext
  tw * dw/dt = a*(V - EL) - w
  Spike condition : V >= V_thresh -> V = V_reset, w += b

References:
  Lapicque (1907). Recherches quantitatives sur l'excitation electrique
      des nerfs. J. Physiol. Pathol. Gen., 9, 620-635.
  Fourcaud-Trocme et al. (2003). J. Neurosci. 23(37), 11628-11640.
  Brette and Gerstner (2005). J. Neurophysiol. 94(5), 3637-3642.
  Naud et al. (2008). Biol. Cybern. 99(4-5), 335-347.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np
from neurons.base_neuron import BaseNeuron, SpikeRecord


class LeakyIntegrateAndFire(BaseNeuron):
    """
    Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    Cm      : Membrane capacitance (pF). Default 200.
    gL      : Leak conductance (nS). Default 10.
    EL      : Leak reversal potential (mV). Default -70.
    V_thresh: Firing threshold (mV). Default -55.
    V_reset : Reset potential after spike (mV). Default -70.
    t_ref   : Absolute refractory period (ms). Default 2.
    """

    def __init__(
        self,
        Cm: float = 200.0,
        gL: float = 10.0,
        EL: float = -70.0,
        V_thresh: float = -55.0,
        V_reset: float = -70.0,
        t_ref: float = 2.0,
    ) -> None:
        super().__init__(name="LIF")
        self.Cm = Cm
        self.gL = gL
        self.EL = EL
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.tau_m = Cm / gL         # membrane time constant (ms)

        self._V = EL
        self._ref_remaining = 0.0

    def step(self, dt: float, I_ext: float = 0.0) -> float:
        if self._ref_remaining > 0.0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            self._t += dt
            return self._V

        dV = (-self.gL * (self._V - self.EL) + I_ext) / self.Cm
        self._V += dV * dt
        self._t += dt

        if self._V >= self.V_thresh:
            self._spike_times.append(self._t)
            self._V = self.V_reset
            self._ref_remaining = self.t_ref

        return self._V

    def simulate(
        self,
        T: float = 500.0,
        dt: float = 0.1,
        I_ext: float | np.ndarray = 300.0,
        t_start: float = 0.0,
    ) -> SpikeRecord:
        self.reset()
        n_steps = int(T / dt)
        t_axis = np.linspace(0, T, n_steps)

        if np.ndim(I_ext) == 0:
            I_array = np.where(t_axis >= t_start, float(I_ext), 0.0)
        else:
            I_array = np.asarray(I_ext, dtype=float)

        V_trace = np.empty(n_steps)
        for i, I in enumerate(I_array):
            V_trace[i] = self.step(dt, I)

        return SpikeRecord(
            times=np.array(self._spike_times),
            voltages=V_trace,
            time_axis=t_axis,
            metadata={"model": "LIF", "tau_m": self.tau_m, "I_array": I_array},
        )

    def analytical_rate(self, I: float) -> float:
        """
        Exact analytical firing rate (Hz) for constant injected current I (pA).
        Returns 0 below rheobase.
        """
        I_rh = self.gL * (self.V_thresh - self.EL)
        if I <= I_rh:
            return 0.0
        arg = 1.0 - self.gL * (self.V_thresh - self.EL) / (I - self.gL * (self.V_reset - self.EL))
        if arg <= 0:
            return 0.0
        isi = self.t_ref - self.tau_m * np.log(arg)
        return 1000.0 / isi if isi > 0 else 0.0

    def reset(self) -> None:
        super().reset()
        self._V = self.EL
        self._ref_remaining = 0.0


class ExponentialIntegrateAndFire(BaseNeuron):
    """
    Exponential Integrate-and-Fire neuron (Fourcaud-Trocme et al. 2003).

    The exponential term gL * dT * exp((V - VT) / dT) models the rapid
    Na+ channel activation near threshold, producing a sharp spike upstroke.
    """

    def __init__(
        self,
        Cm: float = 200.0,
        gL: float = 10.0,
        EL: float = -70.0,
        VT: float = -55.0,
        dT: float = 2.0,
        V_thresh: float = -30.0,
        V_reset: float = -70.0,
        V_peak: float = 20.0,
        t_ref: float = 2.0,
    ) -> None:
        super().__init__(name="EIF")
        self.Cm = Cm
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.dT = dT
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.V_peak = V_peak
        self.t_ref = t_ref
        self._V = EL
        self._ref_remaining = 0.0

    def step(self, dt: float, I_ext: float = 0.0) -> float:
        if self._ref_remaining > 0.0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            self._t += dt
            return self._V

        exp_term = self.gL * self.dT * np.exp(
            min((self._V - self.VT) / self.dT, 20.0)
        )
        dV = (-self.gL * (self._V - self.EL) + exp_term + I_ext) / self.Cm
        self._V += dV * dt
        self._t += dt

        if self._V >= self.V_thresh:
            self._spike_times.append(self._t)
            self._V = self.V_reset
            self._ref_remaining = self.t_ref

        return self._V

    def simulate(
        self,
        T: float = 500.0,
        dt: float = 0.1,
        I_ext: float | np.ndarray = 300.0,
        t_start: float = 0.0,
    ) -> SpikeRecord:
        self.reset()
        n_steps = int(T / dt)
        t_axis = np.linspace(0, T, n_steps)

        if np.ndim(I_ext) == 0:
            I_array = np.where(t_axis >= t_start, float(I_ext), 0.0)
        else:
            I_array = np.asarray(I_ext, dtype=float)

        V_trace = np.empty(n_steps)
        for i, I in enumerate(I_array):
            V_trace[i] = self.step(dt, I)

        return SpikeRecord(
            times=np.array(self._spike_times),
            voltages=V_trace,
            time_axis=t_axis,
            metadata={"model": "EIF", "I_array": I_array},
        )

    def reset(self) -> None:
        super().reset()
        self._V = self.EL
        self._ref_remaining = 0.0


# Canonical AdEx parameter sets from Naud et al. (2008)
ADEX_PRESETS: dict[str, dict] = {
    "RS": {
        "a": 4.0, "b": 80.5, "tau_w": 144.0,
        "I_default": 250.0, "label": "Regular Spiking",
        "description": "Spike-frequency adaptation via incremental w accumulation."
    },
    "IB": {
        "a": 4.0, "b": 80.5, "tau_w": 16.0,
        "I_default": 600.0, "label": "Intrinsic Bursting",
        "description": "Initial burst of spikes followed by adaptation."
    },
    "CH": {
        "a": 4.0, "b": 80.5, "tau_w": 5.0,
        "I_default": 500.0, "label": "Chattering",
        "description": "High-frequency rhythmic bursting pattern."
    },
    "FS": {
        "a": 0.0, "b": 0.0, "tau_w": 40.0,
        "I_default": 250.0, "label": "Fast Spiking",
        "description": "No adaptation; tonic high-frequency firing. PV+ interneurons."
    },
    "LTS": {
        "a": 8.0, "b": 200.0, "tau_w": 200.0,
        "I_default": 300.0, "label": "Low-Threshold Spiking",
        "description": "Strong subthreshold coupling. SST+ interneurons."
    },
    "TC": {
        "a": 40.0, "b": 0.0, "tau_w": 300.0,
        "I_default": 400.0, "label": "Thalamo-Cortical",
        "description": "Subthreshold oscillations and rebound bursting."
    },
}


class AdaptiveExponentialIF(BaseNeuron):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

    The two-dimensional state (V, w) reproduces the full repertoire of
    cortical firing patterns by varying the four parameters (a, b, tau_w, I).

    Parameters
    ----------
    Cm      : Capacitance (pF). Default 200.
    gL      : Leak conductance (nS). Default 10.
    EL      : Leak reversal potential (mV). Default -70.
    VT      : Spike-initiation threshold (mV). Default -50.
    dT      : Sharpness of spike initiation (mV). Default 2.
    V_thresh: Detection threshold for spike (mV). Default -30.
    V_reset : Post-spike reset voltage (mV). Default -58.
    a       : Subthreshold adaptation coupling (nS). Default 4.
    b       : Spike-triggered adaptation increment (pA). Default 80.5.
    tau_w   : Adaptation time constant (ms). Default 144.
    t_ref   : Absolute refractory period (ms). Default 0.
    """

    def __init__(
        self,
        Cm: float = 200.0,
        gL: float = 10.0,
        EL: float = -70.0,
        VT: float = -50.0,
        dT: float = 2.0,
        V_thresh: float = -30.0,
        V_reset: float = -58.0,
        a: float = 4.0,
        b: float = 80.5,
        tau_w: float = 144.0,
        t_ref: float = 0.0,
    ) -> None:
        super().__init__(name="AdEx")
        self.Cm = Cm
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.dT = dT
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.t_ref = t_ref

        self._V = EL
        self._w = 0.0
        self._ref_remaining = 0.0

    @classmethod
    def from_preset(cls, preset: str) -> "AdaptiveExponentialIF":
        """
        Instantiate AdEx with a canonical cortical cell-type preset.

        Parameters
        ----------
        preset : one of 'RS', 'IB', 'CH', 'FS', 'LTS', 'TC'

        Returns
        -------
        AdaptiveExponentialIF instance with preset parameters.
        """
        if preset not in ADEX_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(ADEX_PRESETS.keys())}"
            )
        p = ADEX_PRESETS[preset]
        neuron = cls(a=p["a"], b=p["b"], tau_w=p["tau_w"])
        neuron.name = f"AdEx-{preset}"
        return neuron

    def step(self, dt: float, I_ext: float = 0.0) -> float:
        """Advance state by dt (ms). Returns membrane voltage (mV)."""
        if self._ref_remaining > 0.0:
            self._ref_remaining -= dt
            self._V = self.V_reset
            self._t += dt
            return self._V

        V, w = self._V, self._w

        # Exponential spike initiation
        exp_arg = min((V - self.VT) / self.dT, 20.0)
        exp_term = self.gL * self.dT * np.exp(exp_arg)

        dV = (-self.gL * (V - self.EL) + exp_term - w + I_ext) / self.Cm
        dw = (self.a * (V - self.EL) - w) / self.tau_w

        self._V = V + dV * dt
        self._w = w + dw * dt
        self._t += dt

        if self._V >= self.V_thresh:
            self._spike_times.append(self._t)
            self._V = self.V_reset        # immediate reset (prevents exp overflow)
            self._w += self.b
            self._ref_remaining = self.t_ref

        return self._V

    def simulate(
        self,
        T: float = 1000.0,
        dt: float = 0.1,
        I_ext: float | np.ndarray = 250.0,
        t_start: float = 50.0,
    ) -> SpikeRecord:
        self.reset()
        n_steps = int(T / dt)
        t_axis = np.linspace(0, T, n_steps)

        if np.ndim(I_ext) == 0:
            I_array = np.where(t_axis >= t_start, float(I_ext), 0.0)
        else:
            I_array = np.asarray(I_ext, dtype=float)

        V_trace = np.empty(n_steps)
        w_trace = np.empty(n_steps)

        for i, I in enumerate(I_array):
            V_trace[i] = self.step(dt, I)
            w_trace[i] = self._w

        return SpikeRecord(
            times=np.array(self._spike_times),
            voltages=V_trace,
            time_axis=t_axis,
            metadata={
                "model": self.name,
                "w": w_trace,
                "I_array": I_array,
                "a": self.a, "b": self.b, "tau_w": self.tau_w,
            },
        )

    def reset(self) -> None:
        super().reset()
        self._V = self.EL
        self._w = 0.0
        self._ref_remaining = 0.0
