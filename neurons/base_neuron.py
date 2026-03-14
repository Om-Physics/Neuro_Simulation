"""
base_neuron.py
==============
Abstract base class for all neuron models.

Provides a unified interface for biophysical neuron implementations
following standard computational neuroscience conventions.

References
----------
Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.
Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge University Press.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpikeRecord:
    """Container for spike timing data."""
    times: list = field(default_factory=list)
    neuron_id: int = 0

    def add_spike(self, t: float) -> None:
        self.times.append(t)

    @property
    def count(self) -> int:
        return len(self.times)

    def isi(self) -> np.ndarray:
        """Inter-spike intervals (ms)."""
        t = np.array(self.times)
        return np.diff(t) if len(t) > 1 else np.array([])

    def firing_rate(self, duration_ms: float) -> float:
        """Mean firing rate in Hz."""
        return self.count / (duration_ms * 1e-3) if duration_ms > 0 else 0.0

    def cv_isi(self) -> float:
        """Coefficient of variation of ISI — regularity measure."""
        isi = self.isi()
        if len(isi) < 2:
            return float("nan")
        return float(np.std(isi) / np.mean(isi))


class BaseNeuron(ABC):
    """
    Abstract base class for biophysical neuron models.

    All subclasses must implement `step()` and `reset()`.
    Voltage is stored in millivolts (mV); time in milliseconds (ms);
    current in picoamperes (pA) or microamperes per cm² (µA/cm²)
    depending on whether the model is point or compartmental.
    """

    def __init__(self, neuron_id: int = 0):
        self.neuron_id = neuron_id
        self.spikes = SpikeRecord(neuron_id=neuron_id)
        self.t = 0.0          # current time (ms)
        self._V: float = self.V_rest   # membrane potential

    # ── subclasses must define these class-level constants ────────────────
    @property
    @abstractmethod
    def V_rest(self) -> float:
        """Resting membrane potential (mV)."""

    @property
    @abstractmethod
    def V_thresh(self) -> float:
        """Spike threshold (mV)."""

    # ── required interface ────────────────────────────────────────────────
    @abstractmethod
    def step(self, I_ext: float, dt: float) -> float:
        """
        Advance the model by one timestep dt (ms).

        Parameters
        ----------
        I_ext : float
            External input current (pA or µA/cm²).
        dt : float
            Integration timestep (ms).

        Returns
        -------
        float
            Membrane voltage after the step (mV).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset state variables to resting conditions."""

    # ── shared utilities ──────────────────────────────────────────────────
    @property
    def V(self) -> float:
        return self._V

    def simulate(
        self,
        I_ext: np.ndarray,
        dt: float = 0.025,
        t_start: float = 0.0,
    ) -> dict:
        """
        Run a full simulation given an external current trace.

        Parameters
        ----------
        I_ext : np.ndarray
            Input current array, one value per timestep.
        dt : float
            Timestep in ms (default 0.025 ms → 40 kHz).
        t_start : float
            Start time in ms.

        Returns
        -------
        dict with keys: 't', 'V', 'spikes', 'I_ext'
        """
        self.reset()
        n = len(I_ext)
        t_arr = t_start + np.arange(n) * dt
        V_arr = np.zeros(n)

        for i, (t_i, I_i) in enumerate(zip(t_arr, I_ext)):
            self.t = t_i
            V_arr[i] = self.step(I_i, dt)

        return {
            "t": t_arr,
            "V": V_arr,
            "spikes": np.array(self.spikes.times),
            "I_ext": I_ext,
            "firing_rate_hz": self.spikes.firing_rate(t_arr[-1] - t_arr[0]),
            "cv_isi": self.spikes.cv_isi(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.neuron_id}, "
            f"V={self._V:.2f} mV, spikes={self.spikes.count})"
        )
