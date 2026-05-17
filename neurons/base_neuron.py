"""
base_neuron.py
==============
Abstract base class and shared data structures for all neuron models
in the Neuro_Simulation suite.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpikeRecord:
    """Container for spike train data returned by every neuron model."""

    times: np.ndarray          # spike times in ms
    voltages: np.ndarray       # full voltage trace in mV
    time_axis: np.ndarray      # simulation time axis in ms
    metadata: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def mean_rate(self) -> float:
        """Mean firing rate in Hz over the entire simulation."""
        duration_s = (self.time_axis[-1] - self.time_axis[0]) * 1e-3
        return self.count / duration_s if duration_s > 0 else 0.0

    def isi(self) -> np.ndarray:
        """Inter-spike intervals in ms."""
        return np.diff(self.times) if len(self.times) > 1 else np.array([])

    def cv_isi(self) -> float:
        """Coefficient of variation of the ISI distribution."""
        intervals = self.isi()
        if len(intervals) < 2:
            return float("nan")
        return float(np.std(intervals) / np.mean(intervals))


class BaseNeuron(ABC):
    """
    Abstract base class that every neuron model must implement.

    Subclasses override ``step`` for numerical integration and
    ``simulate`` for full trial simulation.  The ``fI_curve`` method
    sweeps input currents and returns steady-state firing rates.
    """

    def __init__(self, name: str = "neuron"):
        self.name = name
        self._spike_times: list[float] = []
        self._t: float = 0.0

    @abstractmethod
    def step(self, dt: float, I_ext: float = 0.0) -> float:
        """
        Advance the neuron state by one time step.

        Parameters
        ----------
        dt : float
            Integration step size in ms.
        I_ext : float
            External injected current in appropriate units.

        Returns
        -------
        float
            Current membrane voltage in mV.
        """

    @abstractmethod
    def simulate(
        self,
        T: float,
        dt: float,
        I_ext: float | np.ndarray,
        t_start: float = 0.0,
    ) -> SpikeRecord:
        """
        Run a full simulation and return a SpikeRecord.

        Parameters
        ----------
        T : float
            Total simulation duration in ms.
        dt : float
            Time step in ms.
        I_ext : float or ndarray
            Constant current or time-varying current array.
        t_start : float
            Time at which the current is switched on, in ms.
        """

    def fI_curve(
        self,
        I_values: np.ndarray,
        T: float = 500.0,
        dt: float = 0.1,
        warmup: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency-current (f-I) relationship.

        Parameters
        ----------
        I_values : ndarray
            Array of injected current values to sweep.
        T : float
            Simulation duration per current value, in ms.
        dt : float
            Integration step size in ms.
        warmup : float
            Initial period excluded from spike counting, in ms.

        Returns
        -------
        (I_values, rates) : tuple of ndarrays
            Input currents and corresponding mean firing rates in Hz.
        """
        rates = []
        for I in I_values:
            rec = self.simulate(T, dt, I)
            spikes_after_warmup = rec.times[rec.times > warmup]
            duration_s = (T - warmup) * 1e-3
            rates.append(len(spikes_after_warmup) / duration_s)
        return I_values, np.array(rates)

    def reset(self) -> None:
        """Reset internal state to initial conditions."""
        self._spike_times = []
        self._t = 0.0

    @property
    def spike_times(self) -> list[float]:
        return self._spike_times.copy()
