"""
models.py
=========
SQLAlchemy ORM data model for persisting and querying all simulation
results produced by the Neuro_Simulation suite.

Seven tables are defined:

  SimulationRun      Top-level record for each experiment, recording model
                     type, parameters (JSON), timing, and summary statistics.

  NeuronParameters   Full parameter set for a neuron model instance,
                     linked one-to-one with a SimulationRun.

  SpikeTrainRecord   Compressed spike-time arrays for each neuron and trial,
                     stored as a NumPy binary blob.

  VoltageTrace       Downsampled voltage trace for one neuron per run,
                     stored as a NumPy binary blob.

  SynapseRecord      Synapse type, conductance trace, and kinetic parameters
                     for each synaptic simulation.

  PlasticityRecord   STDP or other plasticity rule parameters, final weight
                     distribution, and convergence statistics.

  AnalysisResult     Key-value store for derived scalar metrics (firing rate,
                     CV-ISI, Fano factor, dominant frequency, etc.) linked
                     to a SimulationRun.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import json
import io
import datetime
import numpy as np
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    LargeBinary, ForeignKey, Boolean, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


class Base(DeclarativeBase):
    pass


class SimulationRun(Base):
    """
    Top-level record representing one complete simulation experiment.

    Attributes
    ----------
    id            : Auto-incremented primary key.
    model_type    : Name of the neuron model (e.g. 'HodgkinHuxley', 'AdEx-RS').
    created_at    : UTC timestamp of run creation.
    T_ms          : Total simulation duration in ms.
    dt_ms         : Integration time step in ms.
    I_ext         : Applied external current (uA/cm2 for HH, pA for LIF/AdEx).
    n_spikes      : Total number of spikes detected.
    mean_rate_hz  : Mean firing rate in Hz.
    cv_isi        : Coefficient of variation of the ISI.
    fano_factor   : Fano factor (spike count variance / mean).
    params_json   : Full parameter dictionary serialised as JSON.
    notes         : Optional free-text annotation.
    """

    __tablename__ = "simulation_runs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_type   = Column(String(64), nullable=False, index=True)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)
    T_ms         = Column(Float)
    dt_ms        = Column(Float)
    I_ext        = Column(Float)
    n_spikes     = Column(Integer)
    mean_rate_hz = Column(Float)
    cv_isi       = Column(Float)
    fano_factor  = Column(Float)
    params_json  = Column(Text)
    notes        = Column(Text)

    neuron_params   = relationship("NeuronParameters",  back_populates="run",
                                   cascade="all, delete-orphan", uselist=False)
    spike_trains    = relationship("SpikeTrainRecord",  back_populates="run",
                                   cascade="all, delete-orphan")
    voltage_traces  = relationship("VoltageTrace",      back_populates="run",
                                   cascade="all, delete-orphan")
    synapse_records = relationship("SynapseRecord",     back_populates="run",
                                   cascade="all, delete-orphan")
    analysis_results= relationship("AnalysisResult",    back_populates="run",
                                   cascade="all, delete-orphan")

    def set_params(self, params: dict) -> None:
        self.params_json = json.dumps(params, default=str)

    def get_params(self) -> dict:
        return json.loads(self.params_json) if self.params_json else {}

    def __repr__(self) -> str:
        return (f"<SimulationRun id={self.id} model={self.model_type} "
                f"rate={self.mean_rate_hz:.1f}Hz>")


class NeuronParameters(Base):
    """
    Complete biophysical parameter set for the neuron used in a simulation run.
    Covers HH, LIF, and AdEx parameter namespaces (unused fields are NULL).
    """

    __tablename__ = "neuron_parameters"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    run_id  = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False, unique=True)
    run     = relationship("SimulationRun", back_populates="neuron_params")

    # Shared
    Cm_pF   = Column(Float)
    gL_nS   = Column(Float)
    EL_mV   = Column(Float)

    # HH specific
    gNa_mScm2 = Column(Float)
    gK_mScm2  = Column(Float)
    gL_mScm2  = Column(Float)
    ENa_mV    = Column(Float)
    EK_mV     = Column(Float)

    # LIF specific
    V_thresh_mV = Column(Float)
    V_reset_mV  = Column(Float)
    t_ref_ms    = Column(Float)

    # AdEx specific
    VT_mV    = Column(Float)
    dT_mV    = Column(Float)
    a_nS     = Column(Float)
    b_pA     = Column(Float)
    tau_w_ms = Column(Float)


class SpikeTrainRecord(Base):
    """
    Compressed spike-time array for one neuron / trial.

    The spike_times_blob field stores a NumPy float32 array serialised
    with np.save. Use ``get_spike_times()`` to recover the array.
    """

    __tablename__ = "spike_train_records"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_id           = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    run              = relationship("SimulationRun", back_populates="spike_trains")
    neuron_id        = Column(Integer, default=0)
    n_spikes         = Column(Integer)
    mean_rate_hz     = Column(Float)
    cv_isi           = Column(Float)
    spike_times_blob = Column(LargeBinary)

    def set_spike_times(self, times: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, times.astype(np.float32))
        self.spike_times_blob = buf.getvalue()
        self.n_spikes = len(times)

    def get_spike_times(self) -> np.ndarray:
        if not self.spike_times_blob:
            return np.array([])
        buf = io.BytesIO(self.spike_times_blob)
        return np.load(buf).astype(np.float64)


class VoltageTrace(Base):
    """
    Downsampled membrane voltage trace for one neuron.

    Stored as float32 NumPy array to reduce disk footprint. Use
    ``get_voltage()`` and ``get_time()`` to recover the arrays.
    """

    __tablename__ = "voltage_traces"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    run          = relationship("SimulationRun", back_populates="voltage_traces")
    neuron_id    = Column(Integer, default=0)
    downsample   = Column(Integer, default=10)
    dt_ms        = Column(Float)
    T_ms         = Column(Float)
    V_blob       = Column(LargeBinary)
    t_blob       = Column(LargeBinary)

    def set_trace(self, t: np.ndarray, V: np.ndarray, downsample: int = 10) -> None:
        self.downsample = downsample
        t_ds = t[::downsample].astype(np.float32)
        V_ds = V[::downsample].astype(np.float32)
        tb, vb = io.BytesIO(), io.BytesIO()
        np.save(tb, t_ds); np.save(vb, V_ds)
        self.t_blob = tb.getvalue(); self.V_blob = vb.getvalue()
        self.dt_ms  = float(t_ds[1] - t_ds[0]) if len(t_ds) > 1 else 0.0
        self.T_ms   = float(t_ds[-1]) if len(t_ds) > 0 else 0.0

    def get_voltage(self) -> np.ndarray:
        buf = io.BytesIO(self.V_blob); return np.load(buf).astype(np.float64)

    def get_time(self) -> np.ndarray:
        buf = io.BytesIO(self.t_blob); return np.load(buf).astype(np.float64)


class SynapseRecord(Base):
    """
    Synaptic simulation record: type, conductance trace, and parameters.
    """

    __tablename__ = "synapse_records"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    run            = relationship("SimulationRun", back_populates="synapse_records")
    synapse_type   = Column(String(16))      # AMPA, NMDA, GABA-A, GABA-B
    g_max_nS       = Column(Float)
    E_rev_mV       = Column(Float)
    peak_g_nS      = Column(Float)
    decay_tau_ms   = Column(Float)
    Mg_mM          = Column(Float)           # NMDA only
    conductance_blob = Column(LargeBinary)

    def set_conductance(self, g: np.ndarray) -> None:
        buf = io.BytesIO(); np.save(buf, g.astype(np.float32))
        self.conductance_blob = buf.getvalue()
        self.peak_g_nS = float(np.max(g))

    def get_conductance(self) -> np.ndarray:
        buf = io.BytesIO(self.conductance_blob)
        return np.load(buf).astype(np.float64)


class PlasticityRecord(Base):
    """
    Plasticity rule parameters, convergence statistics, and final weight distribution.
    """

    __tablename__ = "plasticity_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    rule_type       = Column(String(32))     # STDP, BCM, Oja, Triplet
    A_plus          = Column(Float)
    A_minus         = Column(Float)
    tau_plus_ms     = Column(Float)
    tau_minus_ms    = Column(Float)
    n_synapses      = Column(Integer)
    T_ms            = Column(Float)
    rate_pre_hz     = Column(Float)
    rate_post_hz    = Column(Float)
    w_init          = Column(Float)
    w_final_mean    = Column(Float)
    w_final_median  = Column(Float)
    w_final_std     = Column(Float)
    w_final_blob    = Column(LargeBinary)

    def set_weights(self, w: np.ndarray) -> None:
        buf = io.BytesIO(); np.save(buf, w.astype(np.float32))
        self.w_final_blob   = buf.getvalue()
        self.w_final_mean   = float(np.mean(w))
        self.w_final_median = float(np.median(w))
        self.w_final_std    = float(np.std(w))

    def get_weights(self) -> np.ndarray:
        buf = io.BytesIO(self.w_final_blob)
        return np.load(buf).astype(np.float64)


class AnalysisResult(Base):
    """
    Key-value store for derived scalar metrics from a simulation run.
    """

    __tablename__ = "analysis_results"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    run_id   = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    run      = relationship("SimulationRun", back_populates="analysis_results")
    metric   = Column(String(64), nullable=False)
    value    = Column(Float)
    unit     = Column(String(32))
    notes    = Column(Text)


def create_all_tables(engine) -> None:
    """Create all ORM tables in the target database if they do not exist."""
    Base.metadata.create_all(engine)
