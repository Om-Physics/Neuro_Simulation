"""
database/models.py
==================
SQLAlchemy ORM models for persisting simulation results, neuron parameters,
spike trains, and analysis outputs.

Schema
------
  SimulationRun       — top-level experiment record
  NeuronParameters    — biophysical parameters per neuron type
  SpikeTrainRecord    — compressed spike timing data
  VoltageTrace        — sampled membrane voltage (downsampled for storage)
  SynapseRecord       — synapse parameters and weight history
  AnalysisResult      — derived metrics (firing rate, CV-ISI, etc.)
  NetworkState        — network-level summary statistics
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, ForeignKey, Text, JSON,
)
from sqlalchemy.orm import declarative_base, relationship, Session
import json
import numpy as np

Base = declarative_base()


# ── Core experiment record ────────────────────────────────────────────────
class SimulationRun(Base):
    """One complete simulation experiment."""
    __tablename__ = "simulation_runs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(120), nullable=False)
    description = Column(Text,        nullable=True)
    model_type  = Column(String(60),  nullable=False)   # HH, LIF, AdEx, Network
    created_at  = Column(DateTime,    default=datetime.utcnow)
    duration_ms = Column(Float,       nullable=False)
    dt_ms       = Column(Float,       nullable=False)
    seed        = Column(Integer,     default=0)
    tags        = Column(String(200), nullable=True)

    # relationships
    neuron_params  = relationship("NeuronParameters",  back_populates="run",
                                  cascade="all, delete-orphan")
    spike_trains   = relationship("SpikeTrainRecord",  back_populates="run",
                                  cascade="all, delete-orphan")
    voltage_traces = relationship("VoltageTrace",      back_populates="run",
                                  cascade="all, delete-orphan")
    analysis       = relationship("AnalysisResult",    back_populates="run",
                                  cascade="all, delete-orphan")
    network_states = relationship("NetworkState",      back_populates="run",
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return (f"<SimulationRun id={self.id} name='{self.name}' "
                f"model='{self.model_type}' t={self.duration_ms}ms>")


# ── Neuron parameter record ───────────────────────────────────────────────
class NeuronParameters(Base):
    """Biophysical parameters for a neuron in a simulation run."""
    __tablename__ = "neuron_parameters"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    neuron_id   = Column(Integer, nullable=False)
    model_class = Column(String(60))
    population  = Column(String(20), default="E")   # E / I / single

    # stored as JSON for flexibility across model types
    params_json = Column(Text, nullable=False, default="{}")

    run = relationship("SimulationRun", back_populates="neuron_params")

    @property
    def params(self) -> dict:
        return json.loads(self.params_json)

    @params.setter
    def params(self, d: dict):
        self.params_json = json.dumps(d)


# ── Spike train ───────────────────────────────────────────────────────────
class SpikeTrainRecord(Base):
    """Compressed spike timing data for one neuron."""
    __tablename__ = "spike_trains"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    neuron_id   = Column(Integer, nullable=False)
    population  = Column(String(20), default="E")

    # spike times stored as JSON-encoded float list (compact)
    spike_times_json = Column(Text, nullable=False, default="[]")

    # derived (stored for fast query)
    n_spikes        = Column(Integer, default=0)
    mean_rate_hz    = Column(Float,   default=0.0)
    cv_isi          = Column(Float,   nullable=True)
    mean_isi_ms     = Column(Float,   nullable=True)

    run = relationship("SimulationRun", back_populates="spike_trains")

    @property
    def spike_times(self) -> np.ndarray:
        return np.array(json.loads(self.spike_times_json))

    @spike_times.setter
    def spike_times(self, arr):
        times = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)
        self.spike_times_json = json.dumps([round(t, 4) for t in times])
        self.n_spikes = len(times)

    def compute_stats(self, duration_ms: float) -> None:
        """Populate derived columns from spike_times."""
        t = self.spike_times
        self.n_spikes     = len(t)
        self.mean_rate_hz = (len(t) / (duration_ms * 1e-3)) if duration_ms > 0 else 0
        if len(t) > 1:
            isi = np.diff(t)
            self.mean_isi_ms = float(np.mean(isi))
            self.cv_isi      = float(np.std(isi) / np.mean(isi))
        else:
            self.mean_isi_ms = None
            self.cv_isi      = None


# ── Voltage trace ─────────────────────────────────────────────────────────
class VoltageTrace(Base):
    """Membrane voltage trajectory (downsampled for storage efficiency)."""
    __tablename__ = "voltage_traces"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    neuron_id    = Column(Integer, nullable=False)
    downsample   = Column(Integer, default=10)   # every Nth sample stored

    # V(t) stored as JSON float list
    voltage_json = Column(Text, nullable=False, default="[]")
    t_start_ms   = Column(Float, default=0.0)
    t_end_ms     = Column(Float, default=0.0)
    dt_stored_ms = Column(Float, default=0.0)

    # summary stats
    V_mean  = Column(Float, nullable=True)
    V_std   = Column(Float, nullable=True)
    V_min   = Column(Float, nullable=True)
    V_max   = Column(Float, nullable=True)

    run = relationship("SimulationRun", back_populates="voltage_traces")

    @property
    def voltage(self) -> np.ndarray:
        return np.array(json.loads(self.voltage_json))

    @voltage.setter
    def voltage(self, arr: np.ndarray):
        v = arr[::self.downsample]
        self.voltage_json = json.dumps([round(float(x), 4) for x in v])
        self.V_mean = float(np.mean(v))
        self.V_std  = float(np.std(v))
        self.V_min  = float(np.min(v))
        self.V_max  = float(np.max(v))

    @property
    def time(self) -> np.ndarray:
        n = len(self.voltage)
        return self.t_start_ms + np.arange(n) * self.dt_stored_ms


# ── Synapse record ────────────────────────────────────────────────────────
class SynapseRecord(Base):
    """Parameters and weight evolution for a synapse."""
    __tablename__ = "synapse_records"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    pre_id       = Column(Integer, nullable=False)
    post_id      = Column(Integer, nullable=False)
    synapse_type = Column(String(30))   # AMPA, NMDA, GABA_A, GABA_B
    g_max_nS     = Column(Float)
    E_rev_mV     = Column(Float)
    weight_init  = Column(Float, default=1.0)
    weight_final = Column(Float, nullable=True)
    weight_history_json = Column(Text, nullable=True)

    run = relationship("SimulationRun")

    @property
    def weight_history(self) -> np.ndarray:
        if self.weight_history_json:
            return np.array(json.loads(self.weight_history_json))
        return np.array([])


# ── Analysis result ───────────────────────────────────────────────────────
class AnalysisResult(Base):
    """Derived analysis metrics for a simulation run."""
    __tablename__ = "analysis_results"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    analysis_type = Column(String(60))   # e.g. 'fI_curve', 'phase_plane', 'PSD'
    neuron_id    = Column(Integer, nullable=True)

    # flexible key-value metric storage
    metric_json  = Column(Text, nullable=False, default="{}")

    run = relationship("SimulationRun", back_populates="analysis")

    @property
    def metrics(self) -> dict:
        return json.loads(self.metric_json)

    @metrics.setter
    def metrics(self, d: dict):
        # convert numpy arrays to lists for JSON serialisation
        serialisable = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                serialisable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serialisable[k] = float(v)
            else:
                serialisable[k] = v
        self.metric_json = json.dumps(serialisable)


# ── Network state ─────────────────────────────────────────────────────────
class NetworkState(Base):
    """Time-series of network-level population statistics."""
    __tablename__ = "network_states"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)

    # population firing rates (mean ± std over neurons)
    pop_E_rate_mean_hz = Column(Float)
    pop_E_rate_std_hz  = Column(Float)
    pop_I_rate_mean_hz = Column(Float)
    pop_I_rate_std_hz  = Column(Float)

    # synchrony measure (0 = async, 1 = fully sync)
    synchrony_chi  = Column(Float, nullable=True)
    # irregularity: mean CV-ISI across population
    irregularity   = Column(Float, nullable=True)

    # E/I balance
    EI_ratio       = Column(Float, nullable=True)
    network_state  = Column(String(20), nullable=True)   # SR/AI/SIf/SIs

    run = relationship("SimulationRun", back_populates="network_states")


# ── Database engine factory ───────────────────────────────────────────────
def create_db(path: str = "compneuro.db") -> tuple:
    """
    Create (or connect to) a SQLite database and return (engine, Session).

    Parameters
    ----------
    path : str — file path for SQLite DB (use ':memory:' for in-memory)

    Returns
    -------
    (engine, SessionFactory)
    """
    engine = create_engine(f"sqlite:///{path}", echo=False)
    Base.metadata.create_all(engine)
    return engine, Session


def get_session(engine) -> Session:
    return Session(engine)
