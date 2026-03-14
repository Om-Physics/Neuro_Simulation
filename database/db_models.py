"""
db_models.py
============
SQLAlchemy ORM models for persisting simulation results.

Schema
------
  Experiment        — top-level run metadata
  NeuronRecord      — per-neuron parameter snapshot
  SimulationResult  — scalar summary metrics per simulation
  SpikeEvent        — individual spike times (high-resolution)
  VoltageTrace      — downsampled V(t) stored as JSON-encoded arrays
  SynapticWeight    — weight evolution snapshots for plasticity runs
  NetworkState      — population-level network metrics

All timestamps use UTC.  Float arrays are stored as JSON blobs for
portability (no dependency on PostgreSQL array types).

Usage
-----
    from database.db_models import init_db, Session
    engine, Session = init_db("sqlite:///compneuro.db")
    with Session() as sess:
        exp = Experiment(name="HH_fI_curve", description="...")
        sess.add(exp); sess.commit()
"""

from __future__ import annotations
import json
import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, ForeignKey, Text, JSON,
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship, sessionmaker, Session as _Session,
)
from sqlalchemy.pool import StaticPool


# ── ORM base ──────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── 1. Experiment ──────────────────────────────────────────────────────────
class Experiment(Base):
    """
    Top-level container for a simulation experiment.
    One experiment may contain multiple simulation runs.
    """
    __tablename__ = "experiments"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(128), nullable=False, index=True)
    description = Column(Text, default="")
    model_type  = Column(String(64), default="HH")        # HH / LIF / AdEx
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
    tags        = Column(String(256), default="")          # comma-separated

    # relationships
    simulations = relationship("SimulationResult", back_populates="experiment",
                               cascade="all, delete-orphan")
    neurons     = relationship("NeuronRecord",     back_populates="experiment",
                               cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} name='{self.name}'>"


# ── 2. NeuronRecord ────────────────────────────────────────────────────────
class NeuronRecord(Base):
    """
    Stores biophysical parameters for a single neuron instance.
    Enables parameter sweeps and reproducibility.
    """
    __tablename__ = "neuron_records"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    neuron_id     = Column(Integer, default=0)
    model_type    = Column(String(32), nullable=False)  # HH / LIF / AdEx / EIF
    is_excitatory = Column(Boolean, default=True)

    # HH parameters
    C_m   = Column(Float, default=1.0)
    g_Na  = Column(Float, default=120.0)
    g_K   = Column(Float, default=36.0)
    g_L   = Column(Float, default=0.3)
    E_Na  = Column(Float, default=55.0)
    E_K   = Column(Float, default=-77.0)
    E_L   = Column(Float, default=-54.387)

    # LIF / AdEx extra parameters
    V_thresh = Column(Float, default=-55.0)
    V_reset  = Column(Float, default=-70.0)
    t_ref    = Column(Float, default=2.0)
    delta_T  = Column(Float, default=2.0)
    a_adapt  = Column(Float, default=4.0)
    b_adapt  = Column(Float, default=80.5)
    tau_w    = Column(Float, default=144.0)
    preset   = Column(String(16), default="")

    experiment = relationship("Experiment", back_populates="neurons")

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name)
                for c in self.__table__.columns}


# ── 3. SimulationResult ────────────────────────────────────────────────────
class SimulationResult(Base):
    """
    Scalar summary metrics for one simulation run.
    Foreign-keyed to an Experiment.
    """
    __tablename__ = "simulation_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    run_index     = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)

    # Stimulus
    I_ext_mean    = Column(Float, default=0.0)   # µA/cm² or pA
    I_ext_std     = Column(Float, default=0.0)
    duration_ms   = Column(Float, default=0.0)
    dt_ms         = Column(Float, default=0.025)

    # Output metrics
    n_spikes      = Column(Integer, default=0)
    firing_rate_hz= Column(Float,   default=0.0)
    cv_isi        = Column(Float,   default=float("nan"))
    mean_V_mV     = Column(Float,   default=0.0)
    std_V_mV      = Column(Float,   default=0.0)
    V_rest_mV     = Column(Float,   default=-65.0)
    first_spike_ms= Column(Float,   default=float("nan"))
    latency_ms    = Column(Float,   default=float("nan"))

    # Relationships
    experiment    = relationship("Experiment",    back_populates="simulations")
    spikes        = relationship("SpikeEvent",    back_populates="simulation",
                                 cascade="all, delete-orphan")
    voltage_trace = relationship("VoltageTrace",  back_populates="simulation",
                                 uselist=False,   cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (f"<SimResult id={self.id} "
                f"I={self.I_ext_mean:.2f} "
                f"fr={self.firing_rate_hz:.1f} Hz>")


# ── 4. SpikeEvent ──────────────────────────────────────────────────────────
class SpikeEvent(Base):
    """Individual spike times — high-resolution record."""
    __tablename__ = "spike_events"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulation_results.id"),
                           nullable=False)
    neuron_id     = Column(Integer, default=0)
    spike_time_ms = Column(Float,   nullable=False)

    simulation    = relationship("SimulationResult", back_populates="spikes")

    def __repr__(self) -> str:
        return f"<Spike t={self.spike_time_ms:.3f} ms>"


# ── 5. VoltageTrace ────────────────────────────────────────────────────────
class VoltageTrace(Base):
    """
    Downsampled membrane voltage stored as JSON arrays.
    Compression: store every `downsample` timestep.
    """
    __tablename__ = "voltage_traces"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulation_results.id"),
                           nullable=False, unique=True)
    neuron_id     = Column(Integer, default=0)
    downsample    = Column(Integer, default=10)   # store every Nth point
    t_json        = Column(Text,    nullable=False)  # JSON float list
    V_json        = Column(Text,    nullable=False)

    simulation    = relationship("SimulationResult",
                                 back_populates="voltage_trace")

    def set_arrays(self, t, V, downsample: int = 10) -> None:
        self.downsample = downsample
        self.t_json = json.dumps(t[::downsample].tolist())
        self.V_json = json.dumps(V[::downsample].tolist())

    def get_arrays(self):
        import numpy as np
        return (np.array(json.loads(self.t_json)),
                np.array(json.loads(self.V_json)))


# ── 6. SynapticWeight ──────────────────────────────────────────────────────
class SynapticWeight(Base):
    """Snapshot of synaptic weight during a plasticity simulation."""
    __tablename__ = "synaptic_weights"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    pre_id        = Column(Integer, default=0)
    post_id       = Column(Integer, default=0)
    synapse_type  = Column(String(32), default="AMPA")
    plasticity    = Column(String(32), default="STDP")
    time_ms       = Column(Float,   nullable=False)
    weight        = Column(Float,   nullable=False)


# ── 7. NetworkState ────────────────────────────────────────────────────────
class NetworkState(Base):
    """Population-level metrics from a network simulation."""
    __tablename__ = "network_states"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id   = Column(Integer, ForeignKey("experiments.id"))
    N_E             = Column(Integer, default=0)
    N_I             = Column(Integer, default=0)
    p_conn          = Column(Float,   default=0.1)
    duration_ms     = Column(Float,   default=0.0)
    mean_rate_E_hz  = Column(Float,   default=0.0)
    mean_rate_I_hz  = Column(Float,   default=0.0)
    synchrony_index = Column(Float,   default=0.0)
    irregularity    = Column(Float,   default=0.0)   # mean CV ISI
    network_state   = Column(String(16), default="")  # AI / SR / SIf / SIs
    spikes_json     = Column(Text,    default="[]")   # list of [nid, t] pairs

    def set_spikes(self, spikes_list: list) -> None:
        """Store flat spike list [[nid, t], ...] as JSON."""
        self.spikes_json = json.dumps(spikes_list)

    def get_spikes(self) -> list:
        return json.loads(self.spikes_json)


# ── DB initialisation helper ──────────────────────────────────────────────
def init_db(url: str = "sqlite:///compneuro.db"):
    """
    Create database engine and session factory.

    Parameters
    ----------
    url : SQLAlchemy database URL
          sqlite:///compneuro.db    (file-based, default)
          sqlite:///:memory:        (in-memory, for tests)

    Returns
    -------
    (engine, sessionmaker)
    """
    connect_args = {}
    pool_kwargs  = {}
    if url == "sqlite:///:memory:":
        connect_args = {"check_same_thread": False}
        pool_kwargs  = {"poolclass": StaticPool}

    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=False,
        **pool_kwargs,
    )
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, SessionFactory


# ── Convenience data-access helpers ───────────────────────────────────────
class DBManager:
    """Thin CRUD wrapper for the compneuro database."""

    def __init__(self, url: str = "sqlite:///compneuro.db"):
        self.engine, self.Session = init_db(url)

    # ── Experiment ────────────────────────────────────────────────────────
    def create_experiment(self, name: str, **kwargs) -> Experiment:
        with self.Session() as s:
            exp = Experiment(name=name, **kwargs)
            s.add(exp); s.commit(); s.refresh(exp)
            return exp

    def get_experiment(self, exp_id: int) -> Optional[Experiment]:
        with self.Session() as s:
            return s.get(Experiment, exp_id)

    def list_experiments(self) -> list[Experiment]:
        with self.Session() as s:
            return s.query(Experiment).order_by(Experiment.created_at).all()

    # ── SimulationResult ──────────────────────────────────────────────────
    def save_simulation(
        self,
        exp_id:    int,
        result:    dict,
        I_ext_val: float = 0.0,
        run_index: int   = 0,
        neuron_id: int   = 0,
    ) -> SimulationResult:
        """
        Persist a simulation result dict (from neuron.simulate()) to DB.
        """
        import numpy as np
        spikes = result.get("spikes", np.array([]))
        t      = result["t"]
        V      = result["V"]

        sr = SimulationResult(
            experiment_id  = exp_id,
            run_index      = run_index,
            I_ext_mean     = I_ext_val,
            duration_ms    = float(t[-1] - t[0]),
            dt_ms          = float(t[1] - t[0]) if len(t) > 1 else 0.0,
            n_spikes       = len(spikes),
            firing_rate_hz = float(result.get("firing_rate_hz", 0.0)),
            cv_isi         = float(result.get("cv_isi", float("nan"))),
            mean_V_mV      = float(np.mean(V)),
            std_V_mV       = float(np.std(V)),
            first_spike_ms = float(spikes[0]) if len(spikes) > 0 else float("nan"),
        )

        vt = VoltageTrace(neuron_id=neuron_id)
        vt.set_arrays(t, V, downsample=10)
        sr.voltage_trace = vt

        for t_sp in spikes:
            sr.spikes.append(SpikeEvent(neuron_id=neuron_id,
                                        spike_time_ms=float(t_sp)))
        with self.Session() as s:
            s.add(sr); s.commit(); s.refresh(sr)
            return sr

    def get_fI_data(self, exp_id: int) -> dict:
        """Fetch I and firing_rate_hz arrays for an f-I curve experiment."""
        import numpy as np
        with self.Session() as s:
            rows = (s.query(SimulationResult)
                    .filter_by(experiment_id=exp_id)
                    .order_by(SimulationResult.I_ext_mean)
                    .all())
        if not rows:
            return {"I": np.array([]), "f_hz": np.array([])}
        return {
            "I":    np.array([r.I_ext_mean     for r in rows]),
            "f_hz": np.array([r.firing_rate_hz for r in rows]),
        }

    def save_network_state(self, exp_id: int, result: dict,
                           network_params: dict) -> NetworkState:
        import numpy as np
        spikes_flat = []
        for nid, spike_list in enumerate(result["spikes"]):
            for t_sp in spike_list:
                spikes_flat.append([nid, float(t_sp)])

        isi_cvs = []
        for spike_list in result["spikes"]:
            if len(spike_list) > 2:
                isi = np.diff(spike_list)
                isi_cvs.append(np.std(isi) / np.mean(isi))

        rates_E = []
        for nid in range(network_params.get("N_E", 400)):
            sl = result["spikes"][nid]
            dur = result["t"][-1]
            rates_E.append(len(sl) / (dur * 1e-3) if dur > 0 else 0.0)

        ns = NetworkState(
            experiment_id  = exp_id,
            N_E            = network_params.get("N_E", 0),
            N_I            = network_params.get("N_I", 0),
            p_conn         = network_params.get("p_conn", 0.1),
            duration_ms    = float(result["t"][-1]),
            mean_rate_E_hz = float(np.mean(rates_E)) if rates_E else 0.0,
            irregularity   = float(np.mean(isi_cvs)) if isi_cvs else 0.0,
        )
        ns.set_spikes(spikes_flat)

        with self.Session() as s:
            s.add(ns); s.commit(); s.refresh(ns)
            return ns
