"""
db.py
=====
Database engine, session factory, and context managers.

Provides thread-safe SQLite access with WAL journaling for concurrent
read/write during long simulation runs. Also ships a lightweight
repository layer so simulation code never touches SQL directly.
"""

import os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from .models import (
    Base, SimulationRun, NeuronParameters, SpikeTrainRecord,
    VoltageTrace, SynapseRecord, AnalysisResult, NetworkState,
)

# ── Engine factory ────────────────────────────────────────────────────────
_DEFAULT_DB = Path(__file__).parent.parent / "compneuro.db"

def build_engine(db_path: str | Path = _DEFAULT_DB, echo: bool = False):
    """
    Create a SQLAlchemy engine with WAL journaling enabled.

    Parameters
    ----------
    db_path : path to SQLite file (use ':memory:' for tests)
    echo    : if True, log all SQL statements

    Returns
    -------
    engine : SQLAlchemy Engine
    """
    url = f"sqlite:///{db_path}" if str(db_path) != ":memory:" else "sqlite:///:memory:"
    engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False})

    # Enable WAL mode + foreign keys at connection open
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine):
    """
    Transactional context manager — auto-commits on success, rolls back on error.

    Usage:
        with session_scope(engine) as session:
            session.add(obj)
    """
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Repository: SimulationRun ─────────────────────────────────────────────
class SimulationRepository:
    """
    High-level data access for simulation results.
    All database I/O goes through here — no raw SQL in simulation code.
    """

    def __init__(self, engine):
        self.engine = engine

    # ── Create ────────────────────────────────────────────────────────────
    def save_run(
        self,
        name:        str,
        model_type:  str,
        duration_ms: float,
        dt_ms:       float,
        description: str = "",
        seed:        int = 0,
        tags:        str = "",
    ) -> SimulationRun:
        """Persist a new simulation run and return the ORM object."""
        with session_scope(self.engine) as sess:
            run = SimulationRun(
                name=name,
                description=description,
                model_type=model_type,
                duration_ms=duration_ms,
                dt_ms=dt_ms,
                seed=seed,
                tags=tags,
            )
            sess.add(run)
            sess.flush()
            run_id = run.id
        return self.get_run(run_id)

    def save_spike_train(
        self,
        run_id:     int,
        neuron_id:  int,
        spike_times: np.ndarray,
        duration_ms: float,
        population: str = "E",
    ) -> SpikeTrainRecord:
        """Store spike times for one neuron."""
        with session_scope(self.engine) as sess:
            st = SpikeTrainRecord(
                run_id=run_id,
                neuron_id=neuron_id,
                population=population,
            )
            st.spike_times = spike_times
            st.compute_stats(duration_ms)
            sess.add(st)
            sess.flush()
            rec_id = st.id
        return self.get_spike_train(rec_id)

    def save_voltage_trace(
        self,
        run_id:     int,
        neuron_id:  int,
        V:          np.ndarray,
        t_start_ms: float,
        t_end_ms:   float,
        dt_ms:      float,
        downsample: int = 10,
    ) -> VoltageTrace:
        """Store (downsampled) voltage trace for one neuron."""
        with session_scope(self.engine) as sess:
            vt = VoltageTrace(
                run_id=run_id,
                neuron_id=neuron_id,
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
                dt_stored_ms=dt_ms * downsample,
                downsample=downsample,
            )
            vt.voltage = V
            sess.add(vt)
            sess.flush()
            vt_id = vt.id
        return self.get_voltage_trace(vt_id)

    def save_neuron_params(
        self,
        run_id:     int,
        neuron_id:  int,
        model_class: str,
        params:     dict,
        population: str = "E",
    ) -> NeuronParameters:
        with session_scope(self.engine) as sess:
            np_rec = NeuronParameters(
                run_id=run_id,
                neuron_id=neuron_id,
                model_class=model_class,
                population=population,
            )
            np_rec.params = params
            sess.add(np_rec)
            sess.flush()
            np_id = np_rec.id
        with session_scope(self.engine) as sess:
            return sess.get(NeuronParameters, np_id)

    def save_analysis(
        self,
        run_id:        int,
        analysis_type: str,
        metrics:       dict,
        neuron_id:     int | None = None,
    ) -> AnalysisResult:
        with session_scope(self.engine) as sess:
            ar = AnalysisResult(
                run_id=run_id,
                analysis_type=analysis_type,
                neuron_id=neuron_id,
            )
            ar.metrics = metrics
            sess.add(ar)
            sess.flush()
            ar_id = ar.id
        return self.get_analysis(ar_id)

    def save_network_state(
        self,
        run_id: int,
        **kwargs,
    ) -> NetworkState:
        with session_scope(self.engine) as sess:
            ns = NetworkState(run_id=run_id, **kwargs)
            sess.add(ns)
            sess.flush()
            ns_id = ns.id
        with session_scope(self.engine) as sess:
            return sess.get(NetworkState, ns_id)

    # ── Read ──────────────────────────────────────────────────────────────
    def get_run(self, run_id: int) -> SimulationRun | None:
        with session_scope(self.engine) as sess:
            return sess.get(SimulationRun, run_id)

    def get_spike_train(self, rec_id: int) -> SpikeTrainRecord | None:
        with session_scope(self.engine) as sess:
            return sess.get(SpikeTrainRecord, rec_id)

    def get_voltage_trace(self, vt_id: int) -> VoltageTrace | None:
        with session_scope(self.engine) as sess:
            return sess.get(VoltageTrace, vt_id)

    def get_analysis(self, ar_id: int) -> AnalysisResult | None:
        with session_scope(self.engine) as sess:
            return sess.get(AnalysisResult, ar_id)

    def list_runs(self, model_type: str = None) -> list[SimulationRun]:
        with session_scope(self.engine) as sess:
            q = sess.query(SimulationRun)
            if model_type:
                q = q.filter(SimulationRun.model_type == model_type)
            return q.order_by(SimulationRun.created_at.desc()).all()

    def get_all_spike_trains(self, run_id: int) -> list[SpikeTrainRecord]:
        with session_scope(self.engine) as sess:
            return (sess.query(SpikeTrainRecord)
                    .filter(SpikeTrainRecord.run_id == run_id)
                    .order_by(SpikeTrainRecord.neuron_id).all())

    def get_analysis_by_type(self, run_id: int, analysis_type: str) -> list:
        with session_scope(self.engine) as sess:
            return (sess.query(AnalysisResult)
                    .filter(AnalysisResult.run_id == run_id,
                            AnalysisResult.analysis_type == analysis_type)
                    .all())

    # ── Convenience: full run summary ─────────────────────────────────────
    def run_summary(self, run_id: int) -> dict:
        """Return a summary dict for a run — useful for reports."""
        run = self.get_run(run_id)
        if run is None:
            return {}
        spike_trains = self.get_all_spike_trains(run_id)
        total_spikes = sum(st.n_spikes for st in spike_trains)
        rates = [st.mean_rate_hz for st in spike_trains if st.mean_rate_hz]
        cvs   = [st.cv_isi for st in spike_trains if st.cv_isi is not None]
        return {
            "run_id":     run_id,
            "name":       run.name,
            "model":      run.model_type,
            "duration_ms": run.duration_ms,
            "n_neurons":  len(spike_trains),
            "total_spikes": total_spikes,
            "mean_rate_hz": float(np.mean(rates)) if rates else 0.0,
            "mean_cv_isi":  float(np.mean(cvs))   if cvs   else None,
        }
