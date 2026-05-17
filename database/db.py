"""
db.py
=====
Database engine factory, session management, and high-level repository
class for persisting and querying all Neuro_Simulation results.

The repository pattern isolates all SQLAlchemy interaction behind a clean
Python interface, so simulation code never calls ORM operations directly.

Usage example:

    from database.db import build_engine, SimulationRepository
    from database.models import create_all_tables

    engine = build_engine("neuro_sim.db")
    create_all_tables(engine)
    repo   = SimulationRepository(engine)

    # Persist a completed HH simulation
    run_id = repo.save_hh_run(hh_data, params)

    # Query results
    summary = repo.run_summary(run_id)
    df      = repo.all_runs_dataframe()

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import datetime
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from database.models import (
    Base, SimulationRun, NeuronParameters, SpikeTrainRecord,
    VoltageTrace, SynapseRecord, PlasticityRecord, AnalysisResult,
    create_all_tables,
)


def build_engine(db_path: str = "neuro_sim.db", echo: bool = False) -> sa.Engine:
    """
    Create a SQLAlchemy engine connected to a SQLite database file.

    WAL (Write-Ahead Logging) mode is enabled via a connection event
    to allow concurrent reads during long simulation runs.

    Parameters
    ----------
    db_path : Path to the SQLite file. Default 'neuro_sim.db'.
    echo    : If True, SQLAlchemy echoes all SQL to stdout.

    Returns
    -------
    sqlalchemy.Engine
    """
    url = f"sqlite:///{db_path}"
    engine = sa.create_engine(url, echo=echo)

    @sa.event.listens_for(engine, "connect")
    def set_wal_mode(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    create_all_tables(engine)
    return engine


@contextmanager
def session_scope(engine: sa.Engine) -> Iterator[Session]:
    """
    Provide a transactional database session as a context manager.

    Commits on clean exit; rolls back and re-raises on any exception.

    Usage:
        with session_scope(engine) as session:
            session.add(some_orm_object)
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class SimulationRepository:
    """
    High-level interface for persisting and querying simulation results.

    All public methods accept plain Python / NumPy objects and handle
    the conversion to ORM records internally.
    """

    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine

    def save_hh_run(self, data: dict, params: dict | None = None) -> int:
        """
        Persist a completed Hodgkin-Huxley simulation.

        Parameters
        ----------
        data   : Output dict from HodgkinHuxley.simulate_detailed().
        params : Optional dict of model parameters to store.

        Returns
        -------
        int : The new SimulationRun primary key.
        """
        from analysis.spike_analysis import isi_statistics, fano_factor

        spikes = data["spikes"]
        T      = float(data["t"][-1])
        isi    = isi_statistics(spikes)
        rate   = float(len(spikes) / (T * 1e-3)) if T > 0 else 0.0
        ff     = fano_factor(spikes, T) if len(spikes) > 1 else float("nan")

        with session_scope(self.engine) as s:
            run = SimulationRun(
                model_type="HodgkinHuxley",
                T_ms=T,
                dt_ms=params.get("dt", 0.025) if params else 0.025,
                I_ext=params.get("I_ext", 10.0) if params else 10.0,
                n_spikes=len(spikes),
                mean_rate_hz=rate,
                cv_isi=isi["cv"],
                fano_factor=ff,
            )
            if params:
                run.set_params(params)
            s.add(run)
            s.flush()

            vt = VoltageTrace()
            vt.run_id = run.id
            vt.set_trace(data["t"], data["V"], downsample=4)
            s.add(vt)

            st = SpikeTrainRecord()
            st.run_id = run.id
            st.set_spike_times(spikes)
            st.mean_rate_hz = rate
            st.cv_isi = isi["cv"]
            s.add(st)

            for metric, value, unit in [
                ("mean_rate_hz", rate, "Hz"),
                ("cv_isi", isi["cv"], ""),
                ("fano_factor", ff, ""),
                ("n_spikes", float(len(spikes)), "count"),
            ]:
                s.add(AnalysisResult(run_id=run.id, metric=metric,
                                     value=value, unit=unit))

            run_id = run.id

        return run_id

    def save_adex_run(self, rec, preset: str, I_ext: float) -> int:
        """
        Persist a completed AdEx simulation.

        Parameters
        ----------
        rec    : SpikeRecord returned by AdaptiveExponentialIF.simulate().
        preset : Preset name, e.g. 'RS'.
        I_ext  : Injected current in pA.

        Returns
        -------
        int : The new SimulationRun primary key.
        """
        from analysis.spike_analysis import isi_statistics, fano_factor

        spikes = rec.times
        T      = float(rec.time_axis[-1])
        isi    = isi_statistics(spikes)
        rate   = float(len(spikes) / (T * 1e-3)) if T > 0 else 0.0
        ff     = fano_factor(spikes, T) if len(spikes) > 1 else float("nan")

        with session_scope(self.engine) as s:
            run = SimulationRun(
                model_type=f"AdEx-{preset}",
                T_ms=T,
                dt_ms=rec.metadata.get("dt", 0.1),
                I_ext=I_ext,
                n_spikes=len(spikes),
                mean_rate_hz=rate,
                cv_isi=isi["cv"],
                fano_factor=ff,
            )
            run.set_params({"preset": preset, "I_ext": I_ext,
                            "a": rec.metadata.get("a"),
                            "b": rec.metadata.get("b"),
                            "tau_w": rec.metadata.get("tau_w")})
            s.add(run)
            s.flush()

            vt = VoltageTrace()
            vt.run_id = run.id
            vt.set_trace(rec.time_axis, rec.voltages, downsample=5)
            s.add(vt)

            st = SpikeTrainRecord()
            st.run_id = run.id
            st.set_spike_times(spikes)
            st.mean_rate_hz = rate
            st.cv_isi = isi["cv"]
            s.add(st)

            run_id = run.id

        return run_id

    def save_stdp_run(self, result: dict, rule_params: dict) -> int:
        """
        Persist a completed STDP simulation.

        Parameters
        ----------
        result     : Output dict from STDPRule.run().
        rule_params: Dict of STDP rule hyperparameters.

        Returns
        -------
        int : New PlasticityRecord primary key.
        """
        with session_scope(self.engine) as s:
            rec = PlasticityRecord(
                rule_type="STDP",
                A_plus=rule_params.get("A_plus", 0.010),
                A_minus=rule_params.get("A_minus", 0.0105),
                tau_plus_ms=rule_params.get("tau_plus", 20.0),
                tau_minus_ms=rule_params.get("tau_minus", 20.0),
                n_synapses=result["w_final"].size,
                T_ms=float(result["t"][-1]),
                rate_pre_hz=rule_params.get("rate_pre", 20.0),
                rate_post_hz=rule_params.get("rate_post", 20.0),
                w_init=rule_params.get("w_init", 0.5),
            )
            rec.set_weights(result["w_final"])
            s.add(rec)
            s.flush()
            rec_id = rec.id

        return rec_id

    def save_network_run(self, result: dict, net_params: dict) -> int:
        """
        Persist a completed E/I network simulation.

        Parameters
        ----------
        result    : Output dict from SpikingNetwork.run().
        net_params: Dict of network parameters.

        Returns
        -------
        int : New SimulationRun primary key.
        """
        T    = float(result["T"])
        rate = float(result["mean_rate_E"])

        with session_scope(self.engine) as s:
            run = SimulationRun(
                model_type="EI_Network",
                T_ms=T,
                dt_ms=net_params.get("dt", 0.2),
                I_ext=net_params.get("I_dc", 220.0),
                mean_rate_hz=rate,
            )
            run.set_params(net_params)
            s.add(run)
            s.flush()

            for metric, value, unit in [
                ("mean_rate_E", result["mean_rate_E"], "Hz"),
                ("mean_rate_I", result["mean_rate_I"], "Hz"),
                ("N_E", float(result["N_E"]), "count"),
                ("N_I", float(result["N_I"]), "count"),
            ]:
                s.add(AnalysisResult(run_id=run.id, metric=metric,
                                     value=value, unit=unit))

            run_id = run.id

        return run_id

    def run_summary(self, run_id: int) -> dict:
        """
        Return a summary dictionary for a given SimulationRun.

        Returns
        -------
        dict with keys: id, model_type, T_ms, I_ext, n_spikes,
        mean_rate_hz, cv_isi, fano_factor, created_at, params.
        """
        with Session(self.engine) as s:
            run = s.get(SimulationRun, run_id)
            if run is None:
                raise KeyError(f"SimulationRun id={run_id} not found.")
            return {
                "id":           run.id,
                "model_type":   run.model_type,
                "T_ms":         run.T_ms,
                "I_ext":        run.I_ext,
                "n_spikes":     run.n_spikes,
                "mean_rate_hz": run.mean_rate_hz,
                "cv_isi":       run.cv_isi,
                "fano_factor":  run.fano_factor,
                "created_at":   str(run.created_at),
                "params":       run.get_params(),
            }

    def list_runs(self, model_type: str | None = None) -> list[dict]:
        """
        Return a list of summary dicts for all (or filtered) simulation runs.

        Parameters
        ----------
        model_type : If specified, filter to runs of this model type.
        """
        with Session(self.engine) as s:
            q = s.query(SimulationRun)
            if model_type:
                q = q.filter(SimulationRun.model_type == model_type)
            return [
                {"id": r.id, "model_type": r.model_type,
                 "mean_rate_hz": r.mean_rate_hz, "cv_isi": r.cv_isi,
                 "created_at": str(r.created_at)}
                for r in q.all()
            ]

    def delete_run(self, run_id: int) -> None:
        """Delete a simulation run and all related records."""
        with session_scope(self.engine) as s:
            run = s.get(SimulationRun, run_id)
            if run:
                s.delete(run)
