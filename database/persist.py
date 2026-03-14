"""
database/persist.py
====================
Persist all simulation results to the authorised SQLite database
and demonstrate query capabilities.

Run after all simulations complete:
    python database/persist.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import warnings; warnings.filterwarnings("ignore")

from database.models import (
    create_db, get_session,
    SimulationRun, NeuronParameters, SpikeTrainRecord,
    VoltageTrace, AnalysisResult, NetworkState,
)
from neurons.hodgkin_huxley import HodgkinHuxley
from neurons.integrate_fire import LeakyIntegrateAndFire, AdaptiveExponentialIF
from analysis.spike_analysis import spike_train_statistics

print("=" * 56)
print("  CompNeuro Database — Persist & Query Demo")
print("=" * 56)

engine, SessionClass = create_db("compneuro.db")
session = get_session(engine)

# ── 1. HH simulation ─────────────────────────────────────────────────────
print("\n[1] Persisting Hodgkin-Huxley simulation …")
hh = HodgkinHuxley()
n = int(100/0.025)
I_arr = np.zeros(n); I_arr[int(5/0.025):] = 10.0
res = hh.simulate_detailed(I_arr, dt=0.025)

run = SimulationRun(
    name="HH_10uA_100ms",
    description="HH step current 10 µA/cm², 100 ms",
    model_type="HodgkinHuxley",
    duration_ms=100.0, dt_ms=0.025, seed=0,
    tags="HH,action_potential,squid_axon",
)
session.add(run); session.flush()

np_rec = NeuronParameters(run_id=run.id, neuron_id=0, model_class="HodgkinHuxley")
np_rec.params = {
    "C_m":  hh.C_m,  "g_Na": hh.g_Na, "g_K": hh.g_K,  "g_L": hh.g_L,
    "E_Na": hh.E_Na, "E_K":  hh.E_K,  "E_L": hh.E_L,
    "I_ext_uA_cm2": 10.0,
}
session.add(np_rec)

sp = SpikeTrainRecord(run_id=run.id, neuron_id=0, population="E")
sp.spike_times = res["spikes"]
sp.compute_stats(100.0)
session.add(sp)

vt = VoltageTrace(run_id=run.id, neuron_id=0, downsample=4,
                  t_start_ms=0, t_end_ms=100.0, dt_stored_ms=0.1)
vt.voltage = res["V"]
session.add(vt)

stats = spike_train_statistics(res["spikes"], 100.0)
ar = AnalysisResult(run_id=run.id, analysis_type="spike_stats", neuron_id=0)
ar.metrics = stats
session.add(ar)
session.commit()
print(f"   ✓ Run ID={run.id}: {sp.n_spikes} spikes, {sp.mean_rate_hz:.1f} Hz, "
      f"CV={sp.cv_isi:.3f}" if sp.cv_isi else f"   ✓ Run ID={run.id}: {sp.n_spikes} spikes")

# ── 2. AdEx firing patterns ───────────────────────────────────────────────
print("\n[2] Persisting AdEx firing patterns …")
preset_I = {"RS": 400, "IB": 400, "CH": 420, "FS": 300, "LTS": 420}
T_a, dt_a = 400.0, 0.1

run_adex = SimulationRun(
    name="AdEx_patterns",
    description="AdEx firing pattern gallery: RS, IB, CH, FS, LTS",
    model_type="AdaptiveExponentialIF",
    duration_ms=T_a, dt_ms=dt_a,
    tags="AdEx,firing_patterns,cortical_neurons",
)
session.add(run_adex); session.flush()

for preset, I_val in preset_I.items():
    n_s = int(T_a/dt_a)
    I_step = np.zeros(n_s); I_step[int(50/dt_a):] = I_val
    nn = AdaptiveExponentialIF.from_preset(preset); nn.t_ref = 2.0
    res_a = nn.simulate_detailed(I_step, dt=dt_a)

    sp_a = SpikeTrainRecord(run_id=run_adex.id, neuron_id=0, population=preset)
    sp_a.spike_times = res_a["spikes"]
    sp_a.compute_stats(T_a)
    session.add(sp_a)

    ar_a = AnalysisResult(run_id=run_adex.id, analysis_type=f"spike_stats_{preset}")
    ar_a.metrics = spike_train_statistics(res_a["spikes"], T_a)
    session.add(ar_a)
    print(f"   {preset}: {sp_a.n_spikes} spikes, {sp_a.mean_rate_hz:.1f} Hz")

session.commit()

# ── 3. LIF f-I curve ─────────────────────────────────────────────────────
print("\n[3] Persisting LIF f-I curve analysis …")
lif = LeakyIntegrateAndFire()
I_range = np.linspace(0, 500, 15)
f_arr = []
for I_v in I_range:
    I_a = np.full(int(300/0.1), I_v)
    r = lif.simulate(I_a, dt=0.1)
    f_arr.append(r["firing_rate_hz"])
    lif.reset()

run_fi = SimulationRun(
    name="LIF_fI_curve",
    description="LIF f-I curve: 0–500 pA",
    model_type="LIF",
    duration_ms=300.0, dt_ms=0.1,
    tags="LIF,fI_curve,gain,rheobase",
)
session.add(run_fi); session.flush()

ar_fi = AnalysisResult(run_id=run_fi.id, analysis_type="fI_curve")
ar_fi.metrics = {
    "I_pA":   I_range,
    "f_hz":   np.array(f_arr),
    "rheobase_pA": float(I_range[np.array(f_arr) > 1.0][0])
                   if (np.array(f_arr) > 1.0).any() else None,
}
session.add(ar_fi)
session.commit()
print(f"   ✓ f-I curve stored: {len(I_range)} points, "
      f"max={max(f_arr):.1f} Hz")

# ── 4. Query examples ─────────────────────────────────────────────────────
print("\n[4] Database query examples:")
print("-" * 50)

all_runs = session.query(SimulationRun).all()
print(f"\n   All SimulationRuns ({len(all_runs)} total):")
for r in all_runs:
    print(f"     [{r.id}] {r.name:<25} model={r.model_type}")

all_trains = session.query(SpikeTrainRecord).all()
print(f"\n   All SpikeTrainRecords ({len(all_trains)} total):")
for s in all_trains:
    cv_str = f"{s.cv_isi:.3f}" if s.cv_isi else "N/A"
    print(f"     [{s.run_id}:{s.population}] "
          f"n={s.n_spikes}, rate={s.mean_rate_hz:.1f} Hz, CV={cv_str}")

# Query: highest-CV spike train (most irregular)
irregular = (session.query(SpikeTrainRecord)
             .filter(SpikeTrainRecord.cv_isi != None)
             .order_by(SpikeTrainRecord.cv_isi.desc())
             .first())
if irregular:
    print(f"\n   Most irregular: run_id={irregular.run_id}, "
          f"population={irregular.population}, CV={irregular.cv_isi:.3f}")

# Query: all HH runs
hh_runs = (session.query(SimulationRun)
           .filter(SimulationRun.model_type == "HodgkinHuxley")
           .all())
print(f"\n   HH runs: {len(hh_runs)}")

print("\n" + "=" * 56)
print("  Database persist complete. File: compneuro.db")
print("=" * 56)
session.close()
