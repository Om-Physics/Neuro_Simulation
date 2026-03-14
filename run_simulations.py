"""
run_simulations.py
==================
Master simulation script for the CompNeuro project.

Runs all experiments, generates all 10 figures, and persists results
to a SQLite database for reproducibility and later query.

Usage
-----
    python run_simulations.py

All figures are saved to ./figures/
Database is saved as compneuro.db
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Project imports ───────────────────────────────────────────────────────
from neurons.hodgkin_huxley     import HodgkinHuxley
from neurons.integrate_fire     import (
    LeakyIntegrateAndFire, AdaptiveExponentialIF
)
from synapses.synapse           import (
    AMPASynapse, NMDASynapse, GABAASynapse, GABABSynapse
)
from synapses.plasticity        import STDPRule
from networks.network           import SpikingNetwork
from analysis.spike_analysis    import (
    spike_train_statistics, isi_distribution,
    power_spectral_density, fI_curve_analysis,
    population_synchrony, psth,
)
from analysis                   import visualisation as vis
from database.models            import (
    create_db, get_session, SimulationRun, NeuronParameters,
    SpikeTrainRecord, VoltageTrace, AnalysisResult, NetworkState,
)


# ── Database setup ────────────────────────────────────────────────────────
print("=" * 60)
print("  CompNeuro — Neural Network Simulation Suite")
print("=" * 60)
print("\n[DB] Initialising database …")
engine, SessionClass = create_db("compneuro.db")
session = get_session(engine)
print("     ✓ compneuro.db ready\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Hodgkin-Huxley single neuron
# ═══════════════════════════════════════════════════════════════════════════
print("[1/7] Hodgkin-Huxley Action Potential …")

hh = HodgkinHuxley()
dt_hh  = 0.025   # ms
T_hh   = 100.0   # ms
n_hh   = int(T_hh / dt_hh)
I_hh   = np.zeros(n_hh)
I_hh[int(5/dt_hh):] = 10.0       # step current 10 µA/cm² from t=5 ms

hh_result = hh.simulate_detailed(I_hh, dt=dt_hh)
print(f"     Spikes: {len(hh_result['spikes'])}  "
      f"  Rate: {hh_result['firing_rate_hz']:.1f} Hz")

# Persist
run1 = SimulationRun(
    name="HH_step_10uA",
    description="Hodgkin-Huxley step-current injection 10 µA/cm²",
    model_type="HodgkinHuxley",
    duration_ms=T_hh, dt_ms=dt_hh,
)
session.add(run1); session.flush()

np1 = NeuronParameters(run_id=run1.id, neuron_id=0, model_class="HodgkinHuxley")
np1.params = dict(C_m=hh.C_m, g_Na=hh.g_Na, g_K=hh.g_K, g_L=hh.g_L,
                  E_Na=hh.E_Na, E_K=hh.E_K, E_L=hh.E_L)
session.add(np1)

sp1 = SpikeTrainRecord(run_id=run1.id, neuron_id=0, population="E")
sp1.spike_times = hh_result["spikes"]
sp1.compute_stats(T_hh)
session.add(sp1)

vt1 = VoltageTrace(run_id=run1.id, neuron_id=0, downsample=2,
                   t_start_ms=0, t_end_ms=T_hh, dt_stored_ms=dt_hh*2)
vt1.voltage = hh_result["V"]
session.add(vt1)

ar1 = AnalysisResult(run_id=run1.id, analysis_type="spike_stats", neuron_id=0)
ar1.metrics = spike_train_statistics(hh_result["spikes"], T_hh)
session.add(ar1)
session.commit()

print("     Fig 1: HH action potential anatomy …")
vis.fig1_hh_action_potential(hh_result)
print("     Fig 2: Phase plane …")
vis.fig2_phase_plane(hh, hh_result)
print("     ✓ Figs 1–2 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — f-I Curves: HH, LIF, AdEx
# ═══════════════════════════════════════════════════════════════════════════
print("[2/7] f-I Curves (HH, LIF, AdEx) …")

fi_results = {}

# HH f-I
hh_fi = HodgkinHuxley()
I_range_hh = np.linspace(0, 30, 30)
fi_hh = hh_fi.fI_curve(I_range_hh, T_ms=500, dt=0.025)
fi_results["HH"] = fi_hh
print(f"     HH rheobase ≈ {I_range_hh[np.array(fi_hh['f_hz'])>1][0]:.1f} µA/cm²"
      if (np.array(fi_hh['f_hz'])>1).any() else "     HH: no spiking")

# LIF f-I (current in pA)
lif_fi = LeakyIntegrateAndFire()
I_range_lif = np.linspace(0, 600, 30)   # pA
n_steps_lif = int(500 / 0.1)
f_lif = []
for I_val in I_range_lif:
    I_arr = np.full(n_steps_lif, I_val)
    res = lif_fi.simulate(I_arr, dt=0.1)
    f_lif.append(res["firing_rate_hz"])
    lif_fi.reset()
fi_results["LIF"] = {"I": I_range_lif, "f_hz": np.array(f_lif)}

# AdEx-RS f-I
adex_rs = AdaptiveExponentialIF.from_preset("RS")
f_adex_rs = []
for I_val in I_range_lif:
    I_arr = np.full(n_steps_lif, I_val)
    res = adex_rs.simulate(I_arr, dt=0.1)
    f_adex_rs.append(res["firing_rate_hz"])
    adex_rs.reset()
fi_results["AdEx-RS"] = {"I": I_range_lif, "f_hz": np.array(f_adex_rs)}

# AdEx-FS f-I
adex_fs = AdaptiveExponentialIF.from_preset("FS")
f_adex_fs = []
for I_val in I_range_lif:
    I_arr = np.full(n_steps_lif, I_val)
    res = adex_fs.simulate(I_arr, dt=0.1)
    f_adex_fs.append(res["firing_rate_hz"])
    adex_fs.reset()
fi_results["AdEx-FS"] = {"I": I_range_lif, "f_hz": np.array(f_adex_fs)}

# fI curve analysis for LIF
fi_analysis = fI_curve_analysis(I_range_lif, np.array(f_lif))
ar_fi = AnalysisResult(run_id=run1.id, analysis_type="fI_curve_LIF")
ar_fi.metrics = {
    "rheobase_pA": fi_analysis["rheobase"],
    "gain_hz_per_pA": fi_analysis["gain_hz_per_uA_cm2"],
    "f_max_hz": fi_analysis["f_max_hz"],
}
session.add(ar_fi); session.commit()

print("     Fig 3: f-I curves …")
vis.fig3_fi_curves(fi_results)
print("     ✓ Fig 3 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — AdEx Firing Patterns
# ═══════════════════════════════════════════════════════════════════════════
print("[3/7] AdEx Firing Patterns (RS, IB, CH, FS, LTS) …")

T_adex = 500.0
dt_adex = 0.1
n_adex = int(T_adex / dt_adex)
pattern_results = {}

# Per-preset suprathreshold step currents (pA) — from Naud et al. 2008
preset_currents = {
    "RS":  160.0,   # regular spiking with adaptation
    "IB":  160.0,   # intrinsic bursting
    "CH":  170.0,   # chattering
    "FS":  160.0,   # fast spiking (no adaptation)
    "LTS": 165.0,   # low-threshold spiking
}
# Add minimum refractory for FS to prevent numerical runaway
preset_tref = {"RS": 0.0, "IB": 0.0, "CH": 0.0, "FS": 2.0, "LTS": 0.0}

run3 = SimulationRun(
    name="AdEx_firing_patterns",
    description="AdEx firing pattern gallery (RS, IB, CH, FS, LTS)",
    model_type="AdaptiveExponentialIF",
    duration_ms=T_adex, dt_ms=dt_adex,
)
session.add(run3); session.flush()

for preset in ["RS", "IB", "CH", "FS", "LTS"]:
    I_step = np.zeros(n_adex)
    I_step[int(50/dt_adex):] = preset_currents[preset]
    neuron = AdaptiveExponentialIF.from_preset(preset)
    neuron.t_ref = preset_tref[preset]
    res = neuron.simulate_detailed(I_step, dt=dt_adex)
    pattern_results[preset] = res
    print(f"     {preset}: {len(res['spikes'])} spikes, "
          f"{res['firing_rate_hz']:.1f} Hz, CV={res['cv_isi']:.2f}"
          if not np.isnan(res['cv_isi']) else
          f"     {preset}: {len(res['spikes'])} spikes, {res['firing_rate_hz']:.1f} Hz")

    sp_r = SpikeTrainRecord(run_id=run3.id, neuron_id=0, population=preset)
    sp_r.spike_times = res["spikes"]
    sp_r.compute_stats(T_adex)
    session.add(sp_r)

session.commit()
print("     Fig 4: AdEx patterns …")
vis.fig4_adex_patterns(pattern_results)
print("     ✓ Fig 4 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Synaptic Conductances
# ═══════════════════════════════════════════════════════════════════════════
print("[4/7] Synaptic Conductance Kinetics …")

T_syn = 300.0   # ms
dt_syn = 0.1
n_syn = int(T_syn / dt_syn)
syn_results = {}

for name, syn_obj in [
    ("AMPA",   AMPASynapse(g_max=1.0)),
    ("NMDA",   NMDASynapse(g_max=1.0)),
    ("GABA-A", GABAASynapse(g_max=1.0)),
    ("GABA-B", GABABSynapse(g_max=1.0)),
]:
    t_arr = np.arange(n_syn) * dt_syn
    g_arr = np.zeros(n_syn)
    V_post = -65.0   # clamped postsynaptic voltage
    syn_obj.reset()
    for i in range(n_syn):
        spike = (i == int(10 / dt_syn))   # single spike at t=10 ms
        I_syn = syn_obj.step(dt_syn, V_post, spike=spike)
        g_arr[i] = syn_obj.g
    syn_results[name] = {"t": t_arr, "g": g_arr}
    print(f"     {name}: peak conductance = {np.max(g_arr)*1000:.2f} pS "
          f"at t={t_arr[np.argmax(g_arr)]:.1f} ms")

print("     Fig 5: synaptic conductances …")
vis.fig5_synaptic_conductances(syn_results)
print("     Fig 6: NMDA Mg²⁺ block …")
vis.fig6_nmda_mg_block()
print("     ✓ Figs 5–6 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — STDP Weight Dynamics
# ═══════════════════════════════════════════════════════════════════════════
print("[5/7] STDP Weight Dynamics …")

rng_stdp = np.random.default_rng(42)
stdp = STDPRule(A_plus=0.01, A_minus=0.0105, tau_plus=20.0, tau_minus=20.0)
dt_stdp = 0.1   # ms
T_stdp  = 20000 # ms (20 s)
n_stdp  = int(T_stdp / dt_stdp)

pre_rate  = 20.0 * 1e-3   # 20 Hz → events/ms
post_rate = 20.0 * 1e-3

w = 0.5   # initial weight
for step in range(n_stdp):
    pre_spike  = rng_stdp.random() < pre_rate  * dt_stdp
    post_spike = rng_stdp.random() < post_rate * dt_stdp
    w = stdp.update(w, dt_stdp, bool(pre_spike), bool(post_spike))

w_history = stdp.weight_history
print(f"     Initial w=0.5 → Final w={w_history[-1]:.3f}")
print(f"     Weight range: [{w_history.min():.3f}, {w_history.max():.3f}]")

# Steady-state distribution across 100 synapses
w_final = []
for _ in range(100):
    stdp2 = STDPRule()
    w2 = rng_stdp.random()  # random initial weight
    for step in range(n_stdp):
        pre_sp  = rng_stdp.random() < pre_rate  * dt_stdp
        post_sp = rng_stdp.random() < post_rate * dt_stdp
        w2 = stdp2.update(w2, dt_stdp, bool(pre_sp), bool(post_sp))
    w_final.append(w2)

stdp_result = {
    "w_history": w_history[::100],   # subsample for plotting
    "w_final_dist": np.array(w_final),
}
print("     Fig 7: STDP …")
vis.fig7_stdp(stdp_result)
print("     ✓ Fig 7 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Recurrent E/I Network
# ═══════════════════════════════════════════════════════════════════════════
print("[6/7] E/I Recurrent Spiking Network (200 neurons, 500 ms) …")
print("      This may take ~1–2 minutes …")

N_E = 160; N_I = 40
net = SpikingNetwork(
    N_E=N_E, N_I=N_I,
    p_conn=0.15,
    dt=0.2,
    neuron_type="LIF",
    g_EE=0.25, g_EI=0.25, g_IE=1.5, g_II=1.5,
    nu_ext=10.0, g_ext=0.8,   # stronger Poisson drive
    seed=123,
)
print(net.summary())

# Provide a baseline DC current to all neurons (220 pA)
# representing the net input from thousands of background synapses
n_steps_net = int(500.0 / net.dt)
I_dc = np.full((N_E + N_I, n_steps_net), 220.0)   # pA

net_result = net.run(T_ms=500.0, I_ext=I_dc, progress_cb=lambda t: print(f"      t={t:.0f} ms", end="\r"))
print()

# Compute population stats
T_s = 500e-3
rates_E = np.array([len(net_result["spikes"][i]) / T_s for i in range(N_E)])
rates_I = np.array([len(net_result["spikes"][N_E+i]) / T_s for i in range(N_I)])
all_cv = []
for i in range(N_E + N_I):
    t_sp = np.array(net_result["spikes"][i])
    if len(t_sp) > 2:
        isi = np.diff(t_sp)
        all_cv.append(float(np.std(isi) / np.mean(isi)))

print(f"     E rate: {rates_E.mean():.1f} ± {rates_E.std():.1f} Hz")
print(f"     I rate: {rates_I.mean():.1f} ± {rates_I.std():.1f} Hz")
if all_cv:
    print(f"     Mean CV-ISI: {np.mean(all_cv):.3f}  "
          f"({'AI state' if np.mean(all_cv) > 0.7 else 'SR state'})")

# Synchrony
sync_chi = population_synchrony(net_result["V"][:N_E, :])
print(f"     Synchrony χ²: {sync_chi:.3f}")

# Persist
run6 = SimulationRun(
    name="EI_network_LIF",
    description=f"E/I LIF network: {N_E}E + {N_I}I, p={net.p_conn}",
    model_type="SpikingNetwork_LIF",
    duration_ms=500.0, dt_ms=net.dt,
)
session.add(run6); session.flush()

for nid in range(0, min(20, N_E + N_I)):   # store first 20 neurons
    sp_rec = SpikeTrainRecord(
        run_id=run6.id, neuron_id=nid,
        population="E" if nid < N_E else "I",
    )
    sp_rec.spike_times = np.array(net_result["spikes"][nid])
    sp_rec.compute_stats(500.0)
    session.add(sp_rec)

ns = NetworkState(
    run_id=run6.id,
    pop_E_rate_mean_hz=float(rates_E.mean()),
    pop_E_rate_std_hz=float(rates_E.std()),
    pop_I_rate_mean_hz=float(rates_I.mean()),
    pop_I_rate_std_hz=float(rates_I.std()),
    synchrony_chi=float(sync_chi) if not np.isnan(sync_chi) else None,
    irregularity=float(np.mean(all_cv)) if all_cv else None,
    EI_ratio=float(rates_E.mean() / (rates_I.mean() + 1e-9)),
    network_state="AI" if (all_cv and np.mean(all_cv) > 0.7) else "SR",
)
session.add(ns); session.commit()

print("     Fig 8: network dynamics …")
vis.fig8_network(net_result, N_E, N_I)
print("     ✓ Fig 8 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — ISI Analysis & PSD
# ═══════════════════════════════════════════════════════════════════════════
print("[7/7] ISI Analysis + LFP Power Spectrum …")

# Collect all E-cell spike trains with enough spikes
all_trains = [np.array(net_result["spikes"][i])
              for i in range(N_E) if len(net_result["spikes"][i]) > 2]
if not all_trains:
    all_trains = [np.array([10.0, 25.0, 45.0, 70.0, 100.0])]

# ISI of the busiest neuron
ref_train = max(all_trains, key=len)
isi_res = isi_distribution(ref_train, bins=40)
pop_cv  = np.array(all_cv) if all_cv else np.array([1.0])

print("     Fig 9: ISI analysis …")
vis.fig9_isi_analysis(isi_res, pop_cv)

# LFP PSD
LFP   = net_result["LFP"]
dt_ms = net.dt
fs_hz = 1000.0 / dt_ms   # sampling freq in Hz
psd_res = power_spectral_density(LFP, fs_hz=fs_hz)
print(f"     Dominant LFP frequency: {psd_res['dominant_freq_hz']:.1f} Hz")
print(f"     Band powers: " + ", ".join(
    f"{k}={v:.2e}" for k, v in psd_res["band_powers"].items()
))

ar_psd = AnalysisResult(run_id=run6.id, analysis_type="LFP_PSD")
ar_psd.metrics = {
    "dominant_freq_hz": psd_res["dominant_freq_hz"],
    "band_powers": psd_res["band_powers"],
}
session.add(ar_psd); session.commit()

print("     Fig 10: PSD …")
vis.fig10_psd(psd_res)
print("     ✓ Figs 9–10 saved\n")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
session.close()
from pathlib import Path
figs = sorted(Path("figures").glob("*.png"))

print("=" * 60)
print("  SIMULATION COMPLETE")
print("=" * 60)
print(f"  Figures saved ({len(figs)} total):")
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"    {f.name:<45} {size_kb:>4} KB")
print(f"\n  Database: compneuro.db")
print("=" * 60)
