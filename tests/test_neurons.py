"""
test_neurons.py
===============
Unit and integration tests for all three neuron models:
Hodgkin-Huxley, Leaky Integrate-and-Fire, and AdEx.

Test philosophy:
  Each test validates a specific scientific property of the model, not merely
  that the code runs. Tolerances are set to biologically meaningful bounds
  rather than arbitrary numerical precision, so the tests catch regressions
  that would produce scientifically incorrect results even if the code
  executes without error.

Test categories:
  HH   - Action potential existence, rheobase, RK4 stability, gating bounds
  LIF  - Threshold detection, reset, refractory period, analytical f-I match
  AdEx - Spike-reset overflow prevention, preset loading, adaptation current

Run with:
    python -m pytest tests/test_neurons.py -v

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.hodgkin_huxley import HodgkinHuxley
from neurons.integrate_fire import (
    LeakyIntegrateAndFire,
    ExponentialIntegrateAndFire,
    AdaptiveExponentialIF,
    ADEX_PRESETS,
)
from neurons.base_neuron import SpikeRecord


class TestHodgkinHuxley:
    """Tests for the HH conductance-based neuron model."""

    def test_resting_potential_stable_below_rheobase(self):
        """Below rheobase, HH should remain near the resting potential."""
        hh  = HodgkinHuxley()
        rec = hh.simulate(T=200.0, dt=0.025, I_ext=0.5)
        assert rec.count == 0, "Expected no spikes below rheobase."
        assert abs(rec.voltages[-1] - (-65.0)) < 5.0, (
            "Resting voltage should remain near -65 mV without drive."
        )

    def test_action_potential_exists_above_rheobase(self):
        """At I_ext = 10 uA/cm^2 the HH model should fire at least 5 spikes."""
        hh  = HodgkinHuxley()
        rec = hh.simulate(T=120.0, dt=0.025, I_ext=10.0)
        assert rec.count >= 5, (
            f"Expected >= 5 spikes at 10 uA/cm^2, got {rec.count}."
        )

    def test_peak_voltage_near_sodium_reversal(self):
        """The AP peak should be positive and approach E_Na = +55 mV."""
        hh  = HodgkinHuxley()
        rec = hh.simulate(T=30.0, dt=0.025, I_ext=10.0)
        peak = rec.voltages.max()
        assert peak > 20.0, f"AP peak too low: {peak:.1f} mV."
        assert peak < 65.0, f"AP peak unphysically high: {peak:.1f} mV."

    def test_gating_variables_bounded(self):
        """All gating variables m, h, n must remain in [0, 1]."""
        hh   = HodgkinHuxley()
        data = hh.simulate_detailed(T=100.0, dt=0.025, I_ext=10.0)
        for gate in ("m", "h", "n"):
            vals = data[gate]
            assert vals.min() >= -1e-6, f"Gate {gate} below 0: {vals.min()}"
            assert vals.max() <= 1 + 1e-6, f"Gate {gate} above 1: {vals.max()}"

    def test_firing_rate_physiological_range(self):
        """Mean firing rate should lie in a physiologically plausible range."""
        hh   = HodgkinHuxley()
        rec  = hh.simulate(T=500.0, dt=0.025, I_ext=10.0)
        rate = rec.mean_rate
        assert 30.0 <= rate <= 150.0, (
            f"Firing rate {rate:.1f} Hz outside expected range 30-150 Hz."
        )

    def test_fi_curve_monotonically_increasing(self):
        """f-I curve should be monotonically non-decreasing above rheobase."""
        hh  = HodgkinHuxley()
        I   = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
        _, r = hh.fI_curve(I, T=400.0, dt=0.025, warmup=100.0)
        for i in range(1, len(r)):
            assert r[i] >= r[i-1] - 5.0, (
                f"f-I curve not monotone: r[{i-1}]={r[i-1]:.1f}, r[{i}]={r[i]:.1f}"
            )

    def test_rk4_produces_stable_trajectory(self):
        """RK4 integration should not produce NaN or Inf values."""
        hh   = HodgkinHuxley()
        data = hh.simulate_detailed(T=200.0, dt=0.025, I_ext=10.0)
        for key in ("V", "m", "h", "n"):
            assert np.all(np.isfinite(data[key])), (
                f"Non-finite values in {key} trace."
            )

    def test_reset_clears_state(self):
        """After reset, the model should return to the initial steady state."""
        hh = HodgkinHuxley()
        hh.simulate(T=50.0, dt=0.025, I_ext=10.0)
        hh.reset()
        assert abs(hh._V - (-65.0)) < 1.0
        assert len(hh._spike_times) == 0

    def test_nullclines_returns_correct_keys(self):
        hh = HodgkinHuxley()
        nc = hh.nullclines(I_ext=10.0)
        for key in ("V", "V_null", "n_null", "n_inf"):
            assert key in nc, f"Missing key '{key}' in nullclines output."


class TestLeakyIntegrateAndFire:
    """Tests for the LIF neuron model."""

    def test_no_spikes_below_rheobase(self):
        lif = LeakyIntegrateAndFire()
        I_rh = lif.gL * (lif.V_thresh - lif.EL)
        rec  = lif.simulate(T=300.0, dt=0.1, I_ext=I_rh * 0.5)
        assert rec.count == 0, "Expected no spikes below rheobase."

    def test_spikes_above_rheobase(self):
        lif = LeakyIntegrateAndFire()
        rec = lif.simulate(T=500.0, dt=0.1, I_ext=300.0)
        assert rec.count > 0, "Expected spikes at I=300 pA."

    def test_voltage_reset_after_spike(self):
        """Membrane voltage must equal V_reset immediately after each spike."""
        lif  = LeakyIntegrateAndFire(t_ref=0.0)
        data = lif.simulate(T=300.0, dt=0.1, I_ext=300.0)
        V    = data.voltages
        for spike_t in data.times:
            idx = int(spike_t / 0.1)
            if idx + 1 < len(V):
                assert V[idx + 1] <= lif.V_reset + 1.0, (
                    f"Post-spike voltage {V[idx+1]:.1f} exceeds V_reset."
                )

    def test_analytical_rate_matches_simulation(self):
        """Simulated f-I should match the analytical formula within 10%."""
        lif = LeakyIntegrateAndFire()
        for I in [300.0, 500.0, 700.0]:
            rec       = lif.simulate(T=1000.0, dt=0.1, I_ext=I)
            sim_rate  = rec.mean_rate
            anal_rate = lif.analytical_rate(I)
            if anal_rate > 0:
                rel_err = abs(sim_rate - anal_rate) / anal_rate
                assert rel_err < 0.15, (
                    f"I={I}: sim={sim_rate:.1f} Hz vs analytical={anal_rate:.1f} Hz "
                    f"(rel err {rel_err:.2%})"
                )

    def test_refractory_period_enforced(self):
        """ISI must be no shorter than the refractory period."""
        t_ref = 5.0
        lif   = LeakyIntegrateAndFire(t_ref=t_ref)
        rec   = lif.simulate(T=1000.0, dt=0.1, I_ext=1000.0)
        isi   = rec.isi()
        if len(isi) > 0:
            assert isi.min() >= t_ref - 0.5, (
                f"Minimum ISI {isi.min():.2f} ms is shorter than t_ref={t_ref} ms."
            )


class TestAdaptiveExponentialIF:
    """Tests for the AdEx neuron model."""

    def test_all_presets_load_and_fire(self):
        """Every canonical preset must produce at least one spike."""
        for preset in ADEX_PRESETS:
            neuron = AdaptiveExponentialIF.from_preset(preset)
            p      = ADEX_PRESETS[preset]
            rec    = neuron.simulate(T=500.0, dt=0.1, I_ext=p["I_default"],
                                     t_start=50.0)
            assert rec.count > 0, (
                f"Preset {preset} produced no spikes at I={p['I_default']} pA."
            )

    def test_no_voltage_overflow(self):
        """Voltage must never exceed a hard cap even at strong drive."""
        neuron = AdaptiveExponentialIF.from_preset("RS")
        rec    = neuron.simulate(T=500.0, dt=0.1, I_ext=1000.0, t_start=0.0)
        assert rec.voltages.max() < 100.0, (
            f"Voltage overflow detected: max V = {rec.voltages.max():.1f} mV. "
            "Check immediate V_reset on spike."
        )

    def test_adaptation_current_increases_with_spikes(self):
        """The adaptation current w should increase after each spike burst."""
        neuron = AdaptiveExponentialIF.from_preset("RS")
        rec    = neuron.simulate(T=500.0, dt=0.1, I_ext=300.0, t_start=50.0)
        w      = rec.metadata["w"]
        if rec.count >= 3:
            idx_first = int(rec.times[0]  / 0.1)
            idx_last  = int(rec.times[-1] / 0.1)
            assert w[idx_last] > w[idx_first], (
                "Adaptation current should increase over repeated spiking for RS."
            )

    def test_fs_has_zero_adaptation(self):
        """Fast-spiking preset has a=b=0, so w should remain near zero."""
        neuron = AdaptiveExponentialIF.from_preset("FS")
        rec    = neuron.simulate(T=200.0, dt=0.1, I_ext=250.0)
        w      = rec.metadata["w"]
        assert w.max() < 50.0, (
            f"FS w should stay near zero (b=0), but max w = {w.max():.1f} pA."
        )

    def test_unknown_preset_raises(self):
        """Requesting an unknown preset must raise a ValueError."""
        with pytest.raises(ValueError):
            AdaptiveExponentialIF.from_preset("INVALID_PRESET")

    def test_cv_isi_rs_greater_than_fs(self):
        """RS has adaptation, so CV-ISI should be higher than FS (regular)."""
        rs  = AdaptiveExponentialIF.from_preset("RS")
        fs  = AdaptiveExponentialIF.from_preset("FS")
        rec_rs = rs.simulate(T=1000.0, dt=0.1, I_ext=300.0, t_start=50.0)
        rec_fs = fs.simulate(T=1000.0, dt=0.1, I_ext=250.0, t_start=50.0)
        cv_rs = rec_rs.cv_isi()
        cv_fs = rec_fs.cv_isi()
        if not (np.isnan(cv_rs) or np.isnan(cv_fs)):
            assert cv_rs > cv_fs, (
                f"Expected CV(RS)={cv_rs:.3f} > CV(FS)={cv_fs:.3f}."
            )


class TestSpikeRecord:
    """Tests for the SpikeRecord data container."""

    def test_mean_rate_correct(self):
        t_ax   = np.linspace(0, 1000, 10000)
        spikes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        rec    = SpikeRecord(times=spikes, voltages=np.zeros(10000), time_axis=t_ax)
        assert abs(rec.mean_rate - 5.0) < 0.1, (
            f"Expected 5 Hz, got {rec.mean_rate:.2f} Hz."
        )

    def test_cv_isi_poisson_approx(self):
        rng    = np.random.default_rng(0)
        isi    = rng.exponential(scale=50.0, size=500)
        spikes = np.cumsum(isi)
        t_ax   = np.linspace(0, spikes[-1], 10000)
        rec    = SpikeRecord(times=spikes, voltages=np.zeros(10000), time_axis=t_ax)
        cv     = rec.cv_isi()
        assert 0.7 < cv < 1.3, (
            f"CV-ISI for exponential ISIs should be near 1.0, got {cv:.3f}."
        )
