"""
test_synapses.py
================
Unit tests for the synaptic transmission models (AMPA, NMDA, GABA-A, GABA-B)
and plasticity rules (STDP, BCM, Oja, Triplet-STDP).

Each test validates a specific experimentally observed or theoretically
predicted property of the model, ensuring that the kinetic parameters
produce biologically correct time courses and that the plasticity rules
implement the correct learning equations.

Run with:
    python -m pytest tests/test_synapses.py -v

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapses.synapse import AMPASynapse, NMDASynapse, GABAAsynapse, GABABsynapse
from synapses.plasticity import STDPRule, BCMRule, OjaRule, TripletSTDP


class TestAMPASynapse:

    def test_zero_conductance_at_rest(self):
        syn = AMPASynapse()
        assert syn.r == 0.0

    def test_conductance_rises_after_spike(self):
        syn = AMPASynapse()
        d   = syn.simulate(T=100.0, dt=0.1, spike_times=[10.0])
        assert d["g"].max() > 0.01, "AMPA conductance should rise after spike."

    def test_conductance_decays_to_zero(self):
        syn = AMPASynapse()
        d   = syn.simulate(T=200.0, dt=0.1, spike_times=[10.0])
        assert d["g"][-1] < 1e-3, "AMPA conductance should decay to near zero."

    def test_peak_occurs_within_10ms_of_spike(self):
        syn  = AMPASynapse()
        d    = syn.simulate(T=100.0, dt=0.1, spike_times=[10.0])
        t_pk = d["t"][np.argmax(d["g"])]
        assert 10.0 <= t_pk <= 20.0, (
            f"AMPA peak at {t_pk:.1f} ms, expected within 10-20 ms of spike at 10 ms."
        )

    def test_excitatory_reversal_potential(self):
        syn = AMPASynapse(E_rev=0.0)
        assert syn.E_rev == 0.0, "AMPA E_rev should be 0 mV."


class TestNMDASynapse:

    def test_mg_block_at_resting_potential(self):
        """At resting potential (-65 mV) with 1 mM Mg, block should be > 85%."""
        syn = NMDASynapse(Mg=1.0)
        B   = syn.B(-65.0)
        assert B < 0.15, f"Expected near-complete Mg block at -65 mV, B(V)={B:.3f}."

    def test_mg_block_relieved_at_depolarised_voltage(self):
        """At 0 mV the Mg block should be mostly relieved (B > 0.7)."""
        syn = NMDASynapse(Mg=1.0)
        B   = syn.B(0.0)
        assert B > 0.70, f"Expected Mg block relief at 0 mV, B(V)={B:.3f}."

    def test_zero_mg_gives_no_block(self):
        """At [Mg] = 0, B(V) should equal 1 at all voltages."""
        syn = NMDASynapse(Mg=0.0)
        for V in [-80, -65, -40, 0, 20]:
            B = syn.B(V)
            assert abs(B - 1.0) < 1e-9, f"B({V})={B} with zero Mg, expected 1.0."

    def test_slow_kinetics_relative_to_ampa(self):
        """NMDA decay should be substantially slower than AMPA."""
        ampa = AMPASynapse()
        nmda = NMDASynapse()
        d_a  = ampa.simulate(T=300.0, dt=0.1, spike_times=[10.0])
        d_n  = nmda.simulate(T=300.0, dt=0.1, spike_times=[10.0], V_post=0.0)
        t_half_ampa = 0.0
        t_half_nmda = 0.0
        peak_a = d_a["g"].max()
        peak_n = d_n["g"].max()
        if peak_a > 0 and peak_n > 0:
            for i, (ta, ga) in enumerate(zip(d_a["t"], d_a["g"])):
                if ta >= 15.0 and ga < peak_a * 0.5:
                    t_half_ampa = ta; break
            for i, (tn, gn) in enumerate(zip(d_n["t"], d_n["g"])):
                if tn >= 15.0 and gn < peak_n * 0.5:
                    t_half_nmda = tn; break
            assert t_half_nmda > t_half_ampa, (
                f"NMDA half-decay at {t_half_nmda} ms should be > AMPA at {t_half_ampa} ms."
            )

    def test_iv_curve_has_negative_slope_region(self):
        """The NMDA I-V curve should have a negative slope region (N-shape)."""
        syn = NMDASynapse(g_max=1.0, Mg=1.0)
        iv  = syn.iv_curve()
        V   = iv["V"]
        I   = iv["I"]
        dI  = np.diff(I) / np.diff(V)
        assert np.any(dI < 0), "NMDA I-V should have negative slope (N-shaped curve)."


class TestGABAASynapse:

    def test_inhibitory_reversal_potential(self):
        syn = GABAAsynapse()
        assert syn.E_rev == -70.0, "GABA-A E_rev should be -70 mV."

    def test_conductance_rises_after_spike(self):
        syn = GABAAsynapse()
        d   = syn.simulate(T=100.0, dt=0.1, spike_times=[10.0])
        assert d["g"].max() > 0.01

    def test_current_is_inhibitory_at_rest(self):
        """At V = -65 mV, GABA-A current should be outward (positive), inhibitory."""
        syn = GABAAsynapse(g_max=1.0, E_rev=-70.0)
        syn.receive_spike()
        for _ in range(5):
            I = syn.step(0.1, V_post=-65.0)
        assert I > 0, (
            f"GABA-A should produce outward (inhibitory) current at -65 mV, got {I:.4f}."
        )


class TestGABABSynapse:

    def test_slow_peak_timing(self):
        """GABA-B should peak much later than GABA-A due to G-protein cascade."""
        gabab = GABABsynapse()
        d     = gabab.simulate(T=600.0, dt=0.1, spike_times=[10.0, 30.0, 50.0, 70.0])
        t_pk  = d["t"][np.argmax(d["g"])]
        assert t_pk > 80.0, (
            f"GABA-B peak at {t_pk:.1f} ms; expected > 80 ms after first spike."
        )

    def test_k_reversal_potential(self):
        """GABA-B acts on K+ channels: E_rev should be near -95 mV."""
        syn = GABABsynapse()
        assert syn.E_rev == -95.0, "GABA-B E_rev should be -95 mV (K+ Nernst)."

    def test_hill_cooperativity(self):
        """Hill exponent n=4 reflects G-protein stoichiometry."""
        syn = GABABsynapse()
        assert syn.n == 4.0


class TestSTDPRule:

    def test_ltp_for_positive_delta_t(self):
        """When post fires after pre (delta_t > 0), weight should increase."""
        stdp = STDPRule(w_min=0.0, w_max=1.0)
        result = stdp.run(T=5000.0, dt=0.1, rate_pre=10.0, rate_post=10.0,
                          w_init=0.3, n_synapses=10, seed=7)
        dw_sum = result["dw_history"].sum()
        assert np.any(result["dw_history"] > 0), "Expected some LTP events."

    def test_weight_stays_bounded(self):
        """Weights must never leave [w_min, w_max] under any input."""
        stdp   = STDPRule(w_min=0.0, w_max=1.0)
        result = stdp.run(T=10000.0, dt=0.1, n_synapses=20, seed=99)
        assert result["w_final"].min() >= 0.0 - 1e-9
        assert result["w_final"].max() <= 1.0 + 1e-9

    def test_learning_window_ltp_at_positive_dt(self):
        stdp = STDPRule()
        dw   = stdp.learning_window(np.array([10.0]))
        assert dw[0] > 0, "STDP window should be positive (LTP) at delta_t > 0."

    def test_learning_window_ltd_at_negative_dt(self):
        stdp = STDPRule()
        dw   = stdp.learning_window(np.array([-10.0]))
        assert dw[0] < 0, "STDP window should be negative (LTD) at delta_t < 0."

    def test_ltd_amplitude_slightly_larger_than_ltp(self):
        """A_minus > A_plus ensures net depression at equal pre/post rates."""
        stdp = STDPRule()
        assert stdp.A_minus > stdp.A_plus, (
            f"A_minus={stdp.A_minus} should exceed A_plus={stdp.A_plus} "
            "to prevent runaway potentiation."
        )

    def test_weight_convergence_from_two_initial_values(self):
        """Weights initialised at 0.2 and 0.8 should converge toward same mean."""
        stdp = STDPRule()
        r1   = stdp.run(T=20000.0, dt=0.5, w_init=0.2, n_synapses=30, seed=0)
        r2   = stdp.run(T=20000.0, dt=0.5, w_init=0.8, n_synapses=30, seed=0)
        assert abs(r1["w_final"].mean() - r2["w_final"].mean()) < 0.25, (
            "Weights initialised differently should converge to similar means."
        )


class TestBCMRule:

    def test_returns_float(self):
        bcm = BCMRule()
        dw  = bcm.update(pre=1.0, post=0.5, dt=0.1)
        assert isinstance(dw, float)

    def test_positive_dw_when_post_above_theta(self):
        bcm       = BCMRule()
        bcm.theta = 0.1
        dw        = bcm.update(pre=1.0, post=0.5, dt=0.1)
        assert dw > 0

    def test_negative_dw_when_post_below_theta(self):
        bcm       = BCMRule()
        bcm.theta = 0.8
        dw        = bcm.update(pre=1.0, post=0.1, dt=0.1)
        assert dw < 0


class TestOjaRule:

    def test_normalising_property(self):
        """Oja rule should keep weight magnitude from growing without bound."""
        oja = OjaRule(eta=0.01)
        w   = 0.5
        for _ in range(5000):
            pre  = np.random.randn()
            post = w * pre
            w   += oja.update(pre, post, w)
        assert abs(w) < 5.0, f"Oja weight diverged to {w:.3f}."
