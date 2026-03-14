"""
plasticity.py
=============
Synaptic plasticity rules — activity-dependent modification of synaptic
strength based on spike timing correlations.

Implemented rules
-----------------
1. STDPRule        — classical asymmetric spike-timing dependent plasticity
                     (Bi & Poo, 1998; Song et al., 2000)
2. BCMRule         — Bienenstock-Cooper-Munro sliding threshold rule
                     (Bienenstock et al., 1982)
3. OjaRule         — Hebbian learning with Oja normalization (Oja, 1982)
4. TripletSTDP     — triplet model resolving BCM from STDP
                     (Pfister & Gerstner, 2006)

All rules operate on abstract synaptic weight w ∈ [w_min, w_max].

References
----------
Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured
    hippocampal neurons: dependence on spike timing, synaptic strength,
    and postsynaptic cell type. J Neurosci, 18, 10464-10472.
Song, S., Miller, K.D. & Abbott, L.F. (2000). Competitive Hebbian
    learning through spike-timing-dependent synaptic plasticity.
    Nat Neurosci, 3, 919-926.
Pfister, J.P. & Gerstner, W. (2006). Triplets of spikes in a model of
    spike timing-dependent plasticity. J Neurosci, 26, 9673-9682.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── STDP trace container ──────────────────────────────────────────────────
@dataclass
class EligibilityTrace:
    """Exponentially decaying pre/post spike traces for STDP."""
    tau_plus:  float = 20.0   # ms — LTP time window
    tau_minus: float = 20.0   # ms — LTD time window
    x_pre:  float = 0.0       # presynaptic trace
    x_post: float = 0.0       # postsynaptic trace

    def update(self, dt: float, pre_spike: bool, post_spike: bool) -> None:
        self.x_pre  *= np.exp(-dt / self.tau_plus)
        self.x_post *= np.exp(-dt / self.tau_minus)
        if pre_spike:
            self.x_pre  += 1.0
        if post_spike:
            self.x_post += 1.0

    def reset(self) -> None:
        self.x_pre = 0.0
        self.x_post = 0.0


# ── 1. Classical STDP ─────────────────────────────────────────────────────
class STDPRule:
    """
    Asymmetric STDP learning rule (Bi & Poo 1998; Song et al. 2000).

    Weight update:
        Δw_+ = A_+ · exp(-Δt / τ_+)   if Δt = t_post - t_pre > 0  (LTP)
        Δw_- = A_- · exp( Δt / τ_-)   if Δt < 0                   (LTD)

    Implements soft weight bounds via multiplicative scaling.

    Parameters
    ----------
    A_plus    : LTP amplitude
    A_minus   : LTD amplitude (positive; sign handled internally)
    tau_plus  : ms — LTP time constant
    tau_minus : ms — LTD time constant
    w_min, w_max : weight bounds
    mu        : soft-bound exponent (0 = additive, 1 = multiplicative)
    """

    def __init__(
        self,
        A_plus:    float = 0.01,
        A_minus:   float = 0.0105,   # slightly asymmetric → weight competition
        tau_plus:  float = 20.0,
        tau_minus: float = 20.0,
        w_min:     float = 0.0,
        w_max:     float = 1.0,
        mu:        float = 1.0,      # 0=additive, 1=multiplicative
    ):
        self.A_plus    = A_plus
        self.A_minus   = A_minus
        self.tau_plus  = tau_plus
        self.tau_minus = tau_minus
        self.w_min     = w_min
        self.w_max     = w_max
        self.mu        = mu

        self.trace = EligibilityTrace(tau_plus=tau_plus, tau_minus=tau_minus)
        self._weight_history: list[float] = []

    def update(
        self,
        w: float,
        dt: float,
        pre_spike: bool,
        post_spike: bool,
    ) -> float:
        """
        Advance traces and update weight.

        Returns the new weight w(t+dt).
        """
        self.trace.update(dt, pre_spike, post_spike)

        dw = 0.0
        if post_spike:   # LTP: post after pre
            dw += self.A_plus * (self.w_max - w)**self.mu * self.trace.x_pre
        if pre_spike:    # LTD: pre after post
            dw -= self.A_minus * (w - self.w_min)**self.mu * self.trace.x_post

        w_new = np.clip(w + dw, self.w_min, self.w_max)
        self._weight_history.append(w_new)
        return w_new

    def reset(self) -> None:
        self.trace.reset()
        self._weight_history.clear()

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history)


# ── 2. BCM rule ───────────────────────────────────────────────────────────
class BCMRule:
    """
    Bienenstock-Cooper-Munro (BCM) sliding threshold rule.

    Synaptic change proportional to postsynaptic activity relative to
    a dynamic modification threshold θ_M:

        dw/dt = φ(r_post, θ_M) · r_pre
        φ(y, θ) = y(y - θ)       (upward parabola)
        dθ_M/dt = (r_post² - θ_M) / τ_θ   (sliding threshold)

    LTP occurs when r_post > θ_M, LTD when r_post < θ_M.
    The sliding threshold is essential for stability and selectivity.

    Reference: Bienenstock, E.L., Cooper, L.N. & Munro, P.W. (1982).
    Theory for the development of neuron selectivity. J Neurosci, 2, 32-48.
    """

    def __init__(
        self,
        eta:     float = 0.01,    # learning rate
        tau_theta: float = 1000.0,  # ms  — threshold time constant
        w_min:   float = 0.0,
        w_max:   float = 5.0,
    ):
        self.eta       = eta
        self.tau_theta = tau_theta
        self.w_min     = w_min
        self.w_max     = w_max
        self.theta_M   = 1.0      # initial modification threshold

    def phi(self, r_post: float) -> float:
        """BCM nonlinearity: φ(y) = y(y - θ_M)."""
        return r_post * (r_post - self.theta_M)

    def update(
        self,
        w: float,
        dt: float,
        r_pre: float,
        r_post: float,
    ) -> float:
        """Update weight and sliding threshold."""
        dw = self.eta * self.phi(r_post) * r_pre
        self.theta_M += dt * (r_post**2 - self.theta_M) / self.tau_theta
        return float(np.clip(w + dw * dt, self.w_min, self.w_max))


# ── 3. Oja Hebbian rule ───────────────────────────────────────────────────
class OjaRule:
    """
    Oja learning rule — Hebbian with weight normalization.

        dw_i/dt = η · y · (x_i - y · w_i)

    where x is presynaptic input, y = Σ w_i x_i is postsynaptic output.
    The y·w_i term prevents runaway weight growth; the fixed point extracts
    the first principal component of the input distribution.

    Reference: Oja, E. (1982). J Math Biol, 15, 267-273.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = eta

    def update(
        self,
        w: np.ndarray,
        x: np.ndarray,
        y: float,
        dt: float,
    ) -> np.ndarray:
        dw = self.eta * y * (x - y * w)
        return w + dw * dt


# ── 4. Triplet STDP (Pfister & Gerstner 2006) ────────────────────────────
class TripletSTDP:
    """
    Triplet model of STDP.

    Uses pairs AND triplets of spikes for weight updates, resolving
    the discrepancy between pair-based STDP and BCM theory.

    Variables:
        r1 : fast pre-synaptic trace  (τ_+)
        r2 : slow pre-synaptic trace  (τ_x)
        o1 : fast post-synaptic trace (τ_-)
        o2 : slow post-synaptic trace (τ_y)

    At each post-spike:
        Δw_+ = A2_+ · r1 + A3_+ · r1 · o2   (pair + triplet LTP)
    At each pre-spike:
        Δw_- = A2_- · o1 + A3_- · o1 · r2   (pair + triplet LTD)

    Reference: Pfister & Gerstner (2006). J Neurosci, 26, 9673-9682.
    """

    def __init__(
        self,
        tau_plus:  float = 16.8,  # ms
        tau_minus: float = 33.7,  # ms
        tau_x:     float = 101.0, # ms
        tau_y:     float = 125.0, # ms
        A2_plus:   float = 5e-10,
        A3_plus:   float = 6.2e-3,
        A2_minus:  float = 7e-3,
        A3_minus:  float = 2.3e-4,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        self.tau_plus  = tau_plus
        self.tau_minus = tau_minus
        self.tau_x     = tau_x
        self.tau_y     = tau_y
        self.A2_plus   = A2_plus
        self.A3_plus   = A3_plus
        self.A2_minus  = A2_minus
        self.A3_minus  = A3_minus
        self.w_min     = w_min
        self.w_max     = w_max
        # traces
        self.r1 = 0.0; self.r2 = 0.0
        self.o1 = 0.0; self.o2 = 0.0

    def update(
        self,
        w: float,
        dt: float,
        pre_spike: bool,
        post_spike: bool,
    ) -> float:
        # decay traces
        self.r1 *= np.exp(-dt / self.tau_plus)
        self.r2 *= np.exp(-dt / self.tau_x)
        self.o1 *= np.exp(-dt / self.tau_minus)
        self.o2 *= np.exp(-dt / self.tau_y)

        dw = 0.0
        if post_spike:
            dw += self.A2_plus * self.r1 + self.A3_plus * self.r1 * self.o2
            self.o1 += 1.0
            self.o2 += 1.0
        if pre_spike:
            dw -= self.A2_minus * self.o1 + self.A3_minus * self.o1 * self.r2
            self.r1 += 1.0
            self.r2 += 1.0

        return float(np.clip(w + dw, self.w_min, self.w_max))

    def reset(self) -> None:
        self.r1 = self.r2 = self.o1 = self.o2 = 0.0
