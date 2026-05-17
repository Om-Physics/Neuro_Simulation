"""
plasticity.py
=============
Activity-dependent synaptic plasticity rules governing the modification
of synaptic weight w in response to pre- and postsynaptic spike activity.

Four rules are implemented:

1. STDPRule  -- Asymmetric spike-timing-dependent plasticity with
               multiplicative soft weight bounds (Bi and Poo 1998;
               Song, Miller and Abbott 2000).
2. BCMRule   -- Bienenstock-Cooper-Munro sliding threshold rule.
3. OjaRule   -- Oja's normalized Hebbian learning rule.
4. TripletSTDP -- Triplet-based STDP (Pfister and Gerstner 2006).

STDP equations:
  LTP (post after pre, delta_t > 0):
    delta_w = A_plus * (w_max - w)^mu * x_pre * delta(t - t_post)
  LTD (pre after post, delta_t < 0):
    delta_w = -A_minus * (w - w_min)^mu * x_post * delta(t - t_pre)

  Presynaptic trace  : dx_pre/dt  = -x_pre / tau_plus + spike_pre
  Postsynaptic trace : dx_post/dt = -x_post / tau_minus + spike_post

References:
  Bi, G. Q. and Poo, M. M. (1998). Synaptic modifications in cultured
      hippocampal neurons. J. Neurosci. 18(24), 10464-10472.
  Song, S., Miller, K. D. and Abbott, L. F. (2000). Competitive Hebbian
      learning through spike-timing-dependent synaptic plasticity.
      Nat. Neurosci. 3, 919-926.
  Pfister, J. P. and Gerstner, W. (2006). Triplets of spikes in a model
      of spike timing-dependent plasticity. J. Neurosci. 26(38), 9673-9682.

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np


class STDPRule:
    """
    Asymmetric STDP with multiplicative (soft) weight bounds.

    The multiplicative rule ensures weights remain in [w_min, w_max]
    and produces a unimodal steady-state weight distribution under
    Poisson input statistics.

    Parameters
    ----------
    A_plus     : LTP amplitude. Default 0.010.
    A_minus    : LTD amplitude. Default 0.0105 (> A_plus ensures net LTD
                 at equal pre/post firing rates, preventing runaway).
    tau_plus   : LTP time constant (ms). Default 20.
    tau_minus  : LTD time constant (ms). Default 20.
    w_min      : Lower weight bound. Default 0.
    w_max      : Upper weight bound. Default 1.
    mu         : Exponent for multiplicative rule (0=additive, 1=multiplicative).
                 Default 1.
    """

    def __init__(
        self,
        A_plus: float = 0.010,
        A_minus: float = 0.0105,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        mu: float = 1.0,
    ) -> None:
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        self.mu = mu

    def learning_window(self, dt_range: np.ndarray) -> np.ndarray:
        """
        Compute the STDP learning window K(delta_t).

        Parameters
        ----------
        dt_range : array of delta_t = t_post - t_pre values (ms)

        Returns
        -------
        delta_w : array of weight changes (normalized, w=0.5)
        """
        dw = np.where(
            dt_range > 0,
            self.A_plus  * 0.5**self.mu * np.exp(-np.abs(dt_range) / self.tau_plus),
            -self.A_minus * 0.5**self.mu * np.exp(-np.abs(dt_range) / self.tau_minus),
        )
        return dw

    def run(
        self,
        T: float = 15000.0,
        dt: float = 0.1,
        rate_pre: float = 20.0,
        rate_post: float = 20.0,
        w_init: float = 0.5,
        n_synapses: int = 1,
        seed: int = 42,
    ) -> dict:
        """
        Simulate STDP learning under Poisson pre- and postsynaptic firing.

        Parameters
        ----------
        T          : Simulation duration (ms).
        dt         : Time step (ms).
        rate_pre   : Presynaptic Poisson rate (Hz).
        rate_post  : Postsynaptic Poisson rate (Hz).
        w_init     : Initial weight value.
        n_synapses : Number of independent synapses to simulate.
        seed       : Random seed for reproducibility.

        Returns
        -------
        dict with keys 'w_final', 'w_history', 't', 'dw_history'
        """
        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        t_axis = np.arange(n_steps) * dt
        p_pre  = rate_pre  * dt * 1e-3
        p_post = rate_post * dt * 1e-3

        weights = np.full(n_synapses, w_init)
        x_pre  = np.zeros(n_synapses)
        x_post = np.zeros(n_synapses)

        decay_pre  = np.exp(-dt / self.tau_plus)
        decay_post = np.exp(-dt / self.tau_minus)

        w_history  = np.empty((n_steps, n_synapses))
        dw_history = np.zeros((n_steps, n_synapses))

        for i in range(n_steps):
            x_pre  *= decay_pre
            x_post *= decay_post

            spike_pre  = rng.random(n_synapses) < p_pre
            spike_post = rng.random(n_synapses) < p_post

            # LTD: presynaptic spike, x_post captures recent post activity
            dw = np.zeros(n_synapses)
            dw[spike_pre] -= (
                self.A_minus
                * ((weights[spike_pre] - self.w_min) / (self.w_max - self.w_min)) ** self.mu
                * x_post[spike_pre]
            )
            x_pre[spike_pre] += 1.0

            # LTP: postsynaptic spike, x_pre captures recent pre activity
            dw[spike_post] += (
                self.A_plus
                * ((self.w_max - weights[spike_post]) / (self.w_max - self.w_min)) ** self.mu
                * x_pre[spike_post]
            )
            x_post[spike_post] += 1.0

            weights = np.clip(weights + dw, self.w_min, self.w_max)
            w_history[i]  = weights
            dw_history[i] = dw

        return {
            "w_final": weights,
            "w_history": w_history,
            "t": t_axis,
            "dw_history": dw_history,
        }


class BCMRule:
    """
    Bienenstock-Cooper-Munro (BCM) sliding threshold rule.

    The modification threshold theta slides as a function of the
    mean squared postsynaptic activity, ensuring competitive learning.

    Reference:
        Bienenstock, E. L., Cooper, L. N. and Munro, P. W. (1982).
        Theory for the development of neuron selectivity. J. Neurosci. 2, 32-48.
    """

    def __init__(
        self,
        eta: float = 0.01,
        tau_theta: float = 1000.0,
        w_max: float = 1.0,
    ) -> None:
        self.eta = eta
        self.tau_theta = tau_theta
        self.w_max = w_max
        self.theta = 0.1

    def update(self, pre: float, post: float, dt: float) -> float:
        """Update weight and return delta_w."""
        dw = self.eta * pre * post * (post - self.theta)
        self.theta += dt / self.tau_theta * (post**2 - self.theta)
        return dw


class OjaRule:
    """
    Oja's normalized Hebbian learning rule.

    Converges to the principal component of the input distribution.

    Reference:
        Oja, E. (1982). A simplified neuron model as a principal component
        analyzer. J. Math. Biology. 15, 267-273.
    """

    def __init__(self, eta: float = 0.01) -> None:
        self.eta = eta

    def update(self, pre: float, post: float, w: float) -> float:
        """Return delta_w = eta * post * (pre - post * w)."""
        return self.eta * post * (pre - post * w)


class TripletSTDP:
    """
    Triplet spike-timing-dependent plasticity (Pfister and Gerstner 2006).

    Extends the pair-based STDP rule by including interactions between
    triplets of spikes, providing a closer match to experimental data
    from visual cortex and hippocampus.

    State variables (four traces):
      r1, r2 : pre-synaptic traces (fast and slow)
      o1, o2 : post-synaptic traces (fast and slow)

    LTP: A2_plus * o1 * r1 + A3_plus * o1 * r2
    LTD: A2_minus * r1 * o2 + A3_minus * r1 * o1  (all-to-all)
    """

    def __init__(
        self,
        A2_plus: float = 5e-3,
        A3_plus: float = 6.2e-3,
        A2_minus: float = 7e-3,
        A3_minus: float = 2.3e-4,
        tau_plus: float = 16.8,
        tau_x: float = 101.0,
        tau_minus: float = 33.7,
        tau_y: float = 125.0,
    ) -> None:
        self.A2_plus  = A2_plus
        self.A3_plus  = A3_plus
        self.A2_minus = A2_minus
        self.A3_minus = A3_minus
        self.tau_plus  = tau_plus
        self.tau_x     = tau_x
        self.tau_minus = tau_minus
        self.tau_y     = tau_y
        self.r1 = self.r2 = self.o1 = self.o2 = 0.0

    def step(self, dt: float, spike_pre: bool, spike_post: bool) -> float:
        decay = lambda x, tau: x * np.exp(-dt / tau)
        self.r1 = decay(self.r1, self.tau_plus)
        self.r2 = decay(self.r2, self.tau_x)
        self.o1 = decay(self.o1, self.tau_minus)
        self.o2 = decay(self.o2, self.tau_y)

        dw = 0.0
        if spike_post:
            dw += self.A2_plus * self.r1 + self.A3_plus * self.r2
            self.o1 += 1.0
            self.o2 += 1.0
        if spike_pre:
            dw -= self.A2_minus * self.o1 + self.A3_minus * self.r1 * self.o2
            self.r1 += 1.0
            self.r2 += 1.0
        return dw
