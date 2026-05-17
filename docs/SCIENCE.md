# Scientific Documentation

## Neuro_Simulation: Biophysical Neural Network Simulation Suite

---

## 1. Overview

This repository implements a complete hierarchy of computational neuroscience models,
from single ion channels to recurrent neural networks. Every model is grounded in
experimental data and peer-reviewed theory. The code is designed so that each
module can be used independently or composed into larger simulations.

The hierarchy proceeds as follows:

    Ion channels (HH gating kinetics)
        |
    Single neuron membrane dynamics (HH, LIF, AdEx)
        |
    Synaptic transmission (AMPA, NMDA, GABA-A, GABA-B)
        |
    Synaptic plasticity (STDP, BCM, Oja, Triplet-STDP)
        |
    Recurrent network dynamics (E/I sparse random network)
        |
    Population analysis (ISI, CV, Fano, PSD, LFP)

---

## 2. The Hodgkin-Huxley Model

### 2.1 Historical Context

In 1952, Alan Hodgkin and Andrew Huxley published a series of five papers
describing the ionic basis of the action potential in the squid giant axon,
for which they received the Nobel Prize in Physiology in 1963. Their model
remains the gold standard for biophysical neuron simulation.

### 2.2 Governing Equations

The membrane is modelled as an electrical circuit: a capacitance Cm in parallel
with three conductance branches representing Na+, K+, and leak currents.

    Cm * dV/dt = I_ext - I_Na - I_K - I_L

    I_Na = gNa * m^3 * h * (V - ENa)     fast inward current
    I_K  = gK  * n^4      * (V - EK)     slow outward current
    I_L  = gL             * (V - EL)     passive leak

Each gating variable x in {m, h, n} satisfies first-order kinetics:

    dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x

At steady state: x_inf = alpha/(alpha + beta), tau_x = 1/(alpha + beta)

### 2.3 Biophysical Interpretation

The m^3 factor reflects three independent activation subunits of the Na+ channel.
The h factor is the inactivation gate. Because tau_m << tau_h, activation is
near-instantaneous while inactivation is slow, producing the transient inward
current responsible for the action potential upstroke.

The n^4 factor reflects four independent subunits of the delayed rectifier K+
channel. Because tau_n > tau_m, K+ activation lags Na+ activation, producing the
sustained outward current that repolarises the membrane.

### 2.4 Parameters (squid axon, 6.3 degrees Celsius)

| Parameter | Value     | Unit      |
|-----------|-----------|-----------|
| Cm        | 1.0       | uF/cm^2   |
| gNa       | 120.0     | mS/cm^2   |
| gK        | 36.0      | mS/cm^2   |
| gL        | 0.3       | mS/cm^2   |
| ENa       | +55.0     | mV        |
| EK        | -77.0     | mV        |
| EL        | -54.387   | mV        |

### 2.5 Numerical Integration

Fourth-order Runge-Kutta (RK4) is used with dt = 0.025 ms (40 kHz). The HH
system is stiff near threshold due to the exponential voltage dependence of
the rate functions. Euler integration at this time step would accumulate
O(dt) local truncation error leading to incorrect spike timing. RK4 reduces
this to O(dt^4).

---

## 3. Reduced Neuron Models

### 3.1 Leaky Integrate-and-Fire (LIF)

The LIF model reduces the membrane to a single RC circuit with a hard threshold:

    Cm * dV/dt = -gL * (V - EL) + I_ext

Spike condition: if V >= V_thresh, emit spike; V = V_reset; apply t_ref

The membrane time constant is tau_m = Cm / gL = 20 ms.

The analytical firing rate for constant input I is:

    f(I) = [t_ref - tau_m * ln(1 - gL*(V_thresh - EL)/(I - gL*(V_reset - EL)))]^(-1)

The LIF captures rate coding and subthreshold integration but misses:
spike shape, sodium channel inactivation, and spike-frequency adaptation.

### 3.2 Adaptive Exponential Integrate-and-Fire (AdEx)

The AdEx model (Brette and Gerstner 2005) adds two features to LIF:

Feature 1: Exponential spike initiation
    gL * dT * exp((V - VT) / dT)
    This reproduces the sharp Na+ channel activation observed in cortical neurons.

Feature 2: Adaptation current w(t)
    Cm * dV/dt = -gL*(V-EL) + gL*dT*exp((V-VT)/dT) - w + I_ext
    tau_w * dw/dt = a*(V-EL) - w
    Spike rule: V -> V_reset, w -> w + b

The parameters (a, b, tau_w) control the firing pattern:

| Preset | a (nS) | b (pA) | tau_w (ms) | Cell type               |
|--------|--------|--------|------------|-------------------------|
| RS     | 4.0    | 80.5   | 144        | Regular Spiking         |
| IB     | 4.0    | 80.5   | 16         | Intrinsic Bursting      |
| CH     | 4.0    | 80.5   | 5          | Chattering              |
| FS     | 0.0    | 0.0    | 40         | Fast Spiking            |
| LTS    | 8.0    | 200.0  | 200        | Low-Threshold Spiking   |

---

## 4. Synaptic Models

### 4.1 Kinetic Two-State Scheme

All fast ionotropic receptors use the kinetic scheme of Destexhe et al. (1994):

    dr/dt = alpha * [T](t) * (1 - r) - beta * r

    I_syn = g_max * r(t) * (V_post - E_rev)

where r is the fraction of open channels, [T] is transmitter concentration
(modelled as a 1 ms square pulse after each presynaptic spike), and alpha, beta
are binding/unbinding rate constants.

| Receptor | alpha (1/mM/ms) | beta (1/ms) | E_rev (mV) | tau_decay (ms) |
|----------|-----------------|-------------|------------|----------------|
| AMPA     | 1.1             | 0.19        | 0          | ~5             |
| NMDA     | 0.072           | 0.0066      | 0          | ~150           |
| GABA-A   | 5.0             | 0.18        | -70        | ~5             |

### 4.2 NMDA Mg2+ Block

The NMDA receptor has a critical voltage-dependent nonlinearity: at resting
membrane potential, extracellular Mg2+ ions physically occlude the channel pore.
This block is relieved by postsynaptic depolarisation.

    B(V) = 1 / (1 + exp(-0.062 * V) * [Mg2+] / 3.57)

At V = -65 mV with [Mg2+] = 1 mM: B = 0.08 (92% blocked)
At V =   0 mV with [Mg2+] = 1 mM: B = 0.80 (20% blocked)

This makes NMDA a molecular coincidence detector: significant current flows
only when presynaptic glutamate release AND postsynaptic depolarisation
co-occur. This is the molecular basis of Hebbian synaptic strengthening.

The resulting current-voltage relationship is N-shaped, with a region of
negative-slope conductance at hyperpolarised potentials. This bistability
is hypothesised to underlie persistent activity in working memory circuits.

### 4.3 GABA-B G-Protein Cascade

GABA-B receptors are metabotropic: they act through a G-protein second
messenger cascade to activate inwardly rectifying K+ channels. The model
uses a four-state kinetic scheme:

    dR/dt = K1 * [T] * (1-R) - K2 * R       receptor activation
    dG/dt = K3 * R - K4 * G                  G-protein activation
    g = g_max * G^n / (G^n + Kd)             Hill equation (n=4)

The Hill exponent n=4 reflects the stoichiometry of G-protein activation.
Time to peak is approximately 120 ms, much slower than GABA-A (5 ms).
E_rev = -95 mV (K+ Nernst potential).

---

## 5. Spike-Timing Dependent Plasticity

### 5.1 Learning Rule

The STDP rule (Bi and Poo 1998) updates synaptic weight w based on
the relative timing of pre- and postsynaptic spikes:

    delta_t = t_post - t_pre

    If delta_t > 0 (post after pre):  LTP
        dw = A_plus * (w_max - w)^mu * exp(-delta_t / tau_plus)

    If delta_t < 0 (pre after post):  LTD
        dw = -A_minus * (w - w_min)^mu * exp(delta_t / tau_minus)

Parameters: A_plus = 0.010, A_minus = 0.0105, tau_plus = tau_minus = 20 ms

### 5.2 Multiplicative Weight Bounds

The exponent mu = 1 implements multiplicative (soft) weight bounds. This
ensures weights remain in [w_min, w_max] while promoting a unimodal steady-
state weight distribution under uncorrelated Poisson input. The additive rule
(mu = 0) produces a bimodal distribution.

### 5.3 Asymmetry and Stability

A_minus / A_plus = 1.05 > 1 ensures net LTD at equal pre/post firing rates.
Without this asymmetry, random coincidences under Poisson input would cause
runaway potentiation. The slight depression bias produces stable intermediate
weight values.

### 5.4 Causal Interpretation

The temporal asymmetry of STDP encodes causality: a synapse is potentiated
only when the presynaptic neuron fires BEFORE the postsynaptic neuron
(consistent with the presynaptic neuron contributing to firing the post).
Synapses that do not predict postsynaptic firing are depressed.

---

## 6. E/I Recurrent Network

### 6.1 Architecture

The network implements the Brunel (2000) sparse random network:

    N_E = 100  excitatory LIF neurons (glutamatergic)
    N_I = 25   inhibitory LIF neurons (GABAergic)
    Total N = 125

    Connectivity: Erdos-Renyi random graph, p = 0.15, no autapses
    Average in-degree: p * (N-1) = 18.6 connections per neuron

    E -> E, E -> I: AMPA-like, g_EE = g_EI = 0.35 nS
    I -> E, I -> I: GABA-A-like, g_IE = g_II = 2.0 nS

    External drive: Poisson process at 15 kHz, g_ext = 0.9 nS
    DC baseline: I_dc = 220 pA per neuron

### 6.2 Network States

Brunel (2000) identified four dynamical regimes depending on the ratio
g = g_IE / g_EE and the external firing rate nu_ext:

    SR (Synchronous Regular): driven by external input, regular firing
    SI (Synchronous Irregular): network-generated synchrony
    AI (Asynchronous Irregular): balanced E/I, irregular spiking (CV ~ 1)
    Fast oscillations: gamma-band population bursts

The AI state is considered the physiologically realistic operating mode of
awake cortex. Our network operates near the SR state due to the relatively
small network size and strong Poisson drive.

### 6.3 Inhibitory Stabilisation

The inhibitory population fires slightly faster than the excitatory population
(26 Hz vs 21 Hz). This reflects the inhibitory stabilisation mechanism: strong
feedback inhibition prevents excitatory runaway and constrains the network to
a stable operating point. Removing inhibition (g_IE = 0) causes the network
to synchronise and exhibit epileptiform bursting.

### 6.4 LFP Proxy and Frequency Analysis

The mean membrane voltage of excitatory neurons serves as a proxy for the
local field potential (LFP). In real experiments, the LFP reflects the
aggregate dendritic currents of nearby neurons. Our proxy captures the
dominant oscillation frequency of the network.

Welch's power spectral density estimate reveals dominant gamma-band activity
(30-100 Hz), consistent with the PING mechanism (Pyramidal-Interneuron Network
Gamma): fast AMPA-driven excitation triggers inhibitory feedback, producing
oscillations at approximately 1 / (synaptic delay + tau_GABA).

---

## 7. Spike Train Statistics

### 7.1 Inter-Spike Interval (ISI)

The ISI distribution characterises single-neuron firing regularity:

    Regular firing (clock-like):  ISI distribution is delta-function, CV = 0
    Poisson firing (irregular):   ISI distribution is exponential, CV = 1
    Bursty firing:                ISI distribution is right-skewed, CV > 1

### 7.2 Fano Factor

The Fano factor F = Var(spike count) / Mean(spike count) across time bins:

    F < 1: sub-Poisson (regular, e.g. cortical neurons in VI)
    F = 1: Poisson process
    F > 1: super-Poisson (bursty, e.g. neurons during UP/DOWN states)

### 7.3 Phase-Locking Value (PLV)

PLV measures synchronisation between spike trains and an oscillation:

    PLV = |mean(exp(i * phi_k))| for spike phases phi_k

PLV = 0: spikes uniformly distributed across oscillation cycles
PLV = 1: all spikes locked to the same phase

---

## 8. Key References

Hodgkin, A. L. and Huxley, A. F. (1952). A quantitative description of
membrane current and its application to conduction and excitation in nerve.
Journal of Physiology, 117(4), 500-544.

Lapicque, L. (1907). Recherches quantitatives sur l'excitation electrique
des nerfs. J. Physiol. Pathol. Gen., 9, 620-635.

Fourcaud-Trocme, N. et al. (2003). How spike generation mechanisms determine
the neuronal response to fluctuating inputs. J. Neurosci. 23(37), 11628-11640.

Brette, R. and Gerstner, W. (2005). Adaptive exponential integrate-and-fire
model as an effective description of neuronal activity. J. Neurophysiol. 94,
3637-3642.

Naud, R. et al. (2008). Firing patterns in the adaptive exponential
integrate-and-fire model. Biol. Cybern. 99, 335-347.

Destexhe, A., Mainen, Z. F. and Sejnowski, T. J. (1994). Synthesis of models
for excitable membranes, synaptic transmission and neuromodulation using a
common kinetic formalism. J. Comput. Neurosci. 1, 195-230.

Jahr, C. E. and Stevens, C. F. (1990). Voltage dependence of NMDA-activated
macroscopic conductances predicted by single-channel kinetics.
J. Neurosci. 10(9), 3178-3182.

Bi, G. Q. and Poo, M. M. (1998). Synaptic modifications in cultured
hippocampal neurons: dependence on spike timing, synaptic strength, and
postsynaptic cell type. J. Neurosci. 18(24), 10464-10472.

Song, S., Miller, K. D. and Abbott, L. F. (2000). Competitive Hebbian learning
through spike-timing-dependent synaptic plasticity. Nat. Neurosci. 3, 919-926.

Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and
inhibitory spiking neurons. J. Comput. Neurosci. 8(3), 183-208.
