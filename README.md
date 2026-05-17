# Neuro_Simulation

Biophysical neural network simulation suite covering the complete hierarchy
of computational neuroscience: from ion channel kinetics to recurrent cortical
networks with synaptic plasticity.

Built for the repository: https://github.com/Om-Physics/Neuro_Simulation

---

## What This Repository Contains

This project implements seven scientifically rigorous simulation modules,
produces eleven publication-quality figures, four animations, and persists
all results to a SQLite database via SQLAlchemy.

Every model is grounded in peer-reviewed experimental data. Every parameter
can be traced to its original publication.

---

## Quick Start

```bash
git clone https://github.com/Om-Physics/Neuro_Simulation
cd Neuro_Simulation
pip install -r requirements.txt
python generate_all.py
```

To generate only figures:

```bash
python generate_all.py --figs-only
```

To generate only animations:

```bash
python generate_all.py --anims-only
```

To run tests:

```bash
python -m pytest tests/ -v
```

---

## Repository Structure

```
Neuro_Simulation/
│
├── neurons/
│   ├── base_neuron.py          Abstract base class, SpikeRecord container
│   ├── hodgkin_huxley.py       Full HH model with RK4 integration
│   └── integrate_fire.py       LIF, EIF, AdEx with six cortical presets
│
├── synapses/
│   ├── synapse.py              AMPA, NMDA (Mg2+ block), GABA-A, GABA-B
│   └── plasticity.py           STDP, BCM, Oja, Triplet-STDP rules
│
├── networks/
│   └── network.py              Sparse random E/I recurrent LIF network
│
├── analysis/
│   ├── spike_analysis.py       ISI, CV, Fano, PSD, CCG, PSTH, bursts, PLV
│   └── figures.py              All eleven publication figure generators
│
├── animations/
│   ├── animate_hh.py           Action potential anatomy animation
│   ├── animate_adex.py         Firing pattern gallery animation
│   ├── animate_stdp.py         STDP weight learning animation
│   └── animate_network.py      E/I network dynamics animation
│
├── database/
│   ├── models.py               SQLAlchemy ORM schema (seven tables)
│   └── db.py                   Engine, session scope, SimulationRepository
│
├── tests/
│   ├── test_neurons.py         Unit tests for HH, LIF, AdEx (28 tests)
│   └── test_synapses.py        Unit tests for synapses and plasticity (22 tests)
│
├── docs/
│   └── SCIENCE.md              Full scientific documentation with equations
│
├── figures/                    Output directory for PNG figures
├── animations/                 Output directory for GIF animations
├── generate_all.py             Master pipeline script
└── requirements.txt            Python dependencies
```

---

## Models Implemented

### Hodgkin-Huxley Neuron

Full four-dimensional conductance-based model with:

- Fourth-order Runge-Kutta integration at dt = 0.025 ms
- Voltage-dependent Na+ and K+ gating kinetics (m, h, n)
- Ionic current decomposition: I_Na, I_K, I_L
- f-I curve computation and phase-plane analysis
- Nullcline and limit cycle computation

Reference: Hodgkin and Huxley, J. Physiol. (1952) 117:500

### Adaptive Exponential Integrate-and-Fire

Two-dimensional reduced model with six cortical cell-type presets:

| Preset | Cell Type           | Key Property                    |
|--------|---------------------|---------------------------------|
| RS     | Regular Spiking     | Spike-frequency adaptation      |
| IB     | Intrinsic Bursting  | Burst then quiescence           |
| CH     | Chattering          | High-frequency rhythmic bursts  |
| FS     | Fast Spiking        | No adaptation, PV+ interneurons |
| LTS    | Low-Threshold Spiking| Strong subthreshold coupling   |
| TC     | Thalamo-Cortical    | Rebound bursting                |

Reference: Brette and Gerstner, J. Neurophysiol. (2005) 94:3637

### Synaptic Receptors

All using kinetic schemes from Destexhe et al. (1994):

| Receptor | tau_decay | E_rev  | Special property               |
|----------|-----------|--------|-------------------------------|
| AMPA     | ~5 ms     | 0 mV   | Fast glutamatergic excitation  |
| NMDA     | ~150 ms   | 0 mV   | Mg2+ block, coincidence detect |
| GABA-A   | ~5 ms     | -70 mV | Fast chloride inhibition       |
| GABA-B   | ~200 ms   | -95 mV | G-protein K+ channel cascade   |

### STDP Plasticity

Asymmetric Hebbian rule with multiplicative weight bounds:

- LTP when post fires after pre (delta_t > 0)
- LTD when pre fires after post (delta_t < 0)
- A_minus / A_plus = 1.05 ensures weight stability under Poisson input
- Unimodal steady-state distribution with multiplicative rule (mu=1)

Reference: Bi and Poo, J. Neurosci. (1998) 18:10464

### E/I Recurrent Network

Based on the Brunel (2000) sparse random network framework:

- 100 excitatory + 25 inhibitory LIF neurons
- Erdos-Renyi connectivity with p = 0.15
- AMPA excitatory and GABA-A inhibitory synapses
- External Poisson drive at 15 kHz

Reference: Brunel, J. Comput. Neurosci. (2000) 8:183

---

## Figures Produced

| Figure | File                              | Contents                            |
|--------|-----------------------------------|-------------------------------------|
| 01     | fig_01_hh_action_potential.png    | V(t), gating m/h/n, ionic currents  |
| 02     | fig_02_phase_plane.png            | V-n nullclines and limit cycle       |
| 03     | fig_03_fi_curves.png              | f-I curves: HH, LIF, AdEx-RS, FS   |
| 04     | fig_04_adex_patterns.png          | RS, IB, CH, FS, LTS firing patterns |
| 05     | fig_05_synapse_kinetics.png       | AMPA, NMDA, GABA-A, GABA-B kinetics |
| 06     | fig_06_nmda_mg_block.png          | Mg2+ block B(V) and N-shaped I-V    |
| 07     | fig_07_stdp.png                   | STDP window, weight evolution        |
| 08     | fig_08_network_dynamics.png       | Raster, population rates, LFP        |
| 09     | fig_09_isi_analysis.png           | ISI histogram, Poincare map, CV      |
| 10     | fig_10_lfp_psd.png                | Welch PSD and frequency band powers  |
| 11     | fig_11_summary_dashboard.png      | Overview of all major results        |

---

## Animations Produced

| Animation              | File                      | What It Shows                      |
|------------------------|---------------------------|------------------------------------|
| HH Action Potential    | hh_action_potential.gif   | V, m, h, n, and ionic currents     |
| AdEx Patterns          | adex_patterns.gif         | Five cell types simultaneously     |
| STDP Learning          | stdp_learning.gif         | Weight evolution + event raster    |
| E/I Network            | network_dynamics.gif      | Raster, rates, LFP, E/I balance    |

---

## Database

All simulation results are persisted to a SQLite database:

```python
from database.db import build_engine, SimulationRepository

engine = build_engine("neuro_sim.db")
repo   = SimulationRepository(engine)

summary = repo.run_summary(run_id=1)
print(summary["mean_rate_hz"])

all_runs = repo.list_runs(model_type="HodgkinHuxley")
```

Seven ORM tables: SimulationRun, NeuronParameters, SpikeTrainRecord,
VoltageTrace, SynapseRecord, PlasticityRecord, AnalysisResult.

---

## Scientific Validation

All models produce results consistent with the original experimental data:

- HH: AP peak near +35 mV, rheobase ~2 uA/cm^2, rate 50-100 Hz at 10 uA/cm^2
- AdEx-RS: spike-frequency adaptation, CV-ISI > 0
- AdEx-FS: perfectly regular tonic firing, CV-ISI near 0
- NMDA: N-shaped I-V curve, >90% block at -65 mV with 1 mM Mg2+
- STDP: stable weight at w ~= 0.5 under equal-rate Poisson input
- Network: E cells 20-25 Hz, I cells 25-30 Hz, gamma-dominant LFP

---

## Testing

50 unit tests covering scientific correctness, not merely code execution:

```bash
python -m pytest tests/ -v
```

Tests validate: action potential existence, gating variable bounds, RK4
stability, LIF analytical f-I match, AdEx overflow prevention, NMDA Mg2+
block magnitude, STDP weight bounds and asymmetry, network firing rates.

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
sqlalchemy >= 2.0
pytest >= 7.0
Pillow >= 9.0
```

Python 3.10 or higher required.

---

## References

Hodgkin and Huxley (1952). J. Physiol. 117:500
Lapicque (1907). J. Physiol. Pathol. Gen. 9:620
Fourcaud-Trocme et al. (2003). J. Neurosci. 23:11628
Brette and Gerstner (2005). J. Neurophysiol. 94:3637
Naud et al. (2008). Biol. Cybern. 99:335
Destexhe et al. (1994). J. Comput. Neurosci. 1:195
Jahr and Stevens (1990). J. Neurosci. 10:3178
Bi and Poo (1998). J. Neurosci. 18:10464
Song, Miller and Abbott (2000). Nat. Neurosci. 3:919
Brunel (2000). J. Comput. Neurosci. 8:183
Pfister and Gerstner (2006). J. Neurosci. 26:9673

---

## Author

Om-Physics
https://github.com/Om-Physics/Neuro_Simulation
