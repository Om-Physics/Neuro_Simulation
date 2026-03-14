# Neural Network Basics — Computational Neuroscience Simulation Suite

Complete Python implementation: Hodgkin-Huxley · LIF · AdEx · AMPA/NMDA/GABA synapses
· STDP plasticity · E/I recurrent network · SQLAlchemy database · 11 publication figures.

## Quick Start

    pip install numpy scipy matplotlib sqlalchemy
    python generate_figures.py

Output: figures/fig_01_hh_action_potential.png ... fig_11_summary_dashboard.png

## Project Structure

    neurons/
        base_neuron.py       Abstract base class, SpikeRecord
        hodgkin_huxley.py    HH model (RK4, f-I curve, phase plane)
        integrate_fire.py    LIF / EIF / AdEx (presets: RS, IB, CH, FS, LTS)

    synapses/
        synapse.py           AMPA, NMDA (Mg2+ block), GABA-A, GABA-B kinetics
        plasticity.py        STDP, BCM, Oja, Triplet-STDP rules

    networks/
        network.py           Sparse random E/I LIF network (Brunel 2000)

    analysis/
        spike_analysis.py    ISI, CV, Fano, PSD, CCG, PSTH, burst detection
        visualisation.py     Figure helpers and plotting utilities

    database/
        models.py            SQLAlchemy ORM schema (7 tables)
        db.py                Engine, WAL session, SimulationRepository

    generate_figures.py      ← MAIN SCRIPT — runs everything, makes 11 figures
    run_simulations.py       Standalone simulation runner with DB persistence
    run_all.py               Full pipeline runner

## Figures Generated

  Fig 01  HH action potential: V(t), gating m/h/n, ion currents
  Fig 02  Phase plane: V-n nullclines + limit cycle
  Fig 03  f-I curves: HH vs LIF vs AdEx-RS vs AdEx-FS
  Fig 04  AdEx firing patterns: RS / IB / CH / FS / LTS
  Fig 05  Synapse kinetics: AMPA / NMDA / GABA-A / GABA-B
  Fig 06  NMDA Mg2+ block: B(V) and N-shaped I-V curve
  Fig 07  STDP: learning window, weight convergence, distribution
  Fig 08  E/I network: raster, population rates, LFP proxy
  Fig 09  ISI analysis: histogram, Poincare map, CV-ISI distribution
  Fig 10  LFP power spectrum: Welch PSD + frequency band powers
  Fig 11  Summary dashboard: 11-panel overview

## Key References

  Hodgkin & Huxley (1952)  J Physiol 117:500
  Brette & Gerstner (2005) J Neurophysiol 94:3637   [AdEx]

  ## Author

**Om Jha**  
B.Sc. Physics, St. Xavier’s College Kathmandu  
Research interests: Biophysics, Nanomedicine, Computational Physics

## License

This repository is released under the MIT License.

  Naud et al. (2008)       Biol Cybern 99:335        [AdEx presets]
  Destexhe et al. (1994)   J Comput Neurosci 1:195   [Synapses]
  Jahr & Stevens (1990)    J Neurosci 10:3178        [NMDA Mg block]
  Bi & Poo (1998)          J Neurosci 18:10464       [STDP]
  Song et al. (2000)       Nat Neurosci 3:919        [STDP competition]
  Brunel (2000)            J Comput Neurosci 8:183   [E/I network]
