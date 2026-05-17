"""
generate_all.py
===============
Master pipeline script for the Neuro_Simulation repository.

Executes in order:
  1. All eleven publication figures (analysis/figures.py)
  2. All three animations: HH, AdEx patterns, STDP, E/I network
  3. All unit tests via pytest
  4. Database persistence demonstration
  5. Summary report printed to console

Usage:
    python generate_all.py                 # full pipeline
    python generate_all.py --figs-only     # figures only
    python generate_all.py --anims-only    # animations only
    python generate_all.py --tests-only    # tests only

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_figures():
    print("\n" + "="*60)
    print("STAGE 1  Generating publication figures")
    print("="*60)
    from analysis.figures import generate_all
    paths = generate_all(out_dir=os.path.join(ROOT, "figures"))
    print(f"\n  {len(paths)} figures saved to figures/")
    return paths


def run_animations():
    print("\n" + "="*60)
    print("STAGE 2  Generating animations")
    print("="*60)
    anim_dir = os.path.join(ROOT, "animations")
    gif_paths = []

    modules = [
        ("animate_hh",      "make_animation", dict(fps=20, speed_factor=6,  T=100.0)),
        ("animate_adex",    "make_animation", dict(fps=18, speed_factor=20, T=800.0)),
        ("animate_stdp",    "make_animation", dict(fps=18, speed_factor=30, T=10000.0)),
        ("animate_network", "make_animation", dict(fps=18, speed_factor=10, T=400.0)),
    ]

    for mod_name, fn_name, kwargs in modules:
        try:
            print(f"\n  [{mod_name}]")
            import importlib
            spec = importlib.util.spec_from_file_location(
                mod_name, os.path.join(anim_dir, f"{mod_name}.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            path = getattr(mod, fn_name)(**kwargs)
            gif_paths.append(path)
        except Exception as exc:
            print(f"  WARNING: {mod_name} failed: {exc}")

    print(f"\n  {len(gif_paths)} animations saved to animations/")
    return gif_paths


def run_tests():
    print("\n" + "="*60)
    print("STAGE 3  Running unit tests")
    print("="*60)
    try:
        import pytest
        result = pytest.main([
            os.path.join(ROOT, "tests"),
            "-v", "--tb=short", "-q",
        ])
        return result == 0
    except ImportError:
        print("  pytest not installed. Run: pip install pytest")
        return False


def run_db_demo():
    print("\n" + "="*60)
    print("STAGE 4  Database persistence demonstration")
    print("="*60)
    from database.db import build_engine, SimulationRepository
    from neurons.hodgkin_huxley import HodgkinHuxley
    from synapses.plasticity import STDPRule

    db_path = os.path.join(ROOT, "neuro_sim.db")
    engine  = build_engine(db_path)
    repo    = SimulationRepository(engine)

    hh   = HodgkinHuxley()
    data = hh.simulate_detailed(T=120.0, dt=0.025, I_ext=10.0)
    rid  = repo.save_hh_run(data, {"I_ext": 10.0, "dt": 0.025, "T": 120.0})
    s    = repo.run_summary(rid)
    print(f"  HH run saved  id={s['id']}  rate={s['mean_rate_hz']:.1f} Hz  "
          f"CV={s['cv_isi']:.3f}")

    stdp   = STDPRule()
    result = stdp.run(T=5000.0, dt=0.5, n_synapses=20, seed=0)
    pid    = repo.save_stdp_run(result, {"A_plus": 0.010, "A_minus": 0.0105,
                                          "rate_pre": 20, "rate_post": 20,
                                          "w_init": 0.5})
    print(f"  STDP run saved  id={pid}")
    print(f"  Database: {db_path}")


def print_summary(t0: float, fig_paths, gif_paths, tests_ok: bool):
    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  Figures    : {len(fig_paths)}/11")
    print(f"  Animations : {len(gif_paths)}/4")
    print(f"  Tests      : {'PASS' if tests_ok else 'FAIL'}")
    print(f"  Time       : {elapsed:.1f} s")
    print(f"\n  All outputs in: {ROOT}/")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs-only",  action="store_true")
    parser.add_argument("--anims-only", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    fig_paths, gif_paths, tests_ok = [], [], True

    if args.figs_only:
        fig_paths = run_figures()
    elif args.anims_only:
        gif_paths = run_animations()
    elif args.tests_only:
        tests_ok = run_tests()
    else:
        fig_paths = run_figures()
        gif_paths = run_animations()
        tests_ok  = run_tests()
        run_db_demo()

    print_summary(t0, fig_paths, gif_paths, tests_ok)


if __name__ == "__main__":
    main()
