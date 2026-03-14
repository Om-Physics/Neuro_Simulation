"""
analysis/spike_analysis.py
===========================
Quantitative analysis of spike trains and neural population activity.

Methods
-------
  spike_train_statistics  — ISI, CV, Fano factor, firing rate
  interspike_interval     — full ISI distribution analysis
  instantaneous_rate      — kernel-smoothed firing rate estimate
  pairwise_correlation    — spike-train cross-correlogram
  population_synchrony    — χ² synchrony measure (Golomb & Rinzel 1994)
  power_spectral_density  — LFP/voltage PSD via Welch method
  fI_curve_analysis       — f-I gain, rheobase, saturation
  phase_plane_analysis    — nullclines and trajectory for 2D reductions
  burst_detection         — ISI-threshold burst segmentation
  PSTH                    — peri-stimulus time histogram

References
----------
Softky, W.R. & Koch, C. (1993). The highly irregular firing of cortical
    cells is inconsistent with temporal integration of random EPSPs.
    J Neurosci, 13, 334-350.
Golomb, D. & Rinzel, J. (1994). Clustering in globally coupled inhibitory
    neurons. Physica D, 72, 259-282.
Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge.
"""

import numpy as np
from scipy import signal, stats
from typing import Optional


# ── 1. Spike-train statistics ─────────────────────────────────────────────
def spike_train_statistics(
    spike_times: np.ndarray,
    duration_ms: float,
) -> dict:
    """
    Comprehensive spike train statistics.

    Parameters
    ----------
    spike_times : ms — array of spike times
    duration_ms : ms — total recording duration

    Returns
    -------
    dict with:
      n_spikes, mean_rate_hz, mean_isi_ms, std_isi_ms,
      cv_isi, fano_factor, burst_index, local_variation
    """
    t = np.asarray(spike_times)
    n = len(t)
    stats_out = {"n_spikes": n}

    if n == 0:
        stats_out.update({
            "mean_rate_hz": 0.0, "mean_isi_ms": None,
            "std_isi_ms": None, "cv_isi": None,
            "fano_factor": None, "burst_index": None,
            "local_variation": None,
        })
        return stats_out

    stats_out["mean_rate_hz"] = n / (duration_ms * 1e-3)

    if n < 2:
        stats_out.update({
            "mean_isi_ms": None, "std_isi_ms": None,
            "cv_isi": None, "fano_factor": None,
            "burst_index": None, "local_variation": None,
        })
        return stats_out

    isi = np.diff(t)
    stats_out["mean_isi_ms"] = float(np.mean(isi))
    stats_out["std_isi_ms"]  = float(np.std(isi))
    stats_out["cv_isi"]      = float(np.std(isi) / np.mean(isi))

    # Fano factor (spike count variability in 50 ms bins)
    stats_out["fano_factor"] = _fano_factor(t, duration_ms, bin_ms=50.0)

    # Burst index: fraction of ISIs < 10 ms
    stats_out["burst_index"] = float(np.mean(isi < 10.0))

    # Local variation LV (Shinomoto et al. 2009)
    if len(isi) > 1:
        lv_terms = (isi[:-1] - isi[1:])**2 / (isi[:-1] + isi[1:])**2
        stats_out["local_variation"] = float(3 * np.mean(lv_terms))
    else:
        stats_out["local_variation"] = None

    return stats_out


def _fano_factor(spike_times: np.ndarray, duration_ms: float, bin_ms: float) -> float:
    """Compute Fano factor (var/mean) of spike counts in bins."""
    edges = np.arange(0, duration_ms + bin_ms, bin_ms)
    counts, _ = np.histogram(spike_times, bins=edges)
    if np.mean(counts) < 1e-9:
        return float("nan")
    return float(np.var(counts) / np.mean(counts))


# ── 2. ISI distribution ───────────────────────────────────────────────────
def isi_distribution(
    spike_times: np.ndarray,
    bins: int = 50,
    density: bool = True,
) -> dict:
    """
    ISI histogram + fitted distributions (exponential, gamma, log-normal).

    Returns dict with: isi, hist, bin_edges, fits (each model's params + KS stat)
    """
    t = np.asarray(spike_times)
    if len(t) < 2:
        return {"isi": np.array([]), "hist": np.array([]), "bin_edges": np.array([])}

    isi = np.diff(t)
    hist, edges = np.histogram(isi, bins=bins, density=density)

    fits = {}
    # Exponential fit (Poisson process)
    loc_e, scale_e = stats.expon.fit(isi, floc=0)
    ks_e, _ = stats.kstest(isi, "expon", args=(loc_e, scale_e))
    fits["exponential"] = {"scale": scale_e, "ks_stat": ks_e}

    # Gamma fit (renewal process)
    a_g, loc_g, scale_g = stats.gamma.fit(isi, floc=0)
    ks_g, _ = stats.kstest(isi, "gamma", args=(a_g, loc_g, scale_g))
    fits["gamma"] = {"a": a_g, "scale": scale_g, "ks_stat": ks_g}

    # Log-normal fit
    s_ln, loc_ln, scale_ln = stats.lognorm.fit(isi, floc=0)
    ks_ln, _ = stats.kstest(isi, "lognorm", args=(s_ln, loc_ln, scale_ln))
    fits["lognormal"] = {"sigma": s_ln, "mu": np.log(scale_ln), "ks_stat": ks_ln}

    # Best fit by KS statistic
    best = min(fits, key=lambda k: fits[k]["ks_stat"])
    return {
        "isi": isi,
        "hist": hist,
        "bin_edges": edges,
        "fits": fits,
        "best_fit": best,
    }


# ── 3. Instantaneous firing rate (kernel smoothing) ───────────────────────
def instantaneous_rate(
    spike_times: np.ndarray,
    t_arr: np.ndarray,
    kernel: str = "gaussian",
    sigma_ms: float = 20.0,
) -> np.ndarray:
    """
    Kernel-smoothed instantaneous firing rate (Hz).

    Parameters
    ----------
    spike_times : ms
    t_arr       : ms — time axis for output
    kernel      : 'gaussian' | 'causal_exp' | 'box'
    sigma_ms    : bandwidth in ms

    Returns
    -------
    rate_hz : np.ndarray of length len(t_arr)
    """
    dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 0.1
    rate = np.zeros(len(t_arr))

    for sp in spike_times:
        if kernel == "gaussian":
            k = np.exp(-0.5 * ((t_arr - sp) / sigma_ms)**2)
            k /= (sigma_ms * np.sqrt(2 * np.pi)) * 1e-3  # → Hz
        elif kernel == "causal_exp":
            k = np.where(t_arr >= sp,
                         np.exp(-(t_arr - sp) / sigma_ms) / (sigma_ms * 1e-3), 0)
        else:  # box
            k = np.where(np.abs(t_arr - sp) < sigma_ms / 2,
                         1.0 / (sigma_ms * 1e-3), 0)
        rate += k

    return rate


# ── 4. Cross-correlogram ──────────────────────────────────────────────────
def cross_correlogram(
    spike_times_ref: np.ndarray,
    spike_times_target: np.ndarray,
    max_lag_ms: float = 100.0,
    bin_ms: float = 1.0,
) -> dict:
    """
    Pairwise cross-correlogram (CCG).

    Returns dict with lag_ms, counts, and normalized CCG.
    """
    bins = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    counts = np.zeros(len(bins) - 1)

    for sp_ref in spike_times_ref:
        diffs = spike_times_target - sp_ref
        hist, _ = np.histogram(diffs, bins=bins)
        counts += hist

    # remove auto-correlation peak for auto-CCG
    lag_ms = 0.5 * (bins[:-1] + bins[1:])
    norm = len(spike_times_ref) * len(spike_times_target) * bin_ms * 1e-3
    return {
        "lag_ms": lag_ms,
        "counts": counts,
        "ccg_hz": counts / norm if norm > 0 else counts,
    }


# ── 5. Population synchrony ───────────────────────────────────────────────
def population_synchrony(
    V_matrix: np.ndarray,
    bin_ms: float = 0.5,
) -> float:
    """
    χ² synchrony measure (Golomb & Rinzel 1994).

    χ² → 1 : fully synchronised
    χ² → 1/√N : asynchronous

    Parameters
    ----------
    V_matrix : (N, T) — membrane voltages of N neurons over T steps

    Returns
    -------
    float : χ² synchrony index
    """
    N, T = V_matrix.shape
    V_pop = np.mean(V_matrix, axis=0)
    var_pop = np.var(V_pop)
    mean_var = np.mean(np.var(V_matrix, axis=1))
    if mean_var < 1e-12:
        return float("nan")
    return float(np.sqrt(var_pop / mean_var))


# ── 6. Power spectral density ─────────────────────────────────────────────
def power_spectral_density(
    signal_arr: np.ndarray,
    fs_hz: float,
    nperseg: Optional[int] = None,
    band_labels: bool = True,
) -> dict:
    """
    Welch power spectral density estimate.

    Parameters
    ----------
    signal_arr : time series (LFP or V)
    fs_hz      : sampling frequency (Hz)
    nperseg    : Welch segment length (default 256 or len/4)

    Returns
    -------
    dict with: freqs, psd, dominant_freq, band_powers
    """
    if nperseg is None:
        nperseg = min(256, len(signal_arr) // 4)
    freqs, psd = signal.welch(signal_arr, fs=fs_hz, nperseg=nperseg)

    dominant_freq = float(freqs[np.argmax(psd[freqs > 0.5])])  # ignore DC

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 100),
    }
    band_powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        band_powers[name] = float(_trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

    return {
        "freqs": freqs,
        "psd":   psd,
        "dominant_freq_hz": dominant_freq,
        "band_powers": band_powers,
    }


# ── 7. f-I curve analysis ─────────────────────────────────────────────────
def fI_curve_analysis(I_arr: np.ndarray, f_arr: np.ndarray) -> dict:
    """
    Fit and analyse a frequency-current (f-I) curve.

    Fits a sigmoid:
        f(I) = f_max / (1 + exp(-(I - I_half) / k))

    Returns rheobase, gain (df/dI at I_half), f_max, I_half, k.
    """
    from scipy.optimize import curve_fit

    def sigmoid(I, f_max, I_half, k):
        return f_max / (1 + np.exp(-(I - I_half) / k))

    # Rheobase: first I giving f > 1 Hz
    firing_mask = f_arr > 1.0
    rheobase = float(I_arr[firing_mask][0]) if firing_mask.any() else float("nan")

    # Gain in linear region (Δf/ΔI between 10% and 90% of f_max)
    f_max_obs = float(np.max(f_arr))
    lin_mask = (f_arr > 0.1 * f_max_obs) & (f_arr < 0.9 * f_max_obs)
    if lin_mask.sum() > 1:
        gain = float(np.polyfit(I_arr[lin_mask], f_arr[lin_mask], 1)[0])
    else:
        gain = float("nan")

    # Sigmoid fit
    fit_params = None
    try:
        if firing_mask.sum() >= 3:
            p0 = [f_max_obs, float(I_arr[firing_mask].mean()), 1.0]
            popt, _ = curve_fit(sigmoid, I_arr, f_arr, p0=p0, maxfev=5000)
            fit_params = {"f_max": popt[0], "I_half": popt[1], "k": popt[2]}
    except Exception:
        pass

    return {
        "rheobase": rheobase,
        "gain_hz_per_uA_cm2": gain,
        "f_max_hz": f_max_obs,
        "sigmoid_fit": fit_params,
        "I": I_arr,
        "f_hz": f_arr,
    }


# ── 8. Phase plane analysis ───────────────────────────────────────────────
def phase_plane_trajectory(
    V_arr: np.ndarray,
    dt_ms: float,
) -> dict:
    """
    Compute dV/dt vs V phase-plane trajectory.

    Returns
    -------
    dict with V, dVdt (numerical derivative), and threshold estimate
    """
    dVdt = np.gradient(V_arr, dt_ms)
    # Threshold: V at maximum dV/dt (before spike peak)
    peak_idx = np.argmax(V_arr)
    pre_peak_dVdt = dVdt[:peak_idx]
    if len(pre_peak_dVdt) > 0:
        thresh_idx = np.argmax(pre_peak_dVdt)
        V_thresh_est = float(V_arr[thresh_idx])
    else:
        V_thresh_est = float("nan")

    return {
        "V": V_arr,
        "dVdt": dVdt,
        "V_thresh_estimate": V_thresh_est,
    }


# ── 9. Burst detection ────────────────────────────────────────────────────
def detect_bursts(
    spike_times: np.ndarray,
    isi_threshold_ms: float = 20.0,
    min_spikes_per_burst: int = 3,
) -> dict:
    """
    ISI-threshold burst detection (Legendy & Salcman 1985 variant).

    A burst is a sequence of ≥ min_spikes_per_burst consecutive spikes
    with ISI < isi_threshold_ms.

    Returns
    -------
    dict with: bursts (list of spike-time arrays), burst_rate_hz,
               mean_burst_duration_ms, mean_spikes_per_burst, in_burst_fraction
    """
    t = np.asarray(spike_times)
    if len(t) < 2:
        return {"bursts": [], "burst_rate_hz": 0.0,
                "mean_burst_duration_ms": 0.0, "mean_spikes_per_burst": 0.0,
                "in_burst_fraction": 0.0}

    isi = np.diff(t)
    in_burst = isi < isi_threshold_ms

    bursts = []
    i = 0
    while i < len(in_burst):
        if in_burst[i]:
            start = i
            while i < len(in_burst) and in_burst[i]:
                i += 1
            burst_spikes = t[start:i + 1]
            if len(burst_spikes) >= min_spikes_per_burst:
                bursts.append(burst_spikes)
        else:
            i += 1

    total_dur = (t[-1] - t[0]) * 1e-3 if len(t) > 1 else 1.0
    burst_spikes_count = sum(len(b) for b in bursts)

    return {
        "bursts": bursts,
        "n_bursts": len(bursts),
        "burst_rate_hz": len(bursts) / total_dur,
        "mean_burst_duration_ms": (
            float(np.mean([b[-1] - b[0] for b in bursts])) if bursts else 0.0
        ),
        "mean_spikes_per_burst": (
            float(np.mean([len(b) for b in bursts])) if bursts else 0.0
        ),
        "in_burst_fraction": burst_spikes_count / len(t) if len(t) > 0 else 0.0,
    }


# ── 10. PSTH ──────────────────────────────────────────────────────────────
def psth(
    all_spike_trains: list,
    t_start_ms: float,
    t_end_ms: float,
    bin_ms: float = 5.0,
) -> dict:
    """
    Peri-stimulus time histogram across a population.

    Parameters
    ----------
    all_spike_trains : list of np.ndarray, one per trial or neuron
    t_start_ms, t_end_ms : ms — time window
    bin_ms : ms — bin width

    Returns
    -------
    dict with: t_bins, counts, rate_hz
    """
    bins = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    all_counts = np.zeros(len(bins) - 1)
    n_trains = len(all_spike_trains)

    for train in all_spike_trains:
        c, _ = np.histogram(np.asarray(train), bins=bins)
        all_counts += c

    t_bins = 0.5 * (bins[:-1] + bins[1:])
    rate_hz = all_counts / (n_trains * bin_ms * 1e-3) if n_trains > 0 else all_counts

    return {"t_bins": t_bins, "counts": all_counts, "rate_hz": rate_hz}
