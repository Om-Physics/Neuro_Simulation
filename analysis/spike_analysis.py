"""
spike_analysis.py
=================
Comprehensive spike train analysis module providing standard measures
used in computational and systems neuroscience.

Implemented analyses:

  isi_statistics      Inter-spike interval distribution and moments
  cv_isi              Coefficient of variation of the ISI
  fano_factor         Fano factor (spike count variance / mean)
  psth                Peri-stimulus time histogram
  autocorrelogram     Single-unit autocorrelation function
  cross_correlogram   Cross-correlation between two spike trains
  power_spectrum      LFP / population rate power spectral density (Welch)
  burst_detection     ISI-threshold-based burst identification
  phase_locking       Mean vector length for phase-locking to oscillation

Author : Om-Physics
Repository : https://github.com/Om-Physics/Neuro_Simulation
"""

from __future__ import annotations
import numpy as np
from scipy import signal, stats


def isi_statistics(spike_times: np.ndarray) -> dict:
    """
    Compute inter-spike interval (ISI) statistics.

    Parameters
    ----------
    spike_times : 1D array of spike times in ms.

    Returns
    -------
    dict with keys: isi, mean, std, cv, median, skewness, min, max, count.
    """
    spike_times = np.asarray(spike_times)
    if len(spike_times) < 2:
        return {"isi": np.array([]), "mean": np.nan, "std": np.nan,
                "cv": np.nan, "count": len(spike_times)}

    isi = np.diff(spike_times)
    mu  = float(np.mean(isi))
    std = float(np.std(isi))

    return {
        "isi":      isi,
        "mean":     mu,
        "std":      std,
        "cv":       std / mu if mu > 0 else np.nan,
        "median":   float(np.median(isi)),
        "skewness": float(stats.skew(isi)),
        "kurtosis": float(stats.kurtosis(isi)),
        "min":      float(np.min(isi)),
        "max":      float(np.max(isi)),
        "count":    len(spike_times),
    }


def cv_isi(spike_times: np.ndarray) -> float:
    """
    Coefficient of variation of the ISI.

    CV = 0   : perfectly regular (clock-like) firing
    CV = 1   : Poisson-distributed ISIs (irregular, exponential)
    CV > 1   : bursty firing

    Parameters
    ----------
    spike_times : 1D array of spike times in ms.

    Returns
    -------
    float : CV-ISI value, or NaN if fewer than 2 spikes.
    """
    return isi_statistics(spike_times)["cv"]


def fano_factor(
    spike_times: np.ndarray,
    T_total: float,
    bin_size: float = 100.0,
) -> float:
    """
    Fano factor: ratio of spike count variance to mean across time bins.

    F < 1  : sub-Poisson (regular)
    F = 1  : Poisson
    F > 1  : super-Poisson (bursty)

    Parameters
    ----------
    spike_times : 1D array of spike times in ms.
    T_total     : Total recording duration (ms).
    bin_size    : Bin width (ms). Default 100.

    Returns
    -------
    float : Fano factor, or NaN if fewer than 2 bins.
    """
    bins = np.arange(0, T_total + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=bins)
    if len(counts) < 2 or np.mean(counts) == 0:
        return float("nan")
    return float(np.var(counts) / np.mean(counts))


def psth(
    spike_trains: list[np.ndarray],
    T_total: float,
    bin_size: float = 5.0,
    sigma_ms: float | None = None,
) -> dict:
    """
    Peri-stimulus time histogram (population rate).

    Parameters
    ----------
    spike_trains : list of 1D arrays, one per trial or neuron.
    T_total      : Total duration (ms).
    bin_size     : Bin width (ms). Default 5.
    sigma_ms     : Optional Gaussian smoothing sigma (ms). If None, no smoothing.

    Returns
    -------
    dict with keys 't_centers' (ms) and 'rate' (Hz).
    """
    n_units = len(spike_trains)
    bins = np.arange(0, T_total + bin_size, bin_size)
    t_centers = 0.5 * (bins[:-1] + bins[1:])

    counts = np.zeros(len(t_centers))
    for st in spike_trains:
        c, _ = np.histogram(np.asarray(st), bins=bins)
        counts += c

    rate = counts / (n_units * bin_size * 1e-3)   # Hz

    if sigma_ms is not None:
        sigma_bins = sigma_ms / bin_size
        from scipy.ndimage import gaussian_filter1d
        rate = gaussian_filter1d(rate, sigma=sigma_bins)

    return {"t_centers": t_centers, "rate": rate, "n_units": n_units}


def autocorrelogram(
    spike_times: np.ndarray,
    max_lag: float = 100.0,
    bin_size: float = 1.0,
) -> dict:
    """
    Spike autocorrelogram (SAC).

    Counts coincident spikes as a function of lag. The central bin
    (lag=0) is set to zero to exclude self-coincidences.

    Parameters
    ----------
    spike_times : 1D array of spike times in ms.
    max_lag     : Maximum lag (ms). Default 100.
    bin_size    : Bin width (ms). Default 1.

    Returns
    -------
    dict with keys 'lags' (ms) and 'counts'.
    """
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    t_centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.zeros(len(t_centers))

    for i, t_ref in enumerate(spike_times):
        diffs = spike_times - t_ref
        diffs = diffs[(diffs != 0) & (np.abs(diffs) <= max_lag)]
        c, _ = np.histogram(diffs, bins=bins)
        counts += c

    return {"lags": t_centers, "counts": counts}


def cross_correlogram(
    spike_times_a: np.ndarray,
    spike_times_b: np.ndarray,
    max_lag: float = 100.0,
    bin_size: float = 1.0,
) -> dict:
    """
    Cross-correlogram between two spike trains.

    Parameters
    ----------
    spike_times_a, spike_times_b : 1D arrays of spike times (ms).
    max_lag  : Maximum lag (ms). Default 100.
    bin_size : Bin width (ms). Default 1.

    Returns
    -------
    dict with keys 'lags' (ms) and 'counts'.
    """
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    t_centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.zeros(len(t_centers))

    for t_ref in spike_times_a:
        diffs = spike_times_b - t_ref
        diffs = diffs[np.abs(diffs) <= max_lag]
        c, _ = np.histogram(diffs, bins=bins)
        counts += c

    return {"lags": t_centers, "counts": counts}


def power_spectrum(
    signal_trace: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
) -> dict:
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    signal_trace : 1D array (LFP proxy or population rate).
    fs           : Sampling frequency (Hz).
    nperseg      : Segment length for Welch. Default 1024.
    noverlap     : Overlap samples. Default nperseg // 2.

    Returns
    -------
    dict with keys 'freqs' (Hz), 'psd', and 'band_powers' (dict by band name).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = signal.welch(
        signal_trace, fs=fs, nperseg=min(nperseg, len(signal_trace)),
        noverlap=noverlap
    )

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 100.0),
    }
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    band_powers = {
        name: float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]) * df)
        for name, (lo, hi) in bands.items()
    }

    return {"freqs": freqs, "psd": psd, "band_powers": band_powers}


def burst_detection(
    spike_times: np.ndarray,
    max_isi: float = 20.0,
    min_spikes: int = 3,
) -> dict:
    """
    Identify bursts using the ISI threshold method.

    A burst is a sequence of at least min_spikes consecutive spikes
    all separated by ISIs below max_isi.

    Parameters
    ----------
    spike_times : 1D array of spike times in ms.
    max_isi     : Maximum within-burst ISI (ms). Default 20.
    min_spikes  : Minimum spikes per burst. Default 3.

    Returns
    -------
    dict with keys 'bursts' (list of spike-time arrays), 'n_bursts',
    'mean_burst_len', 'burst_rate_hz'.
    """
    if len(spike_times) < min_spikes:
        return {"bursts": [], "n_bursts": 0, "mean_burst_len": 0.0,
                "burst_rate_hz": 0.0}

    isi = np.diff(spike_times)
    in_burst = isi < max_isi

    bursts: list[np.ndarray] = []
    i = 0
    while i < len(in_burst):
        if in_burst[i]:
            j = i
            while j < len(in_burst) and in_burst[j]:
                j += 1
            burst_spikes = spike_times[i:j+2]
            if len(burst_spikes) >= min_spikes:
                bursts.append(burst_spikes)
            i = j + 1
        else:
            i += 1

    mean_len = float(np.mean([len(b) for b in bursts])) if bursts else 0.0
    T_total  = float(spike_times[-1] - spike_times[0]) if len(spike_times) > 1 else 1.0
    burst_rate = len(bursts) / (T_total * 1e-3) if T_total > 0 else 0.0

    return {
        "bursts": bursts,
        "n_bursts": len(bursts),
        "mean_burst_len": mean_len,
        "burst_rate_hz": burst_rate,
    }


def phase_locking(
    spike_times: np.ndarray,
    oscillation_freq: float,
) -> dict:
    """
    Compute phase-locking value (mean vector length) to a reference oscillation.

    Parameters
    ----------
    spike_times     : 1D array of spike times in ms.
    oscillation_freq: Frequency of reference oscillation (Hz).

    Returns
    -------
    dict with keys 'plv' (phase locking value [0,1]), 'mean_angle' (radians).
    """
    if len(spike_times) == 0:
        return {"plv": 0.0, "mean_angle": 0.0}

    omega = 2.0 * np.pi * oscillation_freq * 1e-3   # rad/ms
    phases = (omega * spike_times) % (2.0 * np.pi)
    mean_vec = np.mean(np.exp(1j * phases))

    return {
        "plv": float(np.abs(mean_vec)),
        "mean_angle": float(np.angle(mean_vec)),
    }


def population_synchrony(
    spike_trains: list[np.ndarray],
    T_total: float,
    bin_size: float = 1.0,
) -> float:
    """
    Measure population synchrony as the correlation coefficient of pairwise
    spike count correlations averaged across all neuron pairs.

    Returns a value in [0, 1] where 1 is perfectly synchronous.
    """
    n = len(spike_trains)
    if n < 2:
        return 0.0

    bins = np.arange(0, T_total + bin_size, bin_size)
    counts = np.array([
        np.histogram(st, bins=bins)[0]
        for st in spike_trains
    ], dtype=float)

    # Pairwise Pearson correlations
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(counts[i]) > 0 and np.std(counts[j]) > 0:
                r, _ = stats.pearsonr(counts[i], counts[j])
                corrs.append(r)

    return float(np.mean(corrs)) if corrs else 0.0
