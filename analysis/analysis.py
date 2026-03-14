"""
analysis.py
===========
Analytical tools for computational neuroscience simulations.

Modules
-------
  SpikeAnalysis      — ISI statistics, firing rate, burstiness, CV2
  SpectralAnalysis   — LFP power spectrum, spike-coherence, oscillation detection
  PhasePlane         — nullcline computation, fixed points, bifurcation
  PopulationAnalysis — network synchrony, pairwise correlations, PSTH
  InformationTheory  — mutual information, entropy, neural coding efficiency

All functions return structured dicts or numpy arrays for downstream plotting.

References
----------
Softky, W.R. & Koch, C. (1993). The highly irregular firing of cortical cells
    is inconsistent with temporal integration of random EPSPs. J Neurosci,
    13, 334-350.  [CV of ISI]
Shinomoto, S. et al. (2009). Relating neuronal firing patterns to functional
    differentiation of cerebral cortex. PLoS Comput Biol, 5, e1000433. [LV, IR]
van Rossum, M.C.W. (2001). A novel spike distance. Neural Comput, 13, 751-763.
Tiesinga, P.H.E. & Sejnowski, T.J. (2004). Rapid temporal modulation of
    synchrony by competition in cortical interneuron networks. Neural Comput.
"""

import numpy as np
from scipy import signal, stats
from typing import Optional


# ── Spike Analysis ────────────────────────────────────────────────────────
class SpikeAnalysis:
    """
    Comprehensive spike train statistics.

    Parameters
    ----------
    spike_times : array of spike times in ms
    duration_ms : total recording duration (ms)
    """

    def __init__(self, spike_times: np.ndarray, duration_ms: float):
        self.spikes   = np.sort(np.asarray(spike_times, dtype=float))
        self.duration = duration_ms

    # ── basic metrics ─────────────────────────────────────────────────────
    @property
    def isi(self) -> np.ndarray:
        """Inter-spike intervals (ms)."""
        return np.diff(self.spikes) if len(self.spikes) > 1 else np.array([])

    @property
    def mean_rate(self) -> float:
        """Mean firing rate (Hz)."""
        return len(self.spikes) / (self.duration * 1e-3)

    @property
    def cv_isi(self) -> float:
        """Coefficient of variation of ISI — regularity measure.
        CV=0: clock-like; CV=1: Poisson; CV>1: bursty."""
        isi = self.isi
        if len(isi) < 2:
            return float("nan")
        return float(np.std(isi) / np.mean(isi))

    @property
    def lv(self) -> float:
        """
        Local variation (LV) of ISI — sensitive to local rate changes.
        LV = (3/(n-1)) Σ (ISI_i - ISI_{i+1})² / (ISI_i + ISI_{i+1})²
        Shinomoto et al. (2009).
        """
        isi = self.isi
        if len(isi) < 3:
            return float("nan")
        ratios = ((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:])) ** 2
        return float(3.0 * np.mean(ratios))

    @property
    def ir(self) -> float:
        """
        Revised local variation (IR) — robust to rate non-stationarity.
        IR = mean( |ISI_i - ISI_{i+1}| / (ISI_i + ISI_{i+1}) ) * 3/(n-1)
        """
        isi = self.isi
        if len(isi) < 3:
            return float("nan")
        return float(3.0 * np.mean(np.abs(isi[:-1] - isi[1:]) /
                                   (isi[:-1] + isi[1:])))

    @property
    def burstiness(self) -> float:
        """
        Burstiness parameter B (Goh & Barabási 2008).
        B = (σ_ISI - μ_ISI) / (σ_ISI + μ_ISI)  ∈ [-1, 1]
        B > 0 → bursty;  B < 0 → regular;  B ≈ 0 → Poisson
        """
        isi = self.isi
        if len(isi) < 2:
            return float("nan")
        mu = np.mean(isi); sig = np.std(isi)
        return float((sig - mu) / (sig + mu)) if (sig + mu) > 0 else float("nan")

    @property
    def fano_factor(self) -> float:
        """
        Fano factor of spike counts in 50 ms windows.
        FF = Var(counts) / Mean(counts).  FF=1 → Poisson.
        """
        if self.duration < 100 or len(self.spikes) < 5:
            return float("nan")
        window = 50.0   # ms
        bins = np.arange(0, self.duration + window, window)
        counts, _ = np.histogram(self.spikes, bins=bins)
        return float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0 else float("nan")

    def isi_histogram(self, n_bins: int = 50) -> dict:
        """Normalized ISI histogram for distribution analysis."""
        isi = self.isi
        if len(isi) == 0:
            return {"bins": np.array([]), "counts": np.array([]), "pdf": np.array([])}
        counts, edges = np.histogram(isi, bins=n_bins, density=False)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        pdf = counts / (counts.sum() * np.diff(edges))
        return {"bins": bin_centers, "counts": counts, "pdf": pdf,
                "edges": edges}

    def instantaneous_rate(self, sigma_ms: float = 10.0,
                           dt: float = 1.0) -> dict:
        """
        Gaussian kernel-smoothed instantaneous firing rate.

        Parameters
        ----------
        sigma_ms : kernel SD in ms
        dt       : time resolution (ms)
        """
        t_arr = np.arange(0, self.duration, dt)
        rate  = np.zeros_like(t_arr)
        for sp in self.spikes:
            rate += np.exp(-0.5 * ((t_arr - sp) / sigma_ms) ** 2)
        rate /= (sigma_ms * np.sqrt(2 * np.pi)) * 1e-3   # → Hz
        return {"t": t_arr, "rate": rate, "sigma_ms": sigma_ms}

    def summary(self) -> dict:
        return {
            "n_spikes":    len(self.spikes),
            "mean_rate_hz": self.mean_rate,
            "cv_isi":      self.cv_isi,
            "lv":          self.lv,
            "ir":          self.ir,
            "burstiness":  self.burstiness,
            "fano_factor": self.fano_factor,
            "mean_isi_ms": float(np.mean(self.isi)) if len(self.isi) > 0 else float("nan"),
            "std_isi_ms":  float(np.std(self.isi))  if len(self.isi) > 0 else float("nan"),
        }


# ── Spectral Analysis ─────────────────────────────────────────────────────
class SpectralAnalysis:
    """
    Frequency-domain analysis of LFP / membrane voltage traces.

    Parameters
    ----------
    signal_arr : 1-D array (V or LFP)
    fs_hz      : sampling frequency in Hz (e.g. 1000/dt)
    """

    BANDS = {
        "delta":  (0.5,   4.0),
        "theta":  (4.0,   8.0),
        "alpha":  (8.0,  13.0),
        "beta":  (13.0,  30.0),
        "gamma": (30.0, 100.0),
    }

    def __init__(self, signal_arr: np.ndarray, fs_hz: float):
        self.sig = np.asarray(signal_arr, dtype=float)
        self.fs  = fs_hz

    def psd(self, nperseg: int = 512) -> dict:
        """
        Welch power spectral density estimate.
        Returns frequencies (Hz) and power (dB re 1 V²/Hz).
        """
        f, Pxx = signal.welch(self.sig, fs=self.fs, nperseg=nperseg)
        Pxx_dB = 10 * np.log10(Pxx + 1e-30)
        return {"f": f, "Pxx": Pxx, "Pxx_dB": Pxx_dB}

    def band_power(self, band: str = "gamma") -> float:
        """Relative power in a named frequency band (0–1)."""
        lo, hi = self.BANDS[band]
        psd_d = self.psd()
        f, Pxx = psd_d["f"], psd_d["Pxx"]
        total = np.trapezoid(Pxx, f)
        if total == 0:
            return 0.0
        mask  = (f >= lo) & (f <= hi)
        return float(np.trapezoid(Pxx[mask], f[mask]) / total)

    def spectrogram(self, nperseg: int = 256, noverlap: int = 192) -> dict:
        """Short-time Fourier transform spectrogram."""
        f, t, Sxx = signal.spectrogram(
            self.sig, fs=self.fs, nperseg=nperseg, noverlap=noverlap
        )
        return {"f": f, "t": t, "Sxx": 10 * np.log10(Sxx + 1e-30)}

    def dominant_frequency(self) -> float:
        """Frequency with maximum PSD (Hz)."""
        psd_d = self.psd()
        idx   = np.argmax(psd_d["Pxx"])
        return float(psd_d["f"][idx])

    def all_band_powers(self) -> dict:
        return {band: self.band_power(band) for band in self.BANDS}


# ── Phase-Plane Analysis ──────────────────────────────────────────────────
class PhasePlaneAnalysis:
    """
    Phase-plane and bifurcation tools for 2D neuron reductions.

    Works with the reduced HH system (V, n) or AdEx (V, w).
    """

    def __init__(self, neuron):
        self.neuron = neuron

    def vector_field(
        self,
        V_range: np.ndarray,
        n_range: np.ndarray,
        I_ext: float = 0.0,
    ) -> dict:
        """
        Compute (dV/dt, dn/dt) on a meshgrid for quiver plots.
        Only valid for HH-type neuron with n gating variable.
        """
        VV, NN = np.meshgrid(V_range, n_range)
        dV = np.zeros_like(VV)
        dn = np.zeros_like(NN)

        for i in range(VV.shape[0]):
            for j in range(VV.shape[1]):
                V = VV[i, j]; n_val = NN[i, j]
                # Use HH steady-state m, h for 2D reduction
                m_inf, h_inf, _ = self.neuron._steady_state(V)
                I_Na = self.neuron.g_Na * m_inf**3 * h_inf * (V - self.neuron.E_Na)
                I_K  = self.neuron.g_K  * n_val**4          * (V - self.neuron.E_K)
                I_L  = self.neuron.g_L                       * (V - self.neuron.E_L)
                dV[i, j] = (I_ext - I_Na - I_K - I_L) / self.neuron.C_m
                dn[i, j] = (self.neuron._alpha_n(V) * (1 - n_val)
                            - self.neuron._beta_n(V) * n_val)
        return {"V": VV, "n": NN, "dV": dV, "dn": dn}

    def V_nullcline(self, V_range: np.ndarray, I_ext: float = 0.0) -> np.ndarray:
        """dV/dt = 0 → solve for n on V_nullcline (approx. 2D reduction)."""
        n_nc = []
        for V in V_range:
            m_inf, h_inf, _ = self.neuron._steady_state(V)
            I_Na = self.neuron.g_Na * m_inf**3 * h_inf * (V - self.neuron.E_Na)
            I_L  = self.neuron.g_L                       * (V - self.neuron.E_L)
            # dV/dt = 0: I_ext = I_Na + I_K + I_L
            # I_K = g_K n^4 (V - E_K) → n^4 = (I_ext - I_Na - I_L) / (g_K*(V-E_K))
            num = I_ext - I_Na - I_L
            denom = self.neuron.g_K * (V - self.neuron.E_K)
            if denom == 0 or num / denom < 0:
                n_nc.append(float("nan"))
            else:
                n_nc.append((num / denom) ** 0.25)
        return np.array(n_nc)

    def n_nullcline(self, V_range: np.ndarray) -> np.ndarray:
        """dn/dt = 0 → n_∞(V)."""
        return np.array([
            self.neuron._alpha_n(V) / (self.neuron._alpha_n(V) + self.neuron._beta_n(V))
            for V in V_range
        ])


# ── Population Analysis ───────────────────────────────────────────────────
class PopulationAnalysis:
    """
    Network-level statistics from multi-neuron spike data.

    Parameters
    ----------
    spikes   : list of N arrays, each containing spike times (ms)
    duration : total simulation duration (ms)
    N_E      : number of excitatory neurons (for E/I separation)
    """

    def __init__(
        self,
        spikes:   list,
        duration: float,
        N_E:      Optional[int] = None,
    ):
        self.spikes   = [np.sort(np.asarray(s)) for s in spikes]
        self.N        = len(spikes)
        self.duration = duration
        self.N_E      = N_E or self.N

    def firing_rates(self) -> np.ndarray:
        """Per-neuron mean firing rate (Hz)."""
        return np.array([
            len(s) / (self.duration * 1e-3) for s in self.spikes
        ])

    def psth(self, bin_ms: float = 5.0) -> dict:
        """
        Peri-stimulus time histogram for full population.
        Returns: t (ms), rate (Hz), E_rate, I_rate
        """
        bins = np.arange(0, self.duration + bin_ms, bin_ms)
        all_spikes = np.concatenate(self.spikes) if self.spikes else np.array([])
        counts, edges = np.histogram(all_spikes, bins=bins)
        t    = 0.5 * (edges[:-1] + edges[1:])
        rate = counts / (self.N * bin_ms * 1e-3)   # Hz

        E_spikes = np.concatenate(self.spikes[:self.N_E]) if self.N_E > 0 else np.array([])
        I_spikes = np.concatenate(self.spikes[self.N_E:]) if self.N > self.N_E else np.array([])
        c_E, _ = np.histogram(E_spikes, bins=edges)
        c_I, _ = np.histogram(I_spikes, bins=edges)
        r_E = c_E / (max(self.N_E, 1) * bin_ms * 1e-3)
        r_I = c_I / (max(self.N - self.N_E, 1) * bin_ms * 1e-3)
        return {"t": t, "rate": rate, "E_rate": r_E, "I_rate": r_I}

    def synchrony_index(self, bin_ms: float = 1.0) -> float:
        """
        van Rossum synchrony index:
        χ² = (Var[population spike count] - Mean) / Mean²
        χ² ≈ 0 → asynchronous; χ² ≈ 1 → fully synchronous.
        """
        all_spikes = np.concatenate(self.spikes) if self.spikes else np.array([])
        if len(all_spikes) == 0:
            return 0.0
        bins   = np.arange(0, self.duration + bin_ms, bin_ms)
        counts, _ = np.histogram(all_spikes, bins=bins)
        mu  = np.mean(counts)
        var = np.var(counts)
        return float(var / mu**2) if mu > 0 else 0.0

    def mean_cv_isi(self) -> float:
        """Population-averaged coefficient of variation of ISI."""
        cvs = []
        for s in self.spikes:
            if len(s) > 2:
                isi = np.diff(s)
                if np.mean(isi) > 0:
                    cvs.append(np.std(isi) / np.mean(isi))
        return float(np.mean(cvs)) if cvs else float("nan")

    def pairwise_correlation(
        self,
        n_pairs: int = 200,
        bin_ms:  float = 5.0,
        seed:    int   = 0,
    ) -> dict:
        """
        Estimate mean pairwise spike-count correlation across random neuron pairs.
        Returns: mean_r, std_r, r_values array
        """
        rng  = np.random.default_rng(seed)
        bins = np.arange(0, self.duration + bin_ms, bin_ms)
        # bin all spike trains
        binned = np.array([
            np.histogram(s, bins=bins)[0].astype(float)
            for s in self.spikes
        ])
        n_pairs = min(n_pairs, self.N * (self.N - 1) // 2)
        pairs   = rng.choice(self.N, size=(n_pairs, 2), replace=True)
        r_vals  = []
        for i, j in pairs:
            if i == j:
                continue
            r, _ = stats.pearsonr(binned[i], binned[j])
            r_vals.append(r)
        r_arr = np.array(r_vals)
        return {
            "mean_r": float(np.nanmean(r_arr)),
            "std_r":  float(np.nanstd(r_arr)),
            "r_values": r_arr,
        }

    def network_state_classification(self) -> str:
        """
        Classify network state per Brunel (2000):
          AI  : async. irregular  (χ²≈0, CV>0.7)
          SR  : sync. regular     (χ²>0.5, CV<0.5)
          SIf : sync. irregular fast
          SIs : sync. irregular slow
        """
        chi2 = self.synchrony_index()
        cv   = self.mean_cv_isi()
        if chi2 < 0.1 and (np.isnan(cv) or cv > 0.7):
            return "AI"
        elif chi2 > 0.5 and not np.isnan(cv) and cv < 0.5:
            return "SR"
        elif chi2 > 0.2 and not np.isnan(cv) and cv > 0.7:
            return "SIf"
        else:
            return "SIs"

    def summary(self) -> dict:
        rates = self.firing_rates()
        return {
            "mean_rate_hz":   float(np.mean(rates)),
            "std_rate_hz":    float(np.std(rates)),
            "mean_cv_isi":    self.mean_cv_isi(),
            "synchrony_chi2": self.synchrony_index(),
            "network_state":  self.network_state_classification(),
        }


# ── Information Theory ────────────────────────────────────────────────────
class InformationTheory:
    """
    Basic information-theoretic measures for neural coding.
    """

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """Shannon entropy H(X) = -Σ p log₂ p (bits)."""
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        p /= p.sum()
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def mutual_information(
        response: np.ndarray,
        stimulus: np.ndarray,
        n_bins:   int = 20,
    ) -> float:
        """
        MI(R; S) via histogram estimation (bits).
        response, stimulus: 1-D arrays of equal length.
        """
        r_bins = np.linspace(response.min(), response.max() + 1e-10, n_bins + 1)
        s_bins = np.linspace(stimulus.min(), stimulus.max() + 1e-10, n_bins + 1)

        joint, _, _ = np.histogram2d(response, stimulus,
                                     bins=[r_bins, s_bins], density=True)
        p_r = joint.sum(axis=1)
        p_s = joint.sum(axis=0)
        dr  = (r_bins[1] - r_bins[0])
        ds  = (s_bins[1] - s_bins[0])
        p_r *= dr; p_s *= ds; joint *= dr * ds

        MI = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and p_r[i] > 0 and p_s[j] > 0:
                    MI += joint[i, j] * np.log2(joint[i, j] /
                                                  (p_r[i] * p_s[j]))
        return max(0.0, float(MI))

    @staticmethod
    def spike_train_entropy(
        spike_times: np.ndarray,
        duration_ms: float,
        bin_ms:      float = 5.0,
    ) -> float:
        """Binary word entropy of binned spike train (bits/bin)."""
        bins   = np.arange(0, duration_ms + bin_ms, bin_ms)
        counts, _ = np.histogram(spike_times, bins=bins)
        binary = (counts > 0).astype(int)
        n1 = binary.sum(); n0 = len(binary) - n1
        p1 = n1 / len(binary); p0 = n0 / len(binary)
        ps = np.array([p for p in [p0, p1] if p > 0])
        return float(-np.sum(ps * np.log2(ps)))
