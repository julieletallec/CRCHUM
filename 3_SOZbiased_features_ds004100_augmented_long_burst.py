# -*- coding: utf-8 -*-
"""
ds004100 — Features SOZ-biased (Set B) + Connectivity for GCN

- Pour chaque outcome ∈ {"bad outcome","good outcome"}, patient, seizure:
    * tire K_pre, K_ict ~ Uniform{20..40}
    * preictal : K_pre DERNIERS epochs
      ictal    : K_ict PREMIERS epochs
    * calcule FEATURES NODE (Set B)
      + normalisation baseline préictale (médiane/MAD par électrode & feature)
      + CONNECTIVITÉ (cohérence & PSI 12–45 Hz) sur les mêmes epoch_ids

- Sauvegardes:
    NODE (par epoch + moyennes), RAW + baseline préictale:
      *_NODEfeatures_epoch_<eid>_raw.csv
      *_NODEfeatures_epoch_<eid>_normpre.csv
      *_NODEfeatures_avg_over_epochs_raw.csv
      *_NODEfeatures_avg_over_epochs_normpre.csv

    EDGE (par epoch + moyennes) (inchangé):
      *_EDGE_coh1245_epoch_<eid>.csv (+ matrices si SAVE_EDGE_MATRIX)
      *_EDGE_psi1245_epoch_<eid>.csv (+ matrices si SAVE_EDGE_MATRIX)
      *_EDGE_coh1245_matrix_avg_over_epochs.csv
      *_EDGE_psi1245_matrix_avg_over_epochs.csv
      *_ADJ_coh1245.csv  / *_ADJ_psi1245_sym.csv
      *_LAPL_coh1245_norm.csv / *_LAPL_psi1245_norm.csv

- Manifest global: selection_manifest_ds004100.csv
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mne
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
from scipy.signal import hilbert, find_peaks, coherence
from scipy.stats import kurtosis, skew  # <-- AJOUT

# =========================
# SUPPRESSION DES WARNINGS BRUYANTS
# =========================
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level("ERROR")

# =========================
# CONFIG
# =========================
DATASET_ID = "ds004100"
ROOT_BASE = Path("//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2")
BASE = ROOT_BASE / DATASET_ID / "sc_fc"
OUTCOMES = ["bad outcome", "good outcome"]

# =========================
# Persistent burst (sur NORMPRE)
# =========================
BURST_THR_K = 3.0   # seuil en espace normpre : mean + K*std
BURST_DECAY = 1.0   # 1.0=cumul pur ; <1 intégrateur fuyant



# Tirage aléatoire d'epochs (inclusif)
MIN_EPOCHS, MAX_EPOCHS = 20, 40
K_PREICTAL = 20
K_ICTAL = 10
RANDOM_SEED = 1337
_rng = np.random.default_rng(RANDOM_SEED)

# Parallélisme
N_JOBS_NODE = 14
N_JOBS_EDGE = 14

# Sauver les matrices complètes (utile pour GCN)
SAVE_EDGE_MATRIX = True

# Impression console
PRINT_STATUS = True
def log(msg: str):
    if PRINT_STATUS:
        print(msg)

# =========================
# Utils PSD/Filtering communs
# =========================

def compute_persistent_burst_from_nodes(
    node_pre: Optional[np.ndarray],
    node_ict: Optional[np.ndarray],
    thr_k: float = 3.0,
    decay: float = 1.0,
):
    """
    Persistent burst calculé SUR features normalisées (normpre).

    node_pre/node_ict: (k, n_ch, n_feat) en normpre.
    Seuil thr par canal×feature basé sur le préictal:
        thr = mean(pre) + thr_k * std(pre)
    burst instantané:
        inst = max(0, x - thr)
    intégration:
        S_t = decay*S_{t-1} + inst
    """
    if node_pre is None and node_ict is None:
        return None, None

    base = node_pre if node_pre is not None else node_ict
    mu = np.nanmean(base, axis=0)
    sd = np.nanstd(base, axis=0, ddof=1)
    sd[sd < 1e-9] = 1e-9
    thr = mu + thr_k * sd

    parts = []
    k_pre = 0
    if node_pre is not None:
        parts.append(node_pre)
        k_pre = node_pre.shape[0]
    if node_ict is not None:
        parts.append(node_ict)

    combined = np.concatenate(parts, axis=0)  # (T, n_ch, n_feat)
    T, n_ch, n_feat = combined.shape

    acc = np.zeros((n_ch, n_feat), dtype=float)
    out = np.zeros_like(combined, dtype=float)

    for t in range(T):
        inst = np.maximum(0.0, combined[t] - thr)
        acc = decay * acc + inst
        out[t] = acc

    burst_pre = out[:k_pre] if node_pre is not None else None
    burst_ict = out[k_pre:] if node_pre is not None else out
    return burst_pre, burst_ict



def _robust_psd_array_welch(epoch_data, sfreq, fmin, fmax, welch_nperseg=None, average='mean'):
    nyq = sfreq / 2.0
    fmax_eff = min(fmax, nyq * (1 - 1e-6))
    if fmin >= fmax_eff or fmax_eff <= 0:
        n_ch = epoch_data.shape[0]
        return np.full((n_ch, 1), np.nan, dtype=float), np.array([np.nan])

    n_times = epoch_data.shape[-1]
    if welch_nperseg is None:
        base = int(max(64, min(n_times, sfreq)))
    else:
        base = int(min(welch_nperseg, n_times))
        base = max(base, 64)
    n_per_seg = base

    n_fft = int(2 ** np.ceil(np.log2(max(64, n_per_seg))))
    max_nfft = 2 ** 17
    df = sfreq / n_fft
    lo = int(np.ceil(fmin / df))
    hi = int(np.floor(fmax_eff / df))
    while (hi < lo) and (n_fft < max_nfft):
        n_fft *= 2
        df = sfreq / n_fft
        lo = int(np.ceil(fmin / df))
        hi = int(np.floor(fmax_eff / df))
    if hi < lo:
        n_ch = epoch_data.shape[0]
        return np.full((n_ch, 1), np.nan, dtype=float), np.array([np.nan])

    psd, freqs = psd_array_welch(
        epoch_data, sfreq=sfreq, fmin=fmin, fmax=fmax_eff,
        n_per_seg=n_per_seg, n_fft=n_fft, average=average, verbose=False
    )
    return psd, freqs

def _band_power(epoch_data, sfreq, fmin, fmax):
    psd, freqs = _robust_psd_array_welch(epoch_data, sfreq, fmin, fmax, welch_nperseg=None)
    if np.isnan(freqs).all():
        return np.full(epoch_data.shape[0], np.nan)
    return np.trapezoid(psd, freqs, axis=-1)

def _rms(x, axis=-1):
    return np.sqrt(np.mean(x**2, axis=axis))

# =========================
# NODE FEATURES — Set B (SOZ-biased)
# =========================
def feature_band_ratios(epoch_data, sfreq):
    p_theta = _band_power(epoch_data, sfreq, 4., 8.)
    p_alpha = _band_power(epoch_data, sfreq, 8., 13.)
    p_beta  = _band_power(epoch_data, sfreq, 13., 30.)
    p_gamma = _band_power(epoch_data, sfreq, 30., min(80., sfreq/2-1))
    p_delta = _band_power(epoch_data, sfreq, 1., 4.)
    eps = 1e-12
    return {
        "ratio_bg_ta": (p_beta + p_gamma) / (p_theta + p_alpha + eps),
        "ratio_gamma_delta": p_gamma / (p_delta + eps),
    }

def feature_lvfa_hafa_bursts_scale_invariant(epoch_data, sfreq, p_low=30.0, p_high=70.0):
    """
    LVFA: >14 Hz + amplitude FAIBLE relative (RMS < P30 intra-epoch)
    HAFA: >13 Hz + amplitude ELEVEE relative (RMS > P70 intra-epoch)
    (invariant à l'échelle si epochs z-scorés)
    """
    rms = _rms(epoch_data, axis=1)
    p_lo = np.percentile(rms, p_low)
    p_hi = np.percentile(rms, p_high)
    p_lvfa = _band_power(epoch_data, sfreq, 14., min(150., sfreq/2-1))
    p_hafa = _band_power(epoch_data, sfreq, 13., min(150., sfreq/2-1))
    lvfa_score = (p_lvfa / (p_lvfa + 1e-9)) * (rms < p_lo).astype(float)
    hafa_score = (p_hafa / (p_hafa + 1e-9)) * (rms > p_hi).astype(float)
    return {"lvfa_score": lvfa_score, "hafa_score": hafa_score}

def feature_beta_gamma_slope(epoch_data, sfreq):
    mid = epoch_data.shape[1] // 2
    p1 = _band_power(epoch_data[:, :mid], sfreq, 13., min(80., sfreq/2-1))
    p2 = _band_power(epoch_data[:, mid:], sfreq, 13., min(80., sfreq/2-1))
    eps = 1e-12
    slope = np.log(p2 + eps) - np.log(p1 + eps)
    return {"slope_bg_log": slope}

def feature_dc_shift(epoch_data, sfreq):
    mid = epoch_data.shape[1] // 2
    m1 = epoch_data[:, :mid].mean(axis=1)
    m2 = epoch_data[:, mid:].mean(axis=1)
    delta = m2 - m1
    return {"dc_shift": delta}

def feature_sef95(epoch_data, sfreq, fmin=1., fmax=150.):
    psd, freqs = _robust_psd_array_welch(epoch_data, sfreq, fmin, min(fmax, sfreq/2-1))
    if np.isnan(freqs).all():
        return {"sef95_Hz": np.full(epoch_data.shape[0], np.nan)}
    csum = np.cumsum(psd, axis=1)
    thr = 0.95 * csum[:, -1][:, None]
    idx = (csum >= thr).argmax(axis=1)
    return {"sef95_Hz": freqs[idx]}

def feature_spike_rate_1_3(epoch_data, sfreq, z=3.5):
    """
    Détecteur simple de spikes via TKEO+zscore, puis fréquence 1–3 Hz.
    Robuste aux cas sans pics : sharpness=0, rate=0.
    """
    tkeo = epoch_data[:, 1:-1]**2 - epoch_data[:, :-2]*epoch_data[:, 2:]
    mu = tkeo.mean(axis=1, keepdims=True)
    sd = tkeo.std(axis=1, keepdims=True) + 1e-12
    zsig = (tkeo - mu) / sd

    rates = np.zeros(epoch_data.shape[0], dtype=float)
    sharp = np.zeros(epoch_data.shape[0], dtype=float)

    for c in range(epoch_data.shape[0]):
        peaks, props = find_peaks(zsig[c], height=z)
        heights = props.get("peak_heights", np.array([], dtype=float))
        sharp[c] = float(np.median(heights)) if heights.size > 0 else 0.0
        if peaks.size >= 2:
            isi = np.diff(peaks) / sfreq
            ok = (isi >= 1/3) & (isi <= 1.0)
            rates[c] = float(ok.sum())
        else:
            rates[c] = 0.0

    return {"spike_rate_1_3Hz": rates, "spike_sharpness": sharp}

def feature_polyspike_12_25(epoch_data, sfreq):
    try:
        Xb = filter_data(epoch_data, sfreq, 12., min(25., sfreq/2-1),
                         method="iir", iir_params=dict(order=4, ftype='butter'), verbose=False)
    except Exception:
        return {"polyspike_score": np.full(epoch_data.shape[0], np.nan)}
    rms = _rms(Xb)
    z_rms = (rms - rms.mean()) / (rms.std() + 1e-12)
    return {"polyspike_score": z_rms}

def feature_ll_tkeo(epoch_data, sfreq):
    ll = np.sum(np.abs(np.diff(epoch_data, axis=1)), axis=1)
    tkeo = epoch_data[:,1:-1]**2 - epoch_data[:,:-2]*epoch_data[:,2:]
    tkeo_e = tkeo.mean(axis=1)
    return {"line_length": ll, "tkeo_energy": tkeo_e}

# === Nouvelles features (issues du script CHUM) ===
def feature_high_gamma(epoch_data, sfreq):
    nyq = sfreq / 2.0
    fmin_hg = 80.0
    fmax_hg = min(150.0, nyq - 1.0)
    if fmax_hg <= fmin_hg + 1.0:
        return {
            "hg_power_80_150": np.full(epoch_data.shape[0], np.nan),
            "hg_over_gamma": np.full(epoch_data.shape[0], np.nan),
        }
    p_gamma = _band_power(epoch_data, sfreq, 30., min(80., nyq - 1.0))
    p_hg = _band_power(epoch_data, sfreq, fmin_hg, fmax_hg)
    eps = 1e-12
    return {
        "hg_power_80_150": p_hg,
        "hg_over_gamma": p_hg / (p_gamma + eps),
    }

def feature_spectral_slope(epoch_data, sfreq, fmin=2., fmax=80.):
    psd, freqs = _robust_psd_array_welch(epoch_data, sfreq, fmin, min(fmax, sfreq/2 - 1))
    n_ch, _ = psd.shape
    slopes = np.full(n_ch, np.nan, dtype=float)
    intercepts = np.full(n_ch, np.nan, dtype=float)

    log_f = np.log10(freqs + 1e-12)
    for i in range(n_ch):
        y = psd[i]
        m = np.isfinite(y) & (y > 0)
        if m.sum() < 5:
            continue
        log_p = np.log10(y[m])
        lf = log_f[m]
        a, b = np.polyfit(lf, log_p, 1)
        slopes[i] = a
        intercepts[i] = b

    return {
        "spec_slope_2_80": slopes,
        "spec_intercept_2_80": intercepts,
    }

def feature_hjorth(epoch_data, sfreq):
    x = epoch_data
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)

    var_x = np.var(x, axis=1)
    var_dx = np.var(dx, axis=1)
    var_ddx = np.var(ddx, axis=1)
    eps = 1e-12

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    complexity = np.sqrt(var_ddx / (var_dx + eps)) / (mobility + eps)

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }

def feature_kurtosis_skewness(epoch_data, sfreq):
    k = kurtosis(epoch_data, axis=1, fisher=True, bias=False, nan_policy="omit")
    s = skew(epoch_data, axis=1, bias=False, nan_policy="omit")
    return {
        "kurtosis": k,
        "skewness": s,
    }

NODE_FEATURE_FUNCS = [
    ("band_ratios",      feature_band_ratios, {}),
    ("lvfa_hafa",        feature_lvfa_hafa_bursts_scale_invariant, {}),
    ("bg_slope",         feature_beta_gamma_slope, {}),
    ("dc_shift",         feature_dc_shift, {}),
    ("sef95",            feature_sef95, {}),
    ("spike13",          feature_spike_rate_1_3, {}),
    ("polyspike1225",    feature_polyspike_12_25, {}),
    ("ll_tkeo",          feature_ll_tkeo, {}),
    ("high_gamma",       feature_high_gamma, {}),
    ("spec_slope",       feature_spectral_slope, {}),
    ("hjorth",           feature_hjorth, {}),
    ("kurt_skew",        feature_kurtosis_skewness, {}),
]

def _robust_spatial_z(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback : Z-score spatial robuste par feature (canaux = échantillons),
    médiane/MAD (1.4826*MAD); NaN -> 0.
    """
    med = df.median(axis=0)
    mad = (df - med).abs().median(axis=0)
    scale = 1.4826 * mad.replace(0, np.nan)
    z = (df - med) / scale
    return z.fillna(0.0)

def compute_node_features_for_epoch(epoch_data, sfreq, ch_names):
    cols = {}
    for name, func, kwargs in NODE_FEATURE_FUNCS:
        out = func(epoch_data, sfreq, **kwargs)
        for k, v in out.items():
            cols[k] = np.asarray(v)
    return pd.DataFrame(cols, index=ch_names)


def normalize_nodes_with_preictal_baseline(
    node_pre_raw: Optional[np.ndarray],
    node_ict_raw: Optional[np.ndarray],
):
    """
    Normalisation baseline préictale par électrode & feature :
      - médiane + 1.4826*MAD calculées sur les epochs préictaux
      - fallback sur std si MAD nul
      - z-score robuste appliqué sur preictal & ictal
      - SANS CLIPPING

    node_pre_raw : (n_pre, n_ch, n_feat) ou None
    node_ict_raw : (n_ict, n_ch, n_feat) ou None
    """
    if node_pre_raw is None:
        return None, None

    # --- Baseline : MED & MAD par électrode et feature ---
    med = np.nanmedian(node_pre_raw, axis=0)            # (n_ch, n_feat)
    mad = np.nanmedian(np.abs(node_pre_raw - med), axis=0)
    sigma = 1.4826 * mad

    # fallback : si sigma trop petit ou nan → on utilise std
    std = np.nanstd(node_pre_raw, axis=0)
    bad = (~np.isfinite(sigma)) | (sigma < 1e-9)
    sigma[bad] = std[bad] + 1e-9

    # --- Normalisation SANS CLIPPING ---
    node_pre_norm = (node_pre_raw - med) / (sigma + 1e-9)

    node_ict_norm = None
    if node_ict_raw is not None:
        node_ict_norm = (node_ict_raw - med) / (sigma + 1e-9)

    return node_pre_norm, node_ict_norm




def normalize_nodes_with_preictal_baseline_(
    node_pre_raw: Optional[np.ndarray],
    node_ict_raw: Optional[np.ndarray],
    clip_value: float = 20.0,
):
    """
    Normalisation baseline préictale par électrode & feature :
      - médiane + 1.4826*MAD calculées sur les epochs préictaux sélectionnés
      - z-score robuste appliqué sur preictal & ictal

    node_pre_raw : (n_pre, n_ch, n_feat) ou None
    node_ict_raw : (n_ict, n_ch, n_feat) ou None
    """
    if node_pre_raw is not None:
        base = node_pre_raw  # (n_pre, n_ch, n_feat)
    elif node_ict_raw is not None:
        base = node_ict_raw
    else:
        return None, None

    med = np.nanmedian(base, axis=0)                     # (n_ch, n_feat)
    mad = np.nanmedian(np.abs(base - med), axis=0)
    sigma = 1.4826 * mad

    std = np.nanstd(base, axis=0)
    bad = (~np.isfinite(sigma)) | (sigma < 1e-9)
    sigma[bad] = std[bad] + 1e-9

    node_pre_norm = None
    node_ict_norm = None

    if node_pre_raw is not None:
        node_pre_norm = (node_pre_raw - med) / (sigma + 1e-9)
        if clip_value is not None:
            node_pre_norm = np.clip(node_pre_norm, -clip_value, clip_value)

    if node_ict_raw is not None:
        node_ict_norm = (node_ict_raw - med) / (sigma + 1e-9)
        if clip_value is not None:
            node_ict_norm = np.clip(node_ict_norm, -clip_value, clip_value)

    return node_pre_norm, node_ict_norm

# =========================
# CONNECTIVITÉ — Coherence & PSI (12–45 Hz) + Laplaciens pour GCN
# =========================
def _coherence_matrix(X, sfreq, fmin=12., fmax=45.):
    n_ch, n_t = X.shape
    C = np.ones((n_ch, n_ch), dtype=float)
    nperseg = max(64, min(n_t, int(sfreq//2)))
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            try:
                f, coh = coherence(X[i], X[j], fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
                sel = (f >= fmin) & (f <= fmax) & np.isfinite(coh)
                val = float(np.nanmean(coh[sel])) if np.any(sel) else np.nan
            except Exception:
                val = np.nan
            C[i, j] = C[j, i] = val
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return C

def _stft_multitaper(X, sfreq, fmin, fmax, n_fft=None, time_bw=2.0, n_tapers=None):
    n_ch, n_t = X.shape
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(max(128, n_t))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sfreq)
    fsel = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[fsel]
    try:
        from scipy.signal.windows import dpss
        if n_tapers is None:
            n_tapers = max(1, int(2*time_bw - 1))
        tapers = dpss(n_t, NW=time_bw, Kmax=n_tapers, sym=False).astype(np.float64)
        K = tapers.shape[0]
        Xf = np.zeros((K, n_ch, freqs_sel.size), dtype=np.complex128)
        for k in range(K):
            Xw = X * tapers[k][None, :]
            F = np.fft.rfft(Xw, n=n_fft, axis=1)[:, fsel]
            Xf[k] = F
        Cxy = np.zeros((freqs_sel.size, n_ch, n_ch), dtype=np.complex128)
        for k in range(K):
            Fk = Xf[k]
            Ck = np.einsum('if,jf->fij', Fk, np.conj(Fk), optimize=True)
            Cxy += Ck
        Cxy /= K
    except Exception:
        w = np.hanning(n_t).astype(np.float64)
        Xw = X * w[None, :]
        F = np.fft.rfft(Xw, n=n_fft, axis=1)[:, fsel]
        Cxy = np.einsum('if,jf->fij', F, np.conj(F), optimize=True)
    return freqs_sel, Cxy

def _psi_matrix(X, sfreq, fmin=12., fmax=45.):
    freqs, Cxy = _stft_multitaper(X, sfreq, fmin, fmax)
    F = len(freqs)
    n_ch = X.shape[0]
    if F < 3:
        return np.zeros((n_ch, n_ch), dtype=float)
    num = np.zeros((n_ch, n_ch), dtype=np.float64)
    den = np.zeros((n_ch, n_ch), dtype=np.float64)
    for k in range(F - 1):
        S1 = Cxy[k]; S2 = Cxy[k + 1]
        cross = S2 * np.conj(S1)
        num += np.imag(cross)
        den += np.abs(S2) * np.abs(S1) + 1e-12
    PSI = num / den
    for i in range(n_ch):
        PSI[i, i] = 0.0
    PSI = (PSI - PSI.T) / 2.0
    PSI = np.nan_to_num(PSI, nan=0.0, posinf=0.0, neginf=0.0)
    return PSI

def _normalized_laplacian(A, eps=1e-9):
    A = np.asarray(A, dtype=float)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

# =========================
# DÉCOUVERTE DES SEIZURES DISPONIBLES
# =========================
FIF_RE = re.compile(r"^(.+?)_(ictal|preictal)_(\d+)_processed\.fif$", flags=re.IGNORECASE)

def list_seizures_for_phase(patient_dir: Path, phase: str) -> list[int]:
    out = set()
    if not patient_dir.exists():
        return []
    for fname in os.listdir(patient_dir):
        m = FIF_RE.match(fname)
        if not m:
            continue
        _pat, ph, seiz_str = m.groups()
        if ph.lower() != phase:
            continue
        try:
            out.add(int(seiz_str))
        except Exception:
            pass
    return sorted(out)

# =========================
# SÉLECTION D’EPOCHS
# =========================
def draw_k(total_epochs: int) -> int:
    if total_epochs <= 0:
        return 0
    k = int(_rng.integers(MIN_EPOCHS, MAX_EPOCHS + 1))
    return min(k, total_epochs)

def select_epoch_ids(total_epochs: int, phase: str, k: int):
    if k <= 0 or total_epochs <= 0:
        return []
    if phase == "preictal":
        start = max(0, total_epochs - k)
        return list(range(start, total_epochs))
    else:
        return list(range(0, min(k, total_epochs)))

# =========================
# TRAITEMENT (UN PATIENT, UNE CRISE : PRE + ICT)
# =========================
def process_one_seizure(
    in_pre: Path,
    in_ict: Path,
    out_node_pre: Path,
    out_node_ict: Path,
    out_edge_pre: Path,
    out_edge_ict: Path,
    patient_id: str,
    seizure_num: int,
) -> List[dict]:
    """
    Traite une seizure pour un patient:
      - charge preictal/ictal
      - sélectionne des epochs
      - calcule NODE features (RAW + normpre)
      - calcule PERSISTENT BURSTINT sur NORMPRE et l'ajoute aux CSV (raw + normpre)
      - calcule EDGE features (coh & PSI)
      - retourne deux lignes de manifest (preictal, ictal)
    """

    # ---------- Chargement epochs ----------
    pre_file = in_pre / patient_id / f"{patient_id}_preictal_{seizure_num}_processed.fif"
    ict_file = in_ict / patient_id / f"{patient_id}_ictal_{seizure_num}_processed.fif"

    epochs_pre = None
    epochs_ict = None
    ch_names = None
    sfreq = None

    status_pre = "ok"
    status_ict = "ok"

    if pre_file.exists():
        try:
            epochs_pre = mne.read_epochs(str(pre_file), preload=True, verbose=False)
            ch_names = epochs_pre.ch_names
            sfreq = epochs_pre.info["sfreq"]
        except Exception as e:
            status_pre = f"read_error: {e.__class__.__name__}"
            epochs_pre = None
    else:
        status_pre = "missing_fif"

    if ict_file.exists():
        try:
            epochs_ict = mne.read_epochs(str(ict_file), preload=True, verbose=False)
            if ch_names is None:
                ch_names = epochs_ict.ch_names
                sfreq = epochs_ict.info["sfreq"]
        except Exception as e:
            status_ict = f"read_error: {e.__class__.__name__}"
            epochs_ict = None
    else:
        status_ict = "missing_fif"

    # Si aucun des deux n'existe → rien à faire
    if epochs_pre is None and epochs_ict is None:
        return [
            {
                "patient": patient_id,
                "seizure": seizure_num,
                "phase": "preictal",
                "status": status_pre,
                "n_total_epochs": 0,
                "n_selected_epochs": 0,
                "selected_epoch_ids": json.dumps([]),
                "node_avg_path_raw": "",
                "node_avg_path_normpre": "",
                "edge_avg_path_coh": "",
                "edge_avg_path_psi": "",
            },
            {
                "patient": patient_id,
                "seizure": seizure_num,
                "phase": "ictal",
                "status": status_ict,
                "n_total_epochs": 0,
                "n_selected_epochs": 0,
                "selected_epoch_ids": json.dumps([]),
                "node_avg_path_raw": "",
                "node_avg_path_normpre": "",
                "edge_avg_path_coh": "",
                "edge_avg_path_psi": "",
            },
        ]

    # ---------- Sélection d'epochs ----------
    total_pre = len(epochs_pre) if epochs_pre is not None else 0
    total_ict = len(epochs_ict) if epochs_ict is not None else 0

    k_pre = K_PREICTAL if total_pre > 0 else 0
    k_ict = K_ICTAL if total_ict > 0 else 0

    pre_ids = select_epoch_ids(total_pre, "preictal", k_pre) if k_pre > 0 else []
    ict_ids = select_epoch_ids(total_ict, "ictal",   k_ict) if k_ict > 0 else []

    # ---------- NODE FEATURES RAW ----------
    out_node_pre_patient = out_node_pre / patient_id
    out_node_ict_patient = out_node_ict / patient_id
    out_node_pre_patient.mkdir(parents=True, exist_ok=True)
    out_node_ict_patient.mkdir(parents=True, exist_ok=True)

    node_pre_dfs_raw = []
    node_ict_dfs_raw = []

    # fonctions pour joblib
    def _one_node_pre(eid):
        X = epochs_pre[eid]._data[0]
        return eid, compute_node_features_for_epoch(X, sfreq, ch_names)

    def _one_node_ict(eid):
        X = epochs_ict[eid]._data[0]
        return eid, compute_node_features_for_epoch(X, sfreq, ch_names)

    if pre_ids and epochs_pre is not None:
        res_pre_nodes = Parallel(n_jobs=N_JOBS_NODE)(
            delayed(_one_node_pre)(eid) for eid in pre_ids
        )
        res_pre_nodes.sort(key=lambda t: t[0])  # tri par eid
        node_pre_dfs_raw = [df for eid, df in res_pre_nodes]
    else:
        res_pre_nodes = []

    if ict_ids and epochs_ict is not None:
        res_ict_nodes = Parallel(n_jobs=N_JOBS_NODE)(
            delayed(_one_node_ict)(eid) for eid in ict_ids
        )
        res_ict_nodes.sort(key=lambda t: t[0])
        node_ict_dfs_raw = [df for eid, df in res_ict_nodes]
    else:
        res_ict_nodes = []

    # ---------- Empilement en arrays ----------
    node_pre_raw = None
    node_ict_raw = None
    node_cols = None

    if node_pre_dfs_raw:
        node_cols = node_pre_dfs_raw[0].columns
        node_pre_raw = np.stack([df[node_cols].values for df in node_pre_dfs_raw], axis=0)

    if node_ict_dfs_raw:
        if node_cols is None:
            node_cols = node_ict_dfs_raw[0].columns
        node_ict_raw = np.stack([df[node_cols].values for df in node_ict_dfs_raw], axis=0)

    # ---------- Normalisation baseline préictale ----------
    node_pre_norm = None
    node_ict_norm = None

    if node_pre_raw is not None:
        node_pre_norm, node_ict_norm = normalize_nodes_with_preictal_baseline(
            node_pre_raw, node_ict_raw
        )
    else:
        # fallback : z-score spatial par epoch si pas de préictal
        if node_ict_dfs_raw and node_cols is not None:
            node_ict_norm = np.stack(
                [_robust_spatial_z(df[node_cols].copy()).values for df in node_ict_dfs_raw],
                axis=0,
            )

    # =====================================================================
    # ---------------  AJOUT: PERSISTENT BURSTINT SUR NORMPRE --------------
    # =====================================================================
    burst_pre = burst_ict = None
    burst_cols = None

    if node_cols is not None and (node_pre_norm is not None or node_ict_norm is not None):
        burst_pre, burst_ict = compute_persistent_burst_from_nodes(
            node_pre_norm, node_ict_norm,
            thr_k=BURST_THR_K,
            decay=BURST_DECAY,
        )
        burst_cols = [f"burstint_{c}" for c in list(node_cols)]

    node_cols_ext = list(node_cols) if node_cols is not None else None
    if node_cols_ext is not None and burst_cols is not None:
        node_cols_ext = node_cols_ext + burst_cols
    # =====================================================================

    # ---------- Sauvegarde NODE par epoch ----------
    # PREICTAL
    if node_pre_dfs_raw:
        for (eid, df_raw), idx in zip(res_pre_nodes, range(len(node_pre_dfs_raw))):

            # RAW: on garde les features raw + on ajoute burstint calculé sur normpre
            if node_cols_ext is not None and burst_pre is not None and node_pre_norm is not None:
                raw_base = df_raw[list(node_cols)].values
                raw_ext = np.concatenate([raw_base, burst_pre[idx]], axis=1)
                df_raw_ext = pd.DataFrame(raw_ext, index=ch_names, columns=node_cols_ext)
                df_raw_ext.to_csv(
                    out_node_pre_patient / f"{patient_id}_preictal_{seizure_num}_NODEfeatures_epoch_{eid}_raw.csv",
                    index=True,
                )
            else:
                df_raw.to_csv(
                    out_node_pre_patient / f"{patient_id}_preictal_{seizure_num}_NODEfeatures_epoch_{eid}_raw.csv",
                    index=True,
                )

            # NORMPRE: features normpre + burstint (sur la même échelle normpre)
            if node_pre_norm is not None:
                if node_cols_ext is not None and burst_pre is not None:
                    norm_ext = np.concatenate([node_pre_norm[idx], burst_pre[idx]], axis=1)
                    df_norm = pd.DataFrame(norm_ext, index=ch_names, columns=node_cols_ext)
                else:
                    df_norm = pd.DataFrame(node_pre_norm[idx], index=ch_names, columns=node_cols)

                df_norm.to_csv(
                    out_node_pre_patient / f"{patient_id}_preictal_{seizure_num}_NODEfeatures_epoch_{eid}_normpre.csv",
                    index=True,
                )

    # ICTAL
    if node_ict_dfs_raw:
        for (eid, df_raw), idx in zip(res_ict_nodes, range(len(node_ict_dfs_raw))):

            # RAW: raw + burstint (calculé sur normpre)
            if node_cols_ext is not None and burst_ict is not None and node_ict_norm is not None:
                raw_base = df_raw[list(node_cols)].values
                raw_ext = np.concatenate([raw_base, burst_ict[idx]], axis=1)
                df_raw_ext = pd.DataFrame(raw_ext, index=ch_names, columns=node_cols_ext)
                df_raw_ext.to_csv(
                    out_node_ict_patient / f"{patient_id}_ictal_{seizure_num}_NODEfeatures_epoch_{eid}_raw.csv",
                    index=True,
                )
            else:
                df_raw.to_csv(
                    out_node_ict_patient / f"{patient_id}_ictal_{seizure_num}_NODEfeatures_epoch_{eid}_raw.csv",
                    index=True,
                )

            # NORMPRE
            if node_ict_norm is not None:
                if node_cols_ext is not None and burst_ict is not None:
                    norm_ext = np.concatenate([node_ict_norm[idx], burst_ict[idx]], axis=1)
                    df_norm = pd.DataFrame(norm_ext, index=ch_names, columns=node_cols_ext)
                else:
                    df_norm = pd.DataFrame(node_ict_norm[idx], index=ch_names, columns=node_cols)

                df_norm.to_csv(
                    out_node_ict_patient / f"{patient_id}_ictal_{seizure_num}_NODEfeatures_epoch_{eid}_normpre.csv",
                    index=True,
                )

    # ---------- Moyennes NODE sur epochs ----------
    node_pre_avg_raw_path = ""
    node_pre_avg_norm_path = ""
    node_ict_avg_raw_path = ""
    node_ict_avg_norm_path = ""

    if node_pre_raw is not None and node_cols is not None:
        node_pre_avg_raw = np.nanmean(node_pre_raw, axis=0)

        if node_cols_ext is not None and burst_pre is not None and node_pre_norm is not None:
            burst_pre_avg = np.nanmean(burst_pre, axis=0)
            avg_raw_ext = np.concatenate([node_pre_avg_raw, burst_pre_avg], axis=1)
            node_pre_avg_df_raw = pd.DataFrame(avg_raw_ext, index=ch_names, columns=node_cols_ext)
        else:
            node_pre_avg_df_raw = pd.DataFrame(node_pre_avg_raw, index=ch_names, columns=node_cols)

        node_pre_avg_raw_path = out_node_pre_patient / f"{patient_id}_preictal_{seizure_num}_NODEfeatures_avg_over_epochs_raw.csv"
        node_pre_avg_df_raw.to_csv(node_pre_avg_raw_path, index=True)

        if node_pre_norm is not None:
            node_pre_avg_norm = np.nanmean(node_pre_norm, axis=0)

            if node_cols_ext is not None and burst_pre is not None:
                burst_pre_avg = np.nanmean(burst_pre, axis=0)
                avg_norm_ext = np.concatenate([node_pre_avg_norm, burst_pre_avg], axis=1)
                node_pre_avg_df_norm = pd.DataFrame(avg_norm_ext, index=ch_names, columns=node_cols_ext)
            else:
                node_pre_avg_df_norm = pd.DataFrame(node_pre_avg_norm, index=ch_names, columns=node_cols)

            node_pre_avg_norm_path = out_node_pre_patient / f"{patient_id}_preictal_{seizure_num}_NODEfeatures_avg_over_epochs_normpre.csv"
            node_pre_avg_df_norm.to_csv(node_pre_avg_norm_path, index=True)

    if node_ict_raw is not None and node_cols is not None:
        node_ict_avg_raw = np.nanmean(node_ict_raw, axis=0)

        if node_cols_ext is not None and burst_ict is not None and node_ict_norm is not None:
            burst_ict_avg = np.nanmean(burst_ict, axis=0)
            avg_raw_ext = np.concatenate([node_ict_avg_raw, burst_ict_avg], axis=1)
            node_ict_avg_df_raw = pd.DataFrame(avg_raw_ext, index=ch_names, columns=node_cols_ext)
        else:
            node_ict_avg_df_raw = pd.DataFrame(node_ict_avg_raw, index=ch_names, columns=node_cols)

        node_ict_avg_raw_path = out_node_ict_patient / f"{patient_id}_ictal_{seizure_num}_NODEfeatures_avg_over_epochs_raw.csv"
        node_ict_avg_df_raw.to_csv(node_ict_avg_raw_path, index=True)

        if node_ict_norm is not None:
            node_ict_avg_norm = np.nanmean(node_ict_norm, axis=0)

            if node_cols_ext is not None and burst_ict is not None:
                burst_ict_avg = np.nanmean(burst_ict, axis=0)
                avg_norm_ext = np.concatenate([node_ict_avg_norm, burst_ict_avg], axis=1)
                node_ict_avg_df_norm = pd.DataFrame(avg_norm_ext, index=ch_names, columns=node_cols_ext)
            else:
                node_ict_avg_df_norm = pd.DataFrame(node_ict_avg_norm, index=ch_names, columns=node_cols)

            node_ict_avg_norm_path = out_node_ict_patient / f"{patient_id}_ictal_{seizure_num}_NODEfeatures_avg_over_epochs_normpre.csv"
            node_ict_avg_df_norm.to_csv(node_ict_avg_norm_path, index=True)

    # ---------- EDGE FEATURES (comme avant, phase par phase) ----------
    out_edge_pre_patient = out_edge_pre / patient_id
    out_edge_ict_patient = out_edge_ict / patient_id
    out_edge_pre_patient.mkdir(parents=True, exist_ok=True)
    out_edge_ict_patient.mkdir(parents=True, exist_ok=True)

    edge_pre_results = []
    edge_ict_results = []

    # PREICTAL
    if epochs_pre is not None and pre_ids:
        def _one_edge_pre(eid):
            X = epochs_pre[eid]._data[0]
            Ccoh = _coherence_matrix(X, sfreq, fmin=12., fmax=45.)
            PSI  = _psi_matrix(X, sfreq, fmin=12., fmax=45.)
            return eid, Ccoh, PSI

        edge_pre_results = Parallel(n_jobs=N_JOBS_EDGE)(
            delayed(_one_edge_pre)(eid) for eid in pre_ids
        )

        for eid, Ccoh, PSI in edge_pre_results:
            n = len(ch_names)

            # Coherence (undirected)
            src, tgt, w = [], [], []
            for i in range(n):
                for j in range(i + 1, n):
                    src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(Ccoh[i, j]))
            pd.DataFrame({"source": src, "target": tgt, "coh_12_45": w}).to_csv(
                out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_coh1245_epoch_{eid}.csv", index=False
            )

            # PSI (directed)
            src, tgt, w = [], [], []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(PSI[i, j]))
            pd.DataFrame({"source": src, "target": tgt, "psi_12_45": w}).to_csv(
                out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_psi1245_epoch_{eid}.csv", index=False
            )

            if SAVE_EDGE_MATRIX:
                pd.DataFrame(Ccoh, index=ch_names, columns=ch_names).to_csv(
                    out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_coh1245_matrix_epoch_{eid}.csv", index=True
                )
                pd.DataFrame(PSI, index=ch_names, columns=ch_names).to_csv(
                    out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_psi1245_matrix_epoch_{eid}.csv", index=True
                )

        Ccoh_pre_avg = np.nanmean(np.stack([C for _, C, _ in edge_pre_results], axis=0), axis=0)
        PSI_pre_avg  = np.nanmean(np.stack([P for _, _, P in edge_pre_results], axis=0), axis=0)

        A_coh_pre = Ccoh_pre_avg.copy()
        np.fill_diagonal(A_coh_pre, 0.0)
        PSI_pos_pre = np.clip(PSI_pre_avg, 0, None)
        A_psi_sym_pre = PSI_pos_pre + PSI_pos_pre.T
        np.fill_diagonal(A_psi_sym_pre, 0.0)

        L_coh_pre = _normalized_laplacian(A_coh_pre)
        L_psi_pre = _normalized_laplacian(A_psi_sym_pre)

        edge_pre_avg_coh_path = out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_coh1245_matrix_avg_over_epochs.csv"
        edge_pre_avg_psi_path = out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_EDGE_psi1245_matrix_avg_over_epochs.csv"
        pd.DataFrame(Ccoh_pre_avg, index=ch_names, columns=ch_names).to_csv(edge_pre_avg_coh_path, index=True)
        pd.DataFrame(PSI_pre_avg,  index=ch_names, columns=ch_names).to_csv(edge_pre_avg_psi_path, index=True)

        pd.DataFrame(A_coh_pre, index=ch_names, columns=ch_names).to_csv(
            out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_ADJ_coh1245.csv", index=True
        )
        pd.DataFrame(A_psi_sym_pre, index=ch_names, columns=ch_names).to_csv(
            out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_ADJ_psi1245_sym.csv", index=True
        )
        pd.DataFrame(L_coh_pre, index=ch_names, columns=ch_names).to_csv(
            out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_LAPL_coh1245_norm.csv", index=True
        )
        pd.DataFrame(L_psi_pre, index=ch_names, columns=ch_names).to_csv(
            out_edge_pre_patient / f"{patient_id}_preictal_{seizure_num}_LAPL_psi1245_norm.csv", index=True
        )
    else:
        edge_pre_avg_coh_path = ""
        edge_pre_avg_psi_path = ""

    # ICTAL
    if epochs_ict is not None and ict_ids:
        def _one_edge_ict(eid):
            X = epochs_ict[eid]._data[0]
            Ccoh = _coherence_matrix(X, sfreq, fmin=12., fmax=45.)
            PSI  = _psi_matrix(X, sfreq, fmin=12., fmax=45.)
            return eid, Ccoh, PSI

        edge_ict_results = Parallel(n_jobs=N_JOBS_EDGE)(
            delayed(_one_edge_ict)(eid) for eid in ict_ids
        )

        for eid, Ccoh, PSI in edge_ict_results:
            n = len(ch_names)

            # Coherence (undirected)
            src, tgt, w = [], [], []
            for i in range(n):
                for j in range(i + 1, n):
                    src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(Ccoh[i, j]))
            pd.DataFrame({"source": src, "target": tgt, "coh_12_45": w}).to_csv(
                out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_coh1245_epoch_{eid}.csv", index=False
            )

            # PSI (directed)
            src, tgt, w = [], [], []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(PSI[i, j]))
            pd.DataFrame({"source": src, "target": tgt, "psi_12_45": w}).to_csv(
                out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_psi1245_epoch_{eid}.csv", index=False
            )

            if SAVE_EDGE_MATRIX:
                pd.DataFrame(Ccoh, index=ch_names, columns=ch_names).to_csv(
                    out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_coh1245_matrix_epoch_{eid}.csv", index=True
                )
                pd.DataFrame(PSI, index=ch_names, columns=ch_names).to_csv(
                    out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_psi1245_matrix_epoch_{eid}.csv", index=True
                )

        Ccoh_ict_avg = np.nanmean(np.stack([C for _, C, _ in edge_ict_results], axis=0), axis=0)
        PSI_ict_avg  = np.nanmean(np.stack([P for _, _, P in edge_ict_results], axis=0), axis=0)

        A_coh_ict = Ccoh_ict_avg.copy()
        np.fill_diagonal(A_coh_ict, 0.0)
        PSI_pos_ict = np.clip(PSI_ict_avg, 0, None)
        A_psi_sym_ict = PSI_pos_ict + PSI_pos_ict.T
        np.fill_diagonal(A_psi_sym_ict, 0.0)

        L_coh_ict = _normalized_laplacian(A_coh_ict)
        L_psi_ict = _normalized_laplacian(A_psi_sym_ict)

        edge_ict_avg_coh_path = out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_coh1245_matrix_avg_over_epochs.csv"
        edge_ict_avg_psi_path = out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_EDGE_psi1245_matrix_avg_over_epochs.csv"
        pd.DataFrame(Ccoh_ict_avg, index=ch_names, columns=ch_names).to_csv(edge_ict_avg_coh_path, index=True)
        pd.DataFrame(PSI_ict_avg,  index=ch_names, columns=ch_names).to_csv(edge_ict_avg_psi_path, index=True)

        pd.DataFrame(A_coh_ict, index=ch_names, columns=ch_names).to_csv(
            out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_ADJ_coh1245.csv", index=True
        )
        pd.DataFrame(A_psi_sym_ict, index=ch_names, columns=ch_names).to_csv(
            out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_ADJ_psi1245_sym.csv", index=True
        )
        pd.DataFrame(L_coh_ict, index=ch_names, columns=ch_names).to_csv(
            out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_LAPL_coh1245_norm.csv", index=True
        )
        pd.DataFrame(L_psi_ict, index=ch_names, columns=ch_names).to_csv(
            out_edge_ict_patient / f"{patient_id}_ictal_{seizure_num}_LAPL_psi1245_norm.csv", index=True
        )
    else:
        edge_ict_avg_coh_path = ""
        edge_ict_avg_psi_path = ""

    # ---------- Manifest rows ----------
    pre_row = {
        "patient": patient_id,
        "seizure": seizure_num,
        "phase": "preictal",
        "status": status_pre if total_pre == 0 else "ok",
        "n_total_epochs": int(total_pre),
        "n_selected_epochs": int(len(pre_ids)),
        "selected_epoch_ids": json.dumps(pre_ids),
        "node_avg_path_raw": str(node_pre_avg_raw_path) if node_pre_avg_raw_path else "",
        "node_avg_path_normpre": str(node_pre_avg_norm_path) if node_pre_avg_norm_path else "",
        "edge_avg_path_coh": str(edge_pre_avg_coh_path) if edge_pre_avg_coh_path else "",
        "edge_avg_path_psi": str(edge_pre_avg_psi_path) if edge_pre_avg_psi_path else "",
    }

    ict_row = {
        "patient": patient_id,
        "seizure": seizure_num,
        "phase": "ictal",
        "status": status_ict if total_ict == 0 else "ok",
        "n_total_epochs": int(total_ict),
        "n_selected_epochs": int(len(ict_ids)),
        "selected_epoch_ids": json.dumps(ict_ids),
        "node_avg_path_raw": str(node_ict_avg_raw_path) if node_ict_avg_raw_path else "",
        "node_avg_path_normpre": str(node_ict_avg_norm_path) if node_ict_avg_norm_path else "",
        "edge_avg_path_coh": str(edge_ict_avg_coh_path) if edge_ict_avg_coh_path else "",
        "edge_avg_path_psi": str(edge_ict_avg_psi_path) if edge_ict_avg_psi_path else "",
    }

    return [pre_row, ict_row]


# =========================
# MAIN
# =========================
def main():
    log(f"[INFO] RNG seed = {RANDOM_SEED} | epochs per phase ∈ [{MIN_EPOCHS}, {MAX_EPOCHS}]")
    log("[INFO] Node = Set B (SOZ-biased) avec baseline préictale. Edge = Coherence & PSI in 12–45 Hz.")
    manifest_rows = []

    for outcome in OUTCOMES:
        in_pre  = BASE / outcome / "preictal"
        in_ict  = BASE / outcome / "ictal"
        out_node_pre = in_pre / "NODE_features_SOZ_augmented_20_10_burst"
        out_node_ict = in_ict / "NODE_features_SOZ_augmented_20_10_burst"
        out_edge_pre = in_pre / "EDGE_features_SOZ_augmented_20_10_burst"
        out_edge_ict = in_ict / "EDGE_features_SOZ_augmented_20_10_burst"
        for d in [out_node_pre, out_node_ict, out_edge_pre, out_edge_ict]:
            d.mkdir(parents=True, exist_ok=True)

        pre_patients = [p.name for p in in_pre.iterdir() if p.is_dir()] if in_pre.exists() else []
        ict_patients = [p.name for p in in_ict.iterdir() if p.is_dir()] if in_ict.exists() else []
        patients = sorted(set(pre_patients + ict_patients))

        for patient_id in patients:
            seiz_pre = list_seizures_for_phase(in_pre / patient_id, "preictal")
            seiz_ict = list_seizures_for_phase(in_ict / patient_id, "ictal")
            seiz_all = sorted(set(seiz_pre) | set(seiz_ict))
            if not seiz_all:
                continue

            log(f"==> {outcome} | {patient_id}: seizures {seiz_all}")

            for seiz in seiz_all:
                rows = process_one_seizure(
                    in_pre, in_ict,
                    out_node_pre, out_node_ict,
                    out_edge_pre, out_edge_ict,
                    patient_id, seiz,
                )
                for r in rows:
                    manifest_rows.append({"outcome": outcome, **r})
                    ph = r["phase"]
                    if r.get("status") == "ok":
                        log(f"  [{ph.upper()}] {patient_id} {seiz}: {r['n_selected_epochs']} epochs (/{r['n_total_epochs']})")
                    else:
                        log(f"  [{ph.upper()}] {patient_id} {seiz}: {r.get('status')}")

    man_df = pd.DataFrame(manifest_rows)
    man_path = BASE / "selection_manifest_ds004100.csv"
    man_df.to_csv(man_path, index=False)
    log(f"\n[OK] Manifest sauvegardé: {man_path}")
    log("Colonnes: [outcome, patient, seizure, phase, status, n_total_epochs, n_selected_epochs, selected_epoch_ids, node_avg_path_raw, node_avg_path_normpre, edge_avg_path_coh, edge_avg_path_psi]")

if __name__ == "__main__":
    main()
