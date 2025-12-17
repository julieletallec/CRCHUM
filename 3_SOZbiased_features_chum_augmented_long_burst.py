# -*- coding: utf-8 -*-
"""
CHUM — Nouvelles features SOZ-biased (node) + connectivité pour GCN
+ normalisation par électrode basée sur le préictal.

- Pour chaque patient, seizure, phase ∈ {"preictal","ictal"} :
    * tire K ~ Uniform{20..40}
    * preictal : K DERNIERS epochs (ou tous s'il y en a moins)
      ictal    : K PREMIERS epochs (ou tous s'il y en a moins)
    * calcule FEATURES NODE (nouvelles, sans metrics de connectivité) sur ces epochs
    * calcule EDGE (cohérence & PSI 12–45 Hz) sur les mêmes epochs
    * normalisation des features de nœuds par électrode & feature :
        - baseline = epochs préictaux sélectionnés de cette crise
        - stats robustes : médiane + 1.4826 * MAD
        - z-score : (x - médiane) / sigma, clip [-20,20]

- Sauvegardes :
    NODE (par epoch, RAW & ZNORM) :
        {patient}_{phase}_{seizure}_NODEfeatures_epoch_{eid}_raw.csv
        {patient}_{phase}_{seizure}_NODEfeatures_epoch_{eid}_znorm.csv

    NODE (moyennes sur les epochs sélectionnés de la phase) :
        {patient}_{phase}_{seizure}_NODEfeatures_avg_over_epochs_raw.csv
        {patient}_{phase}_{seizure}_NODEfeatures_avg_over_epochs_znorm.csv

    EDGE (identique à ancien script) :
        *_EDGE_coh1245_epoch_<eid>.csv (+ matrices si besoin)
        *_EDGE_psi1245_epoch_<eid>.csv (+ matrices)
        *_EDGE_coh1245_matrix_avg_over_epochs.csv
        *_EDGE_psi1245_matrix_avg_over_epochs.csv
        *_ADJ_coh1245.csv / *_ADJ_psi1245_sym.csv
        *_LAPL_coh1245_norm.csv / *_LAPL_psi1245_norm.csv

- Manifest global : selection_manifest_CHUM.csv
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import find_peaks, coherence

# =========================
# SUPPRESSION DES WARNINGS BRUYANTS
# =========================
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level("ERROR")

# =========================
# CONFIG GLOBALE
# =========================
DATASET_ID = "CHUM"
ROOT_BASE = Path("//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2")
BASE = ROOT_BASE / DATASET_ID / "sc_fc"

IN_PRE  = BASE / "preictal"
IN_ICT  = BASE / "ictal"

# Paramètres pour le persistent burst (comme dans l'autre script)
BURST_THR_K = 3.0   # seuil : mu_pre + K * sigma_pre
BURST_DECAY = 1.0   # 1.0 = cumul pur ; <1 = atténuation exponentielle


OUT_NODE_PRE = IN_PRE / "NODE_features_SOZ_augmented_20_10_burst"
OUT_NODE_ICT = IN_ICT / "NODE_features_SOZ_augmented_20_10_burst"
OUT_EDGE_PRE = IN_PRE / "EDGE_features_SOZ_augmented_20_10_burst"
OUT_EDGE_ICT = IN_ICT / "EDGE_features_SOZ_augmented_20_10_burst"

for d in [OUT_NODE_PRE, OUT_NODE_ICT, OUT_EDGE_PRE, OUT_EDGE_ICT]:
    d.mkdir(parents=True, exist_ok=True)

# Epochs par phase
MIN_EPOCHS, MAX_EPOCHS = 20, 40

#K_PREICTAL = 30
#K_ICTAL = 60

K_PREICTAL = 20
K_ICTAL = 10

RANDOM_SEED = 1337
_rng = np.random.default_rng(RANDOM_SEED)

# Parallélisme
N_JOBS_NODE = 14
N_JOBS_EDGE = 14

# Normalisation
CLIP_Z = 20.0  # clip des z-scores

# Sauvegarder les matrices complètes d'edges
SAVE_EDGE_MATRIX = True

PRINT_STATUS = True
def log(msg: str):
    if PRINT_STATUS:
        print(msg)



def compute_persistent_burst_from_nodes(
    node_pre: Optional[np.ndarray],
    node_ict: Optional[np.ndarray],
    thr_k: float = 3.0,
    decay: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calcule burstint_* sur des features déjà normalisées (ZNORM).
    node_pre/node_ict : (k, n_ch, n_feat) en ZNORM.

    Seuil par canal x feature défini sur le préictal :
        thr = mean(pre) + thr_k * std(pre)
    Burst instantané :
        inst = max(0, x - thr)
    Intégration :
        S_t = decay * S_{t-1} + inst
    """
    if node_pre is None and node_ict is None:
        return None, None

    # baseline : préictal si dispo
    base = node_pre if node_pre is not None else node_ict

    mu = np.nanmean(base, axis=0)                 # (n_ch, n_feat)
    sd = np.nanstd(base, axis=0, ddof=1)          # (n_ch, n_feat)
    sd[sd < 1e-9] = 1e-9
    thr = mu + thr_k * sd

    parts = []
    k_pre = 0
    if node_pre is not None:
        parts.append(node_pre)
        k_pre = node_pre.shape[0]
    if node_ict is not None:
        parts.append(node_ict)

    combined = np.concatenate(parts, axis=0)      # (T, n_ch, n_feat)
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




# =========================
# Utils PSD / bandpower
# =========================
def _robust_psd_array_welch(epoch_data, sfreq, fmin, fmax,
                             welch_nperseg=None, average='mean'):
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
# NODE FEATURES — Nouvelles (sans métriques de connectivité)
# =========================

def feature_band_ratios(epoch_data, sfreq):
    p_theta = _band_power(epoch_data, sfreq, 4., 8.)
    p_alpha = _band_power(epoch_data, sfreq, 8., 13.)
    p_beta  = _band_power(epoch_data, sfreq, 13., 30.)
    p_gamma = _band_power(epoch_data, sfreq, 30., min(80., sfreq / 2 - 1))
    p_delta = _band_power(epoch_data, sfreq, 1., 4.)
    eps = 1e-12
    return {
        "ratio_bg_ta": (p_beta + p_gamma) / (p_theta + p_alpha + eps),
        "ratio_gamma_delta": p_gamma / (p_delta + eps),
    }

def feature_lvfa_hafa(epoch_data, sfreq, p_low=30.0, p_high=70.0):
    rms = _rms(epoch_data, axis=1)
    p_lo = np.percentile(rms, p_low)
    p_hi = np.percentile(rms, p_high)
    p_lvfa = _band_power(epoch_data, sfreq, 14., min(150., sfreq / 2 - 1))
    p_hafa = _band_power(epoch_data, sfreq, 13., min(150., sfreq / 2 - 1))
    lvfa_score = (p_lvfa / (p_lvfa + 1e-9)) * (rms < p_lo).astype(float)
    hafa_score = (p_hafa / (p_hafa + 1e-9)) * (rms > p_hi).astype(float)
    return {"lvfa_score": lvfa_score, "hafa_score": hafa_score}

def feature_beta_gamma_slope(epoch_data, sfreq):
    mid = epoch_data.shape[1] // 2
    p1 = _band_power(epoch_data[:, :mid], sfreq, 13., min(80., sfreq/2 - 1))
    p2 = _band_power(epoch_data[:, mid:], sfreq, 13., min(80., sfreq/2 - 1))
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
    psd, freqs = _robust_psd_array_welch(epoch_data, sfreq, fmin, min(fmax, sfreq/2 - 1))
    if np.isnan(freqs).all():
        return {"sef95_Hz": np.full(epoch_data.shape[0], np.nan)}
    csum = np.cumsum(psd, axis=1)
    thr = 0.95 * csum[:, -1][:, None]
    idx = (csum >= thr).argmax(axis=1)
    return {"sef95_Hz": freqs[idx]}

def feature_spike_rate_1_3(epoch_data, sfreq, z=3.5):
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
            ok = (isi >= 1/3) & (isi <= 1.0)  # 1–3 Hz
            rates[c] = float(ok.sum())
        else:
            rates[c] = 0.0

    return {"spike_rate_1_3Hz": rates, "spike_sharpness": sharp}

def feature_ll_tkeo(epoch_data, sfreq):
    ll = np.sum(np.abs(np.diff(epoch_data, axis=1)), axis=1)
    tkeo = epoch_data[:, 1:-1]**2 - epoch_data[:, :-2]*epoch_data[:, 2:]
    tkeo_e = tkeo.mean(axis=1)
    return {"line_length": ll, "tkeo_energy": tkeo_e}

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
    from scipy.stats import kurtosis, skew
    k = kurtosis(epoch_data, axis=1, fisher=True, bias=False, nan_policy="omit")
    s = skew(epoch_data, axis=1, bias=False, nan_policy="omit")
    return {
        "kurtosis": k,
        "skewness": s,
    }

NODE_FEATURE_FUNCS = [
    feature_band_ratios,
    feature_lvfa_hafa,
    feature_beta_gamma_slope,
    feature_dc_shift,
    feature_sef95,
    feature_spike_rate_1_3,
    feature_ll_tkeo,
    feature_high_gamma,
    feature_spectral_slope,
    feature_hjorth,
    feature_kurtosis_skewness,
    # PAS de feature_connectivity_node_metrics ici
]

def compute_node_features_for_epoch(epoch_data, sfreq, ch_names):
    cols: Dict[str, np.ndarray] = {}
    for func in NODE_FEATURE_FUNCS:
        out = func(epoch_data, sfreq)
        for k, v in out.items():
            cols[k] = np.asarray(v)
    df = pd.DataFrame(cols, index=ch_names)
    return df  # (n_ch, n_feat)

# =========================
# CONNECTIVITÉ (identique à ton script)
# =========================
def _coherence_matrix(X, sfreq, fmin=12., fmax=45.):
    n_ch, n_t = X.shape
    C = np.ones((n_ch, n_ch), dtype=float)
    nperseg = max(64, min(n_t, int(sfreq // 2)))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            try:
                f, coh = coherence(X[i], X[j], fs=sfreq, nperseg=nperseg,
                                   noverlap=nperseg // 2)
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
            Ck = np.einsum("if,jf->fij", Fk, np.conj(Fk), optimize=True)
            Cxy += Ck
        Cxy /= K
    except Exception:
        w = np.hanning(n_t).astype(np.float64)
        Xw = X * w[None, :]
        F = np.fft.rfft(Xw, n=n_fft, axis=1)[:, fsel]
        Cxy = np.einsum("if,jf->fij", F, np.conj(F), optimize=True)
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
# DÉCOUVERTE DES SEIZURES
# =========================
FIF_RE = re.compile(r"^(.+?)_(ictal|preictal)_(\d+)_processed\.fif$", flags=re.IGNORECASE)

def list_seizures_for_phase(patient_dir: Path, phase: str) -> List[int]:
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

def select_epoch_ids(total_epochs: int, phase: str, k: int) -> List[int]:
    if k <= 0 or total_epochs <= 0:
        return []
    if phase == "preictal":
        start = max(0, total_epochs - k)
        return list(range(start, total_epochs))
    else:
        return list(range(0, min(k, total_epochs)))

# =========================
# TRAITEMENT D'UNE PHASE (NODE RAW + EDGES)
# =========================
def process_phase_node_and_edges(
    in_dir: Path,
    out_node_dir: Path,
    out_edge_dir: Path,
    patient_id: str,
    seizure_num: int,
    phase: str,
) -> Dict:

    result = {
        "patient": patient_id,
        "seizure": seizure_num,
        "phase": phase,
        "status": None,
        "n_total_epochs": 0,
        "n_selected_epochs": 0,
        "selected_epoch_ids": "[]",
        # Pour normalisation :
        "node_stack_raw": None,
        "ch_names": None,
        "feature_names": None,
        # chemins edges :
        "edge_avg_path_coh": None,
        "edge_avg_path_psi": None,
    }

    in_file = in_dir / patient_id / f"{patient_id}_{phase}_{seizure_num}_processed.fif"
    if not in_file.exists():
        result["status"] = "missing_fif"
        return result

    try:
        epochs = mne.read_epochs(str(in_file), preload=True, verbose=False)
    except Exception as e:
        result["status"] = f"read_error:{e.__class__.__name__}"
        return result

    ch_names = epochs.ch_names
    sfreq = epochs.info["sfreq"]
    total_epochs = len(epochs)
    result["n_total_epochs"] = int(total_epochs)

    if total_epochs == 0:
        result["status"] = "no_epochs"
        return result

    #k = draw_k(total_epochs)
    if phase == "preictal":
        desired_k = K_PREICTAL  # 30
    else:  # "ictal"
        desired_k = K_ICTAL     # 60

    k = min(desired_k, total_epochs)

    epoch_ids = select_epoch_ids(total_epochs, phase, k)


    if len(epoch_ids) == 0:
        result["status"] = "no_selection"
        return result

    result["n_selected_epochs"] = int(len(epoch_ids))
    result["selected_epoch_ids"] = json.dumps(epoch_ids)
    result["ch_names"] = ch_names

    # ===== NODE FEATURES (RAW, pas encore normalisées) =====
    def _one_node(eid: int):
        X = epochs[eid]._data[0]  # (n_ch, n_times)
        df_node = compute_node_features_for_epoch(X, sfreq, ch_names)
        return eid, df_node

    node_results = Parallel(n_jobs=N_JOBS_NODE)(
        delayed(_one_node)(eid) for eid in epoch_ids
    )

    # empilement (k, n_ch, n_feat)
    node_stack_raw = np.stack([df.values for _, df in node_results], axis=0)
    feature_names = node_results[0][1].columns.tolist()
    result["node_stack_raw"] = node_stack_raw
    result["feature_names"] = feature_names

    # les edges peuvent être calculés maintenant (indépendants de la normalisation)
    out_edge_dir_patient = out_edge_dir / patient_id
    out_edge_dir_patient.mkdir(parents=True, exist_ok=True)

    def _one_edge(eid: int):
        X = epochs[eid]._data[0]
        Ccoh = _coherence_matrix(X, sfreq, fmin=12., fmax=45.)
        PSI  = _psi_matrix(X, sfreq, fmin=12., fmax=45.)
        return eid, Ccoh, PSI

    edge_results = Parallel(n_jobs=N_JOBS_EDGE)(
        delayed(_one_edge)(eid) for eid in epoch_ids
    )

    # Sauvegarde per-epoch edges
    for eid, Ccoh, PSI in edge_results:
        n = len(ch_names)

        # Coherence undirected
        src, tgt, w = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(Ccoh[i, j]))
        pd.DataFrame({"source": src, "target": tgt, "coh_12_45": w}).to_csv(
            out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_coh1245_epoch_{eid}.csv",
            index=False,
        )

        # PSI directed
        src, tgt, w = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                src.append(ch_names[i]); tgt.append(ch_names[j]); w.append(float(PSI[i, j]))
        pd.DataFrame({"source": src, "target": tgt, "psi_12_45": w}).to_csv(
            out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_psi1245_epoch_{eid}.csv",
            index=False,
        )

        if SAVE_EDGE_MATRIX:
            pd.DataFrame(Ccoh, index=ch_names, columns=ch_names).to_csv(
                out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_coh1245_matrix_epoch_{eid}.csv",
                index=True,
            )
            pd.DataFrame(PSI, index=ch_names, columns=ch_names).to_csv(
                out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_psi1245_matrix_epoch_{eid}.csv",
                index=True,
            )

    # Moyenne sur epochs (edges)
    Ccoh_avg = np.nanmean(np.stack([C for _, C, _ in edge_results], axis=0), axis=0)
    PSI_avg  = np.nanmean(np.stack([P for _, _, P in edge_results], axis=0), axis=0)

    # adjacency & Laplacians
    A_coh = Ccoh_avg.copy()
    np.fill_diagonal(A_coh, 0.0)
    PSI_pos = np.clip(PSI_avg, 0, None)
    A_psi_sym = PSI_pos + PSI_pos.T
    np.fill_diagonal(A_psi_sym, 0.0)

    L_coh = _normalized_laplacian(A_coh)
    L_psi = _normalized_laplacian(A_psi_sym)

    # Sauvegardes matrices / adjacency / laplaciens
    edge_coh_avg_path = out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_coh1245_matrix_avg_over_epochs.csv"
    edge_psi_avg_path = out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_EDGE_psi1245_matrix_avg_over_epochs.csv"

    pd.DataFrame(Ccoh_avg, index=ch_names, columns=ch_names).to_csv(edge_coh_avg_path, index=True)
    pd.DataFrame(PSI_avg,  index=ch_names, columns=ch_names).to_csv(edge_psi_avg_path, index=True)
    pd.DataFrame(A_coh,    index=ch_names, columns=ch_names).to_csv(
        out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_ADJ_coh1245.csv", index=True
    )
    pd.DataFrame(A_psi_sym, index=ch_names, columns=ch_names).to_csv(
        out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_ADJ_psi1245_sym.csv", index=True
    )
    pd.DataFrame(L_coh,    index=ch_names, columns=ch_names).to_csv(
        out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_LAPL_coh1245_norm.csv", index=True
    )
    pd.DataFrame(L_psi,    index=ch_names, columns=ch_names).to_csv(
        out_edge_dir_patient / f"{patient_id}_{phase}_{seizure_num}_LAPL_psi1245_norm.csv", index=True
    )

    result["edge_avg_path_coh"] = str(edge_coh_avg_path)
    result["edge_avg_path_psi"] = str(edge_psi_avg_path)
    result["status"] = "ok"
    return result

# =========================
# NORMALISATION PAR ÉLECTRODE (baseline préictal)
# =========================

def normalize_nodes_with_preictal_baseline(
    node_pre_raw: Optional[np.ndarray],
    node_ict_raw: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Normalisation par électrode et par crise.
    SANS CLIPPING.

    node_pre_raw : (k_pre, n_ch, n_feat) ou None
    node_ict_raw : (k_ict, n_ch, n_feat) ou None

    Retourne:
        node_pre_z, node_ict_z, (med, sigma)
        med et sigma : (n_ch, n_feat)
    """

    # 1) Sélection de la baseline
    if node_pre_raw is not None:
        base = node_pre_raw       # (k_pre, n_ch, n_feat)
    elif node_ict_raw is not None:
        base = node_ict_raw       # fallback si pas de préictal
    else:
        return None, None, None

    # 2) Statistiques robustes canal × feature (une baseline par électrode)
    med = np.nanmedian(base, axis=0)                  # (n_ch, n_feat)
    mad = np.nanmedian(np.abs(base - med), axis=0)    # (n_ch, n_feat)

    sigma = 1.4826 * mad
    sigma[sigma < 1e-9] = 1e-9                        # protection division par zéro

    # 3) Fonction de normalisation SANS CLIPPING
    def _z(stack):
        if stack is None:
            return None
        return (stack - med) / sigma                  # broadcasting (k, n_ch, n_feat)

    # 4) Normalisation (par électrode et par crise)
    node_pre_z = _z(node_pre_raw)
    node_ict_z = _z(node_ict_raw)

    return node_pre_z, node_ict_z, (med, sigma)


def normalize_nodes_with_preictal_baseline_(
    node_pre_raw: Optional[np.ndarray],
    node_ict_raw: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    node_pre_raw : (k_pre, n_ch, n_feat) ou None
    node_ict_raw : (k_ict, n_ch, n_feat) ou None

    Retourne:
        node_pre_z, node_ict_z, (med, sigma)
    """
    if node_pre_raw is not None:
        base = node_pre_raw  # (k_pre, n_ch, n_feat)
    elif node_ict_raw is not None:
        # fallback : baseline sur ictal si pas de préictal
        base = node_ict_raw
    else:
        return None, None, None

    # stats robustes sur l’axe epochs (axis=0)
    med = np.nanmedian(base, axis=0)                       # (n_ch, n_feat)
    mad = np.nanmedian(np.abs(base - med), axis=0)         # (n_ch, n_feat)
    sigma = 1.4826 * mad
    sigma[sigma < 1e-9] = 1e-9

    def _z(stack):
        if stack is None:
            return None
        z = (stack - med) / sigma
        if CLIP_Z is not None:
            z = np.clip(z, -CLIP_Z, CLIP_Z)
        return z

    node_pre_z = _z(node_pre_raw)
    node_ict_z = _z(node_ict_raw)
    return node_pre_z, node_ict_z, (med, sigma)

# =========================
# MAIN
# =========================
def main():
    log(f"[INFO] RNG seed = {RANDOM_SEED} | epochs per phase ∈ [{MIN_EPOCHS}, {MAX_EPOCHS}]")
    log("[INFO] Node = nouvelles features SOZ-biased (sans métriques de connectivité).")
    log("[INFO] Edge = Coherence & PSI 12–45 Hz (comme avant).")
    manifest_rows: List[Dict] = []

    pre_patients = [p.name for p in IN_PRE.iterdir() if p.is_dir()] if IN_PRE.exists() else []
    ict_patients = [p.name for p in IN_ICT.iterdir() if p.is_dir()] if IN_ICT.exists() else []
    patients = sorted(set(pre_patients + ict_patients))

    for patient_id in patients:
        seiz_pre = list_seizures_for_phase(IN_PRE / patient_id, "preictal")
        seiz_ict = list_seizures_for_phase(IN_ICT / patient_id, "ictal")
        seiz_all = sorted(set(seiz_pre) | set(seiz_ict))
        if not seiz_all:
            continue

        log(f"==> {patient_id}: seizures {seiz_all}")

        for seiz in seiz_all:
            # --- Phase PREICTAL ---
            res_pre = process_phase_node_and_edges(
                IN_PRE, OUT_NODE_PRE, OUT_EDGE_PRE,
                patient_id, seiz, "preictal"
            )
            # --- Phase ICTAL ---
            res_ict = process_phase_node_and_edges(
                IN_ICT, OUT_NODE_ICT, OUT_EDGE_ICT,
                patient_id, seiz, "ictal"
            )

            node_pre_raw = res_pre.get("node_stack_raw")
            node_ict_raw = res_ict.get("node_stack_raw")
            ch_names_pre = res_pre.get("ch_names") or res_ict.get("ch_names")
            feat_names_orig = res_pre.get("feature_names") or res_ict.get("feature_names")

            # 1) ZNORM sur les features ORIGINALES
            node_pre_z, node_ict_z, _ = normalize_nodes_with_preictal_baseline(
                node_pre_raw, node_ict_raw
            )

            # 2) burstint calculé SUR ZNORM
            burst_pre_z, burst_ict_z = compute_persistent_burst_from_nodes(
                node_pre_z, node_ict_z,
                thr_k=BURST_THR_K,
                decay=BURST_DECAY,
            )

            # 3) noms étendus
            burst_feat_names = [f"burstint_{f}" for f in feat_names_orig]
            feat_names_ext = feat_names_orig + burst_feat_names

            # 4) Sauvegarde PREICTAL
            if node_pre_raw is not None:
                out_node_dir_patient_pre = OUT_NODE_PRE / patient_id
                out_node_dir_patient_pre.mkdir(parents=True, exist_ok=True)
                epoch_ids_pre = json.loads(res_pre["selected_epoch_ids"])
                k_pre, n_ch, n_feat = node_pre_raw.shape

                for i in range(k_pre):
                    eid = epoch_ids_pre[i]

                    # RAW = raw orig + burst(ZNORM)
                    raw_ext = np.concatenate([node_pre_raw[i], burst_pre_z[i]], axis=1)   # (n_ch, 2*n_feat)
                    zn_ext  = np.concatenate([node_pre_z[i],   burst_pre_z[i]], axis=1)

                    df_raw = pd.DataFrame(raw_ext, index=ch_names_pre, columns=feat_names_ext)
                    df_zn  = pd.DataFrame(zn_ext,  index=ch_names_pre, columns=feat_names_ext)

                    df_raw.to_csv(
                        out_node_dir_patient_pre /
                        f"{patient_id}_preictal_{seiz}_NODEfeatures_epoch_{eid}_raw.csv",
                        index=True
                    )
                    df_zn.to_csv(
                        out_node_dir_patient_pre /
                        f"{patient_id}_preictal_{seiz}_NODEfeatures_epoch_{eid}_znorm.csv",
                        index=True
                    )

                # moyennes sur epochs (phase)
                node_avg_pre_raw = np.nanmean(node_pre_raw, axis=0)
                node_avg_pre_z   = np.nanmean(node_pre_z,   axis=0)
                burst_avg_pre_z  = np.nanmean(burst_pre_z,  axis=0)

                avg_raw_ext = np.concatenate([node_avg_pre_raw, burst_avg_pre_z], axis=1)
                avg_zn_ext  = np.concatenate([node_avg_pre_z,   burst_avg_pre_z], axis=1)

                df_avg_raw_pre = pd.DataFrame(avg_raw_ext, index=ch_names_pre, columns=feat_names_ext)
                df_avg_z_pre   = pd.DataFrame(avg_zn_ext,  index=ch_names_pre, columns=feat_names_ext)

                avg_raw_path_pre = out_node_dir_patient_pre / f"{patient_id}_preictal_{seiz}_NODEfeatures_avg_over_epochs_raw.csv"
                avg_z_path_pre   = out_node_dir_patient_pre / f"{patient_id}_preictal_{seiz}_NODEfeatures_avg_over_epochs_znorm.csv"
                df_avg_raw_pre.to_csv(avg_raw_path_pre, index=True)
                df_avg_z_pre.to_csv(avg_z_path_pre, index=True)

                res_pre["node_avg_path_raw"]   = str(avg_raw_path_pre)
                res_pre["node_avg_path_znorm"] = str(avg_z_path_pre)

                # 5) Sauvegarde ICTAL
                if node_ict_raw is not None:
                    out_node_dir_patient_ict = OUT_NODE_ICT / patient_id
                    out_node_dir_patient_ict.mkdir(parents=True, exist_ok=True)
                    epoch_ids_ict = json.loads(res_ict["selected_epoch_ids"])
                    k_ict, n_ch, n_feat = node_ict_raw.shape

                    for i in range(k_ict):
                        eid = epoch_ids_ict[i]

                        raw_ext = np.concatenate([node_ict_raw[i], burst_ict_z[i]], axis=1)
                        zn_ext  = np.concatenate([node_ict_z[i],   burst_ict_z[i]], axis=1)

                        df_raw = pd.DataFrame(raw_ext, index=ch_names_pre, columns=feat_names_ext)
                        df_zn  = pd.DataFrame(zn_ext,  index=ch_names_pre, columns=feat_names_ext)

                        df_raw.to_csv(
                            out_node_dir_patient_ict /
                            f"{patient_id}_ictal_{seiz}_NODEfeatures_epoch_{eid}_raw.csv",
                            index=True
                        )
                        df_zn.to_csv(
                            out_node_dir_patient_ict /
                            f"{patient_id}_ictal_{seiz}_NODEfeatures_epoch_{eid}_znorm.csv",
                            index=True
                        )

                    node_avg_ict_raw = np.nanmean(node_ict_raw, axis=0)
                    node_avg_ict_z   = np.nanmean(node_ict_z,   axis=0)
                    burst_avg_ict_z  = np.nanmean(burst_ict_z,  axis=0)

                    avg_raw_ext = np.concatenate([node_avg_ict_raw, burst_avg_ict_z], axis=1)
                    avg_zn_ext  = np.concatenate([node_avg_ict_z,   burst_avg_ict_z], axis=1)

                    df_avg_raw_ict = pd.DataFrame(avg_raw_ext, index=ch_names_pre, columns=feat_names_ext)
                    df_avg_z_ict   = pd.DataFrame(avg_zn_ext,  index=ch_names_pre, columns=feat_names_ext)

                    avg_raw_path_ict = out_node_dir_patient_ict / f"{patient_id}_ictal_{seiz}_NODEfeatures_avg_over_epochs_raw.csv"
                    avg_z_path_ict   = out_node_dir_patient_ict / f"{patient_id}_ictal_{seiz}_NODEfeatures_avg_over_epochs_znorm.csv"
                    df_avg_raw_ict.to_csv(avg_raw_path_ict, index=True)
                    df_avg_z_ict.to_csv(avg_z_path_ict, index=True)

                    res_ict["node_avg_path_raw"]   = str(avg_raw_path_ict)
                    res_ict["node_avg_path_znorm"] = str(avg_z_path_ict)

            # logs
            if res_pre.get("status") == "ok":
                log(f"  [PREICTAL] {patient_id} {seiz}: {res_pre['n_selected_epochs']} epochs (/{res_pre['n_total_epochs']})")
            else:
                log(f"  [PREICTAL] {patient_id} {seiz}: {res_pre.get('status')}")

            if res_ict.get("status") == "ok":
                log(f"  [ICTAL]    {patient_id} {seiz}: {res_ict['n_selected_epochs']} epochs (/{res_ict['n_total_epochs']})")
            else:
                log(f"  [ICTAL]    {patient_id} {seiz}: {res_ict.get('status')}")

            # Nettoyage avant manifest (on ne garde pas les gros arrays)
            for r in (res_pre, res_ict):
                r_clean = r.copy()
                r_clean.pop("node_stack_raw", None)
                r_clean.pop("ch_names", None)
                r_clean.pop("feature_names", None)
                manifest_rows.append(r_clean)

    man_df = pd.DataFrame(manifest_rows)
    man_path = BASE / "selection_manifest_CHUM.csv"
    man_df.to_csv(man_path, index=False)
    log(f"\n[OK] Manifest sauvegardé: {man_path}")
    log("Colonnes: [patient, seizure, phase, status, n_total_epochs, n_selected_epochs, "
        "selected_epoch_ids, node_avg_path_raw, node_avg_path_znorm, edge_avg_path_coh, edge_avg_path_psi]")

if __name__ == "__main__":
    main()
