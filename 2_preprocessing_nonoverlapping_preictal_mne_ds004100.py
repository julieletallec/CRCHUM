# -*- coding: utf-8 -*-
# ds004100 – PREICTAL: normalisation en fenêtres (1 s) pour BOTH outcomes
# Parcourt:
#   .../processed_data/ds004100/sc_fc/<good outcome|bad outcome>/preictal/<patient>/
# Lit:    <patient>_preictal_<seiz>_raw.fif
# Écrit:  <patient>_preictal_<seiz>_processed.fif (fenêtres 1 s, notch 60/120, ref moyenne, z-score par canal & epoch)

from pathlib import Path
import os
import warnings
import numpy as np
import mne

# --- Réduire le bruit des warnings MNE/pandas (optionnel) ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# CONFIG
# =========================
DATASET_ID = "ds004100"
ROOT = Path(f"//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data3/{DATASET_ID}/sc_fc")

OUTCOMES = ["good outcome", "bad outcome"]   # traite les deux
PHASE = "preictal"                           # ce script ne traite que le préictal
WINDOW_SIZE_SEC = 1.0
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 125
NOTCH_FREQS = (60.0, 120.0)

def zscore_windows(wins_np: np.ndarray) -> np.ndarray:
    """
    wins_np: shape (n_epochs, n_channels, n_times)
    z-score par (epoch, canal) -> (x - mean) / std, avec epsilon pour éviter /0
    """
    X = wins_np.astype(np.float32, copy=True)
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    sd[sd < 1e-12] = 1e-12
    X = (X - mu) / sd
    return X

def process_one_file(raw_fif_path: Path) -> None:
    """
    Charge <patient>_preictal_<seiz>_raw.fif
    Applique filtrage, notch, ref moyenne, découpe en fenêtres 1 s, z-score
    Sauvegarde <patient>_preictal_<seiz>_processed.fif (EpochsArray)
    """
    patient, phase, seiz = raw_fif_path.name.split("_")[0], raw_fif_path.name.split("_")[1], raw_fif_path.name.split("_")[2]
    out_name = f"{patient}_{phase}_{seiz}_processed.fif"
    out_path = raw_fif_path.parent / out_name

    try:
        raw = mne.io.read_raw_fif(raw_fif_path, preload=True, verbose=False)
    except Exception as e:
        print(f"[ERR] Lecture {raw_fif_path.name}: {e}")
        return

    # 1) filtre passe-bande
    raw_f = raw.copy().filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, verbose=False)

    # 2) notch 60 / 120 Hz
    raw_n = raw_f.copy().notch_filter(freqs=NOTCH_FREQS, verbose=False)

    # 3) référence moyenne
    #raw_r, _ = raw_n.set_eeg_reference(ref_channels="average", verbose=False)
    raw_r = raw_n.copy().set_eeg_reference(ref_channels="average", verbose=False)

    # 4) fenêtres de 1 s
    epochs = mne.make_fixed_length_epochs(raw_r, duration=WINDOW_SIZE_SEC, preload=True, verbose=False)

    # 5) normalisation z-score par (epoch, canal)
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    Xz = zscore_windows(X)
    epochs_z = mne.EpochsArray(Xz, epochs.info, events=epochs.events, tmin=epochs.tmin, verbose=False)

    # 6) sauvegarde
    epochs_z.save(out_path, overwrite=True)
    print(f"[OK] {raw_fif_path.name} -> {out_name} ({Xz.shape[0]} fenêtres)")

def main():
    for outcome in OUTCOMES:
        base_dir = ROOT / outcome / PHASE
        if not base_dir.is_dir():
            print(f"[WARN] Dossier introuvable (skip): {base_dir}")
            continue

        # patients = sous-dossiers
        patients = sorted([p for p in base_dir.iterdir() if p.is_dir()])
        print(f"\n=== Outcome: {outcome} | Patients: {len(patients)} ===")

        for pdir in patients:
            # détecter toutes les crises disponibles via *_raw.fif
            raw_files = sorted([f for f in pdir.glob(f"{pdir.name}_preictal_*_raw.fif") if f.is_file()])
            if not raw_files:
                print(f"[INFO] {pdir.name}: aucun fichier *_raw.fif")
                continue

            print(f"-> {pdir.name}: {len(raw_files)} enregistrements préictaux")
            for rf in raw_files:
                process_one_file(rf)

if __name__ == "__main__":
    main()
