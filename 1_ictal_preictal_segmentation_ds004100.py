# -*- coding: utf-8 -*-
# ds004100 – coupe automatiquement préictal/ictal pour BOTH outcomes (good & bad)
# Sauvegarde:
#   .../processed_data/ds004100/sc_fc/<outcome>/(preictal|ictal)/<patient>/<patient>_<phase>_<idx>_raw.fif

import os
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# --- Réduire le bruit des warnings MNE/pandas (optionnel) ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# CONFIG
# =========================
GENERAL_DIR = Path("/home/julieletallec/smb_share")
RAW_ROOT    = GENERAL_DIR / "Donnee" / "iEEG" / "RAW" / "ds004100"
OUT_ROOT    = GENERAL_DIR / "Equip_Rech" / "LeTallec_Julie" / "processed_data3" / "ds004100" / "sc_fc"

# Exclusions et petites corrections dataset
EXCLUDE_SUBS = {"sub-HUP132", "sub-HUP165"}
TRIM_FIX = {"sub-HUP151 ": "sub-HUP151"}  # espaces parasites dans participants.tsv

# =========================
# Helpers
# =========================
RUN_RE = re.compile(r"_run-(\d{2})", flags=re.IGNORECASE)

def get_run_id(name: str) -> str | None:
    """Extrait run-XX sous forme 'XX' depuis un nom de fichier."""
    m = RUN_RE.search(name)
    return m.group(1) if m else None

def outcome_label_from_participants_code(s: str) -> str | None:
    """Mappe le code outcome du participants.tsv (S/F) vers labels dossier."""
    if s == "S":
        return "good outcome"
    if s == "F":
        return "bad outcome"
    return None

def find_matches_for_patient(ieeg_dir: Path, patient_id: str):
    """
    Liste les triple (edf_path, events_path, channels_path) pour un patient,
    en appariant par 'run-XX'. Ne retient que les triplets complets.
    """
    edfs = sorted([p for p in ieeg_dir.glob(f"{patient_id}_ses-presurgery_task-ictal*.edf") if p.is_file()])
    events = list(ieeg_dir.glob(f"{patient_id}_ses-presurgery*events.tsv"))
    chans  = list(ieeg_dir.glob(f"{patient_id}_ses-presurgery*channels.tsv"))

    # indexer events/channels par run
    ev_by_run = {}
    ch_by_run = {}
    for ev in events:
        r = get_run_id(ev.name)
        if r: ev_by_run[r] = ev
    for ch in chans:
        r = get_run_id(ch.name)
        if r: ch_by_run[r] = ch

    triplets = []
    for edf in edfs:
        r = get_run_id(edf.name)
        ev = ev_by_run.get(r)
        ch = ch_by_run.get(r)
        if r and ev and ch:
            triplets.append((r, edf, ev, ch))
        else:
            print(f"[WARN] Appariement incomplet pour {edf.name} (run={r}) -> events={bool(ev)} channels={bool(ch)}")

    # trier par run pour stabilité
    triplets.sort(key=lambda t: t[0])
    return triplets

def read_seizure_bounds(events_tsv: Path) -> tuple[float, float] | None:
    """
    Lit *_events.tsv et récupère 'sz onset' et 'sz offset' (col trial_type, onset).
    Renvoie (onset_sec, offset_sec) ou None si introuvable.
    """
    df = pd.read_csv(events_tsv, sep="\t")
    if "trial_type" not in df.columns or "onset" not in df.columns:
        print(f"[WARN] Colonnes trial_type/onset manquantes dans {events_tsv.name}")
        return None

    # case-insensitive pour 'sz onset/offset'
    tt = df["trial_type"].astype(str).str.lower()
    onset_rows = df.loc[tt == "sz onset"]
    offset_rows = df.loc[tt == "sz offset"]
    if onset_rows.empty or offset_rows.empty:
        print(f"[WARN] sz onset/offset introuvables dans {events_tsv.name}")
        return None

    onset = float(onset_rows.iloc[0]["onset"])
    offset = float(offset_rows.iloc[0]["onset"])
    if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset:
        print(f"[WARN] bornes invalides dans {events_tsv.name} -> onset={onset}, offset={offset}")
        return None
    return onset, offset

def good_channel_names(channels_tsv: Path) -> list[str]:
    """
    Extrait les noms de canaux avec status == 'good'. Colonne 'name' et 'status' attendues.
    """
    df = pd.read_csv(channels_tsv, sep="\t")
    if "name" not in df.columns or "status" not in df.columns:
        print(f"[WARN] Colonnes name/status manquantes dans {channels_tsv.name}")
        return []
    mask = df["status"].astype(str).str.lower().eq("good")
    return df.loc[mask, "name"].astype(str).tolist()

def save_segments(raw: mne.io.BaseRaw, onset: float, offset: float, out_dir: Path, patient_id: str, idx: int):
    """Découpe et sauvegarde préictal (0→onset) et ictal (onset→offset) en .fif."""
    out_dir_ictal    = out_dir / "ictal" / patient_id
    out_dir_preictal = out_dir / "preictal" / patient_id
    out_dir_ictal.mkdir(parents=True, exist_ok=True)
    out_dir_preictal.mkdir(parents=True, exist_ok=True)

    ictal   = raw.copy().crop(onset,  offset)
    preict  = raw.copy().crop(0.0,    onset)

    ictal.save(   out_dir_ictal    / f"{patient_id}_ictal_{idx}_raw.fif",    overwrite=True)
    preict.save(  out_dir_preictal / f"{patient_id}_preictal_{idx}_raw.fif", overwrite=True)

# =========================
# MAIN
# =========================
def main():
    # participants.tsv
    part_path = RAW_ROOT / "participants.tsv"
    if not part_path.exists():
        raise FileNotFoundError(f"participants.tsv introuvable: {part_path}")
    participants = pd.read_csv(part_path, sep="\t")

    # patient -> outcome_label
    patients_outcome = {}
    for pid, code in zip(participants["participant_id"].astype(str), participants["outcome"].astype(str)):
        pid = TRIM_FIX.get(pid, pid)  # fix éventuel espace final
        if pid in EXCLUDE_SUBS:
            continue
        label = outcome_label_from_participants_code(code)
        if label:
            patients_outcome[pid] = label

    print(f"[INFO] Patients retenus: {len(patients_outcome)} (good/bad confondus)")

    # boucle patients
    for patient_id, outcome_label in sorted(patients_outcome.items()):
        ieeg_dir = RAW_ROOT / patient_id / "ses-presurgery" / "ieeg"
        if not ieeg_dir.is_dir():
            print(f"[WARN] Dossier ieeg introuvable pour {patient_id}: {ieeg_dir}")
            continue

        triplets = find_matches_for_patient(ieeg_dir, patient_id)
        if not triplets:
            print(f"[INFO] Aucun enregistrement task-ictal trouvé pour {patient_id}")
            continue

        print(f"\n==> {patient_id} [{outcome_label}] : {len(triplets)} enregistrements")
        for run, edf_path, events_path, chans_path in triplets:
            # bornes
            bounds = read_seizure_bounds(events_path)
            if bounds is None:
                print(f"  [SKIP] bornes manquantes/invalides pour run-{run}")
                continue
            onset, offset = bounds

            # canaux "good"
            good_chs = good_channel_names(chans_path)
            if not good_chs:
                print(f"  [SKIP] aucun canal 'good' dans {chans_path.name}")
                continue

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"  [ERR] Lecture EDF échouée {edf_path.name}: {e}")
                continue

            # ne garder que l'intersection des canaux
            keep = [ch for ch in good_chs if ch in raw.ch_names]
            if not keep:
                print(f"  [SKIP] aucun des canaux 'good' n'existe dans {edf_path.name}")
                continue

            raw = raw.pick_channels(keep).load_data()

            # répertoire de sortie par outcome
            out_dir = OUT_ROOT / outcome_label
            try:
                save_segments(raw, onset, offset, out_dir, patient_id, idx=int(run))
                print(f"  [OK] run-{run}: préictal/ictal sauvegardés -> {out_dir}")
            except Exception as e:
                print(f"  [ERR] run-{run}: échec sauvegarde -> {e}")

if __name__ == "__main__":
    main()
