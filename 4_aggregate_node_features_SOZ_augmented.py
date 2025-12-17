# build_master_node_features_df.py
import os
import re
from typing import Optional, Set, Tuple
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ROOT = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2"
DEBUG_UNMATCHED = True  # mettre False si tu veux éviter trop de logs

# Racines BIDS / SOZ externes
DS004100_BIDS_ROOT = "//home/julieletallec/smb_share/Donnee/iEEG/RAW/ds004100"
CHUM_SOZ_ROOT = "//home/julieletallec/smb_share/Donnee/iEEG/RAW/CHUM/Resection Zones/Channel_per_seizure"

CHUM_outcomes = {
    "Patient_01": "good outcome",
    "Patient_02": "good outcome",
    "Patient_07": "good outcome",
    "Patient_08": "good outcome",
    "Patient_09": "bad outcome",
    "Patient_11": "good outcome",
    "Patient_12": "good outcome",
    "Patient_14": "good outcome",
    "Patient_15": "good outcome",
    "Patient_16": "bad outcome",
    "Patient_17": "bad outcome",
    "Patient_19": "bad outcome",
    "Patient_21": "bad outcome",
    "Patient_22": "good outcome",
    "Patient_23": "bad outcome",
    "Patient_24": "good outcome",
    "Patient_25": "good outcome",
}

# =========================
# REGEX de nommage des fichiers
# =========================
# Tolère suffixes type _znorm, _prenorm, etc.
FNAME_RE = re.compile(
    r"^(?P<patient>.+?)_(?P<phase>ictal|preictal)_(?P<seizure>\d+)_NODE(?:_|-)?features_"
    r"(?:epoch_(?P<epoch>\d+)(?:_[A-Za-z0-9-]+)?|avg_over_epochs(?:_[A-Za-z0-9-]+)?)"
    r"\.csv$",
    flags=re.IGNORECASE
)

FNAME_RE_ALT = [
    re.compile(
        r"^(?P<patient>.+?)_(?P<phase>ictal|preictal).*?seizure[_-]?(?P<seizure>\d+).*?_NODE(?:_|-)?features_"
        r"(?:epoch_(?P<epoch>\d+)(?:_[A-Za-z0-9-]+)?|avg_over_epochs(?:_[A-Za-z0-9-]+)?)"
        r"\.csv$",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"^(?P<patient>.+?)_(?P<phase>ictal|preictal).*?run-(?P<run>\d+).*?_NODE(?:_|-)?features_"
        r"(?:epoch_(?P<epoch>\d+)(?:_[A-Za-z0-9-]+)?|avg_over_epochs(?:_[A-Za-z0-9-]+)?)"
        r"\.csv$",
        flags=re.IGNORECASE
    ),
]

# =========================
# Extraction du contexte depuis le chemin
# =========================
NODE_DIR_NAMES = "NODE_features_SOZ_augmented_20_10_burst"

def parse_path_metadata(dirpath: str) -> dict:
    rel = os.path.relpath(dirpath, ROOT)
    parts = rel.split(os.sep)

    meta = {"dataset": None, "outcome": None, "phase": None, "patient": None}
    if len(parts) == 0 or parts[0].startswith(".."):
        return meta

    meta["dataset"] = parts[0]  # ex: CHUM, ds004100

    # on cherche "sc_fc"
    try:
        i_sc = parts.index("sc_fc")
    except ValueError:
        return meta

    # outcome (ds004100) ou phase (CHUM)
    if i_sc + 1 < len(parts):
        candidate = parts[i_sc + 1]
        # outcome ds004100 = "bad outcome" / "good outcome"
        if candidate not in ("ictal/"+ NODE_DIR_NAMES, "preictal/"+ NODE_DIR_NAMES) :
            meta["outcome"] = candidate

    # phase
    if "ictal" in parts:
        meta["phase"] = "ictal"
    if "preictal" in parts:
        meta["phase"] = "preictal"

    # patient = dossier juste après NODE_features_SOZ(_augmented)
    if "NODE_features_SOZ_augmented_20_10_burst" in parts:
        i_nf = parts.index("NODE_features_SOZ_augmented_20_10_burst")
        if i_nf + 1 < len(parts):
            meta["patient"] = parts[i_nf + 1]
            

    return meta

# =========================
# Helpers communs
# =========================
RUN_RE = re.compile(r"_run-(\d{2})", flags=re.IGNORECASE)
MODALITY_TOKENS = r"(EEG|ECOG|SEEG|EKG|EMG|DBS|LFP)"

def normalize_elec_basic(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"\s+", "", name).upper()

def normalize_chum_elec(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(rf"^{MODALITY_TOKENS}\s+", "", s, flags=re.IGNORECASE)
    s = s.upper()
    s = re.sub(r"\s+", "", s)
    return s

def normalize_ds004100_elec(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.upper()
    s = s.replace("\t", " ")
    s = re.sub(r"[()]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(rf"\s+{MODALITY_TOKENS}$", "", s)
    s = re.sub(rf"^{MODALITY_TOKENS}\s+", "", s)
    s = re.sub(r"[\s_]+", "", s)
    return s

def patient_number(patient: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(patient))
    return int(m.group(1)) if m else None

def patient_id_formats(patient: str) -> Tuple[str, str]:
    num = patient_number(patient)
    if num is None:
        return patient, patient.lower()
    p_pad = f"Patient_{num:02d}"
    p_pad_lower = f"patient_{num:02d}"
    return p_pad, p_pad_lower

# =========================
# Helpers ds004100 : récupérer SOZ
# =========================
def channels_tsv_path_ds004100(patient: str, seizure: int) -> Optional[str]:
    sub_dir = os.path.join(DS004100_BIDS_ROOT, patient, "ses-presurgery", "ieeg")
    if not os.path.isdir(sub_dir):
        if DEBUG_UNMATCHED:
            print(f"[WARN] Dossier ieeg introuvable pour {patient}: {sub_dir}")
        return None

    target_run = seizure  # (tu avais déjà corrigé ça)
    target_path = None
    available_runs = []

    try:
        for fname in os.listdir(sub_dir):
            low = fname.lower()
            if low.endswith(".tsv") and ("ictal" in low) and ("channels" in low):
                m = RUN_RE.search(fname)
                if m:
                    run_num = int(m.group(1))
                    available_runs.append(run_num)
                    if run_num == target_run:
                        target_path = os.path.join(sub_dir, fname)
                        break
    except Exception as e:
        print(f"[WARN] listdir échoué pour {sub_dir} -> {e}")
        return None

    if target_path is None and DEBUG_UNMATCHED:
        ar = ",".join(f"{r:02d}" for r in sorted(set(available_runs)))
        print(f"[WARN] Pas de match exact ds004100 pour {patient}, seizure {seizure} (attendu run-{target_run:02d}). Runs dispos: [{ar}]")
    return target_path

def load_soz_names_from_channels_tsv(tsv_path: str) -> Set[str]:
    try:
        ch = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        print(f"[WARN] Impossible de lire {tsv_path} -> {e}")
        return set()

    if "name" not in ch.columns:
        print(f"[WARN] 'name' manquant dans {tsv_path}")
        return set()

    desc_col = None
    for c in ("status_description", "description"):
        if c in ch.columns:
            desc_col = c
            break
    if desc_col is None:
        print(f"[WARN] ni 'status_description' ni 'description' présents dans {tsv_path}")
        return set()

    mask_soz = ch[desc_col].astype(str).str.contains("soz", case=False, na=False)
    names_raw = ch.loc[mask_soz, "name"].astype(str).tolist()
    soz_names = {normalize_ds004100_elec(x) for x in names_raw}
    return soz_names

# =========================
# Helpers CHUM : récupérer SOZ
# =========================
def chum_soz_csv_path(patient: str, seizure: int) -> Optional[str]:
    p_pad, _ = patient_id_formats(patient)
    pat_dir = os.path.join(CHUM_SOZ_ROOT, p_pad)
    if not os.path.isdir(pat_dir):
        if DEBUG_UNMATCHED:
            print(f"[WARN] Dossier CHUM SOZ introuvable: {pat_dir}")
        return None

    num = patient_number(patient)
    if num is None:
        pattern = re.compile(rf"^Channel[_ ]?soz[_ ]?patient_.*_seizure_?{int(seizure)}\.csv$", flags=re.IGNORECASE)
    else:
        pattern = re.compile(
            rf"^Channel[_ ]?soz[_ ]?patient_{num:02d}[_ ]?[_-]?seizure_?{int(seizure)}\.csv$",
            flags=re.IGNORECASE
        )

    try:
        for fname in os.listdir(pat_dir):
            if pattern.match(fname):
                return os.path.join(pat_dir, fname)
    except Exception as e:
        print(f"[WARN] listdir échoué pour {pat_dir} -> {e}")
        return None

    if DEBUG_UNMATCHED:
        print(f"[WARN] Fichier SOZ CHUM non trouvé pour {p_pad}, seizure {seizure} dans {pat_dir}")
    return None

def load_soz_names_from_chum_csv(csv_path: str) -> Set[str]:
    if not os.path.exists(csv_path):
        if DEBUG_UNMATCHED:
            print(f"[WARN] Fichier CHUM SOZ introuvable: {csv_path}")
        return set()

    soz = set()
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    elec = parts[1]
                    soz.add(normalize_elec_basic(elec))
    except Exception as e:
        print(f"[WARN] Lecture échouée {csv_path} -> {e}")
        return set()

    return soz

# =========================
# Fonction principale
# =========================

from sklearn.preprocessing import MinMaxScaler

def collect_node_features(root=ROOT) -> pd.DataFrame:
    rows = []
    n_files = 0

    # juste pour le logging "nouveau patient"
    seen_patients = set()

    for dirpath, dirnames, filenames in os.walk(root):
        if "NODE_features_SOZ_augmented_20_10_burst" not in dirpath:
            continue


        meta_path = parse_path_metadata(dirpath)
        if meta_path["dataset"] is None or meta_path["patient"] is None:
            continue
        
        p = meta_path["patient"]
        if p not in seen_patients:
            seen_patients.add(p)
            print(f"\n=== Nouveau patient détecté ===")
            print(f"patient={p} | dataset={meta_path['dataset']} | phase={meta_path['phase']} | dir={dirpath}")

        for fname in filenames:
            # 🚩 On veut maintenant les FEATURES DÉJÀ NORMALISÉES :
            #  - CHUM : *_znorm.csv
            #  - ds004100 : *_prenorm.csv
            if not (fname.endswith("znorm.csv") or fname.endswith("normpre.csv")):
                continue

            if "_NODEfeatures_" not in fname:
                continue

            m = FNAME_RE.match(fname)
            if not m:
                if DEBUG_UNMATCHED and fname.endswith(".csv"):
                    print(f"[SKIP-REGEX] Ne matche pas : {fname}")
                continue

            meta_file = m.groupdict()
            patient_from_name = meta_file["patient"]
            patient = meta_path["patient"] or patient_from_name
            seizure = int(meta_file["seizure"])
            phase_from_name = meta_file["phase"]

            if meta_file.get("epoch") is None:
                epoch = -1
                is_avg = True
            else:
                epoch = int(meta_file["epoch"])
                is_avg = False

            dataset = meta_path["dataset"]
            outcome = meta_path["outcome"]  # None pour CHUM
            phase = meta_path["phase"] or phase_from_name

            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath, index_col=0)
            except Exception as e:
                print(f"[WARN] lecture impossible: {fpath} -> {e}")
                continue

            df = df.copy()
            df["electrode"] = df.index
            df.reset_index(drop=True, inplace=True)

            df.insert(0, "dataset", dataset)
            df.insert(1, "outcome", outcome)
            df.insert(2, "phase", phase)
            df.insert(3, "patient", patient)
            df.insert(4, "seizure", seizure)
            df.insert(5, "epoch", epoch)
            df.insert(6, "is_avg", is_avg)
            df["file_path"] = fpath

            rows.append(df)
            n_files += 1

    if not rows:
        print("[INFO] Aucun fichier Node Features trouvé.")
        return pd.DataFrame()

    big_df = pd.concat(rows, axis=0, ignore_index=True)

    # ==== CHUM outcomes ==== 
    mask_chum = big_df["dataset"] == "CHUM"
    big_df.loc[mask_chum, "outcome"] = big_df.loc[mask_chum, "patient"].map(CHUM_outcomes)

    # ==== is_SOZ (NaN par défaut) ==== 
    big_df["is_SOZ"] = pd.Series([pd.NA] * len(big_df), dtype="boolean")

    # normalisation des noms d'électrode
    def row_norm(row):
        if row["dataset"] == "ds004100":
            return normalize_ds004100_elec(row["electrode"])
        else:
            return normalize_chum_elec(row["electrode"])

    big_df["electrode_norm"] = big_df.apply(row_norm, axis=1)

    # ---- ds004100 ----
    mask_ds = big_df["dataset"] == "ds004100"
    for (patient, seizure), idx_grp in big_df.loc[mask_ds].groupby(["patient", "seizure"]).groups.items():
        tsv_path = channels_tsv_path_ds004100(str(patient), int(seizure))
        if not tsv_path or not os.path.exists(tsv_path):
            continue
        soz_names = load_soz_names_from_channels_tsv(tsv_path)
        big_df.loc[idx_grp, "is_SOZ"] = False
        if soz_names:
            grp_df = big_df.loc[idx_grp]
            elec_mask = grp_df["electrode_norm"].isin(soz_names)
            if elec_mask.any():
                big_df.loc[grp_df.index[elec_mask], "is_SOZ"] = True
        else:
            if DEBUG_UNMATCHED:
                print(f"[INFO] Aucun canal SOZ détecté dans {tsv_path}")

    # ---- CHUM ----
    mask_ch = big_df["dataset"] == "CHUM"
    for (patient, seizure), idx_grp in big_df.loc[mask_ch].groupby(["patient", "seizure"]).groups.items():
        print(patient, seizure)
        csv_path = chum_soz_csv_path(str(patient), int(seizure))
        if not csv_path:
            print("not csv path", csv_path)
            continue
        soz_names = load_soz_names_from_chum_csv(csv_path)
        big_df.loc[idx_grp, "is_SOZ"] = False
        if soz_names:
            grp_df = big_df.loc[idx_grp]
            elec_mask = grp_df["electrode_norm"].isin(soz_names)
            if elec_mask.any():
                big_df.loc[grp_df.index[elec_mask], "is_SOZ"] = True
        else:
            if DEBUG_UNMATCHED:
                print(f"[INFO] Aucun canal SOZ détecté (ou fichier vide) pour {patient}, seizure {seizure} -> {csv_path}")

    # ==== first_ictal_epoch ==== 
    big_df["first_ictal_epoch"] = False
    ict_mask = big_df["phase"].eq("ictal") & big_df["epoch"].ge(0)

    first_ep_df = (
        big_df.loc[ict_mask, ["dataset", "patient", "seizure", "epoch"]]
              .groupby(["dataset", "patient", "seizure"], as_index=False)["epoch"]
              .min()
              .rename(columns={"epoch": "first_epoch"})
    )

    big_df = big_df.merge(first_ep_df, on=["dataset", "patient", "seizure"], how="left")
    big_df["first_ictal_epoch"] = ict_mask & big_df["epoch"].eq(big_df["first_epoch"])
    big_df.drop(columns=["first_epoch"], inplace=True)
    big_df["first_ictal_epoch"] = big_df["first_ictal_epoch"].astype(bool)

    # Colonnes bien ordonnées 
    meta_cols = [
        "dataset", "outcome", "phase", "patient",
        "seizure", "epoch", "is_avg", "electrode",
        "is_SOZ", "first_ictal_epoch", "file_path"
    ]
    aux_cols = ["electrode_norm"]
    feature_cols = [c for c in big_df.columns if c not in meta_cols + aux_cols]
    """
    # 👉 filtrage des features (inchangé pour l’instant)
    FEATURES_TO_KEEP = [
        "ratio_bg_ta",
        "ratio_gamma_delta",
        "sef95_Hz",
        "spike_sharpness",
        "line_length",
        "tkeo_energy",
        "hg_power_80_150",
        "hg_over_gamma",
        "spec_slope_2_80",
        "spec_intercept_2_80",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "kurtosis",
        "skewness",
    ]

    """

    FEATURES_TO_KEEP = [
        "burstint_ratio_bg_ta",
        "burstint_ratio_gamma_delta",
        "burstint_sef95_Hz",
        "burstint_spike_sharpness",
        "burstint_line_length",
        "burstint_tkeo_energy",
        "burstint_hg_power_80_150",
        "burstint_hg_over_gamma",
        "burstint_spec_slope_2_80",
        "burstint_spec_intercept_2_80",
        "burstint_hjorth_activity",
        "burstint_hjorth_mobility",
        "burstint_hjorth_complexity",
        "burstint_kurtosis",
        "burstint_skewness",
    ]

    selected_features = [f for f in FEATURES_TO_KEEP if f in feature_cols]

    big_df = big_df[meta_cols + selected_features]

    # ---- Appliquer Min-Max normalization sur les features ----
    scaler = MinMaxScaler()
    big_df[selected_features] = scaler.fit_transform(big_df[selected_features])

    # Types
    big_df["seizure"] = pd.to_numeric(big_df["seizure"], errors="coerce").astype("Int64")
    big_df["epoch"] = pd.to_numeric(big_df["epoch"], errors="coerce").astype("Int64")
    big_df["is_avg"] = big_df["is_avg"].astype(bool)

    print(f"[OK] agrégation terminée : {big_df.shape[0]} lignes, {big_df.shape[1]} colonnes, depuis {n_files} fichiers.")
    print(big_df[["dataset"]].value_counts(dropna=False))
    return big_df


def collect_node_features_(root=ROOT) -> pd.DataFrame:
    rows = []
    n_files = 0

    # juste pour le logging "nouveau patient"
    seen_patients = set()

    for dirpath, dirnames, filenames in os.walk(root):
        if "NODE_features_SOZ_augmented_20_10_burst" not in dirpath:
            continue


        meta_path = parse_path_metadata(dirpath)
        if meta_path["dataset"] is None or meta_path["patient"] is None:
            continue
        
        p = meta_path["patient"]
        if p not in seen_patients:
            seen_patients.add(p)
            print(f"\n=== Nouveau patient détecté ===")
            print(f"patient={p} | dataset={meta_path['dataset']} | phase={meta_path['phase']} | dir={dirpath}")

        for fname in filenames:
            # 🚩 On veut maintenant les FEATURES DÉJÀ NORMALISÉES :
            #  - CHUM : *_znorm.csv
            #  - ds004100 : *_prenorm.csv
            if not (fname.endswith("znorm.csv") or fname.endswith("normpre.csv")):
                continue

            if "_NODEfeatures_" not in fname:
                continue

            m = FNAME_RE.match(fname)
            if not m:
                if DEBUG_UNMATCHED and fname.endswith(".csv"):
                    print(f"[SKIP-REGEX] Ne matche pas : {fname}")
                continue

            meta_file = m.groupdict()
            patient_from_name = meta_file["patient"]
            patient = meta_path["patient"] or patient_from_name
            seizure = int(meta_file["seizure"])
            phase_from_name = meta_file["phase"]

            if meta_file.get("epoch") is None:
                epoch = -1
                is_avg = True
            else:
                epoch = int(meta_file["epoch"])
                is_avg = False

            dataset = meta_path["dataset"]
            outcome = meta_path["outcome"]  # None pour CHUM
            phase = meta_path["phase"] or phase_from_name

            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath, index_col=0)
            except Exception as e:
                print(f"[WARN] lecture impossible: {fpath} -> {e}")
                continue

            df = df.copy()
            df["electrode"] = df.index
            df.reset_index(drop=True, inplace=True)

            df.insert(0, "dataset", dataset)
            df.insert(1, "outcome", outcome)
            df.insert(2, "phase", phase)
            df.insert(3, "patient", patient)
            df.insert(4, "seizure", seizure)
            df.insert(5, "epoch", epoch)
            df.insert(6, "is_avg", is_avg)
            df["file_path"] = fpath

            rows.append(df)
            n_files += 1

    if not rows:
        print("[INFO] Aucun fichier Node Features trouvé.")
        return pd.DataFrame()

    big_df = pd.concat(rows, axis=0, ignore_index=True)

    # ==== CHUM outcomes ====
    mask_chum = big_df["dataset"] == "CHUM"
    big_df.loc[mask_chum, "outcome"] = big_df.loc[mask_chum, "patient"].map(CHUM_outcomes)

    if DEBUG_UNMATCHED:
        missing = big_df.loc[mask_chum & big_df["outcome"].isna(), "patient"].unique().tolist()
        if len(missing) > 0:
            print("[WARN] Patients CHUM sans outcome dans CHUM_outcomes :", missing)

    # ==== is_SOZ (NaN par défaut) ====
    big_df["is_SOZ"] = pd.Series([pd.NA] * len(big_df), dtype="boolean")

    # normalisation des noms d'électrode
    def row_norm(row):
        if row["dataset"] == "ds004100":
            return normalize_ds004100_elec(row["electrode"])
        else:
            return normalize_chum_elec(row["electrode"])

    big_df["electrode_norm"] = big_df.apply(row_norm, axis=1)

    # ---- ds004100 ----
    mask_ds = big_df["dataset"] == "ds004100"
    for (patient, seizure), idx_grp in big_df.loc[mask_ds].groupby(["patient", "seizure"]).groups.items():
        tsv_path = channels_tsv_path_ds004100(str(patient), int(seizure))
        if not tsv_path or not os.path.exists(tsv_path):
            continue
        soz_names = load_soz_names_from_channels_tsv(tsv_path)
        big_df.loc[idx_grp, "is_SOZ"] = False
        if soz_names:
            grp_df = big_df.loc[idx_grp]
            elec_mask = grp_df["electrode_norm"].isin(soz_names)
            if elec_mask.any():
                big_df.loc[grp_df.index[elec_mask], "is_SOZ"] = True
        else:
            if DEBUG_UNMATCHED:
                print(f"[INFO] Aucun canal SOZ détecté dans {tsv_path}")

    # ---- CHUM ----
    mask_ch = big_df["dataset"] == "CHUM"
    for (patient, seizure), idx_grp in big_df.loc[mask_ch].groupby(["patient", "seizure"]).groups.items():
        print(patient, seizure)
        csv_path = chum_soz_csv_path(str(patient), int(seizure))
        if not csv_path:
            print("not csv path", csv_path)
            continue
        soz_names = load_soz_names_from_chum_csv(csv_path)
        big_df.loc[idx_grp, "is_SOZ"] = False
        if soz_names:
            grp_df = big_df.loc[idx_grp]
            elec_mask = grp_df["electrode_norm"].isin(soz_names)
            if elec_mask.any():
                big_df.loc[grp_df.index[elec_mask], "is_SOZ"] = True
        else:
            if DEBUG_UNMATCHED:
                print(f"[INFO] Aucun canal SOZ détecté (ou fichier vide) pour {patient}, seizure {seizure} -> {csv_path}")

    # ==== first_ictal_epoch ====
    big_df["first_ictal_epoch"] = False
    ict_mask = big_df["phase"].eq("ictal") & big_df["epoch"].ge(0)

    first_ep_df = (
        big_df.loc[ict_mask, ["dataset", "patient", "seizure", "epoch"]]
              .groupby(["dataset", "patient", "seizure"], as_index=False)["epoch"]
              .min()
              .rename(columns={"epoch": "first_epoch"})
    )

    big_df = big_df.merge(first_ep_df, on=["dataset", "patient", "seizure"], how="left")
    big_df["first_ictal_epoch"] = ict_mask & big_df["epoch"].eq(big_df["first_epoch"])
    big_df.drop(columns=["first_epoch"], inplace=True)
    big_df["first_ictal_epoch"] = big_df["first_ictal_epoch"].astype(bool)

    # Colonnes bien ordonnées
    meta_cols = [
        "dataset", "outcome", "phase", "patient",
        "seizure", "epoch", "is_avg", "electrode",
        "is_SOZ", "first_ictal_epoch", "file_path"
    ]
    aux_cols = ["electrode_norm"]
    feature_cols = [c for c in big_df.columns if c not in meta_cols + aux_cols]

    # 👉 filtrage des features (inchangé pour l’instant)
    FEATURES_TO_KEEP = [
        "ratio_bg_ta",
        "ratio_gamma_delta",
        # "lvfa_score",
        # "hafa_score",
        # "slope_bg_log",
        # "dc_shift",
        "sef95_Hz",
        # "spike_rate_1_3Hz",
        "spike_sharpness",
        # "polyspike_score",
        "line_length",
        "tkeo_energy",
    ]

    FEATURES_TO_KEEP = [
    "ratio_bg_ta",
    "ratio_gamma_delta",
    #"lvfa_score",
    #"hafa_score",
    #"slope_bg_log",
    #"dc_shift",
    "sef95_Hz",
    #"spike_rate_1_3Hz",
    "spike_sharpness",
    "line_length",
    "tkeo_energy",
    "hg_power_80_150",
    "hg_over_gamma",
    "spec_slope_2_80",
    "spec_intercept_2_80",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "kurtosis",
    "skewness",
    ]

    FEATURES_TO_KEEP = [
        "burstint_ratio_bg_ta",
        "burstint_ratio_gamma_delta",
        "burstint_sef95_Hz",
        "burstint_spike_sharpness",
        "burstint_line_length",
        "burstint_tkeo_energy",
        "burstint_hg_power_80_150",
        "burstint_hg_over_gamma",
        "burstint_spec_slope_2_80",
        "burstint_spec_intercept_2_80",
        "burstint_hjorth_activity",
        "burstint_hjorth_mobility",
        "burstint_hjorth_complexity",
        "burstint_kurtosis",
        "burstint_skewness",
    ]
    selected_features = [f for f in FEATURES_TO_KEEP if f in feature_cols]

    big_df = big_df[meta_cols + selected_features]

    # ⚠️ IMPORTANT : plus AUCUNE normalisation ici.
    # Les features sont déjà znorm/prenorm dans les fichiers d'entrée.

    # Types
    big_df["seizure"] = pd.to_numeric(big_df["seizure"], errors="coerce").astype("Int64")
    big_df["epoch"] = pd.to_numeric(big_df["epoch"], errors="coerce").astype("Int64")
    big_df["is_avg"] = big_df["is_avg"].astype(bool)

    print(f"[OK] agrégation terminée : {big_df.shape[0]} lignes, {big_df.shape[1]} colonnes, depuis {n_files} fichiers.")
    print(big_df[["dataset"]].value_counts(dropna=False))
    return big_df

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df_all = collect_node_features(ROOT)

    out_csv = os.path.join(ROOT, "master_node_features_SOZ_normalized_aug_20_10_burst_norm.csv")
    out_parquet = os.path.join(ROOT, "master_node_features_SOZ_normalized_aug_20_10_burst_norm.parquet")

    try:
        df_all.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    except Exception as e:
        print(f"[WARN] Échec save CSV: {e}")

    try:
        df_all.to_parquet(out_parquet, index=False)
        print(f"[SAVE] {out_parquet}")
    except Exception as e:
        print(f"[WARN] Échec save Parquet: {e}")
