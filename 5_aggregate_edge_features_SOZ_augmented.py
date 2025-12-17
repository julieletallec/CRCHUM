# build_master_edge_features_df.py
# Version optimisée avec suivi détaillé (prints) et concat ultra-rapide

import os
import re
import pandas as pd
import numpy as np
from itertools import repeat

# ===== CONFIG =====
ROOT = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2"
DEBUG_UNMATCHED = False  # True -> log des fichiers non reconnus

# Mapping outcome pour CHUM (Engel I -> good, sinon bad)
CHUM_outcomes = {
    "Patient_01": "good outcome","Patient_02": "good outcome","Patient_07": "good outcome",
    "Patient_08": "good outcome","Patient_09": "bad outcome","Patient_11": "good outcome",
    "Patient_12": "good outcome","Patient_14": "good outcome","Patient_15": "good outcome",
    "Patient_16": "bad outcome","Patient_17": "bad outcome","Patient_19": "bad outcome",
    "Patient_21": "bad outcome","Patient_22": "good outcome","Patient_23": "bad outcome",
    "Patient_24": "good outcome","Patient_25": "good outcome",
}

IGNORE_TOKENS = ("_matrix_", "_ADJ_", "_LAPL_")

EDGE_RE = re.compile(
    r"^(?P<patient>.+?)_(?P<phase>ictal|preictal)_(?P<seizure>\d+)_EDGE_(?P<feat>corr|psi|granger|coh1245|psi1245)_(?:epoch_(?P<epoch>\d+)|avg_over_epochs)\.csv$",
    flags=re.IGNORECASE
)

FEATURE_VALUE_COL = {
    "corr":    (["pearson_r", "corr", "r"], "pearson_r"),
    "psi":     (["psi"], "psi"),
    "granger": (["granger_f", "granger", "gc", "granger_stat"], "granger_f"),
    "coh1245": (["coh_12_45", "coh"], "coh_12_45"),
    "psi1245": (["psi_12_45", "psi"], "psi_12_45"),
}


def parse_path_metadata(dirpath: str) -> dict:
    rel = os.path.relpath(dirpath, ROOT)
    parts = rel.split(os.sep)
    meta = {"dataset": None, "outcome": None, "phase": None, "patient": None}

    if len(parts) == 0 or parts[0].startswith(".."):
        return meta

    meta["dataset"] = parts[0]

    try:
        i_sc = parts.index("sc_fc")
    except ValueError:
        return meta

    if i_sc + 1 < len(parts):
        cand = parts[i_sc + 1]
        if cand not in ("ictal", "preictal", "EDGE_features_SOZ_augmented_20_10_burst"):
            meta["outcome"] = cand

    if "ictal" in parts:
        meta["phase"] = "ictal"
    if "preictal" in parts:
        meta["phase"] = "preictal"

    if "EDGE_features_SOZ_augmented_20_10_burst" in parts:
        i_ef = parts.index("EDGE_features_SOZ_augmented_20_10_burst")
        if i_ef + 1 < len(parts):
            meta["patient"] = parts[i_ef + 1]

    return meta


def _detect_value_col(df: pd.DataFrame, feat_key: str) -> tuple[str, str]:
    assert feat_key in FEATURE_VALUE_COL
    candidates, std_name = FEATURE_VALUE_COL[feat_key]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return std_name, c
    for c in df.columns:
        if c not in ("source", "target") and pd.api.types.is_numeric_dtype(df[c]):
            return std_name, c
    raise ValueError(f"Aucune colonne numérique trouvée pour '{feat_key}' (colonnes={list(df.columns)})")


def collect_edge_features(root=ROOT) -> pd.DataFrame:
    print(f"[INIT] Parcours du dossier racine : {root}")
    acc = {
        "dataset": [], "outcome": [], "phase": [], "patient": [],
        "seizure": [], "epoch": [], "is_avg": [],
        "source": [], "target": [], "feature": [], "value": [],
        "file_path": []
    }
    n_files = 0
    n_dirs = 0

    for dirpath, _, filenames in os.walk(root):
        if "EDGE_features_SOZ_augmented_20_10_burst" not in dirpath:
            continue
        n_dirs += 1
        print(f"[SCAN] Dossier {n_dirs}: {dirpath} ({len(filenames)} fichiers)")

        meta_path = parse_path_metadata(dirpath)
        if meta_path["dataset"] is None or meta_path["patient"] is None:
            continue

        for fname in filenames:
            if not fname.endswith(".csv"):
                continue
            if any(tok in fname for tok in IGNORE_TOKENS):
                continue

            fullpath = os.path.join(dirpath, fname)
            m = EDGE_RE.match(fname)
            if m is None:
                if DEBUG_UNMATCHED and "_EDGE" in fname:
                    print(f"[SKIP-NON-MATCH] {fullpath}")
                continue

            g = m.groupdict()
            patient = g["patient"]
            phase = g["phase"]
            seizure = int(g["seizure"])
            epoch = int(g["epoch"]) if g.get("epoch") is not None else -1
            is_avg = (epoch == -1)
            feat_key = g["feat"].lower()

            dataset = meta_path["dataset"]
            outcome = meta_path["outcome"]
            path_patient = meta_path["patient"] or patient

            try:
                df = pd.read_csv(fullpath)
            except Exception as e:
                print(f"[WARN] lecture impossible: {fullpath} -> {e}")
                continue

            if not {"source", "target"} <= set(df.columns):
                if DEBUG_UNMATCHED:
                    print(f"[WARN] colonnes manquantes (source/target) dans {fullpath}")
                continue

            try:
                std_feat_name, val_col = _detect_value_col(df, feat_key)
            except Exception as e:
                if DEBUG_UNMATCHED:
                    print(f"[WARN] aucune valeur numérique dans {fullpath}: {e}")
                continue

            n = len(df)
            if n == 0:
                continue

            # Ajout rapide
            acc["dataset"].extend(repeat(dataset, n))
            acc["outcome"].extend(repeat(outcome, n))
            acc["phase"].extend(repeat(phase, n))
            acc["patient"].extend(repeat(path_patient, n))
            acc["seizure"].extend(repeat(seizure, n))
            acc["epoch"].extend(repeat(epoch, n))
            acc["is_avg"].extend(repeat(is_avg, n))
            acc["feature"].extend(repeat(std_feat_name, n))
            acc["file_path"].extend(repeat(fullpath, n))
            acc["source"].extend(df["source"].tolist())
            acc["target"].extend(df["target"].tolist())
            acc["value"].extend(df[val_col].tolist())

            n_files += 1
            if n_files % 200 == 0:
                print(f"[PROGRESS] {n_files} fichiers traités...")

    print(f"[DONE] {n_files} fichiers lus sur {n_dirs} dossiers.")
    if n_files == 0 or len(acc["dataset"]) == 0:
        print("[INFO] Aucun fichier EDGE trouvé.")
        return pd.DataFrame()

    print("[BUILD] Création du DataFrame principal (long_df)")
    long_df = pd.DataFrame(acc)
    print(f"[INFO] long_df: {long_df.shape[0]} lignes, {long_df.shape[1]} colonnes")

    for c in ["dataset","outcome","phase","patient","source","target","feature"]:
        long_df[c] = long_df[c].astype("category")

    print("[STEP] Remplissage outcome CHUM")
    mask_chum = long_df["dataset"].eq("CHUM")
    long_df.loc[mask_chum, "outcome"] = long_df.loc[mask_chum, "patient"].map(CHUM_outcomes)

    long_df["seizure"] = np.int32(long_df["seizure"])
    long_df["epoch"] = np.int32(long_df["epoch"])
    long_df["is_avg"] = long_df["is_avg"].astype(bool)

    # =============================
    # === Agrégation & Pivot ===
    # =============================
    print("[STEP] Agrégation et pivot en cours...")

    idx_cols = ["dataset","outcome","phase","patient","seizure","epoch","is_avg","source","target","feature"]
    long_df = long_df.sort_values(idx_cols, kind="stable")
    flat = (
        long_df.groupby(idx_cols, sort=False, observed=True, as_index=False)
               .agg(value=("value","first"), file_path=("file_path","first"))
    )
    print(f"[INFO] flat: {flat.shape[0]} lignes après agrégation.")

    print("[STEP] Pivot des features (wide_df)")
    wide_df = (
        flat.pivot(index=idx_cols[:-1], columns="feature", values="value")
            .rename_axis(None, axis=1)
            .reset_index()
    )

    print(f"[INFO] wide_df: {wide_df.shape[0]} lignes, {wide_df.shape[1]} colonnes")

    print("[STEP] Ajout de file_path (mapping unique)")
    path_first = flat.drop(columns=["value"]).drop_duplicates(idx_cols, keep="first")
    path_map = (
        path_first.drop(columns=["feature"])
                  .drop_duplicates(idx_cols[:-1], keep="first")
    )
    wide_df = wide_df.merge(path_map, on=idx_cols[:-1], how="left")

    print("[STEP] Calcul de first_ictal_epoch")
    wide_df["first_ictal_epoch"] = False
    ict_mask = wide_df["phase"].eq("ictal") & wide_df["epoch"].ge(0)
    grp = ["dataset","patient","seizure"]
    min_epoch = (
        wide_df.loc[ict_mask]
               .groupby(grp, sort=False)["epoch"]
               .transform("min")
    )
    wide_df.loc[ict_mask, "first_ictal_epoch"] = (
        wide_df.loc[ict_mask, "epoch"].to_numpy() == min_epoch.to_numpy()
    )

    print("[STEP] Ajustement des types finaux")
    if not wide_df["seizure"].isna().any():
        wide_df["seizure"] = wide_df["seizure"].astype("int32")
    else:
        wide_df["seizure"] = wide_df["seizure"].astype("Int64")

    if not wide_df["epoch"].isna().any():
        wide_df["epoch"] = wide_df["epoch"].astype("int32")
    else:
        wide_df["epoch"] = wide_df["epoch"].astype("Int64")

    wide_df["is_avg"] = wide_df["is_avg"].astype(bool)
            
    meta_cols = ["dataset","outcome","phase","patient","seizure","epoch","is_avg","first_ictal_epoch"]
    edge_cols = ["source","target"]
    wanted = ["pearson_r","psi","granger_f","coh_12_45","psi_12_45"]
    feature_cols_all = [c for c in wanted if c in wide_df.columns]

    final_cols = meta_cols + edge_cols + feature_cols_all + (["file_path"] if "file_path" in wide_df.columns else [])
    wide_df = wide_df[final_cols]

    print(f"[OK] Agrégation finale : {wide_df.shape[0]} lignes, {wide_df.shape[1]} colonnes (depuis {n_files} fichiers).")
    return wide_df


if __name__ == "__main__":
    print("[RUN] Début du script principal")
    df_edges = collect_edge_features(ROOT)

    out_csv = os.path.join(ROOT, "master_edge_features_SOZ_aug_20_10_burst.csv")
    out_parquet = os.path.join(ROOT, "master_edge_features_SOZ_aug_20_10_burst.parquet")

    print("[SAVE] Sauvegarde CSV...")
    try:
        df_edges.to_csv(out_csv, index=False)
        print(f"[SAVE-OK] {out_csv}")
    except Exception as e:
        print(f"[WARN] Échec save CSV: {e}")

    print("[SAVE] Sauvegarde Parquet...")
    try:
        df_edges.to_parquet(out_parquet, index=False)
        print(f"[SAVE-OK] {out_parquet}")
    except Exception as e:
        print(f"[WARN] Échec save Parquet: {e}")

    print("[DONE] Script terminé ✅")
