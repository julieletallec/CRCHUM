#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
uv run z_visualise_features_for_M1.py \
    //home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2 \
    CHUM::Patient_01 \
    CHUM::Patient_02 \
    CHUM::Patient_07 \
    CHUM::Patient_11 \
    CHUM::Patient_14 \
    CHUM::Patient_22 \
    ds004100::sub-HUP074 \
    ds004100::sub-HUP082 \
    ds004100::sub-HUP089 \
    ds004100::sub-HUP097 \
    ds004100::sub-HUP107 \
    ds004100::sub-HUP111 \
    ds004100::sub-HUP141 \
    ds004100::sub-HUP144 \
    ds004100::sub-HUP148 \
    ds004100::sub-HUP150 \
    ds004100::sub-HUP157 \
    ds004100::sub-HUP173 \
    ds004100::sub-HUP180 \
    ds004100::sub-HUP185 \
    --save-dir ./figures_features_M1_for_thesis

Exemple d'appel :

uv run z_visualise_features_for_M1.py \
    //home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2 \
    CHUM::Patient_01 \
    CHUM::Patient_02 \
    CHUM::Patient_07 \
    CHUM::Patient_11 \
    CHUM::Patient_14 \
    CHUM::Patient_22 \
    ds004100::sub-HUP074 \
    ds004100::sub-HUP082 \
    ds004100::sub-HUP089 \
    ds004100::sub-HUP097 \
    ds004100::sub-HUP107 \
    ds004100::sub-HUP111 \
    ds004100::sub-HUP141 \
    ds004100::sub-HUP144 \
    ds004100::sub-HUP148 \
    ds004100::sub-HUP150 \
    ds004100::sub-HUP157 \
    ds004100::sub-HUP173 \
    ds004100::sub-HUP180 \
    ds004100::sub-HUP185 \
    CHUM::Patient_09 \
    CHUM::Patient_16 \
    CHUM::Patient_17 \
    CHUM::Patient_21 \
    ds004100::sub-HUP080 \
    ds004100::sub-HUP112 \
    ds004100::sub-HUP114 \
    ds004100::sub-HUP133 \
    ds004100::sub-HUP138 \
    ds004100::sub-HUP151 \
    ds004100::sub-HUP162 \
    ds004100::sub-HUP171 \
    ds004100::sub-HUP172 \
    ds004100::sub-HUP181 \
    ds004100::sub-HUP187 \
    ds004100::sub-HUP188 \
    --save-dir ./new_M1_features_SOZ





uv run z_visualise_features_for_M1.py \
    //home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2 \
    CHUM::Patient_09 \
    CHUM::Patient_16 \
    CHUM::Patient_17 \
    CHUM::Patient_21 \
    ds004100::sub-HUP080 \
    ds004100::sub-HUP112 \
    ds004100::sub-HUP114 \
    ds004100::sub-HUP133 \
    ds004100::sub-HUP138 \
    ds004100::sub-HUP151 \
    ds004100::sub-HUP162 \
    ds004100::sub-HUP171 \
    ds004100::sub-HUP172 \
    ds004100::sub-HUP181 \
    ds004100::sub-HUP187 \
    ds004100::sub-HUP188 \
    --save-dir ./figures_features_M1_10_20_burst_all_patients
"""


ROOT_DEFAULT = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2"

EPOCH_DURATION_SEC = 1.0
N_SEC_PRE = 5
N_SEC_ICT = 5

N_EPOCHS_PRE = int(N_SEC_PRE / EPOCH_DURATION_SEC)
N_EPOCHS_ICT = int(N_SEC_ICT / EPOCH_DURATION_SEC)

# =========================
# Features à tracer / analyser
# =========================
EXPLICIT_NODE_FEATURES = [
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


def pretty_feat(f: str) -> str:
    """Remove 'burstint_' prefix for nicer plotting."""
    return f.replace("burstint_", "")


# =========================
# Helpers IDs
# =========================
def parse_dataset_and_patient(raw_id: str):
    """
    Accepte:
      - 'CHUM::Patient_01'
      - 'ds004100::sub-HUP070'
      - 'Patient_01'   -> dataset=CHUM
      - 'sub-HUP070'   -> dataset=ds004100
    Retourne (dataset, patient_id)
    """
    raw_id = raw_id.strip()
    if "::" in raw_id:
        ds, pat = raw_id.split("::", 1)
        return ds.strip(), pat.strip()

    if raw_id.startswith("sub-"):
        return "ds004100", raw_id
    if raw_id.startswith("Patient_"):
        return "CHUM", raw_id
    return "CHUM", raw_id


# =========================
# Manifest epochs sélectionnés
# =========================
def load_selected_epochs_from_manifest(root_base, dataset, patient_id, seizure_num, verbose=True):
    """
    Cherche:
      - CHUM      -> ROOT/CHUM/sc_fc/selection_manifest_CHUM.csv
      - ds004100  -> ROOT/ds004100/sc_fc/selection_manifest_ds004100.csv (si présent)
    Retourne {'preictal':[...], 'ictal':[...]} ou None si pas de manifest.
    """
    root_base = Path(root_base)
    if dataset == "CHUM":
        manifest_path = root_base / "CHUM" / "sc_fc" / "selection_manifest_CHUM.csv"
    else:
        manifest_path = root_base / dataset / "sc_fc" / f"selection_manifest_{dataset}.csv"

    if verbose:
        print(f"[INFO] Manifest ({dataset}): {manifest_path}")

    if not manifest_path.exists():
        if verbose:
            print(f"[WARN] Manifest introuvable: {manifest_path} (fallback: tous les epochs)")
        return None

    man = pd.read_csv(manifest_path)

    out = {}
    for phase in ["preictal", "ictal"]:
        rows = man[
            (man["patient"] == patient_id)
            & (man["seizure"] == int(seizure_num))
            & (man["phase"] == phase)
            & (man["status"] == "ok")
        ]
        if verbose:
            print(f"[INFO] {dataset} {patient_id} seizure {seizure_num} / {phase}: {len(rows)} ligne(s)")

        if len(rows) == 0:
            out[phase] = []
            continue

        selected_str = rows.iloc[0]["selected_epoch_ids"]
        try:
            epoch_ids = json.loads(selected_str)
        except Exception as e:
            if verbose:
                print(f"[WARN] JSON parse failed selected_epoch_ids ({dataset} {patient_id} {phase}): {e}")
            epoch_ids = []

        out[phase] = epoch_ids
        if verbose:
            print(f"[INFO] {phase}: {epoch_ids}")

    return out


# =========================
# Chargement parquet
# =========================
def load_master_parquet(root_base, verbose=True):
    """
    Charge ROOT/master_node_features_SOZ_normalized_aug_20_10_burst_figures.parquet
    """
    root_base = Path(root_base)
    parquet_path = root_base / "master_node_features_SOZ_normalized_aug_20_10_burst_figure.parquet"
    if verbose:
        print(f"[INFO] Chargement parquet: {parquet_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if verbose:
        print(f"[INFO] Parquet chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# =========================
# Préparation des données
# =========================
def prepare_data_for_dataset_patient_seizure(df_all, root_base, dataset, patient_id, seizure_num, verbose=True):
    """
    Retourne (epochs_selected, df_sub) où df_sub = lignes parquet pour ce triplet (is_avg=False)

    Supporte un "fenêtrage" des epochs (ex: 5 dernières préictales + 5 premières ictales)
    qui s'applique autant si les epochs viennent du manifest que du fallback.
    """
    df_sub = df_all[
        (df_all["dataset"] == dataset)
        & (df_all["patient"] == patient_id)
        & (df_all["seizure"] == int(seizure_num))
        & (df_all["is_avg"] == False)
    ].copy()

    if df_sub.empty:
        if verbose:
            print(f"[ERROR] Aucune ligne parquet pour {dataset}, {patient_id}, seizure {seizure_num}")
        return None, None

    # 1) Essaye de charger la sélection depuis le manifest
    epochs_selected = load_selected_epochs_from_manifest(
        root_base, dataset, patient_id, seizure_num, verbose=verbose
    )

    # 2) Fallback: tous les epochs dispo si pas de manifest
    if epochs_selected is None:
        pre_epochs_all = sorted(
            df_sub.loc[df_sub["phase"] == "preictal", "epoch"].dropna().unique().tolist()
        )
        ict_epochs_all = sorted(
            df_sub.loc[df_sub["phase"] == "ictal", "epoch"].dropna().unique().tolist()
        )
        epochs_selected = {"preictal": pre_epochs_all, "ictal": ict_epochs_all}
        if verbose:
            print(
                f"[INFO] Fallback epochs {dataset} {patient_id} seizure {seizure_num}: "
                f"pre={len(pre_epochs_all)}, ict={len(ict_epochs_all)}"
            )

    # 3) Fenêtrage (marche pour manifest OU fallback)
    pre_list = list(epochs_selected.get("preictal", []))
    ict_list = list(epochs_selected.get("ictal", []))

    if N_EPOCHS_PRE and len(pre_list) > N_EPOCHS_PRE:
        pre_list = pre_list[-N_EPOCHS_PRE:]
    if N_EPOCHS_ICT and len(ict_list) > N_EPOCHS_ICT:
        ict_list = ict_list[:N_EPOCHS_ICT]

    epochs_selected["preictal"] = pre_list
    epochs_selected["ictal"] = ict_list

    if verbose:
        print(f"[INFO] Windowed epochs: pre={epochs_selected['preictal']}, ict={epochs_selected['ictal']}")

    return epochs_selected, df_sub


# =========================
# (1) Courbes temporelles par feature
# =========================
def plot_timecourses_per_patient_seizure_feature(
    df_all,
    root_base,
    dataset,
    patient_id,
    seizure_num,
    show=True,
    save_dir=None,
    verbose=True,
):
    """
    Une figure par feature pour (dataset, patient, seizure).
    Chaque courbe = une électrode (préictal puis ictal concaténés),
    couleur selon SOZ (rouge) vs non-SOZ (noir).
    """
    print(f"\n=== TIMECOURSE {dataset} | {patient_id} | Seizure {seizure_num} ===")

    epochs_selected, df_sub = prepare_data_for_dataset_patient_seizure(
        df_all, root_base, dataset, patient_id, seizure_num, verbose=verbose
    )
    if epochs_selected is None or df_sub is None:
        return

    pre_epochs = epochs_selected.get("preictal", [])
    ict_epochs = epochs_selected.get("ictal", [])

    df_pre = df_sub[(df_sub["phase"] == "preictal") & (df_sub["epoch"].isin(pre_epochs))].copy()
    df_ict = df_sub[(df_sub["phase"] == "ictal") & (df_sub["epoch"].isin(ict_epochs))].copy()

    if df_pre.empty and df_ict.empty:
        print(f"[ERROR] Aucune donnée (epochs sélectionnés) pour {dataset} {patient_id} seizure {seizure_num}")
        return

    if not df_pre.empty:
        df_pre["epoch"] = df_pre["epoch"].astype(int)
    if not df_ict.empty:
        df_ict["epoch"] = df_ict["epoch"].astype(int)

    feature_cols = [c for c in EXPLICIT_NODE_FEATURES if c in df_sub.columns]
    if not feature_cols:
        print(f"[WARN] Aucune feature trouvée parmi EXPLICIT_NODE_FEATURES pour {dataset} {patient_id}")
        return

    df_combined = pd.concat([df_pre, df_ict], ignore_index=True)

    # map SOZ
    def first_non_null_is_soz(s):
        s_valid = s.dropna()
        if len(s_valid) == 0:
            return 0
        return int(bool(s_valid.iloc[0]))

    soz_map = df_combined.groupby("electrode")["is_SOZ"].apply(first_non_null_is_soz).to_dict()

    def color_for_elec(e):
        return "r" if soz_map.get(e, 0) == 1 else "k"

    # save dir
    if save_dir is not None:
        base_save_dir = Path(save_dir)
        save_dir_patient = base_save_dir / dataset / patient_id
        save_dir_patient.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[INFO] Save dir: {save_dir_patient.resolve()}")
    else:
        save_dir_patient = None

    electrodes = sorted(df_combined["electrode"].dropna().unique().tolist())
    total_epochs = len(pre_epochs) + len(ict_epochs)
    x = np.arange(total_epochs)
    transition_idx = len(pre_epochs)

    # pivots pour accès rapide
    pre_pivots, ict_pivots = {}, {}
    for feat in feature_cols:
        pre_pivots[feat] = df_pre.pivot(index="epoch", columns="electrode", values=feat) if not df_pre.empty else None
        ict_pivots[feat] = df_ict.pivot(index="epoch", columns="electrode", values=feat) if not df_ict.empty else None

    for feat in feature_cols:
        plt.figure(figsize=(10, 5))

        pre_piv = pre_pivots[feat]
        ict_piv = ict_pivots[feat]

        non_soz_plotted, soz_plotted = False, False

        for elec in electrodes:
            y = []

            if pre_piv is not None and elec in pre_piv.columns:
                y.extend(pre_piv[elec].reindex(pre_epochs).values.tolist())
            else:
                y.extend([np.nan] * len(pre_epochs))

            if ict_piv is not None and elec in ict_piv.columns:
                y.extend(ict_piv[elec].reindex(ict_epochs).values.tolist())
            else:
                y.extend([np.nan] * len(ict_epochs))

            c = color_for_elec(elec)
            label = None
            if c == "k" and not non_soz_plotted:
                label = "non-SOZ"
                non_soz_plotted = True
            elif c == "r" and not soz_plotted:
                label = "SOZ"
                soz_plotted = True

            plt.plot(x, y, color=c, alpha=0.7, linewidth=1.0, label=label)

        if len(pre_epochs) > 0 and len(ict_epochs) > 0:
            plt.axvline(x=transition_idx - 0.5, color="gray", linestyle="--")
            ymin, ymax = plt.ylim()
            plt.text(
                transition_idx - 0.5,
                ymax,
                "preictal → ictal",
                rotation=90,
                va="top",
                ha="right",
                color="gray",
            )

        plt.xlabel("Epoch (préictal puis ictal)")
        plt.ylabel(feat)
        plt.title(f"{dataset} | {patient_id} | seizure {seizure_num} | {feat}")
        plt.legend()
        plt.tight_layout()

        if save_dir_patient is not None:
            out_path = save_dir_patient / f"{patient_id}_seiz{seizure_num}_{feat}_timecourse.png"
            plt.savefig(out_path, dpi=150)
            if verbose:
                print(f"[INFO] Saved: {out_path.name}")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_timecourses_per_patient_seizure_feature_thesis(
    df_all,
    root_base,
    dataset,
    patient_id,
    seizure_num,
    show=True,
    save_dir=None,
    verbose=True,
):
    """
    Une figure par feature pour (dataset, patient, seizure).
    Chaque courbe = une électrode (préictal puis ictal concaténés),
    couleur selon SOZ (rouge) vs non-SOZ (noir).
    """
    print(f"\n=== TIMECOURSE {dataset} | {patient_id} | Seizure {seizure_num} ===")

    epochs_selected, df_sub = prepare_data_for_dataset_patient_seizure(
        df_all, root_base, dataset, patient_id, seizure_num, verbose=verbose
    )
    if epochs_selected is None or df_sub is None:
        return

    pre_epochs = epochs_selected.get("preictal", [])
    ict_epochs = epochs_selected.get("ictal", [])

    df_pre = df_sub[(df_sub["phase"] == "preictal") & (df_sub["epoch"].isin(pre_epochs))].copy()
    df_ict = df_sub[(df_sub["phase"] == "ictal") & (df_sub["epoch"].isin(ict_epochs))].copy()

    if df_pre.empty and df_ict.empty:
        print(f"[ERROR] Aucune donnée (epochs sélectionnés) pour {dataset} {patient_id} seizure {seizure_num}")
        return

    if not df_pre.empty:
        df_pre["epoch"] = df_pre["epoch"].astype(int)
    if not df_ict.empty:
        df_ict["epoch"] = df_ict["epoch"].astype(int)

    feature_cols = [c for c in EXPLICIT_NODE_FEATURES if c in df_sub.columns]
    if not feature_cols:
        print(f"[WARN] Aucune feature trouvée parmi EXPLICIT_NODE_FEATURES pour {dataset} {patient_id}")
        return

    df_combined = pd.concat([df_pre, df_ict], ignore_index=True)

    # map SOZ
    def first_non_null_is_soz(s):
        s_valid = s.dropna()
        if len(s_valid) == 0:
            return 0
        return int(bool(s_valid.iloc[0]))

    soz_map = df_combined.groupby("electrode")["is_SOZ"].apply(first_non_null_is_soz).to_dict()

    def color_for_elec(e):
        return "r" if soz_map.get(e, 0) == 1 else "k"

    # save dir
    if save_dir is not None:
        base_save_dir = Path(save_dir)
        save_dir_patient = base_save_dir / dataset / patient_id
        save_dir_patient.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[INFO] Save dir: {save_dir_patient.resolve()}")
    else:
        save_dir_patient = None

    electrodes = sorted(df_combined["electrode"].dropna().unique().tolist())
    total_epochs = len(pre_epochs) + len(ict_epochs)
    x = np.arange(total_epochs)
    transition_idx = len(pre_epochs)

    # pivots pour accès rapide
    pre_pivots, ict_pivots = {}, {}
    for feat in feature_cols:
        pre_pivots[feat] = df_pre.pivot(index="epoch", columns="electrode", values=feat) if not df_pre.empty else None
        ict_pivots[feat] = df_ict.pivot(index="epoch", columns="electrode", values=feat) if not df_ict.empty else None

    for feat in feature_cols:
        plt.figure(figsize=(10, 5))

        pre_piv = pre_pivots[feat]
        ict_piv = ict_pivots[feat]

        non_soz_plotted, soz_plotted = False, False

        for elec in electrodes:
            y = []

            if pre_piv is not None and elec in pre_piv.columns:
                y.extend(pre_piv[elec].reindex(pre_epochs).values.tolist())
            else:
                y.extend([np.nan] * len(pre_epochs))

            if ict_piv is not None and elec in ict_piv.columns:
                y.extend(ict_piv[elec].reindex(ict_epochs).values.tolist())
            else:
                y.extend([np.nan] * len(ict_epochs))

            c = color_for_elec(elec)
            label = None
            if c == "k" and not non_soz_plotted:
                label = "non-SOZ"
                non_soz_plotted = True
            elif c == "r" and not soz_plotted:
                label = "SOZ"
                soz_plotted = True

            plt.plot(x, y, color=c, alpha=0.7, linewidth=1.0, label=label)

        if len(pre_epochs) > 0 and len(ict_epochs) > 0:
            plt.axvline(x=transition_idx - 0.5, color="gray", linestyle="--")
            ymin, ymax = plt.ylim()
            plt.text(
                transition_idx - 0.5,
                ymax,
                "preictal → ictal",
                rotation=90,
                va="top",
                ha="right",
                color="gray",
            )

        plt.xlabel("Epoch (1 epoch = 1 sec)")
        plt.ylabel(feat)
        plt.title(f"Burst-integrated per-epoch feature values (persistent activity)")
        plt.legend()
        plt.tight_layout()

        if save_dir_patient is not None:
            out_path = save_dir_patient / f"{patient_id}_seiz{seizure_num}_{feat}_timecourse.png"
            plt.savefig(out_path, dpi=150)
            if verbose:
                print(f"[INFO] Saved: {out_path.name}")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_timecourses_for_multiple_patients(root_base, patient_tokens, show=True, save_dir=None, verbose=True):
    df_all = load_master_parquet(root_base, verbose=verbose)
    parsed = [parse_dataset_and_patient(tok) for tok in patient_tokens]

    for raw_tok, (dataset, patient_id) in zip(patient_tokens, parsed):
        if verbose:
            print(f"\n[INFO] Patient: {dataset} | {patient_id}")

        df_p = df_all[
            (df_all["dataset"] == dataset)
            & (df_all["patient"] == patient_id)
            & (df_all["is_avg"] == False)
        ]
        if df_p.empty:
            print(f"[WARN] Aucun enregistrement pour {dataset} | {patient_id} -> skip")
            continue

        seizures = sorted(df_p["seizure"].dropna().unique().tolist())
        print(f"[INFO] Seizures: {seizures}")

        for seiz in seizures:
            plot_timecourses_per_patient_seizure_feature_thesis(
                df_all=df_all,
                root_base=root_base,
                dataset=dataset,
                patient_id=patient_id,
                seizure_num=int(seiz),
                show=show,
                save_dir=save_dir,
                verbose=verbose,
            )


# =========================
# (2) Métriques simples de changement mean(pre) -> mean(ictal)
# =========================
def compute_mean_change_metrics_for_patients(
    root_base,
    patient_tokens,
    verbose=True,
):
    """
    Pour chaque (patient, seizure, feature):
      - change_elec = mean_ictal(elec) - mean_preictal(elec)

    Puis agrégation par groupes d'électrodes:
      - change_global_abs = mean_elec( |change_elec| )
      - change_soz_abs    = mean_elec_SOZ( |change_elec| )
      - change_nonsoz_abs = mean_elec_nonSOZ( |change_elec| )
      - soz_amp_abs       = change_soz_abs - change_nonsoz_abs

    Sortie: df_metrics (une ligne par patient_key, seizure, feature)
    """
    root_base = Path(root_base)
    df_all = load_master_parquet(root_base, verbose=verbose)

    rows = []
    parsed = [parse_dataset_and_patient(tok) for tok in patient_tokens]

    for (dataset, patient_id) in parsed:
        if verbose:
            print(f"\n[METRIC] {dataset} | {patient_id}")

        df_p = df_all[
            (df_all["dataset"] == dataset)
            & (df_all["patient"] == patient_id)
            & (df_all["is_avg"] == False)
        ]
        if df_p.empty:
            if verbose:
                print("[METRIC] no data -> skip")
            continue

        seizures = sorted(df_p["seizure"].dropna().unique().tolist())
        if verbose:
            print(f"[METRIC] seizures = {seizures}")

        for seiz in seizures:
            epochs_selected, df_sub = prepare_data_for_dataset_patient_seizure(
                df_all, root_base, dataset, patient_id, int(seiz), verbose=verbose
            )
            if epochs_selected is None or df_sub is None:
                continue

            pre_epochs = epochs_selected.get("preictal", [])
            ict_epochs = epochs_selected.get("ictal", [])

            df_pre = df_sub[(df_sub["phase"] == "preictal") & (df_sub["epoch"].isin(pre_epochs))].copy()
            df_ict = df_sub[(df_sub["phase"] == "ictal") & (df_sub["epoch"].isin(ict_epochs))].copy()

            if df_pre.empty or df_ict.empty:
                if verbose:
                    print(f"[METRIC] seizure {seiz}: pre ou ict vide -> skip")
                continue

            feature_cols = [c for c in EXPLICIT_NODE_FEATURES if c in df_sub.columns]
            if not feature_cols:
                continue

            # map SOZ
            df_combined = pd.concat([df_pre, df_ict], ignore_index=True)

            def first_non_null_is_soz(s):
                s_valid = s.dropna()
                if len(s_valid) == 0:
                    return 0
                return int(bool(s_valid.iloc[0]))

            soz_map = df_combined.groupby("electrode")["is_SOZ"].apply(first_non_null_is_soz).to_dict()
            soz_elec = {e for e, v in soz_map.items() if v == 1}
            nonsoz_elec = set(soz_map.keys()) - soz_elec

            electrodes = sorted(
                set(df_pre["electrode"].dropna().unique()).union(df_ict["electrode"].dropna().unique())
            )
            if len(electrodes) == 0:
                continue

            # pré-calcul des moyennes par (electrode, feature)
            pre_mean = df_pre.groupby("electrode")[feature_cols].mean(numeric_only=True)
            ict_mean = df_ict.groupby("electrode")[feature_cols].mean(numeric_only=True)

            for feat in feature_cols:
                s_pre = pre_mean[feat] if feat in pre_mean.columns else pd.Series(dtype=float)
                s_ict = ict_mean[feat] if feat in ict_mean.columns else pd.Series(dtype=float)

                s = (s_ict - s_pre).reindex(electrodes).astype(float)
                s_abs = s.abs()

                if s_abs.dropna().empty:
                    continue

                global_abs = float(s_abs.mean(skipna=True))

                soz_idx = [e for e in electrodes if e in soz_elec]
                nonsoz_idx = [e for e in electrodes if e in nonsoz_elec]

                soz_abs = float(s_abs.reindex(soz_idx).mean(skipna=True)) if soz_idx else np.nan
                nonsoz_abs = float(s_abs.reindex(nonsoz_idx).mean(skipna=True)) if nonsoz_idx else np.nan

                soz_amp_abs = (soz_abs - nonsoz_abs) if (np.isfinite(soz_abs) and np.isfinite(nonsoz_abs)) else np.nan

                rows.append(
                    dict(
                        dataset=dataset,
                        patient=patient_id,
                        patient_key=f"{dataset}::{patient_id}",
                        seizure=int(seiz),
                        feature=feat,
                        change_global_abs=global_abs,
                        change_soz_abs=soz_abs,
                        change_nonsoz_abs=nonsoz_abs,
                        change_soz_amp_abs=soz_amp_abs,
                    )
                )

    return pd.DataFrame(rows)


# =========================
# (3) Heatmaps
# =========================
def plot_heatmap_features_x_patients(df_metrics, value_col, out_path, title, symmetric=False, per_patient_scale=False, verbose=True):
    """
    Heatmap unique:
      - lignes: features
      - colonnes: patients
      - valeur: moyenne (sur seizures) de df_metrics[value_col]
    """
    if df_metrics.empty:
        if verbose:
            print("[HEATMAP] df_metrics vide -> skip")
        return
    if value_col not in df_metrics.columns:
        raise ValueError(f"{value_col} absent de df_metrics")

    features = sorted(df_metrics["feature"].unique().tolist())
    patients = sorted(df_metrics["patient_key"].unique().tolist())
    mat = np.full((len(features), len(patients)), np.nan, dtype=float)

    feat_to_i = {f: i for i, f in enumerate(features)}
    pat_to_j = {p: j for j, p in enumerate(patients)}

    for (pk, feat), g in df_metrics.groupby(["patient_key", "feature"]):
        i = feat_to_i[feat]
        j = pat_to_j[pk]
        mat[i, j] = float(g[value_col].mean())

    if symmetric:
        vmax = float(np.nanmax(np.abs(mat)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmax = float(np.nanmax(mat))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = 0.0
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(patients), 0.5 * len(features) + 1))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    ax.set_xticks(np.arange(len(patients)))
    ax.set_xticklabels(patients, rotation=90)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[HEATMAP] Saved: {out_path.resolve()}")


def plot_heatmaps_per_patient_features_x_seizures(
    df_metrics,
    value_col,
    out_path,
    title,
    symmetric=False,
    per_patient_scale=True,
    verbose=True,
):
    """
    Une figure avec un subplot (heatmap) par patient:
      - lignes: features
      - colonnes: seizures
      - valeurs: df_metrics[value_col]
    Si per_patient_scale=True -> échelle rescalée pour chaque patient.

    MODIFS:
      - enlève 'burstint_' sur les y-ticks
      - ajoute UNE seule colorbar globale en bas même si per_patient_scale=True
        (échelle = moyenne des échelles patient)
    """
    if df_metrics.empty:
        if verbose:
            print("[HEATMAP-PER-PAT] df_metrics vide -> skip")
        return
    if value_col not in df_metrics.columns:
        raise ValueError(f"{value_col} absent de df_metrics")

    patients = sorted(df_metrics["patient_key"].unique().tolist())
    features = sorted(df_metrics["feature"].unique().tolist())
    if len(patients) == 0:
        return

    feat_to_i = {f: i for i, f in enumerate(features)}

    n_pat = len(patients)
    nrows = min(2, n_pat)
    ncols = int(math.ceil(n_pat / nrows))

    fig_height = max(2.5, 0.25 * len(features)) * nrows
    fig_width = max(10.0, 3.0 * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    last_im = None

    cmap = "coolwarm" if symmetric else "viridis"

    # échelle globale si pas per_patient_scale
    if not per_patient_scale:
        vals = df_metrics[value_col].values.astype(float)
        if symmetric:
            vmax = float(np.nanmax(np.abs(vals)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            vmin = -vmax
        else:
            vmax = float(np.nanmax(vals))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            vmin = 0.0

    # collecter les échelles patient pour fabriquer UNE colorbar "moyenne"
    patient_vmins = []
    patient_vmaxs = []

    for idx, pk in enumerate(patients):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        g_pat = df_metrics[df_metrics["patient_key"] == pk].copy()
        seizures = sorted(g_pat["seizure"].unique().tolist())
        if len(seizures) == 0:
            ax.set_title(pk)
            ax.axis("off")
            continue

        seiz_to_j = {s: j for j, s in enumerate(seizures)}
        mat = np.full((len(features), len(seizures)), np.nan, dtype=float)

        for (feat, seiz), g in g_pat.groupby(["feature", "seizure"]):
            i = feat_to_i[feat]
            j = seiz_to_j[seiz]
            mat[i, j] = float(g[value_col].mean())

        # échelle locale si demandé
        if per_patient_scale:
            vals_pat = g_pat[value_col].values.astype(float)
            if symmetric:
                vmax_use = float(np.nanmax(np.abs(vals_pat)))
                if not np.isfinite(vmax_use) or vmax_use == 0:
                    vmax_use = 1.0
                vmin_use = -vmax_use
            else:
                vmax_use = float(np.nanmax(vals_pat))
                if not np.isfinite(vmax_use) or vmax_use == 0:
                    vmax_use = 1.0
                vmin_use = 0.0

            patient_vmins.append(vmin_use)
            patient_vmaxs.append(vmax_use)
        else:
            vmin_use, vmax_use = vmin, vmax

        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin_use, vmax=vmax_use, cmap=cmap)
        last_im = im

        ax.set_title(pk, fontsize=9)
        if c == 0:
            ax.set_yticks(np.arange(len(features)))
            ax.set_yticklabels([pretty_feat(f) for f in features], fontsize=7)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_xticks(np.arange(len(seizures)))
        ax.set_xticklabels(seizures, rotation=90, fontsize=7)
        ax.set_xlabel("seizure", fontsize=8)

    # supprime axes en trop
    for idx in range(n_pat, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        fig.delaxes(axes[r, c])

    # UNE colorbar globale en bas
    if last_im is not None:
        if per_patient_scale:
            # moyenne des échelles patient
            vmin_bar = float(np.nanmean(patient_vmins)) if patient_vmins else (-1.0 if symmetric else 0.0)
            vmax_bar = float(np.nanmean(patient_vmaxs)) if patient_vmaxs else 1.0

            if not np.isfinite(vmin_bar):
                vmin_bar = -1.0 if symmetric else 0.0
            if not np.isfinite(vmax_bar) or vmax_bar == 0:
                vmax_bar = 1.0

            if symmetric:
                vmax_bar = float(np.nanmean([abs(v) for v in patient_vmaxs])) if patient_vmaxs else 1.0
                if not np.isfinite(vmax_bar) or vmax_bar == 0:
                    vmax_bar = 1.0
                vmin_bar = -vmax_bar

            import matplotlib as mpl

            sm = mpl.cm.ScalarMappable(
                cmap=cmap,
                norm=mpl.colors.Normalize(vmin=vmin_bar, vmax=vmax_bar),
            )
            sm.set_array([])

            # Titre un peu plus haut (optionnel mais souvent mieux)
            fig.suptitle(title, fontsize=12, y=0.98)

            # Colorbar un peu plus haut
            cbar_ax = fig.add_axes([0.15, 0.085, 0.7, 0.02])  # <-- y=0.085, height=0.02
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
            #cbar.set_label(value_col , fontsize=9)

            # Réserver un peu plus d'espace en bas pour la colorbar, et un peu moins en haut
            plt.tight_layout(rect=[0.01, 0.13, 0.99, 0.95])

        else:
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
            cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(value_col, fontsize=9)

            fig.suptitle(title, fontsize=15)
            plt.tight_layout(rect=[0.01, 0.10, 0.99, 0.94])
    else:
        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.94])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[HEATMAP-PER-PAT] Saved: {out_path.resolve()}")














# ============================================================
# (5) Patient "profile" + stabilité inter-seizures + regroupement
# ============================================================
# Ajoute ce bloc à la suite de ton code (après les fonctions heatmaps, avant main() ou après).
# Ensuite, dans main(), appelle run_patient_profile_and_stability_plots(df_metrics, save_dir, ...)

from itertools import combinations

def _safe_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None


def _get_patient_seizure_feature_matrix(
    df_metrics: pd.DataFrame,
    value_col: str,
    normalize_within_seizure: bool = True,
) -> dict:
    """
    Retourne un dict patient_key -> DataFrame(index=seizure, columns=features)
    Valeurs = df_metrics[value_col] (moyenne si doublons), optionnellement normalisées par seizure (somme=1).
    """
    df = df_metrics.copy()
    df = df[np.isfinite(df[value_col].astype(float))].copy()
    df[value_col] = df[value_col].astype(float)

    out = {}
    features = sorted(df["feature"].unique().tolist())

    for pk, g in df.groupby("patient_key"):
        # pivot seizure x feature
        mat = (
            g.groupby(["seizure", "feature"])[value_col]
            .mean()
            .unstack("feature")
            .reindex(columns=features)
        )

        # normalisation par seizure (pour comparer "pattern" plutôt que "amplitude")
        if normalize_within_seizure:
            row_sums = mat.sum(axis=1).replace(0.0, np.nan)
            mat = mat.div(row_sums, axis=0)

        out[pk] = mat

    return out


def compute_patient_stability_scores(
    df_metrics: pd.DataFrame,
    value_col: str = "change_global_abs",
    top_n: int = 5,
    corr_method: str = "spearman",
    normalize_within_seizure: bool = True,
) -> pd.DataFrame:
    """
    Pour chaque patient:
      - stability_corr_mean: moyenne des corrélations pairwise (entre seizures) sur vecteurs feature
      - stability_corr_median: médiane des corrélations pairwise
      - stability_jaccard_topN_mean: moyenne des Jaccard pairwise sur top-N features
      - n_seizures

    Note:
      - corr_method = 'spearman' recommandé ici.
      - normalize_within_seizure=True pour capturer "quelles features bougent".
    """
    pk_to_mat = _get_patient_seizure_feature_matrix(
        df_metrics, value_col=value_col, normalize_within_seizure=normalize_within_seizure
    )

    rows = []
    for pk, mat in pk_to_mat.items():
        seizures = mat.index.tolist()
        n_seiz = len(seizures)

        if n_seiz < 2:
            rows.append(
                dict(
                    patient_key=pk,
                    n_seizures=n_seiz,
                    stability_corr_mean=np.nan,
                    stability_corr_median=np.nan,
                    stability_jaccard_topN_mean=np.nan,
                )
            )
            continue

        # ---------- Corrélation pairwise ----------
        # corr sur seizures (mat: seizure x feature) -> corr entre lignes = corr(mat.T)
        # pandas corr calcule entre colonnes, donc on transpose:
        corr_mat = mat.T.corr(method=corr_method, min_periods=3)

        # moyenne/mediane sur triangle supérieur (sans diag)
        vals = []
        for i in range(n_seiz):
            for j in range(i + 1, n_seiz):
                v = corr_mat.iloc[i, j]
                if np.isfinite(v):
                    vals.append(float(v))

        corr_mean = float(np.mean(vals)) if vals else np.nan
        corr_median = float(np.median(vals)) if vals else np.nan

        # ---------- Jaccard top-N ----------
        # top-N features par seizure sur les valeurs (déjà normalisées si option True)
        top_sets = {}
        for seiz in seizures:
            s = mat.loc[seiz].dropna()
            if s.empty:
                top_sets[seiz] = set()
            else:
                top_feats = s.sort_values(ascending=False).head(top_n).index.tolist()
                top_sets[seiz] = set(top_feats)

        j_vals = []
        for a, b in combinations(seizures, 2):
            A, B = top_sets[a], top_sets[b]
            if len(A) == 0 and len(B) == 0:
                continue
            j = len(A & B) / max(1, len(A | B))
            j_vals.append(float(j))

        j_mean = float(np.mean(j_vals)) if j_vals else np.nan

        rows.append(
            dict(
                patient_key=pk,
                n_seizures=n_seiz,
                stability_corr_mean=corr_mean,
                stability_corr_median=corr_median,
                stability_jaccard_topN_mean=j_mean,
            )
        )

    return pd.DataFrame(rows).sort_values(["n_seizures", "patient_key"], ascending=[False, True])


def build_patient_profile_matrix(
    df_metrics: pd.DataFrame,
    value_col: str = "change_global_abs",
    normalize_within_seizure: bool = True,
    normalize_within_patient: bool = True,
) -> pd.DataFrame:
    """
    Matrice patient x feature:
      - On part des vecteurs seizure (optionnellement normalisés somme=1)
      - On moyenne sur seizures
      - Optionnellement: re-normalisation par patient (somme=1) pour un "profil relatif patient"
    """
    pk_to_mat = _get_patient_seizure_feature_matrix(
        df_metrics, value_col=value_col, normalize_within_seizure=normalize_within_seizure
    )

    rows = []
    for pk, mat in pk_to_mat.items():
        if mat.empty:
            continue
        v = mat.mean(axis=0, skipna=True)  # moyenne sur seizures -> profil patient
        rows.append(pd.Series(v, name=pk))

    if not rows:
        return pd.DataFrame()

    prof = pd.DataFrame(rows)
    # normalisation par patient (somme=1) -> compare "quelles features dominent" vs amplitude globale
    if normalize_within_patient:
        row_sums = prof.sum(axis=1).replace(0.0, np.nan)
        prof = prof.div(row_sums, axis=0)

    return prof


def _order_patients_by_hclust_or_pca(X: np.ndarray, patient_keys: list[str]) -> list[int]:
    """
    Retourne un ordre d'indices patients pour la heatmap:
      - essaie clustering hiérarchique (scipy)
      - fallback: tri par 1ère composante PCA
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore

        # distance euclidienne sur profils
        Z = linkage(X, method="average", metric="euclidean")
        order = leaves_list(Z).tolist()
        return order
    except Exception:
        # fallback PCA
        try:
            from sklearn.decomposition import PCA  # type: ignore

            pc1 = PCA(n_components=1).fit_transform(X)[:, 0]
            order = np.argsort(pc1).tolist()
            return order
        except Exception:
            return list(range(len(patient_keys)))


def plot_patient_profile_heatmap(
    patient_profile: pd.DataFrame,
    out_path: Path,
    title: str,
    cluster_patients: bool = True,
    cluster_features: bool = False,
    verbose: bool = True,
):
    """
    Heatmap patient x feature (valeurs = profil patient).
    - cluster_patients: réordonne les patients (hclust ou PCA fallback)
    - cluster_features: optionnel (souvent moins utile avec 15 features, mais possible)
    """
    if patient_profile.empty:
        if verbose:
            print("[PROFILE-HEATMAP] patient_profile vide -> skip")
        return

    prof = patient_profile.copy()
    patients = prof.index.tolist()
    features = prof.columns.tolist()

    X = prof.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    if cluster_patients:
        p_order = _order_patients_by_hclust_or_pca(X, patients)
        prof = prof.iloc[p_order, :]
        patients = prof.index.tolist()
        X = prof.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)

    if cluster_features:
        # réordonne features via hclust sur colonnes
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore

            Zf = linkage(X.T, method="average", metric="euclidean")
            f_order = leaves_list(Zf).tolist()
            prof = prof.iloc[:, f_order]
            features = prof.columns.tolist()
            X = prof.values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(0.55 * len(features) + 6, 0.35 * len(patients) + 3))
    im = ax.imshow(X, aspect="auto", interpolation="nearest", cmap="viridis")

    ax.set_yticks(np.arange(len(patients)))
    ax.set_yticklabels(patients, fontsize=8)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([pretty_feat(f) for f in features], rotation=45, ha="right", fontsize=8)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("patient profile (relative magnitude)")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[PROFILE-HEATMAP] Saved: {out_path.resolve()}")


def plot_patient_umap_or_tsne(
    patient_profile: pd.DataFrame,
    out_path: Path,
    title: str,
    random_state: int = 0,
    verbose: bool = True,
):
    """
    UMAP (si dispo) sinon t-SNE (sklearn) sur les profils patients.
    """
    if patient_profile.empty:
        if verbose:
            print("[PATIENT-EMBED] patient_profile vide -> skip")
        return

    X = patient_profile.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    patients = patient_profile.index.tolist()

    umap_mod = _safe_import_umap()
    coords = None
    method = None

    if umap_mod is not None:
        try:
            reducer = umap_mod.UMAP(
                n_neighbors=min(10, max(2, len(patients) - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
            method = "UMAP"
        except Exception:
            coords = None

    if coords is None:
        try:
            from sklearn.manifold import TSNE  # type: ignore

            tsne = TSNE(
                n_components=2,
                perplexity=max(2, min(10, (len(patients) - 1) // 2)),
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(X)
            method = "t-SNE"
        except Exception:
            # fallback PCA 2D
            try:
                from sklearn.decomposition import PCA  # type: ignore
                coords = PCA(n_components=2).fit_transform(X)
                method = "PCA"
            except Exception:
                if verbose:
                    print("[PATIENT-EMBED] Aucun backend embedding dispo (umap/sklearn) -> skip")
                return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1])

    # annoter points
    for i, pk in enumerate(patients):
        ax.text(coords[i, 0], coords[i, 1], pk, fontsize=7)

    ax.set_title(f"{title}\n({method})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[PATIENT-EMBED] Saved: {out_path.resolve()}")


def plot_patient_stability_barplots(
    df_stability: pd.DataFrame,
    out_path_prefix: Path,
    title_prefix: str,
    verbose: bool = True,
):
    if df_stability.empty:
        if verbose:
            print("[STABILITY-PLOT] df_stability vide -> skip")
        return

    import matplotlib.lines as mlines
    good_h = mlines.Line2D([], [], color="g", marker="s", linestyle="None", markersize=8, label="good outcome")
    bad_h  = mlines.Line2D([], [], color="r", marker="s", linestyle="None", markersize=8, label="bad outcome")
    unk_h  = mlines.Line2D([], [], color="0.6", marker="s", linestyle="None", markersize=8, label="unknown")

    # ========= (A) CORR MEAN =========
    df1 = df_stability.sort_values("stability_corr_mean", ascending=False).copy()
    patient_keys = df1["patient_key"].tolist()
    colors = [outcome_color(pk) for pk in patient_keys]
    hatches = [SOZ_HATCH.get(soz_location_group(pk), "..") for pk in patient_keys]

    fig, ax = plt.subplots(figsize=(0.45 * len(df1) + 6, 4))
    bars = ax.bar(patient_keys, df1["stability_corr_mean"].values, color=colors)
    _apply_hatch_to_bars(bars, hatches)

    ax.set_xticks(np.arange(len(df1)))
    ax.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax.set_ylabel("Mean pairwise Spearman correlation\n(seizure patterns)")
    ax.set_title(f"{title_prefix} — Pattern stability\nColor=outcome, hatch=SOZ")

    ax.legend(handles=[good_h, bad_h, unk_h], fontsize=8, frameon=True)
    plt.tight_layout()
    out1 = Path(str(out_path_prefix) + "_corr_mean.png")
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"[STABILITY-PLOT] Saved: {out1.resolve()}")

    # ========= (B) JACCARD TOP-N =========
    df2 = df_stability.sort_values("stability_jaccard_topN_mean", ascending=False).copy()
    patient_keys = df2["patient_key"].tolist()
    colors = [outcome_color(pk) for pk in patient_keys]
    hatches = [SOZ_HATCH.get(soz_location_group(pk), "..") for pk in patient_keys]

    fig, ax = plt.subplots(figsize=(0.45 * len(df2) + 6, 4))
    bars = ax.bar(patient_keys, df2["stability_jaccard_topN_mean"].values, color=colors)
    _apply_hatch_to_bars(bars, hatches)

    ax.set_xticks(np.arange(len(df2)))
    ax.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax.set_ylabel("Mean pairwise Jaccard\n(top-N features)")
    ax.set_title(f"{title_prefix} — Feature overlap stability\nColor=outcome, hatch=SOZ")

    ax.legend(handles=[good_h, bad_h, unk_h], fontsize=8, frameon=True)
    plt.tight_layout()
    out2 = Path(str(out_path_prefix) + "_jaccard_topN.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"[STABILITY-PLOT] Saved: {out2.resolve()}")



def run_patient_profile_and_stability_plots(
    df_metrics: pd.DataFrame,
    save_dir: Path,
    value_col: str = "change_global_abs",
    top_n: int = 5,
    normalize_within_seizure: bool = True,
    normalize_within_patient: bool = True,
    verbose: bool = True,
):
    """
    Pipeline complet:
      1) calcule stabilité patient (corr + jaccard)
      2) construit profil patient (patient x feature)
      3) heatmap + embedding UMAP/tSNE
      4) sauvegarde CSVs
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # (A) stabilité
    df_stab = compute_patient_stability_scores(
        df_metrics=df_metrics,
        value_col=value_col,
        top_n=top_n,
        corr_method="spearman",
        normalize_within_seizure=normalize_within_seizure,
    )
    df_stab.to_csv(save_dir / f"patient_stability_{value_col}.csv", index=False)
    if verbose:
        print(f"[STABILITY] CSV saved: {(save_dir / f'patient_stability_{value_col}.csv').resolve()}")

    plot_patient_stability_barplots(
        df_stability=df_stab,
        out_path_prefix=save_dir / f"patient_stability_{value_col}",
        title_prefix=f"{value_col} (normalize_within_seizure={normalize_within_seizure})",
        verbose=verbose,
    )

    # (B) profil patient
    prof = build_patient_profile_matrix(
        df_metrics=df_metrics,
        value_col=value_col,
        normalize_within_seizure=normalize_within_seizure,
        normalize_within_patient=normalize_within_patient,
    )
    prof.to_csv(save_dir / f"patient_profile_{value_col}.csv", index=True)
    if verbose:
        print(f"[PROFILE] CSV saved: {(save_dir / f'patient_profile_{value_col}.csv').resolve()}")

    # (C) heatmap + embedding
    """
    plot_patient_profile_heatmap(
        patient_profile=prof,
        out_path=save_dir / f"patient_profile_heatmap_{value_col}.png",
        title=f"Patient profiles (patients × features) — {value_col}\n"
              f"(normalize_within_seizure={normalize_within_seizure}, normalize_within_patient={normalize_within_patient})",
        cluster_patients=True,
        cluster_features=False,
        verbose=verbose,
    )

    plot_patient_umap_or_tsne(
        patient_profile=prof,
        out_path=save_dir / f"patient_profile_embedding_{value_col}.png",
        title=f"Patient embedding from profiles — {value_col}",
        random_state=0,
        verbose=verbose,
    )
    """

    return df_stab, prof




# ============================================================
# (PATCH) Colorer l'embedding par surgery outcome (good=vert, bad=rouge)
# ============================================================
# 1) Ajoute ces constantes (par ex. après EXPLICIT_NODE_FEATURES)

GOOD_OUTCOME_PATIENTS = {
    "CHUM::Patient_01",
    "CHUM::Patient_02",
    "CHUM::Patient_07",
    "CHUM::Patient_11",
    "CHUM::Patient_14",
    "CHUM::Patient_22",
    "ds004100::sub-HUP074",
    "ds004100::sub-HUP082",
    "ds004100::sub-HUP089",
    "ds004100::sub-HUP097",
    "ds004100::sub-HUP107",
    "ds004100::sub-HUP111",
    "ds004100::sub-HUP126",
    "ds004100::sub-HUP141",
    "ds004100::sub-HUP144",
    "ds004100::sub-HUP148",
    "ds004100::sub-HUP150",
    "ds004100::sub-HUP157",
    "ds004100::sub-HUP173",
    "ds004100::sub-HUP180",
    "ds004100::sub-HUP185",
}

BAD_OUTCOME_PATIENTS = {
    "CHUM::Patient_09",
    "CHUM::Patient_16",
    "CHUM::Patient_17",
    "CHUM::Patient_21",
    "ds004100::sub-HUP080",
    "ds004100::sub-HUP112",
    "ds004100::sub-HUP114",
    "ds004100::sub-HUP133",
    "ds004100::sub-HUP138",
    "ds004100::sub-HUP151",
    "ds004100::sub-HUP162",
    "ds004100::sub-HUP171",
    "ds004100::sub-HUP172",
    "ds004100::sub-HUP181",
    "ds004100::sub-HUP187",
    "ds004100::sub-HUP188",
}


def surgery_outcome_label(patient_key: str) -> str:
    """
    Retourne: 'good', 'bad', ou 'unknown'
    """
    if patient_key in GOOD_OUTCOME_PATIENTS:
        return "good"
    if patient_key in BAD_OUTCOME_PATIENTS:
        return "bad"
    return "unknown"


def surgery_outcome_color(patient_key: str) -> str:
    """
    Couleurs matplotlib:
      - good -> vert
      - bad -> rouge
      - unknown -> gris
    """
    lab = surgery_outcome_label(patient_key)
    if lab == "good":
        return "g"
    if lab == "bad":
        return "r"
    return "0.6"


# ============================================================
# 2) Remplace ta fonction plot_patient_umap_or_tsne par cette version
#    (même nom, elle écrase l'ancienne)
# ============================================================

def plot_patient_umap_or_tsne(
    patient_profile: pd.DataFrame,
    out_path: Path,
    title: str,
    random_state: int = 0,
    verbose: bool = True,
):
    """
    UMAP (si dispo) sinon t-SNE (sklearn) sur les profils patients.
    Coloration: surgery outcome (good=vert, bad=rouge).
    """
    if patient_profile.empty:
        if verbose:
            print("[PATIENT-EMBED] patient_profile vide -> skip")
        return

    X = patient_profile.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    patients = patient_profile.index.tolist()

    umap_mod = _safe_import_umap()
    coords = None
    method = None

    if umap_mod is not None:
        try:
            reducer = umap_mod.UMAP(
                n_neighbors=min(10, max(2, len(patients) - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
            method = "UMAP"
        except Exception:
            coords = None

    if coords is None:
        try:
            from sklearn.manifold import TSNE  # type: ignore

            tsne = TSNE(
                n_components=2,
                perplexity=max(2, min(10, (len(patients) - 1) // 2)),
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(X)
            method = "t-SNE"
        except Exception:
            # fallback PCA 2D
            try:
                from sklearn.decomposition import PCA  # type: ignore
                coords = PCA(n_components=2).fit_transform(X)
                method = "PCA"
            except Exception:
                if verbose:
                    print("[PATIENT-EMBED] Aucun backend embedding dispo (umap/sklearn) -> skip")
                return

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [surgery_outcome_color(pk) for pk in patients]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors)

    # annoter points
    for i, pk in enumerate(patients):
        ax.text(coords[i, 0], coords[i, 1], pk, fontsize=7)

    # petite légende
    import matplotlib.lines as mlines
    good_h = mlines.Line2D([], [], color="g", marker="o", linestyle="None", markersize=6, label="good outcome")
    bad_h = mlines.Line2D([], [], color="r", marker="o", linestyle="None", markersize=6, label="bad outcome")
    unk_h = mlines.Line2D([], [], color="0.6", marker="o", linestyle="None", markersize=6, label="unknown")
    ax.legend(handles=[good_h, bad_h], loc="best", fontsize=8, frameon=True)

    ax.set_title(f"{title}\n({method})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[PATIENT-EMBED] Saved: {out_path.resolve()}")





import numpy as np
import pandas as pd
from itertools import combinations

def compute_patient_constant_change_score(
    df_metrics: pd.DataFrame,
    value_col: str = "change_global_abs",
    eps: float = 1e-12,
    tau_quantile: float = 0.10,   # bruit plancher = quantile bas des normes globales
    min_pairs: int = 1,
) -> pd.DataFrame:
    """
    Score patient qui met en valeur:
      - un pattern de changement CONSISTANT entre seizures
      - MAIS pénalise les patients où le changement est ~0 partout

    Étapes:
      1) Pour chaque seizure: x (15 features)
      2) u = x / sum(|x|)  (pattern relatif, amplitude-invariant)
      3) pattern_stability = moyenne cos(u_i, u_j)
      4) nontriviality = 1 - exp(- median(||x||) / tau)  (tau = plancher bruit global, saturant)
      5) score = pattern_stability * nontriviality

    Retourne df avec:
      - patient_key, n_seizures, pattern_stability, median_amplitude, nontriviality, constant_change_score
    """

    df = df_metrics.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df[value_col].astype(float))].copy()
    df[value_col] = df[value_col].astype(float)

    # ---------- Estimation automatique d'un "bruit plancher" tau ----------
    # on regarde toutes les seizures de tous les patients: norme L2 du vecteur x_s
    all_seiz = (
        df.groupby(["patient_key", "seizure", "feature"])[value_col]
          .mean()
          .unstack("feature")
    )
    X_all = np.nan_to_num(all_seiz.values.astype(float), nan=0.0)
    norms_all = np.linalg.norm(X_all, axis=1)

    # tau = quantile bas des normes > 0 (sinon fallback)
    norms_pos = norms_all[np.isfinite(norms_all) & (norms_all > 0)]
    if len(norms_pos) == 0:
        tau = 1.0
    else:
        tau = float(np.quantile(norms_pos, tau_quantile))
        tau = max(tau, 1e-6)

    rows = []

    for pk, g_pat in df.groupby("patient_key"):
        mat = (
            g_pat.groupby(["seizure", "feature"])[value_col]
                 .mean()
                 .unstack("feature")
        )

        # remplacer NaN par 0 (important: sinon les normes/cos partent en NaN)
        X = np.nan_to_num(mat.values.astype(float), nan=0.0)
        seizures = mat.index.tolist()
        n_seiz = len(seizures)

        if n_seiz < 2:
            rows.append(dict(
                patient_key=pk,
                n_seizures=n_seiz,
                pattern_stability=np.nan,
                median_amplitude=np.nan,
                nontriviality=np.nan,
                constant_change_score=np.nan,
            ))
            continue

        # amplitude brute par seizure (sert uniquement à rejeter "tout petit")
        amp = np.linalg.norm(X, axis=1)
        median_amp = float(np.median(amp))

        # pattern relatif par seizure: L1-normalisation
        l1 = np.sum(np.abs(X), axis=1)
        U = X / (l1[:, None] + eps)

        # cosine pairwise sur U
        # (cosine = dot/(||u|| ||v||))
        u_norm = np.linalg.norm(U, axis=1) + eps

        sims = []
        for i in range(n_seiz):
            for j in range(i + 1, n_seiz):
                cos_ij = float(np.dot(U[i], U[j]) / (u_norm[i] * u_norm[j]))
                if np.isfinite(cos_ij):
                    sims.append(cos_ij)

        if len(sims) < min_pairs:
            pattern = np.nan
        else:
            pattern = float(np.mean(sims))

        # nontriviality: saturant, ne "récompense" pas les gros amplitudes
        nontriv = float(1.0 - np.exp(- median_amp / tau))

        # score final
        score = float(pattern * nontriv) if (np.isfinite(pattern) and np.isfinite(nontriv)) else np.nan

        rows.append(dict(
            patient_key=pk,
            n_seizures=n_seiz,
            pattern_stability=pattern,
            median_amplitude=median_amp,
            nontriviality=nontriv,
            constant_change_score=score,
        ))

    out = pd.DataFrame(rows)
    return out.sort_values("constant_change_score", ascending=False)



def plot_amplitude_aware_stability_barplot(
    df_amp_stab: pd.DataFrame,
    out_path: Path,
    title: str,
    verbose: bool = True,
):
    if df_amp_stab.empty:
        if verbose:
            print("[AMP-STAB] df vide -> skip")
        return

    df = df_amp_stab.copy()
    df = df.sort_values("stability_cosine_weighted_mean", ascending=False)

    patient_keys = df["patient_key"].tolist()
    values = df["stability_cosine_weighted_mean"].values
    colors = [surgery_outcome_color(pk) for pk in patient_keys]

    fig, ax = plt.subplots(figsize=(0.45 * len(df) + 6, 4))
    ax.bar(patient_keys, values, color=colors)

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax.set_ylabel("Weighted cosine similarity\n(pattern + amplitude)")
    ax.set_title(title)

    import matplotlib.lines as mlines
    good_h = mlines.Line2D([], [], color="g", marker="s", linestyle="None", markersize=8, label="good outcome")
    bad_h = mlines.Line2D([], [], color="r", marker="s", linestyle="None", markersize=8, label="bad outcome")
    unk_h = mlines.Line2D([], [], color="0.6", marker="s", linestyle="None", markersize=8, label="unknown")
    ax.legend(handles=[good_h, bad_h, unk_h], fontsize=8, frameon=True)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[AMP-STAB] Saved: {out_path.resolve()}")


def _mannwhitney_or_permutation_pvalue(x, y, n_perm=20000, seed=0):
    """
    Retourne un p-value pour la différence de distributions entre x et y.
    - essaie Mann-Whitney (scipy)
    - sinon permutation test sur différence de moyennes (robuste et simple)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, "insufficient N"

    # try scipy
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p), "Mann–Whitney U"
    except Exception:
        rng = np.random.default_rng(seed)
        obs = abs(np.mean(x) - np.mean(y))

        pooled = np.concatenate([x, y])
        n_x = len(x)

        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            x_perm = pooled[:n_x]
            y_perm = pooled[n_x:]
            diff = abs(np.mean(x_perm) - np.mean(y_perm))
            if diff >= obs:
                cnt += 1

        p = (cnt + 1) / (n_perm + 1)
        return float(p), f"Permutation test (mean diff, n={n_perm})"


def plot_amp_stability_2panel(
    df_amp_stab: pd.DataFrame,
    out_path: Path,
    title: str,
    color_by: str = "outcome",  # 'outcome' or 'soz'
    value_col: str = "stability_cosine_weighted_mean",
    verbose: bool = True,
):
    """
    Figure 2 panneaux:
      - gauche: barplot par patient (trié), couleur = outcome OU SOZ
      - droite: boxplots good vs bad + points (1 point/patient)
               + test stat (uniquement basé sur outcome)
    """
    if df_amp_stab.empty:
        if verbose:
            print("[AMP-2PANEL] df vide -> skip")
        return

    df = df_amp_stab.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df[value_col].astype(float))].copy()
    df[value_col] = df[value_col].astype(float)

    if df.empty:
        if verbose:
            print("[AMP-2PANEL] aucune valeur finite -> skip")
        return

    # tri barplot
    df_sorted = df.sort_values(value_col, ascending=False)
    patient_keys = df_sorted["patient_key"].tolist()
    vals = df_sorted[value_col].values

    color_func = get_color_func(color_by)
    colors = [color_func(pk) for pk in patient_keys]

    # --- stats good vs bad (toujours outcome-based) ---
    df["outcome"] = df["patient_key"].apply(surgery_outcome_label)
    df_gb = df[df["outcome"].isin(["good", "bad"])].copy()
    good_vals = df_gb.loc[df_gb["outcome"] == "good", value_col].values
    bad_vals  = df_gb.loc[df_gb["outcome"] == "bad", value_col].values
    pval, test_name = _mannwhitney_or_permutation_pvalue(good_vals, bad_vals)

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # panel gauche
    ax0.bar(patient_keys, vals, color=colors)
    ax0.set_xticks(np.arange(len(patient_keys)))
    ax0.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax0.set_ylabel("Weighted cosine similarity\n(pattern + amplitude)")
    ax0.set_title(f"{title}\nColor by: {color_by}")
    add_legend(ax0, color_by)

    # panel droit: boxplots good vs bad (toujours outcome)
    data = [good_vals, bad_vals]
    ax1.boxplot(data, labels=["good", "bad"], showfliers=False, widths=0.6)

    # points overlay (couleur = même règle que le plot, mais uniquement sur good/bad patients)
    rng = np.random.default_rng(0)
    for i, outc in enumerate(["good", "bad"], start=1):
        sub = df_gb[df_gb["outcome"] == outc]
        if sub.empty:
            continue
        x = i + rng.normal(0.0, 0.04, size=len(sub))
        y = sub[value_col].values
        cols = [color_func(pk) for pk in sub["patient_key"].tolist()]
        ax1.scatter(x, y, c=cols)

    ax1.set_ylabel("Weighted cosine similarity")
    ax1.set_title("Good vs Bad (stats)")
    if np.isfinite(pval):
        txt = f"{test_name}\np = {pval:.3g}\nN_good={len(good_vals)}  N_bad={len(bad_vals)}"
    else:
        txt = f"{test_name}\n(p-value NA)\nN_good={len(good_vals)}  N_bad={len(bad_vals)}"
    ax1.text(0.5, 0.02, txt, transform=ax1.transAxes, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[AMP-2PANEL] Saved: {out_path.resolve()}")





# ============================================================
# (SOZ) Location mapping + styles (outcome color + SOZ marker/hatch)
# ============================================================

# --- SOZ groups (harmonisés) ---
SOZ_TEMPORAL = {
    "CHUM::Patient_01",
    "ds004100::sub-HUP074",
    "ds004100::sub-HUP082",
    "ds004100::sub-HUP089",
    "ds004100::sub-HUP097",
    "ds004100::sub-HUP107",
    "ds004100::sub-HUP111",
    "ds004100::sub-HUP114",
    "ds004100::sub-HUP133",
    "ds004100::sub-HUP138",
    "ds004100::sub-HUP151",
    "ds004100::sub-HUP162",
    "ds004100::sub-HUP172",
    "ds004100::sub-HUP181",
    "ds004100::sub-HUP188",
}

SOZ_FRONTAL = {
    "CHUM::Patient_02",
    "CHUM::Patient_09",
    "CHUM::Patient_16",
    "ds004100::sub-HUP150",
    "ds004100::sub-HUP173",
    "ds004100::sub-HUP180",
}

SOZ_INSULAR = {
    "CHUM::Patient_07",
    "CHUM::Patient_17",
    "ds004100::sub-HUP141",
    "ds004100::sub-HUP148",
    "ds004100::sub-HUP157",
    "ds004100::sub-HUP171",
}

SOZ_MIXED = {
    "CHUM::Patient_11",   # fronto-temporal
    "CHUM::Patient_14",   # fronto-parietal
    "CHUM::Patient_21",   # temporal opercular + generalized
    "ds004100::sub-HUP080",
    "ds004100::sub-HUP112",
    "ds004100::sub-HUP187",
}

def soz_location_group(patient_key: str) -> str:
    """
    Retourne un groupe harmonisé: temporal/frontal/insular/mixed/unknown
    """
    if patient_key in SOZ_TEMPORAL:
        return "temporal"
    if patient_key in SOZ_FRONTAL:
        return "frontal"
    if patient_key in SOZ_INSULAR:
        return "insular"
    if patient_key in SOZ_MIXED:
        return "mixed"
    return "unknown"


# --- style maps ---
SOZ_MARKER = {
    "temporal": "o",
    "frontal": "s",
    "insular": "^",
    "mixed": "D",
    "unknown": "x",
}

SOZ_HATCH = {
    "temporal": None,
    "frontal": "//",
    "insular": "xx",
    "mixed": "\\\\",
    "unknown": "..",
}


def _apply_hatch_to_bars(bar_container, hatches):
    """
    bar_container = retour de ax.bar(...)
    hatches = list[str|None] same length
    """
    for rect, hatch in zip(bar_container, hatches):
        if hatch is not None:
            rect.set_hatch(hatch)


def outcome_color(patient_key: str) -> str:
    # alias pour cohérence
    return surgery_outcome_color(patient_key)


# ============================================================
# (AMP-STAB) 2-panel figure: barplot (outcome + hatch SOZ) + boxplot (good/bad) with points (shape SOZ)
# ============================================================

def plot_amp_stability_2panel_outcome_soz(
    df_amp_stab: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str = "stability_cosine_weighted_mean",
    verbose: bool = True,
):
    if df_amp_stab.empty:
        if verbose:
            print("[AMP-STAB-2PANEL-SOZ] df vide -> skip")
        return

    df = df_amp_stab.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df[value_col].astype(float))].copy()
    df[value_col] = df[value_col].astype(float)

    if df.empty:
        if verbose:
            print("[AMP-STAB-2PANEL-SOZ] aucune valeur finite -> skip")
        return

    df["outcome"] = df["patient_key"].apply(surgery_outcome_label)
    df["soz_group"] = df["patient_key"].apply(soz_location_group)

    # pour stats good vs bad (ignore unknown)
    df_gb = df[df["outcome"].isin(["good", "bad"])].copy()
    good_vals = df_gb.loc[df_gb["outcome"] == "good", value_col].values
    bad_vals  = df_gb.loc[df_gb["outcome"] == "bad", value_col].values
    pval, test_name = _mannwhitney_or_permutation_pvalue(good_vals, bad_vals)

    # tri barplot
    df_sorted = df.sort_values(value_col, ascending=False)
    patient_keys = df_sorted["patient_key"].tolist()
    vals = df_sorted[value_col].values

    colors = [outcome_color(pk) for pk in patient_keys]
    hatches = [SOZ_HATCH.get(soz_location_group(pk), "..") for pk in patient_keys]

    fig = plt.figure(figsize=(17, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    bars = ax0.bar(patient_keys, vals, color=colors)
    _apply_hatch_to_bars(bars, hatches)

    ax0.set_xticks(np.arange(len(patient_keys)))
    ax0.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax0.set_ylabel("Weighted cosine similarity\n(pattern + amplitude)")
    ax0.set_title(title + "\nBar color = outcome, hatch = SOZ location group")

    # legend outcome
    import matplotlib.lines as mlines
    good_h = mlines.Line2D([], [], color="g", marker="s", linestyle="None", markersize=8, label="good outcome")
    bad_h  = mlines.Line2D([], [], color="r", marker="s", linestyle="None", markersize=8, label="bad outcome")
    unk_h  = mlines.Line2D([], [], color="0.6", marker="s", linestyle="None", markersize=8, label="unknown outcome")

    # legend SOZ (hatch proxy)
    import matplotlib.patches as mpatches
    soz_leg = []
    for k in ["temporal", "frontal", "insular", "mixed", "unknown"]:
        patch = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=SOZ_HATCH[k] if SOZ_HATCH[k] is not None else "",
            label=f"SOZ {k}",
        )
        soz_leg.append(patch)

    ax0.legend(handles=[good_h, bad_h, unk_h] + soz_leg, loc="upper right", fontsize=8, frameon=True)

    # right: boxplot good vs bad + points (colored by outcome, marker by SOZ)
    data = [
        df_gb.loc[df_gb["outcome"] == "good", value_col].values,
        df_gb.loc[df_gb["outcome"] == "bad", value_col].values,
    ]
    labels = ["good", "bad"]
    ax1.boxplot(data, labels=labels, showfliers=False, widths=0.6)

    # points with jitter, colored by outcome, shaped by SOZ
    rng = np.random.default_rng(0)
    for i, outc in enumerate(["good", "bad"], start=1):
        sub = df_gb[df_gb["outcome"] == outc].copy()
        if sub.empty:
            continue
        x = i + rng.normal(0.0, 0.04, size=len(sub))
        y = sub[value_col].values
        cols = [outcome_color(pk) for pk in sub["patient_key"].tolist()]
        # plot per marker group to respect shapes
        for soz_g in sub["soz_group"].unique():
            idx = sub["soz_group"].values == soz_g
            ax1.scatter(x[idx], y[idx], c=np.array(cols, dtype=object)[idx], marker=SOZ_MARKER.get(soz_g, "x"))

    ax1.set_ylabel("Weighted cosine similarity")
    ax1.set_title("Good vs Bad")

    if np.isfinite(pval):
        txt = f"{test_name}\np = {pval:.3g}\nN_good={len(good_vals)}  N_bad={len(bad_vals)}"
    else:
        txt = f"{test_name}\n(p-value NA)\nN_good={len(good_vals)}  N_bad={len(bad_vals)}"
    ax1.text(0.5, 0.02, txt, transform=ax1.transAxes, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[AMP-STAB-2PANEL-SOZ] Saved: {out_path.resolve()}")


# ============================================================
# (EMBED) override: outcome color + SOZ marker
# ============================================================

def plot_patient_umap_or_tsne(
    patient_profile: pd.DataFrame,
    out_path: Path,
    title: str,
    color_by: str = "outcome",  # 'outcome' or 'soz'
    random_state: int = 0,
    verbose: bool = True,
):
    """
    UMAP (si dispo) sinon t-SNE / PCA.
    Couleur = outcome OU SOZ (pas de mixing).
    """
    if patient_profile.empty:
        if verbose:
            print("[PATIENT-EMBED] patient_profile vide -> skip")
        return

    X = patient_profile.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    patients = patient_profile.index.tolist()

    umap_mod = _safe_import_umap()
    coords = None
    method = None

    if umap_mod is not None:
        try:
            reducer = umap_mod.UMAP(
                n_neighbors=min(10, max(2, len(patients) - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
            method = "UMAP"
        except Exception:
            coords = None

    if coords is None:
        try:
            from sklearn.manifold import TSNE  # type: ignore
            tsne = TSNE(
                n_components=2,
                perplexity=max(2, min(10, (len(patients) - 1) // 2)),
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(X)
            method = "t-SNE"
        except Exception:
            try:
                from sklearn.decomposition import PCA  # type: ignore
                coords = PCA(n_components=2).fit_transform(X)
                method = "PCA"
            except Exception:
                if verbose:
                    print("[PATIENT-EMBED] Aucun backend embedding dispo -> skip")
                return

    color_func = get_color_func(color_by)
    colors = [color_func(pk) for pk in patients]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors)
    for i, pk in enumerate(patients):
        ax.text(coords[i, 0], coords[i, 1], pk, fontsize=7)

    ax.set_title(f"{title}\n({method}) — Color by: {color_by}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    add_legend(ax, color_by)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[PATIENT-EMBED] Saved: {out_path.resolve()}")


# ============================================================
# Color helpers: outcome OR SOZ (no mixing)
# ============================================================

# --- SOZ groups (harmonisés) ---
SOZ_TEMPORAL = {
    "CHUM::Patient_01",
    "ds004100::sub-HUP074",
    "ds004100::sub-HUP082",
    "ds004100::sub-HUP089",
    "ds004100::sub-HUP097",
    "ds004100::sub-HUP107",
    "ds004100::sub-HUP111",
    "ds004100::sub-HUP114",
    "ds004100::sub-HUP133",
    "ds004100::sub-HUP138",
    "ds004100::sub-HUP151",
    "ds004100::sub-HUP162",
    "ds004100::sub-HUP172",
    "ds004100::sub-HUP181",
    "ds004100::sub-HUP188",
}

SOZ_FRONTAL = {
    "CHUM::Patient_02",
    "CHUM::Patient_09",
    "CHUM::Patient_16",
    "ds004100::sub-HUP150",
    "ds004100::sub-HUP173",
    "ds004100::sub-HUP180",
}

SOZ_INSULAR = {
    "CHUM::Patient_07",
    "CHUM::Patient_17",
    "ds004100::sub-HUP141",
    "ds004100::sub-HUP148",
    "ds004100::sub-HUP157",
    "ds004100::sub-HUP171",
}

SOZ_MIXED = {
    "CHUM::Patient_11",
    "CHUM::Patient_14",
    "CHUM::Patient_21",
    "ds004100::sub-HUP080",
    "ds004100::sub-HUP112",
    "ds004100::sub-HUP187",
}

def soz_location_group(patient_key: str) -> str:
    if patient_key in SOZ_TEMPORAL:
        return "temporal"
    if patient_key in SOZ_FRONTAL:
        return "frontal"
    if patient_key in SOZ_INSULAR:
        return "insular"
    if patient_key in SOZ_MIXED:
        return "mixed"
    return "unknown"

def color_outcome(patient_key: str) -> str:
    # good=vert, bad=rouge, unknown=gris
    return surgery_outcome_color(patient_key)

# couleurs distinctes pour SOZ (tu peux changer si tu veux)
SOZ_COLOR = {
    "temporal": "tab:blue",
    "frontal": "tab:orange",
    "insular": "tab:purple",
    "mixed": "tab:brown",
    "unknown": "0.6",
}

def color_soz(patient_key: str) -> str:
    return SOZ_COLOR.get(soz_location_group(patient_key), "0.6")

def get_color_func(color_by: str):
    if color_by == "outcome":
        return color_outcome
    if color_by == "soz":
        return color_soz
    raise ValueError("color_by must be 'outcome' or 'soz'")

def add_legend(ax, color_by: str):
    import matplotlib.lines as mlines
    if color_by == "outcome":
        h1 = mlines.Line2D([], [], color="g", marker="s", linestyle="None", markersize=8, label="good outcome")
        h2 = mlines.Line2D([], [], color="r", marker="s", linestyle="None", markersize=8, label="bad outcome")
        h3 = mlines.Line2D([], [], color="0.6", marker="s", linestyle="None", markersize=8, label="unknown outcome")
        ax.legend(handles=[h1, h2, h3], fontsize=8, frameon=True)
    else:
        hs = []
        for k in ["temporal", "frontal", "insular", "mixed", "unknown"]:
            hs.append(mlines.Line2D([], [], color=SOZ_COLOR[k], marker="s", linestyle="None", markersize=8, label=f"SOZ {k}"))
        ax.legend(handles=hs, fontsize=8, frameon=True)



# ============================================================
# (SOZ vs OUTCOME) association plot + stats (Fisher/Chi2)
# ============================================================

def plot_soz_vs_outcome_association(
    patient_keys: list[str],
    out_path: Path,
    title: str = "SOZ location group vs surgery outcome",
    verbose: bool = True,
):
    """
    Produit:
      - barplot: proportion good/bad dans chaque groupe SOZ
      - stats: Fisher exact (2x2) si possible, sinon Chi2 sur RxC
    """
    rows = []
    for pk in patient_keys:
        rows.append(
            dict(
                patient_key=pk,
                outcome=surgery_outcome_label(pk),
                soz_group=soz_location_group(pk),
            )
        )
    df = pd.DataFrame(rows)
    df = df[df["outcome"].isin(["good", "bad"])].copy()  # ignore unknown outcome
    if df.empty:
        if verbose:
            print("[SOZ-OUTCOME] no good/bad -> skip")
        return

    # contingency
    tab = pd.crosstab(df["soz_group"], df["outcome"]).reindex(
        index=["temporal", "frontal", "insular", "mixed", "unknown"],
        columns=["good", "bad"],
        fill_value=0,
    )

    # proportions
    denom = tab.sum(axis=1).replace(0, np.nan)
    prop = tab.div(denom, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(prop.index))
    width = 0.35

    ax.bar(x - width/2, prop["good"].values, width=width, label="good", color="g")
    ax.bar(x + width/2, prop["bad"].values,  width=width, label="bad",  color="r")

    ax.set_xticks(x)
    ax.set_xticklabels(prop.index.tolist(), rotation=0)
    ax.set_ylabel("Proportion within SOZ group")
    ax.set_title(title)
    ax.legend()

    # stats: Chi2 RxC
    stat_txt = ""
    try:
        from scipy.stats import chi2_contingency  # type: ignore
        chi2, p, dof, exp = chi2_contingency(tab.values)
        stat_txt = f"Chi2 test: p={p:.3g} (dof={dof})"
    except Exception:
        stat_txt = "Chi2 test unavailable (no scipy)"

    ax.text(0.5, 0.02, stat_txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[SOZ-OUTCOME] Saved: {out_path.resolve()}")
        print("[SOZ-OUTCOME] contingency table:\n", tab)



def _kruskal_or_permutation_kgroups(groups: dict, n_perm: int = 20000, seed: int = 0):
    """
    groups: dict[label -> 1D array]
    Retourne (pvalue, test_name)
    - essaie Kruskal-Wallis (scipy)
    - sinon permutation test sur stat = variance des moyennes de groupe (pondérée)
    """
    # clean
    clean = {}
    for k, arr in groups.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) > 0:
            clean[k] = a

    # besoin d'au moins 2 groupes non vides
    if len(clean) < 2:
        return np.nan, "insufficient groups"

    # try scipy
    try:
        from scipy.stats import kruskal  # type: ignore
        stat, p = kruskal(*[clean[k] for k in clean.keys()])
        return float(p), "Kruskal–Wallis"
    except Exception:
        rng = np.random.default_rng(seed)

        labels = []
        values = []
        for k, a in clean.items():
            labels.extend([k] * len(a))
            values.extend(a.tolist())
        labels = np.array(labels)
        values = np.array(values, dtype=float)

        # observed stat: variance of group means (weighted by group size)
        def stat_fn(vals, labs):
            uniq = np.unique(labs)
            means = []
            weights = []
            for u in uniq:
                v = vals[labs == u]
                means.append(np.mean(v))
                weights.append(len(v))
            means = np.array(means, dtype=float)
            weights = np.array(weights, dtype=float)
            m = np.average(means, weights=weights)
            return float(np.average((means - m) ** 2, weights=weights))

        obs = stat_fn(values, labels)

        cnt = 0
        for _ in range(n_perm):
            perm = rng.permutation(values)
            s = stat_fn(perm, labels)
            if s >= obs:
                cnt += 1
        p = (cnt + 1) / (n_perm + 1)
        return float(p), f"Permutation test (k-groups, n={n_perm})"


def plot_amp_stability_bar_and_groupbox_by_soz(
    df_amp_stab: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str = "stability_cosine_weighted_mean",
    include_unknown: bool = True,
    verbose: bool = True,
):
    """
    2 panels:
      - gauche: barplot trié, couleur = SOZ location (temporal/frontal/insular/mixed/unknown)
      - droite: boxplots par SOZ location (xticks = SOZ groups) + points (1 pt/patient)
               + test stat k-group (Kruskal ou permutation)
    """
    if df_amp_stab.empty:
        if verbose:
            print("[AMP-SOZ-2PANEL] df vide -> skip")
        return

    df = df_amp_stab.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df[value_col].astype(float))].copy()
    df[value_col] = df[value_col].astype(float)
    if df.empty:
        if verbose:
            print("[AMP-SOZ-2PANEL] aucune valeur finite -> skip")
        return

    # ajoute groupe SOZ
    df["soz_group"] = df["patient_key"].apply(soz_location_group)

    if not include_unknown:
        df = df[df["soz_group"] != "unknown"].copy()

    # ordre standard des groupes sur le panel droit
    group_order = ["temporal", "frontal", "insular", "mixed"]
    if include_unknown:
        group_order.append("unknown")

    # garder uniquement les groupes présents
    present = [g for g in group_order if (df["soz_group"] == g).any()]

    # --- panel gauche: barplot trié ---
    df_sorted = df.sort_values(value_col, ascending=False)
    patient_keys = df_sorted["patient_key"].tolist()
    vals = df_sorted[value_col].values
    colors = [color_soz(pk) for pk in patient_keys]

    # --- panel droit: boxplots par groupe ---
    groups = {}
    data = []
    labels = []
    for g in present:
        arr = df.loc[df["soz_group"] == g, value_col].values
        groups[g] = arr
        data.append(arr)
        labels.append(g)

    pval, test_name = _kruskal_or_permutation_kgroups(groups)

    # --- figure layout ---
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # gauche
    ax0.bar(patient_keys, vals, color=colors)
    ax0.set_xticks(np.arange(len(patient_keys)))
    ax0.set_xticklabels(patient_keys, rotation=90, fontsize=8)
    ax0.set_ylabel("Weighted cosine similarity\n(pattern + amplitude)")
    ax0.set_title(f"{title}\nColor by: SOZ location")
    add_legend(ax0, "soz")

    # droite
    ax1.boxplot(data, labels=labels, showfliers=False, widths=0.6)

    # points (couleur = groupe SOZ)
    rng = np.random.default_rng(0)
    for i, g in enumerate(labels, start=1):
        arr = np.asarray(groups[g], dtype=float)
        if len(arr) == 0:
            continue
        jitter = rng.normal(0.0, 0.04, size=len(arr))
        x = i + jitter
        c = SOZ_COLOR.get(g, "0.6")
        ax1.scatter(x, arr, c=c)

    ax1.set_ylabel("Weighted cosine similarity")
    ax1.set_title("By SOZ location (stats)")

    # annotation p-value
    Ns = [len(groups[g]) for g in labels]
    if np.isfinite(pval):
        txt = f"{test_name}\np = {pval:.3g}\n" + "  ".join([f"N_{g}={n}" for g, n in zip(labels, Ns)])
    else:
        txt = f"{test_name}\n(p-value NA)\n" + "  ".join([f"N_{g}={n}" for g, n in zip(labels, Ns)])
    ax1.text(0.5, 0.02, txt, transform=ax1.transAxes, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[AMP-SOZ-2PANEL] Saved: {out_path.resolve()}")





def compute_patient_consistent_change(
    df_metrics: pd.DataFrame,
    value_col: str = "change_global_abs",
) -> pd.DataFrame:
    """
    Nouvelle métrique qui met en valeur un CHANGEMENT constant (non-nul).

    Pour chaque patient:
      - x_s = vecteur (features) pour chaque seizure s
      - x_bar = moyenne des x_s
      - ConsistentChangeScore = ||x_bar||   (=> 0 si tout est ~0)
      - ConsistencyIndex = ||x_bar|| / mean(||x_s||)  in [0,1] (si mean norm > 0)
      - MeanAmplitude = mean(||x_s||)

    Remarque: marche aussi avec value_col="change_soz_amp_abs" (signé).
    """
    rows = []

    for pk, g_pat in df_metrics.groupby("patient_key"):
        mat = (
            g_pat.groupby(["seizure", "feature"])[value_col]
            .mean()
            .unstack("feature")
        )

        # garder lignes seizure valides (au moins 1 valeur)
        mat = mat.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
        if mat.shape[0] < 2:
            rows.append(dict(
                patient_key=pk,
                n_seizures=int(mat.shape[0]),
                ConsistentChangeScore=np.nan,
                ConsistencyIndex=np.nan,
                MeanAmplitude=np.nan,
            ))
            continue

        X = np.nan_to_num(mat.values.astype(float), nan=0.0)
        norms = np.linalg.norm(X, axis=1)  # amplitude par seizure
        mean_amp = float(np.mean(norms)) if len(norms) else np.nan

        x_bar = np.mean(X, axis=0)
        score = float(np.linalg.norm(x_bar))  # <-- metric principale

        consistency = (score / mean_amp) if (np.isfinite(mean_amp) and mean_amp > 0) else np.nan

        rows.append(dict(
            patient_key=pk,
            n_seizures=int(mat.shape[0]),
            ConsistentChangeScore=score,
            ConsistencyIndex=consistency,
            MeanAmplitude=mean_amp,
        ))

    return pd.DataFrame(rows)




# =========================
# Main / CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Timecourses + heatmaps simples (diff de mean preictal->ictal, et SOZ amp)."
    )
    p.add_argument(
        "root_base",
        nargs="?",
        default=ROOT_DEFAULT,
        help=f"Dossier racine contenant le parquet master (défaut: {ROOT_DEFAULT})",
    )
    p.add_argument(
        "patients",
        nargs="+",
        help="Liste tokens patients: CHUM::Patient_01 ds004100::sub-HUP070 ...",
    )
    p.add_argument("--no-show", action="store_true", help="Ne pas afficher les figures")
    p.add_argument("--save-dir", type=str, default=None, help="Dossier de sortie (png + csv)")
    p.add_argument("--quiet", action="store_true", help="Moins de logs")
    p.add_argument("--no-timecourses", action="store_true", help="Ne pas générer les courbes temporelles")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    show = not args.no_show

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # (1) courbes temporelles
    if not args.no_timecourses:
        plot_timecourses_for_multiple_patients(
            root_base=args.root_base,
            patient_tokens=args.patients,
            show=show,
            save_dir=save_dir,
            verbose=verbose,
        )


    # (2) métriques simples
    df_metrics = compute_mean_change_metrics_for_patients(
        root_base=args.root_base,
        patient_tokens=args.patients,
        verbose=verbose,
    )
    if df_metrics.empty:
        print("[MAIN] df_metrics vide -> rien à heatmap.")
        return

    # CSV
    if save_dir is not None:
        csv_path = save_dir / "mean_change_metrics_per_seizure.csv"
        df_metrics.to_csv(csv_path, index=False)
        print(f"[MAIN] CSV saved: {csv_path.resolve()}")

    if save_dir is not None:

        """
    # (3) heatmaps features × patients
    
        out1 = save_dir / "heatmap_features_x_patients_change_global_abs.png"
        plot_heatmap_features_x_patients(
            df_metrics=df_metrics,
            value_col="change_global_abs",
            out_path=out1,
            title="Feature change magnitude |mean_ictal - mean_preictal| (avg over seizures)",
            symmetric=False,
            verbose=verbose,
        )

        out2 = save_dir / "heatmap_features_x_patients_change_soz_amp_abs.png"
        plot_heatmap_features_x_patients(
            df_metrics=df_metrics,
            value_col="change_soz_amp_abs",
            out_path=out2,
            title="SOZ amplification of change magnitude (SOZ - non-SOZ), avg over seizures",
            symmetric=True,
            verbose=verbose,
        )
        """

        # (4) 2 figures par patient : features × seizures
        """
        out3 = save_dir / "heatmaps_per_patient_features_x_seizures_change_global_abs.png"
        plot_heatmaps_per_patient_features_x_seizures(
            df_metrics=df_metrics,
            value_col="change_global_abs",
            out_path=out3,
            title="Per Patient Preictal-to-Ictal Change Magnitude: |μictal − μpre| Per Feature and Seizure\nBAD Surgery Outcome",
            symmetric=False,
            per_patient_scale=True,
            verbose=verbose,
        )

        out4 = save_dir / "heatmaps_per_patient_features_x_seizures_change_soz_amp_abs.png"
        plot_heatmaps_per_patient_features_x_seizures(
            df_metrics=df_metrics,
            value_col="change_soz_amp_abs",
            out_path=out4,
            title="Per Patient SOZ-specific Amplification of Change (SOZ − non-SOZ) Per Feature and Seizure\nBAD Surgery Outcome",
            symmetric=True,
            per_patient_scale=True,
            verbose=verbose,
        )
        """
        """
        # (5) stabilité inter-seizures + profils patients + clustering/embedding
        run_patient_profile_and_stability_plots(
            df_metrics=df_metrics,
            save_dir=save_dir,
            value_col="change_global_abs",
            top_n=15,
            normalize_within_seizure=True,   # compare "pattern" (quelles features dominent)
            normalize_within_patient=True,
            verbose=verbose,
        )

        # (optionnel) version "amplitude" (souvent utile à comparer)
        run_patient_profile_and_stability_plots(
            df_metrics=df_metrics,
            save_dir="test",
            value_col="change_global_abs",
            top_n=15,
            normalize_within_seizure=False,  # compare amplitude globale + pattern
            normalize_within_patient=False,
            verbose=verbose,
        )
        """
        """
        # (6) stabilité amplitude-aware (pattern + amplitude)
        df_amp_stab = compute_patient_amplitude_aware_stability(
            df_metrics=df_metrics,
            value_col="change_global_abs",
        )

        # CSV
        df_amp_stab.to_csv(save_dir / "patient_stability_amplitude_aware.csv", index=False)

        # Plot
        plot_amplitude_aware_stability_barplot(
            df_amp_stab=df_amp_stab,
            out_path=save_dir / "patient_stability_amplitude_aware.png",
            title="Amplitude-aware inter-seizure stability\n(weighted cosine similarity)",
            verbose=verbose,
        )

        df_amp_stab = compute_patient_amplitude_aware_stability(df_metrics=df_metrics, value_col="change_global_abs")
        df_amp_stab.to_csv(save_dir / "patient_stability_amplitude_aware.csv", index=False)


        # (5) stabilité inter-seizures + profils patients + clustering/embedding
        df_stab, prof = run_patient_profile_and_stability_plots(
            df_metrics=df_metrics,
            save_dir=save_dir,
            value_col="change_global_abs",
            top_n=15,
            normalize_within_seizure=True,
            normalize_within_patient=True,
            verbose=verbose,
        )

        # (6) amplitude-aware stability + 2-panel figure (outcome + SOZ)
        df_amp_stab = compute_patient_amplitude_aware_stability(
            df_metrics=df_metrics,
            value_col="change_global_abs",
        )
        df_amp_stab.to_csv(save_dir / "patient_stability_amplitude_aware.csv", index=False)

        plot_amp_stability_2panel_outcome_soz(
            df_amp_stab=df_amp_stab,
            out_path=save_dir / "patient_stability_amplitude_aware_2panel_outcome_soz.png",
            title="Amplitude-aware inter-seizure stability (weighted cosine)\n1 bar = 1 patient",
            value_col="stability_cosine_weighted_mean",
            verbose=verbose,
        )

        # (7) lien SOZ location ↔ outcome
        patient_keys = sorted(df_metrics["patient_key"].unique().tolist())
        plot_soz_vs_outcome_association(
            patient_keys=patient_keys,
            out_path=save_dir / "soz_location_vs_outcome.png",
            title="SOZ location group vs surgery outcome",
            verbose=verbose,
        )


        # --- amplitude-aware stability ---
        df_amp_stab = compute_patient_amplitude_aware_stability(df_metrics=df_metrics, value_col="change_global_abs")
        df_amp_stab.to_csv(save_dir / "patient_stability_amplitude_aware.csv", index=False)

        plot_amp_stability_2panel(
            df_amp_stab=df_amp_stab,
            out_path=save_dir / "patient_stability_amplitude_aware_2panel_colorBY_outcome.png",
            title="Amplitude-aware inter-seizure stability (weighted cosine)\n1 bar = 1 patient",
            color_by="outcome",
            value_col="stability_cosine_weighted_mean",
            verbose=verbose,
        )

        plot_amp_stability_2panel(
            df_amp_stab=df_amp_stab,
            out_path=save_dir / "patient_stability_amplitude_aware_2panel_colorBY_soz.png",
            title="Amplitude-aware inter-seizure stability (weighted cosine)\n1 bar = 1 patient",
            color_by="soz",
            value_col="stability_cosine_weighted_mean",
            verbose=verbose,
        )

        """
        #df_amp_stab = compute_patient_amplitude_aware_stability(df_metrics=df_metrics, value_col="change_soz_amp_abs")
        #df_amp_stab.to_csv(save_dir / "patient_stability_amplitude_aware.csv", index=False)

        # --- patient profile + embedding/clustering ---
        df_stab, prof = run_patient_profile_and_stability_plots(
            df_metrics=df_metrics,
            save_dir=save_dir,
            value_col="change_soz_amp_abs",
            top_n=15,
            normalize_within_seizure=True,
            normalize_within_patient=True,
            verbose=verbose,
        )

        plot_patient_umap_or_tsne(
            patient_profile=prof,
            out_path=save_dir / "patient_profile_embedding_colorBY_outcome.png",
            title="Patient embedding from profiles — change_soz_amp_abs",
            color_by="outcome",
            random_state=0,
            verbose=verbose,
        )

        plot_patient_umap_or_tsne(
            patient_profile=prof,
            out_path=save_dir / "patient_profile_embedding_colorBY_soz.png",
            title="Patient embedding from profiles — change_soz_amp_abs",
            color_by="soz",
            random_state=0,
            verbose=verbose,
        )

        patient_keys = sorted(df_metrics["patient_key"].unique().tolist())
        plot_soz_vs_outcome_association(
            patient_keys=patient_keys,
            out_path=save_dir / "soz_location_vs_outcome.png",
            title="SOZ location group vs surgery outcome",
            verbose=verbose,
        )
        """
        plot_amp_stability_bar_and_groupbox_by_soz(
        df_amp_stab=df_amp_stab,
        out_path=save_dir / "patient_stability_amplitude_aware_2panel_colorBY_soz.png",
        title="Amplitude-aware inter-seizure stability (weighted cosine)\n1 bar = 1 patient",
        value_col="stability_cosine_weighted_mean",
        include_unknown=True,
        verbose=verbose,
    )
        
        plot_amp_stability_2panel(
            df_amp_stab=df_amp_stab,
            out_path=save_dir / "patient_stability_amplitude_aware_2panel_colorBY_outcome.png",
            title="Amplitude-aware inter-seizure stability (weighted cosine)\n1 bar = 1 patient",
            color_by="outcome",
            value_col="stability_cosine_weighted_mean",
            verbose=verbose,
        )
        """
        # (6) nouvelle stabilité "constant change" (anti-0)
        df_cc = compute_patient_constant_change_score(
            df_metrics=df_metrics,
            value_col="change_soz_amp_abs",   # ou "change_soz_amp_abs"
        )
        df_cc.to_csv(save_dir / "patient_constant_change_score.csv", index=False)

        # 2-panel — couleur outcome
        plot_amp_stability_2panel(
            df_amp_stab=df_cc.rename(columns={"constant_change_score":"stability_value"}),
            out_path=save_dir / "patient_constant_change_2panel_colorBY_outcome.png",
            title="Constant change score (pattern stable + non-trivial)\nColor by outcome",
            color_by="outcome",
            value_col="stability_value",
            verbose=verbose,
        )

        # 2-panel — couleur SOZ
        plot_amp_stability_2panel(
            df_amp_stab=df_cc.rename(columns={"constant_change_score":"stability_value"}),
            out_path=save_dir / "patient_constant_change_2panel_colorBY_soz.png",
            title="Constant change score (pattern stable + non-trivial)\nColor by SOZ location",
            color_by="soz",
            value_col="stability_value",
            verbose=verbose,
        )






if __name__ == "__main__":
    main()
