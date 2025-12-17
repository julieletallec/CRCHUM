
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
    ds004100::sub-HUP126 \
    ds004100::sub-HUP130 \
    ds004100::sub-HUP141 \
    ds004100::sub-HUP144 \
    ds004100::sub-HUP148 \
    ds004100::sub-HUP150 \
    ds004100::sub-HUP157 \
    ds004100::sub-HUP173 \
    ds004100::sub-HUP180 \
    ds004100::sub-HUP185 \
    --save-dir ./figures_features_M1_10_20_burst_new 
"""


ROOT_DEFAULT = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data2"

# =========================
# Features à tracer / analyser
# =========================
EXPLICIT_NODE_FEATURES = [
    'burstint_ratio_bg_ta',
    'burstint_ratio_gamma_delta',
    'burstint_sef95_Hz',
    'burstint_spike_sharpness',
    'burstint_line_length',
    'burstint_tkeo_energy',
    'burstint_hg_power_80_150',
    'burstint_hg_over_gamma',
    'burstint_spec_slope_2_80',
    'burstint_spec_intercept_2_80',
    'burstint_hjorth_activity',
    'burstint_hjorth_mobility',
    'burstint_hjorth_complexity',
    'burstint_kurtosis',
    'burstint_skewness'
]


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
    Charge ROOT/master_node_features_SOZ_normalized_aug_20_10_burst_norm.parquet
    """
    root_base = Path(root_base)
    parquet_path = root_base / "master_node_features_SOZ_normalized_aug_20_10_burst_norm.parquet"
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

    epochs_selected = load_selected_epochs_from_manifest(
        root_base, dataset, patient_id, seizure_num, verbose=verbose
    )

    # fallback: tous les epochs dispo
    if epochs_selected is None:
        pre_epochs = sorted(df_sub.loc[df_sub["phase"] == "preictal", "epoch"].dropna().unique().tolist())
        ict_epochs = sorted(df_sub.loc[df_sub["phase"] == "ictal", "epoch"].dropna().unique().tolist())
        epochs_selected = {"preictal": pre_epochs, "ictal": ict_epochs}
        if verbose:
            print(f"[INFO] Fallback epochs {dataset} {patient_id} seizure {seizure_num}: "
                  f"pre={len(pre_epochs)}, ict={len(ict_epochs)}")

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
            plt.text(transition_idx - 0.5, ymax, "preictal → ictal", rotation=90,
                     va="top", ha="right", color="gray")

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
            plot_timecourses_per_patient_seizure_feature(
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
                # si une des phases est vide, pas de diff de mean fiable
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

            electrodes = sorted(set(df_pre["electrode"].dropna().unique()).union(df_ict["electrode"].dropna().unique()))
            if len(electrodes) == 0:
                continue

            # pré-calcul des moyennes par (electrode, feature)
            pre_mean = df_pre.groupby("electrode")[feature_cols].mean(numeric_only=True)
            ict_mean = df_ict.groupby("electrode")[feature_cols].mean(numeric_only=True)

            for feat in feature_cols:
                # change par électrode
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

                rows.append(dict(
                    dataset=dataset,
                    patient=patient_id,
                    patient_key=f"{dataset}::{patient_id}",
                    seizure=int(seiz),
                    feature=feat,
                    change_global_abs=global_abs,
                    change_soz_abs=soz_abs,
                    change_nonsoz_abs=nonsoz_abs,
                    change_soz_amp_abs=soz_amp_abs,
                ))

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

    # échelle
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

    # colormap par défaut
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
        else:
            vmin_use, vmax_use = vmin, vmax

        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin_use, vmax=vmax_use, cmap=cmap)
        last_im = im

        ax.set_title(pk, fontsize=9)
        if c == 0:
            ax.set_yticks(np.arange(len(features)))
            ax.set_yticklabels(features, fontsize=7)
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

    # colorbar globale seulement si échelle globale
    if last_im is not None and not per_patient_scale:
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(value_col, fontsize=9)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0.01, 0.08 if not per_patient_scale else 0.02, 0.99, 0.94])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[HEATMAP-PER-PAT] Saved: {out_path.resolve()}")


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

    # (3) heatmaps features × patients
    if save_dir is not None:
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

        # (4) 2 figures par patient : features × seizures
        out3 = save_dir / "heatmaps_per_patient_features_x_seizures_change_global_abs.png"
        plot_heatmaps_per_patient_features_x_seizures(
            df_metrics=df_metrics,
            value_col="change_global_abs",
            out_path=out3,
            title="Per patient: |mean_ictal - mean_preictal| (features × seizures)",
            symmetric=False,
            per_patient_scale=True,
            verbose=verbose,
        )

        out4 = save_dir / "heatmaps_per_patient_features_x_seizures_change_soz_amp_abs.png"
        plot_heatmaps_per_patient_features_x_seizures(
            df_metrics=df_metrics,
            value_col="change_soz_amp_abs",
            out_path=out4,
            title="Per patient: SOZ amplification (features × seizures)",
            symmetric=True,
            per_patient_scale=True,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
