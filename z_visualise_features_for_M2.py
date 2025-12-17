#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualisation des features par électrode à partir d'un fichier features.parquet.

Génère automatiquement 3 types de figures :

1) features_per_seizure_<patient>.png
   - 1 subplot par seizure du patient
   - normalisation locale par seizure/feature
   - SOZ en rouge, non-SOZ en noir
   - highlight des N features les plus discriminantes (|d'| max)

2) features_all_seizures_<patient>.png
   - un seul plot pour toutes les seizures du patient
   - normalisation globale par feature (au niveau du patient)

3) features_GLOBAL_all_patients.png
   - un seul plot avec *tous les patients et toutes les seizures*
   - normalisation globale par feature (global dataset)
   - permet de voir la tendance globale des features

MODIF (important) :
- Pour chaque patient et chaque seizure, on lit les CSV temporels dans :
    results/<patient>/series/seiz_*_seq_*.csv
  et on détecte les électrodes dont la série temporelle est à 0 (tolérance atol) pour
  tous les temps. Ces électrodes sont retirées des plots de features.
- Print, pour chaque (patient, seizure), la liste des électrodes retirées.
- Filtrage possible :
    * par défaut: on retire toutes les électrodes plates (SOZ et non-SOZ)
    * option: ne retirer que celles qui sont SOZ (voir flag --drop_only_soz)


uv run z_visualise_features_for_M2.py     --parquet results_grid_search_kwta_20_10_burst/results_20251213_011858/features/features_augmented.parquet \
--out figures_features_M2_10_20_burst \
--drop_only_soz
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

offset = 0.1


# ---------------------------------------------------------
# d-prime
# ---------------------------------------------------------
def d_prime(soz_vals: np.ndarray, non_vals: np.ndarray) -> float:
    soz_vals = soz_vals[~np.isnan(soz_vals)]
    non_vals = non_vals[~np.isnan(non_vals)]
    if len(soz_vals) < 1 or len(non_vals) < 1:
        return np.nan
    mu1, mu2 = np.mean(soz_vals), np.mean(non_vals)
    sd1, sd2 = np.std(soz_vals), np.std(non_vals)
    denom = np.sqrt(0.5 * (sd1**2 + sd2**2))
    if denom == 0:
        return np.nan
    return (mu1 - mu2) / denom


# ---------------------------------------------------------
# Utils: normalisations + paths
# ---------------------------------------------------------
def normalize_seizure_id(x) -> Optional[int]:
    """
    Convertit seizure_id (df parquet) en int si possible.
    Supporte int/float/str ; si str contient un nombre, l'extrait.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        return int(x)
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def normalize_electrode_name(x) -> str:
    """
    Normalise un nom d'électrode pour matcher parquet <-> CSV.
    Ex: "SEEG F23" -> "F23"
        "F23" -> "F23"
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # enlève "SEEG" où qu'il soit
    s = s.replace("SEEG", "").strip()
    # nettoyage espaces multiples
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_results_root(parquet_path: Path) -> Optional[Path]:
    """
    Remonte dans les parents du parquet jusqu'à trouver un dossier contenant 'results/'.
    Retourne le Path vers ce dossier 'results', sinon None.
    """
    for p in [parquet_path.parent] + list(parquet_path.parents):
        candidate = p / "results"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------
# CSV -> électrodes all-zero
# ---------------------------------------------------------
def find_zero_electrodes_from_csv(csv_path: Path, atol: float = 0.0) -> List[str]:
    """
    Retourne la liste des electrode_name (ex: 'I64') dont la colonne est à 0
    sur tous les temps dans ce CSV.
    - atol permet une tolérance (0 strict si atol=0).
    """
    df_csv = pd.read_csv(csv_path, comment="#")
    zero_electrodes: List[str] = []

    for col in df_csv.columns:
        if not str(col).startswith("node_"):
            continue

        # ex: "node_066:SEEG I64" -> "I64" (tout ce qui est après 'SEEG')
        parts = str(col).split("SEEG", 1)
        if len(parts) != 2:
            continue
        electrode = parts[1].strip()
        electrode = normalize_electrode_name(electrode)  # au cas où
        if electrode == "":
            continue

        values = pd.to_numeric(df_csv[col], errors="coerce").fillna(0.0).values
        if values.size == 0:
            continue

        if np.all(np.abs(values) <= atol):
            zero_electrodes.append(electrode)

    return sorted(set(zero_electrodes))


# ---------------------------------------------------------
# PLOTTER PRINCIPAL
# ---------------------------------------------------------
def plot_features_for_all_patients(
    parquet_path: str,
    output_dir: Optional[str] = None,
    figsize_per_feature: float = 0.3,
    base_width: float = 12.0,
    height_per_seizure: float = 3.0,
    n_top_features: int = 5,
    atol_csv_zero: float = 0.0,      # tolérance pour "zéro" dans les CSV
    drop_only_soz: bool = True,     # True = ne retirer que les électrodes plates SOZ
):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Fichier parquet introuvable : {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Sortie
    if output_dir is None:
        output_dir = parquet_path.parent / "feature_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Meta à exclure
    meta_cols = {
        "patient", "seizure_id", "seq_idx",
        "node_index", "electrode_name",
        "is_SOZ", "is_flat_zero"
    }

    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    feature_cols = sorted(feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("Aucune colonne de features numérique détectée (après exclusion des meta_cols).")

    # ------------------------------------------------------------
    # Normalisation des clés pour merge parquet <-> CSV
    # ------------------------------------------------------------
    df = df.copy()
    df["seizure_id_norm"] = df["seizure_id"].apply(normalize_seizure_id)
    df["seizure_id_norm"] = pd.to_numeric(df["seizure_id_norm"], errors="coerce").astype("Int64")
    df["electrode_name_norm"] = df["electrode_name"].apply(normalize_electrode_name)

    # ------------------------------------------------------------
    # Filtrage via CSV temporels
    # ------------------------------------------------------------
    results_root = find_results_root(parquet_path)

    if results_root is None:
        print("[WARN] Impossible de trouver un dossier 'results/' en remontant depuis le parquet.")
        print("       -> Aucun filtrage CSV appliqué, on continue avec le parquet tel quel.")
    else:
        patients = sorted(df["patient"].dropna().unique().tolist())

        # (patient, seizure_id_norm) -> liste electrodes à retirer
        removed_map: Dict[Tuple[str, int], List[str]] = {}

        for patient in patients:
            patient_dir = results_root / patient / "series"
            if not patient_dir.exists():
                print(f"[WARN] Dossier manquant : {patient_dir}")
                continue

            csv_files = sorted(patient_dir.glob("seiz_*_seq_*.csv"))
            if len(csv_files) == 0:
                print(f"[WARN] Aucun CSV trouvé dans : {patient_dir}")
                continue

            for csv_file in csv_files:
                m = re.search(r"seiz_(\d+)_seq_", csv_file.name)
                if m is None:
                    continue
                seiz_id = int(m.group(1))

                zero_electrodes = find_zero_electrodes_from_csv(csv_file, atol=atol_csv_zero)
                if len(zero_electrodes) > 0:
                    removed_map[(patient, seiz_id)] = sorted(
                        set(removed_map.get((patient, seiz_id), [])) | set(zero_electrodes)
                    )

        # Print demandé : par seizure, quelles electrodes retirées
        if len(removed_map) == 0:
            print("[INFO] Filtrage CSV: aucune électrode tout-à-0 détectée.")
        else:
            print("\n[INFO] Filtrage CSV: électrodes retirées (tout-à-0 sur tous les temps) :")
            for (patient, seiz_id), elecs in sorted(removed_map.items(), key=lambda x: (x[0][0], x[0][1])):
                print(f"  - {patient} | seizure {seiz_id} : {len(elecs)} électrodes -> {', '.join(elecs)}")

        # Appliquer le filtrage au parquet features
        if len(removed_map) > 0:
            rows = []
            for (patient, seiz_id), elecs in removed_map.items():
                for e in elecs:
                    rows.append({
                        "patient": patient,
                        "seizure_id_norm": seiz_id,
                        "electrode_name_norm": normalize_electrode_name(e),
                    })
            df_zero = pd.DataFrame(rows).drop_duplicates()
            df_zero["seizure_id_norm"] = pd.to_numeric(df_zero["seizure_id_norm"], errors="coerce").astype("Int64")

            # Debug match sur clés normalisées
            probe = df.merge(
                df_zero[["patient", "seizure_id_norm", "electrode_name_norm"]],
                on=["patient", "seizure_id_norm", "electrode_name_norm"],
                how="inner"
            )
            print(f"[DEBUG] Matches parquet/features vs CSV-zero (norm) : {len(probe)}")
            if len(probe) > 0:
                print("[DEBUG] Exemple match:\n",
                      probe[["patient", "seizure_id_norm", "electrode_name_norm"]]
                      .head(10).to_string(index=False))

            before = len(df)
            df = df.merge(
                df_zero[["patient", "seizure_id_norm", "electrode_name_norm"]].assign(_drop=1),
                on=["patient", "seizure_id_norm", "electrode_name_norm"],
                how="left"
            )

            if drop_only_soz:
                print("YESSSSSSSSSSSSSSS")
                # Supprime seulement si (plate) ET SOZ
                to_remove = (df["_drop"] == 1) & (df["is_SOZ"] == 1)
                removed = int(to_remove.sum())
                df = df.loc[~to_remove].drop(columns=["_drop"])
                print(f"\n[INFO] Lignes supprimées dans le parquet features (plates & SOZ) : {removed} (reste {len(df)})")
            else:
                df = df[df["_drop"].isna()].drop(columns=["_drop"])
                after = len(df)
                print(f"\n[INFO] Lignes supprimées dans le parquet features (après filtrage CSV) : {before - after} (reste {after})")

    # X
    x_positions = np.arange(len(feature_cols))

    patients = sorted(df["patient"].unique().tolist())
    print(f"\n[INFO] Patients trouvés : {patients}")

    # ============================================================
    #  FIGURE 3 : GLOBAL
    # ============================================================
    def plot_global_figure():
        fig, ax = plt.subplots(1, 1, figsize=(max(base_width, figsize_per_feature * len(feature_cols)), 6))

        df_global = df.copy()
        df_global["is_SOZ_bin"] = (df_global["is_SOZ"] == 1).astype(int)

        df_non = df_global[df_global["is_SOZ_bin"] == 0]
        df_soz = df_global[df_global["is_SOZ_bin"] == 1]

        print("[INFO] Construction de la figure globale...")

        for i, col in enumerate(feature_cols):
            vals = df_global[col].astype(float).values
            if np.all(np.isnan(vals)):
                continue

            min_v, max_v = np.nanmin(vals), np.nanmax(vals)

            def norm(v):
                v = np.asarray(v, float)
                if max_v == min_v or np.isnan(max_v) or np.isnan(min_v):
                    return np.full_like(v, 0.5)
                return (v - min_v) / (max_v - min_v)

            if not df_non.empty:
                y = norm(df_non[col].values)
                ax.scatter(np.full(len(y), i + offset), y, s=10, c="black", alpha=0.5, linewidths=0)

            if not df_soz.empty:
                y = norm(df_soz[col].values)
                ax.scatter(np.full(len(y), i - offset), y, s=25, c="red", alpha=0.85, edgecolors="none")

        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_cols, rotation=90)
        ax.set_title("Features – GLOBAL (tous patients + seizures)")
        ax.set_xlabel("Features")

        out = output_dir / "features_GLOBAL_all_patients.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[OK] Figure globale : {out}")

    # ============================================================
    #  FIGURES PAR PATIENT
    # ============================================================
    for patient in patients:
        print(f"\n[INFO] Patient : {patient}")

        df_pat = df[df["patient"] == patient].copy()
        df_pat["is_SOZ_bin"] = (df_pat["is_SOZ"] == 1).astype(int)

        seizures = sorted(df_pat["seizure_id_norm"].dropna().astype(int).unique().tolist())

        width = max(base_width, figsize_per_feature * len(feature_cols))

        # ------------------------------------------------------------
        # FIGURE 1 : PAR SEIZURE AVEC HIGHLIGHT
        # ------------------------------------------------------------
        fig1, axes = plt.subplots(
            len(seizures), 1,
            figsize=(width, max(height_per_seizure * len(seizures), height_per_seizure)),
            sharex=True
        )
        if len(seizures) == 1:
            axes = [axes]

        for ax, seiz in zip(axes, seizures):
            df_sz = df_pat[df_pat["seizure_id_norm"] == seiz]
            df_non = df_sz[df_sz["is_SOZ_bin"] == 0]
            df_soz = df_sz[df_sz["is_SOZ_bin"] == 1]

            sep = {}
            if not df_soz.empty and not df_non.empty:
                for col in feature_cols:
                    sep[col] = abs(d_prime(df_soz[col].values, df_non[col].values))
                top_cols = sorted(sep, key=lambda c: sep[c], reverse=True)[:n_top_features]
                top_idx = [feature_cols.index(c) for c in top_cols]
            else:
                top_idx = []

            for idx in top_idx:
                ax.axvspan(idx - 0.45, idx + 0.45, color="lightgray", alpha=0.35, zorder=0)

            for i, col in enumerate(feature_cols):
                vals = df_sz[col].values.astype(float)
                if np.all(np.isnan(vals)):
                    continue
                min_v, max_v = np.nanmin(vals), np.nanmax(vals)

                def norm(v):
                    v = v.astype(float)
                    if max_v == min_v or np.isnan(min_v) or np.isnan(max_v):
                        return np.full_like(v, 0.5)
                    return (v - min_v) / (max_v - min_v)

                if not df_non.empty:
                    y = norm(df_non[col].values)
                    ax.scatter(np.full(len(y), i + offset), y, s=10, c="black", alpha=0.5, linewidths=0, zorder=1)

                if not df_soz.empty:
                    y = norm(df_soz[col].values)
                    ax.scatter(np.full(len(y), i - offset), y, s=28, c="red", alpha=0.9, edgecolors="none", zorder=2)

            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([])
            ax.set_title(f"Seizure {seiz}", loc="left")

        axes[-1].set_xticks(x_positions)
        axes[-1].set_xticklabels(feature_cols, rotation=90)
        axes[-1].set_xlabel("Features")

        plt.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))
        out1 = output_dir / f"features_per_seizure_{patient}.png"
        plt.savefig(out1, dpi=300)
        plt.close(fig1)
        print(f"[OK] Figure par seizure : {out1}")

        # ------------------------------------------------------------
        # FIGURE 2 : ALL SEIZURES (patient)
        # ------------------------------------------------------------
        fig2, ax2 = plt.subplots(1, 1, figsize=(width, 5.0))
        df_non = df_pat[df_pat["is_SOZ_bin"] == 0]
        df_soz = df_pat[df_pat["is_SOZ_bin"] == 1]

        for i, col in enumerate(feature_cols):
            vals = df_pat[col].astype(float).values
            if np.all(np.isnan(vals)):
                continue
            min_v, max_v = np.nanmin(vals), np.nanmax(vals)

            def norm(v):
                v = v.astype(float)
                if max_v == min_v or np.isnan(min_v) or np.isnan(max_v):
                    return np.full_like(v, 0.5)
                return (v - min_v) / (max_v - min_v)

            if not df_non.empty:
                y = norm(df_non[col].values)
                ax2.scatter(np.full(len(y), i + offset), y, c="black", s=10, alpha=0.5, linewidths=0)

            if not df_soz.empty:
                y = norm(df_soz[col].values)
                ax2.scatter(np.full(len(y), i - offset), y, c="red", s=25, alpha=0.9, edgecolors="none")

        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(feature_cols, rotation=90)
        ax2.set_xlabel("Features")
        ax2.set_title(f"Features – Patient {patient} – all seizures")

        plt.tight_layout()
        out2 = output_dir / f"features_all_seizures_{patient}.png"
        plt.savefig(out2, dpi=300)
        plt.close(fig2)
        print(f"[OK] Figure all seizures : {out2}")

    # ============================================================
    #  FIGURE GLOBALE
    # ============================================================
    plot_global_figure()

    print("\n[OK] Toutes les figures ont été générées.")


# ---------- MAIN ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--ntop", type=int, default=5)
    parser.add_argument("--atol_csv_zero", type=float, default=0.0,
                        help="Tolérance pour considérer une électrode comme à 0 dans les CSV (0 strict par défaut).")
    parser.add_argument("--drop_only_soz", action="store_true",
                        help="Si activé: retire uniquement les électrodes plates qui sont SOZ (sinon retire toutes les plates).")

    args = parser.parse_args()

    plot_features_for_all_patients(
        parquet_path=args.parquet,
        output_dir=args.outdir,
        n_top_features=args.ntop,
        atol_csv_zero=args.atol_csv_zero,
        drop_only_soz=args.drop_only_soz,
    )
