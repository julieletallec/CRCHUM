#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parcourt tous les dossiers results_* sous --root, puis pour chaque
classifier_experiments_augmented_balanced / exp* :

  - lit cv_fold_metrics.json
  - pour chaque patient (val_patients), récupère confidence_gap,
    mean_score_pos et mean_score_neg
  - pour chaque patient, garde le couple (results_dir, experiment)
    avec le MEILLEUR confidence_gap

Puis:

  - sauvegarde un CSV avec 1 ligne par patient
  - trace un boxplot avec 3 boxplots:
        mean_score_pos, mean_score_neg, confidence_gap
  - lit, pour chaque patient, le cv_val_predictions_ranked.csv
    correspondant au meilleur couple (results_dir, experiment),
    récupère le rang du premier SOZ par (patient, seizure, seq)
  - trace un barplot (1 barre par patient) montrant la médiane
    du rang du premier SOZ, avec barre d'erreur = IQR (25–75%)

Exemple d'appel :

uv run z_visualise_final_performances.py \
  --root /home/julieletallec/test/results_grid_search_kwta_20_10_burst \
  --out figures_grid_search_kwta_20_10_burst_LASSO_0.03_postprocess \
  --criterion AUC_group_mean\
  --mode max

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import roc_auc_score



big_exp_name = "classifier_experiments_augmented_balanced_LASSO_global_0.03_postprocess"

# ----------------------------------------------------------
# Utils : recherche des dossiers & couleur par patient
# ----------------------------------------------------------

def patient_has_min_soz_electrodes_per_seizure(
    root: Path,
    results_dir_name: str,
    exp_dir_name: str,
    patient: str,
    min_soz_electrodes: int = 2,
    count_unique_node_index: bool = True,
) -> bool:
    """
    Vérifie que pour ce patient, pour chaque seizure_id, il existe au moins
    `min_soz_electrodes` électrodes SOZ (is_SOZ==1).

    - Si count_unique_node_index=True : on compte des électrodes distinctes via node_index.
      (recommandé, sinon une même électrode répétée sur plusieurs seq_idx gonfle le compte)
    """
    pred_path = (
        root / results_dir_name
        / big_exp_name
        / exp_dir_name
        / "cv_val_predictions_ranked.csv"
    )
    if not pred_path.is_file():
        return False

    df = pd.read_csv(pred_path)

    needed = {"patient", "seizure_id", "is_SOZ"}
    if not needed.issubset(df.columns):
        return False

    df_pat = df[df["patient"].astype(str) == str(patient)].copy()
    if df_pat.empty:
        return False

    df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
    df_soz = df_pat[df_pat["is_SOZ"] == 1].copy()
    if df_soz.empty:
        return False

    if count_unique_node_index and "node_index" in df_soz.columns:
        counts = df_soz.groupby("seizure_id")["node_index"].nunique()
    else:
        # fallback: compte brut de lignes SOZ par seizure
        counts = df_soz.groupby("seizure_id").size()

    # IMPORTANT: on veut "pour chaque seizure", donc on doit itérer sur toutes les seizures du patient
    all_seizures = df_pat["seizure_id"].unique()
    counts = counts.reindex(all_seizures, fill_value=0)

    return bool((counts >= min_soz_electrodes).all())


def find_results_dirs(root: Path):
    """Retourne la liste des sous-dossiers results_YYYYMMDD_HHMMSS"""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("results_")
    )


def build_patient_color_map(patients):
    """Construit un dict patient -> couleur (palette stable)."""
    unique_patients = sorted(pd.unique(patients))
    palette = sns.color_palette("husl", len(unique_patients))
    color_map = {pat: palette[i] for i, pat in enumerate(unique_patients)}
    return color_map


# ----------------------------------------------------------
# 1) Sélection du meilleur confidence_gap par patient
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_soz_rank_boxplot_points_colored_by_seizure(
    root: Path,
    df_best: pd.DataFrame,
    out_dir: Path,
    title: str = "Per-Patient SOZ Electrode Ranks (Boxplot) with Points Colored by Seizure",
    keep_middle_seq_only: bool = False,
    max_seizures_for_palette: int = 30,
):
    """
    1 boxplot par patient (distribution des ranks des électrodes SOZ),
    1 point par électrode SOZ (chaque ligne is_SOZ==1),
    points colorés par seizure_id.

    Hypothèses colonnes dans cv_val_predictions_ranked.csv :
      patient, seizure_id, seq_idx, is_SOZ, rank
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- collect SOZ ranks from best (results_dir, experiment) per patient ----
    for _, r in df_best.iterrows():
        pat = str(r["patient"])
        res_dir = str(r["results_dir"])
        exp_dir = str(r["experiment"])

        pred_path = (
            Path(root)
            / res_dir
            / big_exp_name
            / exp_dir
            / "cv_val_predictions_ranked.csv"
        )
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat}: {pred_path}")
            continue

        df = pd.read_csv(pred_path)

        need_cols = {"patient", "seizure_id", "seq_idx", "is_SOZ", "rank"}
        if not need_cols.issubset(df.columns):
            print(f"[WARN] Missing columns in {pred_path.name}: {need_cols - set(df.columns)}")
            continue

        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            print(f"[WARN] No rows for {pat} in {pred_path}")
            continue

        # keep only SOZ electrodes
        df_soz = df_pat[df_pat["is_SOZ"].astype(int) == 1].copy()
        if df_soz.empty:
            print(f"[INFO] No SOZ rows for {pat} in {pred_path}")
            continue

        # optional: keep only middle seq per (patient, seizure_id)
        if keep_middle_seq_only:
            def _pick_middle_seq(g):
                g = g.sort_values("seq_idx")
                mid = len(g) // 2
                return g[g["seq_idx"] == g.iloc[mid]["seq_idx"]]

            df_soz = (
                df_soz.groupby(["patient", "seizure_id"], group_keys=False)
                      .apply(_pick_middle_seq)
                      .reset_index(drop=True)
            )

        df_soz["patient"] = pat
        df_soz["seizure_id"] = df_soz["seizure_id"].astype(str)
        df_soz["rank"] = pd.to_numeric(df_soz["rank"], errors="coerce")

        # garder seulement ce qu'on plot
        rows.append(df_soz[["patient", "seizure_id", "rank"]])

    if not rows:
        raise RuntimeError("No SOZ ranks collected. Check your files/columns and df_best.")

    df_plot = pd.concat(rows, ignore_index=True).dropna(subset=["rank"])
    df_plot["rank"] = df_plot["rank"].astype(float)

    # ---- color palette by seizure_id (global) ----
    seizure_ids = sorted(df_plot["seizure_id"].unique())
    if len(seizure_ids) > max_seizures_for_palette:
        print(f"[WARN] {len(seizure_ids)} seizures -> palette may be hard to read. "
              f"Consider filtering or increasing max_seizures_for_palette.")

    palette = sns.color_palette("tab20", min(len(seizure_ids), 20))
    # si >20, on recycle (simple, lisible, pas parfait)
    seizure_color = {s: palette[i % len(palette)] for i, s in enumerate(seizure_ids)}

    # ---- plot ----
    plt.figure(figsize=(max(12, 0.45 * df_plot["patient"].nunique()), 6))

    order = sorted(df_plot["patient"].unique())

    # boxplot (sans outliers, sinon ça double les points)
    sns.boxplot(
        data=df_plot,
        x="patient",
        y="rank",
        order=order,
        showfliers=False,
        color="lightgrey",
    )

    # points (stripplot) colorés par seizure
    sns.stripplot(
    data=df_plot,
    x="patient",
    y="rank",
    hue="seizure_id",
    order=order,
    palette=seizure_color,
    dodge=True,      # 👈 séparation en colonnes par seizure
    jitter=False,    # 👈 alignement strict
    size=8,
    alpha=0.8,
    linewidth=0,
)


    plt.title(title)
    plt.xlabel("")
    plt.ylabel("SOZ Electrode Rank")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=90, fontsize=8)

    # légende : si trop de seizures, elle devient énorme → on la met dehors + on peut la couper
    leg = plt.legend(
        title="seizure_id",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True
    )

    fig_path = out_dir / "boxplot_soz_rank_points_colored_by_seizure.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {fig_path}")

    return df_plot



def collect_best_model_per_patient(
    root: Path,
    criterion: str = "confidence_gap_global",
    mode: str = "max",
    require_min_soz_per_seizure: int | None = 2,   # None -> désactive la contrainte
    count_unique_node_index: bool = True,
):
    """
    Parcourt tous les results_*/classifier_experiments_augmented_balanced/exp*/cv_fold_metrics.json

    Pour chaque patient, sélectionne le couple (results_dir, experiment)
    qui optimise le critère demandé, OPTIONNELLEMENT sous contrainte :

      - pour ce patient, pour chaque seizure_id, il y a au moins N électrodes SOZ
        (is_SOZ==1) dans cv_val_predictions_ranked.csv.

    Args
    ----
    root : Path
        Dossier racine contenant les results_*
    criterion : str
        Nom de la clé dans cv_fold_metrics.json
    mode : str
        'max' si plus grand = meilleur, 'min' si plus petit = meilleur
    require_min_soz_per_seizure : int | None
        Si int, impose au moins cette valeur (>=1) d'électrodes SOZ par seizure.
        Si None, pas de contrainte.
    count_unique_node_index : bool
        Si True, compte des électrodes SOZ distinctes via node_index (recommandé).
        Si False (ou si node_index absent), compte le nombre de lignes SOZ.

    Returns
    -------
    DataFrame avec 1 ligne par patient.
    """

    assert mode in ("max", "min"), "mode doit être 'max' ou 'min'"

    def patient_has_min_soz_electrodes_per_seizure(
        root_: Path,
        results_dir_name: str,
        exp_dir_name: str,
        patient: str,
        min_soz_electrodes: int = 2,
        count_unique_node_index_: bool = True,
    ) -> bool:
        pred_path = (
            root_
            / results_dir_name
            / big_exp_name
            / exp_dir_name
            / "cv_val_predictions_ranked.csv"
        )
        if not pred_path.is_file():
            return False

        try:
            df = pd.read_csv(pred_path)
        except Exception:
            return False

        needed = {"patient", "seizure_id", "is_SOZ"}
        if not needed.issubset(df.columns):
            return False

        df_pat = df[df["patient"].astype(str) == str(patient)].copy()
        if df_pat.empty:
            return False

        df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
        df_soz = df_pat[df_pat["is_SOZ"] == 1].copy()
        if df_soz.empty:
            return False

        if count_unique_node_index_ and "node_index" in df_soz.columns:
            counts = df_soz.groupby("seizure_id")["node_index"].nunique()
        else:
            counts = df_soz.groupby("seizure_id").size()

        all_seizures = df_pat["seizure_id"].unique()
        counts = counts.reindex(all_seizures, fill_value=0)

        return bool((counts >= min_soz_electrodes).all())

    best_per_patient = {}  # patient -> dict

    for results_dir in find_results_dirs(root):
        clf_root = results_dir / big_exp_name
        if not clf_root.is_dir():
            continue

        print(f"[SCAN] {results_dir.name}")

        for exp_dir in sorted(clf_root.iterdir()):
            if not exp_dir.is_dir():
                continue

            metrics_path = exp_dir / "cv_fold_metrics.json"
            if not metrics_path.is_file():
                continue

            try:
                with open(metrics_path, "r") as f:
                    folds_metrics = json.load(f)
            except Exception as e:
                print(f"  [WARN] Impossible de lire {metrics_path}: {e}")
                continue

            # cv_fold_metrics.json = liste de dicts (1 par fold)
            for fold_entry in folds_metrics:
                val_pats = str(fold_entry.get("val_patients", "")).split(",")

                score = fold_entry.get(criterion, None)
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    continue

                mean_pos = fold_entry.get("mean_score_pos_global", None)
                mean_neg = fold_entry.get("mean_score_neg_global", None)
                conf_gap = fold_entry.get("confidence_gap_global", None)

                for pat in val_pats:
                    pat = pat.strip()
                    if not pat:
                        continue

                    # ----------------------------
                    # NOUVELLE CONTRAINTE (optionnelle)
                    # ----------------------------
                    if require_min_soz_per_seizure is not None:
                        ok = patient_has_min_soz_electrodes_per_seizure(
                            root_=root,
                            results_dir_name=results_dir.name,
                            exp_dir_name=exp_dir.name,
                            patient=pat,
                            min_soz_electrodes=require_min_soz_per_seizure,
                            count_unique_node_index_=count_unique_node_index,
                        )
                        if not ok:
                            continue

                    current_best = best_per_patient.get(pat)

                    better = False
                    if current_best is None:
                        better = True
                    elif mode == "max" and score > current_best["criterion_value"]:
                        better = True
                    elif mode == "min" and score < current_best["criterion_value"]:
                        better = True

                    if better:
                        best_per_patient[pat] = {
                            "patient": pat,
                            "results_dir": results_dir.name,
                            "experiment": exp_dir.name,
                            "criterion_name": criterion,
                            "criterion_value": float(score),
                            "mean_score_pos": float(mean_pos) if mean_pos is not None else np.nan,
                            "mean_score_neg": float(mean_neg) if mean_neg is not None else np.nan,
                            "confidence_gap": float(conf_gap) if conf_gap is not None else np.nan,
                            # --- nouveaux metrics top-k (si présents dans cv_fold_metrics.json) ---
                            "Precision@top_10pct": float(fold_entry.get("Precision@top_10pct", np.nan)),
                            "Recall@top_10pct": float(fold_entry.get("Recall@top_10pct", np.nan)),
                            "SOZ_enrichment@top_10pct": float(fold_entry.get("SOZ_enrichment@top_10pct", np.nan)),
                            "Precision@top_20pct": float(fold_entry.get("Precision@top_20pct", np.nan)),
                            "Recall@top_20pct": float(fold_entry.get("Recall@top_20pct", np.nan)),
                            "SOZ_enrichment@top_20pct": float(fold_entry.get("SOZ_enrichment@top_20pct", np.nan)),
                            "AUC_group_mean": float(fold_entry.get("AUC_group_mean", np.nan)),
                        }

    if not best_per_patient:
        raise RuntimeError(
            f"Aucun patient trouvé pour le critère '{criterion}'. "
            f"Vérifie les fichiers cv_fold_metrics.json (et la contrainte SOZ si activée)."
        )

    df = pd.DataFrame(list(best_per_patient.values()))
    df = df.sort_values("patient").reset_index(drop=True)
    return df


# ----------------------------------------------------------
# 2) Boxplot confidence (figure 1)
# ----------------------------------------------------------
def make_boxplot_topk_metrics(df: pd.DataFrame, out_dir: Path, color_map, title: str):
    metrics = [
        "Precision@top_10pct",
        "Recall@top_10pct",
        #"SOZ_enrichment@top_10pct",
        "Precision@top_20pct",
        "Recall@top_20pct",
        #"SOZ_enrichment@top_20pct",
        "AUC_group_mean",
    ]

    pretty_labels = [
        "Precision\n(top 10%)",
        "Recall\n(top 10%)",
        #"SOZ enrichment\n(top 10%)",
        "Precision\n(top 20%)",
        "Recall\n(top 20%)",
        #"SOZ enrichment\n(top 20%)",
        "AUC\n(group mean)",
    ]

    # ne garder que les colonnes présentes (au cas où)
    present = [m for m in metrics if m in df.columns]
    if not present:
        print("[WARN] Aucun des metrics top-k n'est présent dans df_best, figure ignorée.")
        return

    data = [df[m].astype(float).values for m in present]
    labels = [pretty_labels[metrics.index(m)] for m in present]
    patients = df["patient"].tolist()

    plt.figure(figsize=(max(10, len(present) * 1.2), 6))
    plt.boxplot(data, labels=labels, showmeans=False, showfliers=False)

    for i, m in enumerate(present):
        x_center = i + 1
        values = df[m].astype(float).values
        for val, pat in zip(values, patients):
            if np.isnan(val):
                continue
            jitter = np.random.uniform(-0.12, 0.12)
            plt.scatter(
                x_center + jitter,
                val,
                color=color_map.get(pat, "grey"),
                s=55,
                alpha=0.75,
                linewidth=0.4,
            )

    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.ylim(0, 1)


    fig_path = out_dir / "boxplot_topk_metrics.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure top-k metrics sauvegardée -> {fig_path}")

def make_boxplot(df: pd.DataFrame, out_dir: Path, color_map, 
                 title: str = "Confidence Metrics Distribution"):
    """
    Boxplot + points individuels des patients pour:
      - mean_score_pos
      - mean_score_neg
      - confidence_gap
    """

    metrics = ["mean_score_pos", "mean_score_neg", "confidence_gap"]

    # Labels jolis sans underscores
    pretty_labels = ["Positive Score\n(SOZ)", "Negative Score\n(non-SOZ)", "Confidence Gap\n(SOZ vs non-SOZ)"]

    data = [df[m].values for m in metrics]
    patients = df["patient"].tolist()

    plt.figure(figsize=(9, 6))

    # Boxplots améliorés
    """
    plt.boxplot(
        data,
        labels=pretty_labels,
        showmeans=False,
        patch_artist=True,
        boxprops=dict(facecolor=None, linewidth=1.5),
        medianprops=dict(color="orange", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )
    """
    plt.boxplot(data, labels=pretty_labels, showmeans=False)

    # Points individuels
    for i, metric in enumerate(metrics):
        x_center = i + 1
        values = df[metric].values

        for val, pat in zip(values, patients):
            jitter = np.random.uniform(-0.1, 0.1)
            plt.scatter(
                x_center + jitter,
                val,
                color=color_map[pat],
                s=60,
                alpha=0.75,
                #edgecolors="black",
                linewidth=0.4,
            )

    #plt.ylabel("Metric value")
    plt.title(title)
    plt.ylim(0, 1)

    plt.grid(axis="y", alpha=0.3)

    fig_path = out_dir / "boxplot_confidence_gap.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OK] Figure sauvegardée -> {fig_path}")


# ----------------------------------------------------------
# 3) Rang du PREMIER SOZ par groupe, pour les meilleurs couples
# ----------------------------------------------------------
def compute_first_soz_rank_and_counts(df_best: pd.DataFrame, root: Path) -> pd.DataFrame:
    """
    Pour chaque patient dans df_best (best couple results_dir / experiment),
    lit cv_val_predictions_ranked.csv et calcule :

      - first_soz_mean_rank
      - first_soz_median_rank
      - first_soz_q25_rank
      - first_soz_q75_rank
      - first_soz_iqr_rank
      - num_groups_with_soz : nb de groupes (seiz/seq) avec au moins un SOZ
      - num_seizures        : nb de seizures différentes
      - max_electrodes      : max du nb d'électrodes sur un groupe
    """
    records = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir = row["results_dir"]
        exp_dir = row["experiment"]

        csv_path = (
            root
            / res_dir
            / big_exp_name
            / exp_dir
            / "cv_val_predictions_ranked.csv"
        )
        if not csv_path.is_file():
            print(f"[WARN] {csv_path} introuvable, on saute {pat}")
            continue

        df = pd.read_csv(csv_path)
        df_pat = df[df["patient"].astype(str) == str(pat)].copy()
        if df_pat.empty:
            print(f"[WARN] aucune ligne pour {pat} dans {csv_path}")
            continue

        # nb de seizures & max nb d'électrodes
        num_seizures = df_pat["seizure_id"].nunique()
        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"])
        max_electrodes = grp["node_index"].nunique().max()

        # rang du premier SOZ dans chaque groupe
        first_ranks = []
        for _, g in grp:
            g_soz = g[g["is_SOZ"] == 1]
            if g_soz.empty:
                continue
            first_ranks.append(int(g_soz["rank"].min()))

        if first_ranks:
            arr = np.array(first_ranks, dtype=float)
            mean_rank = float(np.mean(arr))
            median_rank = float(np.median(arr))
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = float(q75 - q25)
        else:
            mean_rank = np.nan
            median_rank = np.nan
            q25 = np.nan
            q75 = np.nan
            iqr = np.nan

        records.append(
            {
                "patient": pat,
                "first_soz_mean_rank": mean_rank,
                "first_soz_median_rank": median_rank,
                "first_soz_q25_rank": q25,
                "first_soz_q75_rank": q75,
                "first_soz_iqr_rank": iqr,
                "num_groups_with_soz": int(len(first_ranks)),
                "num_seizures": int(num_seizures),
                "max_electrodes": int(max_electrodes),
            }
        )

    df_first = pd.DataFrame(records)
    df_first = df_first.sort_values("patient").reset_index(drop=True)
    return df_first

def collect_first_soz_rank_for_best_models(root: Path, df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque patient et son meilleur couple (results_dir, experiment),
    lit cv_val_predictions_ranked.csv.

    Pour ce patient:
      - pour chaque groupe (patient, seizure_id, seq_idx),
        on sélectionne les électrodes is_SOZ == 1
      - on prend min(rank) = rang du premier SOZ dans ce groupe

    Retourne un DataFrame avec colonnes:
      - patient
      - first_soz_rank
    (une ligne par groupe où il existe ≥1 SOZ)
    """
    rows = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir = row["results_dir"]
        exp_dir = row["experiment"]

        pred_path = (
            root
            / res_dir
            / big_exp_name
            / exp_dir
            / "cv_val_predictions_ranked.csv"
        )

        if not pred_path.is_file():
            print(f"[WARN] Fichier de prédictions manquant pour {pat} : {pred_path}")
            continue

        try:
            df_pred = pd.read_csv(pred_path)
        except Exception as e:
            print(f"[WARN] Impossible de lire {pred_path}: {e}")
            continue

        if "patient" not in df_pred.columns or "is_SOZ" not in df_pred.columns:
            print(f"[WARN] Colonnes patient/is_SOZ manquantes dans {pred_path}")
            continue

        # On ne garde que ce patient
        sub_all = df_pred[df_pred["patient"].astype(str) == str(pat)].copy()
        if sub_all.empty:
            print(f"[INFO] Aucune prédiction pour {pat} dans {pred_path}")
            continue

        # Si 'rank' n'existe pas, on le reconstruit
        if "rank" not in sub_all.columns:
            if not all(col in sub_all.columns for col in group_keys + ["y_score"]):
                print(f"[WARN] Impossible de reconstruire rank pour {pat} dans {pred_path}")
                continue
            sub_all = sub_all.sort_values(group_keys + ["y_score"],
                                          ascending=[True, True, True, False])
            sub_all["rank"] = sub_all.groupby(group_keys)["y_score"].rank(
                method="first", ascending=False
            ).astype(int)

        # On ne garde que les SOZ
        sub_soz = sub_all[sub_all["is_SOZ"] == 1].copy()
        if sub_soz.empty:
            print(f"[INFO] Aucun SOZ trouvé dans les prédictions pour {pat} (combo {res_dir}/{exp_dir}).")
            continue

        # Rang du premier SOZ par groupe
        grouped = sub_soz.groupby(group_keys)["rank"].min().reset_index(name="first_soz_rank")

        for _, g in grouped.iterrows():
            rows.append({
                "patient": pat,
                "first_soz_rank": int(g["first_soz_rank"]),
            })

    if not rows:
        raise RuntimeError("Aucun rang de premier SOZ collecté. Vérifie les fichiers cv_val_predictions_ranked.csv.")

    first_ranks = pd.DataFrame(rows)
    return first_ranks


# ----------------------------------------------------------
# 4) Barplot : rang du premier SOZ par patient (figure 2)
# ----------------------------------------------------------

def plot_first_soz_rank_barplot(
    df_first: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str = "First SOZ rank per patient (median + IQR)",
):
    """
    Barplot par patient :
      - barre pâle = max_electrodes
      - barre opaque = median rank du premier SOZ
      - IQR = erreur verticale
      - annotation S:xx = nb de seizures
    """

    patients = df_first["patient"].tolist()
    median_ranks = df_first["first_soz_median_rank"].values
    q25 = df_first["first_soz_q25_rank"].values
    q75 = df_first["first_soz_q75_rank"].values
    num_seizures = df_first["num_seizures"].values
    max_electrodes = df_first["max_electrodes"].values

    x = np.arange(len(patients))

    # Taille figure
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    # -------------------------
    # 1) BARRES DE FOND (max_electrodes)
    # -------------------------
    for i, (pat, n_elec) in enumerate(zip(patients, max_electrodes)):
        bg_color = (*color_map.get(pat, (0.8, 0.8, 0.8)), 0.25)  # transparent
        plt.bar(
            i,
            n_elec,
            color=bg_color,
            edgecolor="none",
        )

    # -------------------------
    # 2) BARRES PRINCIPALES (médiane du rang)
    # -------------------------
    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue

        c = color_map.get(pat, "grey")
        plt.bar(
            i,
            med,
            color=c,
            edgecolor="black",
            linewidth=1.0,
        )

        # annotation S:xx
        #offset = max(0.5, 0.05 * med)
        offset = 165
        label = f"S:{int(num_seizures[i])}"
        plt.text(
            i,
            offset,
            label,
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=10,
        )

    # -------------------------
    # 3) BARRES D’ERREUR (IQR)
    # -------------------------
    lower_err = median_ranks - q25
    upper_err = q75 - median_ranks
    valid = ~np.isnan(median_ranks)

    plt.errorbar(
        x[valid],
        median_ranks[valid],
        yerr=np.vstack([lower_err[valid], upper_err[valid]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        capthick=1.0,
    )

    # -------------------------
    # 4) AXES / STYLE
    # -------------------------
    # -------------------------
    # 4) AXES / STYLE
    # -------------------------

    # fabriquer les xticks sur 2 lignes
    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("First SOZ Channel Detection Rank")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    # limite haute
    ymax = max(
        np.nanmax(max_electrodes),
        np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks)
    )
    plt.ylim(0, ymax * 1.05)

    # yticks réguliers SANS 170
    ymin, ymax = plt.ylim()
    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 170]
    plt.yticks(ticks)
    # -------------------------
    # 5) SAVE
    # -------------------------
    fig_path = out_dir / "barplot_first_soz_rank_median_IQR_background_electrodes.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OK] Figure barplot avec background électrodes -> {fig_path}")











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
def clean_patient_name(raw):
    """
    Transforme un nom du type 'CHUM::Patient_01'
    en 'Patient 01 (CHUM)'
    """
    if "::" not in raw:
        return raw  # fallback si format inconnu

    center, ident = raw.split("::", 1)
    ident = ident.replace("_", " ")   # Patient_01 → Patient 01
    ident = ident.strip()

    return f"{ident} ({center})"


def replay_prob_figure(
    seizure_csv_path,
    nodes_csv_path,
    npz_path,
    pat_name,
    out_path=None,
    show=True,
    patient_color_map=None,  # dict: patient_name -> (r, g, b)
):
    """
    Recrée une figure type `plot_probs` à partir des fichiers sauvés par evaluate_and_plot_on_test,
    avec un style plus propre.

    - Courbes des canaux SOZ / non-SOZ
    - p_graph lissé uniquement
    - True seizure onset en rouge avec texte
    """

    seizure_csv_path = Path(seizure_csv_path)
    nodes_csv_path = Path(nodes_csv_path)
    npz_path = Path(npz_path)

    # ---------- 1) Charger le NPZ ----------
    data = np.load(npz_path, allow_pickle=True)

    # séries globales
    p_graph_smooth = np.asarray(data["p_graph"], dtype=float)  # (T,)

    # proba par noeud
    per_node_probs = data.get("per_node_probs", None)                  # (N, T)
    if per_node_probs is None:
        raise ValueError("per_node_probs is missing from npz – cannot replay plot.")

    per_node_probs = np.asarray(per_node_probs, dtype=float)
    N, T_nodes = per_node_probs.shape

    # meta / onset
    patient = str(data.get("patient", ""))
    seizure_id = str(data.get("seizure_id", ""))
    seq_index = int(data.get("seq_index", -1))

    t_true = int(data.get("t_true", -1))
    if t_true < 0:
        t_true = None

    # masque SOZ & noms électrodes
    soz_mask_np = data.get("is_SOZ", None)
    electrode_names = data.get("electrode_names", None)

    # ---------- 2) Compléter avec nodes_csv si besoin ----------
    try:
        df_nodes = pd.read_csv(nodes_csv_path)
    except Exception:
        df_nodes = None

    if soz_mask_np is None and df_nodes is not None:
        soz_mask_np = df_nodes["is_SOZ"].to_numpy().astype(bool)
    if electrode_names is None and df_nodes is not None:
        electrode_names = df_nodes["electrode_name"].to_numpy()

    if soz_mask_np is None:
        soz_mask_np = np.zeros((N,), dtype=bool)
    else:
        soz_mask_np = np.asarray(soz_mask_np).astype(bool)
        if soz_mask_np.shape[0] != N:
            M = min(N, soz_mask_np.shape[0])
            tmp = np.zeros((N,), dtype=bool)
            tmp[:M] = soz_mask_np[:M]
            soz_mask_np = tmp

    # ---------- 3) Harmoniser la longueur en temps ----------
    T_smooth = len(p_graph_smooth)
    T = min(T_smooth, T_nodes)
    t_axis = np.arange(T)

    p_graph_smooth = p_graph_smooth[:T]
    per_node_probs = per_node_probs[:, :T]

    # ---------- 4) Couleur SOZ par patient ----------

    if patient_color_map is not None and pat_name in patient_color_map:
        soz_color = patient_color_map[pat_name]
    else:
        soz_color = "tab:green"  # fallback

    non_soz_color = "grey"

    # ---------- 5) Figure ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    # (a) courbes par noeud
    for i in range(N):
        y = per_node_probs[i]
        if soz_mask_np[i]:
            # SOZ : couleur patient
            ax.plot(t_axis, y, color=soz_color, linewidth=1.5, alpha=0.9)
        else:
            # non-SOZ : gris
            ax.plot(t_axis, y, color=non_soz_color, linewidth=1.0, alpha=0.7)

    # (b) p_graph lissé : courbe magenta (ou autre si tu préfères)
    agg_line, = ax.plot(
        t_axis,
        p_graph_smooth,
        color="black",
        #linestyle="--",
        linewidth=2.5,
        label="Ictal Activity Probability: Aggregated over all Channels",
    )

    # (c) true seizure onset : ligne rouge + texte vertical
    if t_true is not None:
        ax.axvline(t_true, color="red", linestyle="--", linewidth=2.0)
        # texte au milieu de l'axe Y
        y_min, y_max = ax.get_ylim()
        y_mid = (y_min + y_max) / 2.0
        ax.text(
            t_true,
            y_mid,
            "Real Seizure Onset Time",
            color="red",
            rotation=90,
            ha="right",
            va="center",
            fontsize=10
        )

    # ---------- 6) Légende ----------
    legend_handles = [
        Line2D(
            [0], [0],
            color="red",
            linewidth=1.5,
            label="Ictal Activity Probability: SOZ Channels",
        ),
        Line2D(
            [0], [0],
            color=non_soz_color,
            linewidth=1.5,
            label="Ictal Activity Probability: non-SOZ Channels",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.5,
            label="Aggregated Ictal Activity Probability: Graph Level",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    # ---------- 7) Axes / titre ----------
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability of Ictal Activity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(MultipleLocator(1))

    patient_display = clean_patient_name(patient)

    title = f"Aggregated and Channel-Wise Ictal Activity Probabilities for Detecting Seizure Onset Time\n(Exemple Output of Model 1 For {patient_display}, Seizure {seizure_id})"
    ax.set_title(title)


    fig.tight_layout()

    # ---------- 8) Save / show ----------
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



def collect_auc_from_npz_for_best_models(root: Path, df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque patient et son meilleur 'results_dir', parcourt tous les .npz
    dans results/<patient>/series et calcule un AUC temporel :

        y_true  = y_graph_true (0/1 par pas de temps)
        y_score = p_graph      (proba graphe par pas de temps)

    Retourne un DataFrame avec colonnes :
        patient, results_dir, seizure_id, seq_index, auc_time, t_true, t_pred
    """

    rows = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir_name = row["results_dir"]

        # même logique que evaluate_and_plot_on_test pour le chemin
        patient_dir_name = pat.replace("::", "__")
        series_dir = (
            root
            / res_dir_name
            / "results"
            / patient_dir_name
            / "series"
        )

        if not series_dir.is_dir():
            print(f"[WARN] Pas de series_dir pour {pat} : {series_dir}")
            continue

        for npz_path in sorted(series_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Impossible de lire {npz_path}: {e}")
                continue

            if "y_graph_true" not in data or "p_graph" not in data:
                print(f"[WARN] y_graph_true ou p_graph manquant dans {npz_path}, on saute.")
                continue

            y_true = np.asarray(data["y_graph_true"], dtype=float)
            y_score = np.asarray(data["p_graph"], dtype=float)

            # on coupe à la longueur commune
            T = min(len(y_true), len(y_score))
            if T == 0:
                continue
            y_true = y_true[:T]
            y_score = y_score[:T]

            # AUC uniquement si on a au moins 1 zero et 1 one
            if len(np.unique(y_true)) < 2:
                auc_time = np.nan
            else:
                try:
                    auc_time = float(roc_auc_score(y_true, y_score))
                except Exception:
                    auc_time = np.nan

            seizure_id = str(data.get("seizure_id", "?"))
            seq_index = int(data.get("seq_index", -1))

            t_true = int(data.get("t_true", -1))
            t_pred = int(data.get("t_pred", -1))
            if t_true < 0:
                t_true = None
            if t_pred < 0:
                t_pred = None

            rows.append({
                "patient": pat,
                "results_dir": res_dir_name,
                "seizure_id": seizure_id,
                "seq_index": seq_index,
                "auc_time": auc_time,
                "t_true": t_true,
                "t_pred": t_pred,
                "npz_path": str(npz_path),
            })

    if not rows:
        raise RuntimeError("Aucun AUC calculé depuis les npz. Vérifie les chemins / contenus.")
    return pd.DataFrame(rows)


import numpy as np

def onset_curve_bce(p_smooth, t_true, eps=1e-7):
    """
    Binary cross-entropy entre p_graph_smooth et une courbe step 0→1 à t_true.

    p_smooth : array-like de probas (longueur T)
    t_true   : index entier de l'onset vrai (None ou <0 -> retourne NaN)
    eps      : petit terme pour éviter log(0)
    """
    p_smooth = np.asarray(p_smooth, dtype=float)
    T = len(p_smooth)

    if t_true is None or t_true < 0 or T == 0:
        return np.nan

    # courbe vraie : 0 avant t_true, 1 après
    y_true = np.zeros(T, dtype=float)
    y_true[t_true:] = 1.0

    # clamp des probas pour éviter log(0)
    p = np.clip(p_smooth, eps, 1.0 - eps)

    bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(bce.mean())


def collect_bce_from_npz_for_best_models(root: Path, df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque patient et son meilleur 'results_dir', parcourt tous les .npz
    dans results/<patient>/series et calcule une BCE temporelle :

        p(t) = p_graph_smooth(t)
        y(t) = 0 avant t_true, 1 après t_true

    Retourne un DataFrame avec colonnes :
        patient, results_dir, seizure_id, seq_index, bce_onset, t_true, npz_path
    """

    rows = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir_name = row["results_dir"]

        patient_dir_name = pat.replace("::", "__")
        series_dir = (
            root
            / res_dir_name
            / "results"
            / patient_dir_name
            / "series"
        )

        if not series_dir.is_dir():
            print(f"[WARN] Pas de series_dir pour {pat} : {series_dir}")
            continue

        for npz_path in sorted(series_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Impossible de lire {npz_path}: {e}")
                continue

            if "p_graph_smooth" not in data:
                print(f"[WARN] p_graph_smooth manquant dans {npz_path}, on saute.")
                continue

            p_smooth = np.asarray(data["p_graph_smooth"], dtype=float)
            t_true = int(data.get("t_true", -1))
            if t_true < 0:
                t_true = None

            bce_onset = onset_curve_bce(p_smooth, t_true)

            seizure_id = str(data.get("seizure_id", "?"))
            seq_index = int(data.get("seq_index", -1))

            rows.append({
                "patient": pat,
                "results_dir": res_dir_name,
                "seizure_id": seizure_id,
                "seq_index": seq_index,
                "bce_onset": bce_onset,
                "t_true": t_true,
                "npz_path": str(npz_path),
            })

    if not rows:
        raise RuntimeError("Aucune BCE calculée depuis les npz. Vérifie les chemins / contenus.")
    return pd.DataFrame(rows)

def plot_bce_boxplot_per_patient(df_bce: pd.DataFrame, out_dir: Path, color_map: dict):
    """
    Boxplot des BCE temporelles par patient (une valeur par séquence).
    Plus la BCE est faible, meilleure est la correspondance entre
    la proba agrégée et une courbe step idéale (0 avant t_true, 1 après).
    """

    # --- ordre stable ---
    pats = sorted(df_bce["patient"].unique())
    data = [df_bce[df_bce["patient"] == p]["bce_onset"].dropna().values for p in pats]

    # --- préparation des labels en deux lignes ---
    xtick_labels = []
    for p in pats:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    # --- figure ---
    plt.figure(figsize=(max(12, len(pats) * 0.45), 3))

    # --- boxplot de base ---
    bp = plt.boxplot(
        data,
        showmeans=False,
        patch_artist=True,    # nécessaire pour changer couleur des boxes
        showfliers=False
    )

    # --- coloriage + style ---
    for i, p in enumerate(pats):
        c = color_map.get(p, "grey")

        plt.setp(bp["boxes"][i], facecolor=(0, 0, 0, 0), edgecolor=c, linewidth=1)
        plt.setp(bp["whiskers"][2*i:2*i+2], color=c, linewidth=1)
        plt.setp(bp["caps"][2*i:2*i+2], color=c, linewidth=1)
        plt.setp(bp["medians"][i], color="orange", linewidth=1.5)

    # --- afficher tous les points individuels ---
    for i, p in enumerate(pats):
        values = df_bce[df_bce["patient"] == p]["bce_onset"].dropna().values
        c = color_map.get(p, "grey")
        for v in values:
            jitter = np.random.uniform(-0.12, 0.12)
            plt.scatter(
                i + 1 + jitter,
                v,
                color=c,
                linewidth=0.4,
                s=15,
                alpha=0.75,
            )

    # --- ticks et axes ---
    plt.xticks(range(1, len(pats) + 1), xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("Time-wise BCE vs\nIdeal Seizure Onset Step Function")
    plt.title("Per-Patient Onset Curve Loss (Aggregated Ictal Activity Probability)\n- Best Model Applied to Held-Out Patient (Nested CV) -")
    plt.grid(axis="y", alpha=0.3)

    # --- sauvegarde ---
    fig_path = out_dir / "boxplot_bce_onset_per_patient.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OK] Figure BCE sauvegardée -> {fig_path}")

def compute_all_soz_rank_and_counts(df_best: pd.DataFrame, root: Path) -> pd.DataFrame:
    """
    Pour chaque patient dans df_best (best couple results_dir / experiment),
    lit cv_val_predictions_ranked.csv et calcule les stats de rang pour
    TOUTES les électrodes SOZ :

      - all_soz_mean_rank
      - all_soz_median_rank
      - all_soz_q25_rank
      - all_soz_q75_rank
      - all_soz_iqr_rank
      - num_groups_with_soz : nb de groupes (seiz/seq) avec au moins un SOZ
      - num_seizures        : nb de seizures différentes
      - max_electrodes      : max du nb d'électrodes sur un groupe
      - max_soz_electrodes  : max du nb d'électrodes SOZ sur un groupe
                               (si ça varie selon les crises, on prend le max)
    """
    records = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir = row["results_dir"]
        exp_dir = row["experiment"]

        csv_path = (
            root
            / res_dir
            / big_exp_name
            / exp_dir
            / "cv_val_predictions_ranked.csv"
        )
        if not csv_path.is_file():
            print(f"[WARN] {csv_path} introuvable, on saute {pat}")
            continue

        df = pd.read_csv(csv_path)
        df_pat = df[df["patient"].astype(str) == str(pat)].copy()
        if df_pat.empty:
            print(f"[WARN] aucune ligne pour {pat} dans {csv_path}")
            continue

        # nb de seizures & max nb d'électrodes (toutes)
        num_seizures = df_pat["seizure_id"].nunique()
        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"])
        max_electrodes = grp["node_index"].nunique().max()

        # on récupère TOUS les rangs des électrodes SOZ + nb max d'électrodes SOZ
        all_ranks = []
        max_soz_electrodes = 0
        num_groups_with_soz = 0

        for _, g in grp:
            g_soz = g[g["is_SOZ"] == 1]
            if g_soz.empty:
                continue

            num_groups_with_soz += 1

            # tous les rangs SOZ de ce groupe
            all_ranks.extend(list(g_soz["rank"].astype(int)))

            # nb d'électrodes SOZ distinctes dans ce groupe
            n_soz = g_soz["node_index"].nunique()
            if n_soz > max_soz_electrodes:
                max_soz_electrodes = n_soz

        if all_ranks:
            arr = np.array(all_ranks, dtype=float)
            mean_rank = float(np.mean(arr))
            median_rank = float(np.median(arr))
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = float(q75 - q25)
        else:
            mean_rank = np.nan
            median_rank = np.nan
            q25 = np.nan
            q75 = np.nan
            iqr = np.nan

        records.append(
            {
                "patient": pat,
                "all_soz_mean_rank": mean_rank,
                "all_soz_median_rank": median_rank,
                "all_soz_q25_rank": q25,
                "all_soz_q75_rank": q75,
                "all_soz_iqr_rank": iqr,
                "num_groups_with_soz": int(num_groups_with_soz),
                "num_seizures": int(num_seizures),
                "max_electrodes": int(max_electrodes),
                "max_soz_electrodes": int(max_soz_electrodes),
            }
        )

    df_all = pd.DataFrame(records)
    df_all = df_all.sort_values("patient").reset_index(drop=True)
    return df_all


def plot_all_soz_rank_barplot(
    df_all: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str = "All SOZ rank per patient (median + IQR)",
):
    """
    Barplot par patient pour TOUTES les électrodes SOZ :
      - barre pâle large  = max_electrodes (toutes les électrodes)
      - barre plus étroite = nb max d'électrodes SOZ pour ce patient
      - barre opaque       = médiane du rang (toutes SOZ)
      - IQR                = barre d'erreur
      - annotation S:xx    = nb de seizures
    """

    patients = df_all["patient"].tolist()
    median_ranks = df_all["all_soz_median_rank"].values
    q25 = df_all["all_soz_q25_rank"].values
    q75 = df_all["all_soz_q75_rank"].values
    num_seizures = df_all["num_seizures"].values
    max_electrodes = df_all["max_electrodes"].values

    # nb max d'électrodes SOZ par patient
    if "max_soz_electrodes" in df_all.columns:
        max_soz_electrodes = df_all["max_soz_electrodes"].values
    else:
        max_soz_electrodes = np.full_like(max_electrodes, np.nan, dtype=float)

    x = np.arange(len(patients))

    # Taille figure
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    # -------------------------
    # 1) BARRES DE FOND
    #    - max_electrodes : barre large, très pâle
    #    - max_soz_electrodes : barre plus étroite, plus visible
    # -------------------------
    for i, (pat, n_elec) in enumerate(zip(patients, max_electrodes)):
        base_color = color_map.get(pat, (0.8, 0.8, 0.8))

        # barre de fond = toutes les électrodes
        bg_color = (*base_color, 0.20)
        plt.bar(
            i,
            n_elec,
            width=0.8,
            color=bg_color,
            edgecolor="none",
        )

        # barre verticale pour nb max d'électrodes SOZ
        n_soz = max_soz_electrodes[i]
        if not np.isnan(n_soz):
            soz_color = (*base_color, 0.65)
            plt.bar(
                i,
                n_soz,
                width=0.8,
                color=soz_color,
                edgecolor="none",
            )

    # -------------------------
    # 2) BARRES PRINCIPALES (médiane des rangs SOZ)
    # -------------------------
    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue

        c = color_map.get(pat, "grey")
        plt.bar(
            i,
            med,
            width=0.25,
            color=c,
            edgecolor="black",
            linewidth=1.0,
        )

        # annotation S:xx (nb de seizures)
        offset = 165  # tu peux adapter bien sûr
        label = f"S:{int(num_seizures[i])}"
        plt.text(
            i,
            offset,
            label,
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=10,
        )

    # -------------------------
    # 3) BARRES D’ERREUR (IQR)
    # -------------------------
    lower_err = median_ranks - q25
    upper_err = q75 - median_ranks
    valid = ~np.isnan(median_ranks)

    plt.errorbar(
        x[valid],
        median_ranks[valid],
        yerr=np.vstack([lower_err[valid], upper_err[valid]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        capthick=1.0,
    )

    # -------------------------
    # 4) AXES / STYLE
    # -------------------------

    # xticks sur 2 lignes, comme dans tes autres figures
    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("All SOZ Channel Detection Rank")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    # limite haute (doit couvrir max_electrodes)
    ymax = max(
        np.nanmax(max_electrodes),
        np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks)
    )
    plt.ylim(0, ymax * 1.05)

    # yticks réguliers SANS 170 (comme dans ton code)
    ymin, ymax = plt.ylim()
    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 170]
    plt.yticks(ticks)

    # -------------------------
    # 5) SAVE
    # -------------------------
    fig_path = out_dir / "barplot_all_soz_rank_median_IQR_background_electrodes_with_soz_count.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OK] Figure barplot (ALL SOZ, avec nb d'électrodes SOZ) -> {fig_path}")



def keep_middle_seq_per_seizure(df_auc: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (patient, seizure_id), garde uniquement la séquence
    dont seq_index est au milieu (sur les 3 séquences).
    Si le nombre de séquences n'est pas 3, on prend quand même
    celle au rang médian après tri.
    """
    def _pick_middle(g):
        g = g.sort_values("seq_index")
        mid = len(g) // 2  # pour 3 -> index 1 (la 2ème)
        return g.iloc[[mid]]

    df_mid = (
        df_auc
        .groupby(["patient", "seizure_id"], group_keys=False)
        .apply(_pick_middle)
        .reset_index(drop=True)
    )
    return df_mid

def plot_auc_boxplot_per_patient(df_auc: pd.DataFrame, out_dir: Path, color_map: dict):
    """
    Boxplot des AUC temporels par patient (une valeur par séquence).
    """

    pats = sorted(df_auc["patient"].unique())
    data = [df_auc[df_auc["patient"] == p]["auc_time"].dropna().values for p in pats]

    plt.figure(figsize=(max(10, len(pats) * 0.4), 6))
    bp = plt.boxplot(data, labels=pats, showmeans=False)

    # colorier les box par patient
    for i, p in enumerate(pats):
        c = color_map.get(p, "grey")
        plt.setp(bp["boxes"][i], color=c)
        plt.setp(bp["whiskers"][2*i:2*i+2], color=c)
        plt.setp(bp["caps"][2*i:2*i+2], color=c)

    plt.ylabel("Time-wise ROC-AUC (graph probability vs. ictal label)")
    plt.title("Per-Patient Seizure Detection AUC (Best Model, All Sequences)")
    plt.grid(axis="y", alpha=0.3)

    fig_path = out_dir / "boxplot_auc_time_per_patient.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OK] Figure AUC sauvegardée -> {fig_path}")


from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def plot_soz_rank_boxplot_points_colored_by_seizure_from_df_all(
    root: Path,
    df_best: pd.DataFrame,
    out_dir: Path,
    title: str = "SOZ Electrode Ranks per Patient (points colored by seizure)",
    color_map: dict | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _get_pat_color(pat: str, alpha: float):
        print("COLOR MAP:", color_map)
        if color_map is None:
            return (0.8, 0.8, 0.8, alpha)

        candidates = [
            pat,
            pat.strip(),
            pat.replace("::", "__"),
            pat.replace("__", "::"),
            pat.strip().replace("::", "__"),
            pat.strip().replace("__", "::"),
        ]
        for k in candidates:
            if k in color_map:
                return to_rgba(color_map[k], alpha=alpha)
        return (0.8, 0.8, 0.8, alpha)

    rows = []
    max_elec_by_patient: dict[str, int] = {}
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = (
            Path(root)
            / res_dir
            / big_exp_name
            / exp_dir
            / "cv_val_predictions_ranked.csv"
        )

        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat}, skipped.")
            continue

        df = pd.read_csv(pred_path)

        needed = {"patient", "seizure_id", "is_SOZ", "rank"}
        if not needed.issubset(df.columns):
            print(f"[WARN] Missing columns in {pred_path}: {needed - set(df.columns)}")
            continue

        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            continue

        # même définition que tes autres fonctions
        if all(k in df_pat.columns for k in group_keys) and "node_index" in df_pat.columns:
            grp = df_pat.groupby(group_keys)
            max_electrodes = int(grp["node_index"].nunique().max())
        else:
            # fallback minimal si node_index/seq_idx absents
            if "node_index" in df_pat.columns and "seizure_id" in df_pat.columns:
                max_electrodes = int(df_pat.groupby("seizure_id")["node_index"].nunique().max())
            elif "seizure_id" in df_pat.columns:
                max_electrodes = int(df_pat.groupby("seizure_id").size().max())
            else:
                max_electrodes = int(len(df_pat))

        max_elec_by_patient[pat] = max(max_elec_by_patient.get(pat, 0), max_electrodes)

        df_soz = df_pat[df_pat["is_SOZ"].astype(int) == 1].copy()
        if df_soz.empty:
            continue

        df_soz["patient"] = pat
        df_soz["seizure_id"] = df_soz["seizure_id"].astype(str)
        df_soz["rank"] = pd.to_numeric(df_soz["rank"], errors="coerce")
        rows.append(df_soz[["patient", "seizure_id", "rank"]])

    if not rows:
        print("[WARN] No SOZ ranks found → plot skipped.")
        return

    df_plot = pd.concat(rows, ignore_index=True).dropna(subset=["rank"])
    df_plot["rank"] = df_plot["rank"].astype(float)

    seizure_ids = sorted(df_plot["seizure_id"].unique())
    palette = sns.color_palette("tab20", min(20, len(seizure_ids)))
    seizure_color = {s: palette[i % len(palette)] for i, s in enumerate(seizure_ids)}

    patients = sorted(df_plot["patient"].unique())
    x = np.arange(len(patients))

    plt.figure(figsize=(max(12, 0.45 * len(patients)), 6))

    # BARRES DE FOND (couleur patient EXACTE via mapping robuste)
    for i, pat in enumerate(patients):
        n_elec = int(max_elec_by_patient.get(pat, 0))
        bg_color = _get_pat_color(pat, alpha=0.25)
        plt.bar(i, n_elec, color=bg_color, edgecolor="none", zorder=0)

    sns.boxplot(
        data=df_plot,
        x="patient",
        y="rank",
        order=patients,
        showfliers=False,
        color="lightgrey",
        zorder=1,
    )

    sns.stripplot(
        data=df_plot,
        x="patient",
        y="rank",
        hue="seizure_id",
        order=patients,
        palette=seizure_color,
        dodge=True,
        jitter=False,
        size=8,
        alpha=0.8,
        linewidth=0,
        zorder=2,
    )

    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.ylabel("SOZ Electrode Rank")
    plt.xlabel("")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)

    plt.legend(
        title="seizure_id",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    out_path = out_dir / "boxplot_soz_rank_per_patient_points_colored_by_seizure.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] SOZ rank boxplot saved -> {out_path}")


from collections import Counter

def collect_selected_features_for_best_models(
    root: Path,
    df_best: pd.DataFrame,
    features_filename: str = "selected_features_GLOBAL.txt",
) -> pd.DataFrame:
    """
    Pour chaque patient (best couple results_dir/experiment dans df_best),
    lit selected_features_GLOBAL.txt et retourne:

    - un DataFrame long: (patient, feature)
    - + gère les fichiers manquants en warning
    """
    rows = []

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        feat_path = (
            root
            / res_dir
            / big_exp_name
            / exp_dir
            / features_filename
        )

        if not feat_path.is_file():
            print(f"[WARN] Missing {features_filename} for {pat}: {feat_path}")
            continue

        try:
            txt = feat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Cannot read {feat_path}: {e}")
            continue

        feats = []
        for line in txt.splitlines():
            f = line.strip()
            if not f:
                continue
            feats.append(f)

        if not feats:
            print(f"[WARN] Empty feature file for {pat}: {feat_path}")
            continue

        for f in feats:
            rows.append({"patient": pat, "feature": f})

    if not rows:
        raise RuntimeError(
            f"No selected features collected from {features_filename}. "
            "Check paths / file names."
        )

    return pd.DataFrame(rows)


def plot_selected_feature_frequencies(
    df_feat_long: pd.DataFrame,
    out_dir: Path,
    title: str = "Selected feature frequency across patients (best model per patient)",
    top_n: int | None = None,
    rotate_xticks: int = 90,
):
    """
    Plot: x = feature, y = number of patients whose best model selected it.
    (On compte une feature AU MAX une fois par patient, même si répétée.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) dédoublonner (patient, feature) au cas où
    df_u = df_feat_long.drop_duplicates(subset=["patient", "feature"]).copy()

    # 2) compter
    counts = (
        df_u.groupby("feature")["patient"]
            .nunique()
            .sort_values(ascending=False)
    )

    if top_n is not None:
        counts = counts.head(int(top_n))

    df_plot = counts.reset_index()
    df_plot.columns = ["feature", "count_patients"]

    # 3) plot
    plt.figure(figsize=(max(12, 0.35 * len(df_plot)), 5))
    plt.bar(df_plot["feature"], df_plot["count_patients"])
    plt.ylabel("Number of folds selecting the feature")
    plt.xlabel("Feature")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    fig_path = out_dir / "barplot_selected_features_frequency.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Feature frequency plot saved -> {fig_path}")

    # optionnel: aussi un CSV
    csv_path = out_dir / "selected_features_frequency.csv"
    df_plot.to_csv(csv_path, index=False)
    print(f"[OK] Feature frequency table saved -> {csv_path}")



# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyse du meilleur confidence_gap par patient.")
    ap.add_argument("--criterion", default="confidence_gap_global")
    ap.add_argument("--mode", choices=["max", "min"], default="max")
    ap.add_argument("--root", required=True, help="Dossier racine contenant les results_*")
    ap.add_argument(
        "--out",
        default="best_confidence",
        help="Dossier de sortie pour le CSV + figures",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out + "/best_" + args.criterion).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"Le dossier root n'existe pas: {root}")

    print(f"[INFO] Root = {root}")
    df_best = collect_best_model_per_patient(root, criterion=args.criterion, mode=args.mode)

    # Sauvegarde CSV
    csv_path = out_dir / "best_confidence_per_patient.csv"
    df_best.to_csv(csv_path, index=False)
    print(f"[OK] Tableau sauvegardé -> {csv_path}")

    # Palette partagée entre les figures
    color_map = build_patient_color_map(df_best["patient"])

    # Figure 1 : boxplot des scores de confiance
    make_boxplot(df_best, out_dir, color_map, title="Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\nand Their Separation (Confidence Gap)\n- Best Model Applied to Held-Out Patient (Nested CV) -")

    # Calcul des rangs du premier SOZ + counts
    df_first = compute_first_soz_rank_and_counts(df_best, root)

    # Barplot avec annotations S / E
    plot_first_soz_rank_barplot(
        df_first,
        color_map,
        out_dir,
        title="Per-Patient First SOZ Channel Detection Rank (Median ± IQR)\nwith Total Implanted Electrode Count and Number of Evaluated Seizures\n- Best Model Applied to Held-Out Patient (Nested CV) -",
    )

    df_all_soz = compute_all_soz_rank_and_counts(df_best, root)
    plot_all_soz_rank_barplot(
        df_all_soz,
        color_map,
        out_dir,
        title="Per-Patient All SOZ Channel Detection Rank (Median ± IQR)\nwith Total Implanted Electrode Count and Number of Evaluated Seizures\n- Best Model Applied to Held-Out Patient (Nested CV) -",
    )

    make_boxplot_topk_metrics(
    df_best,
    out_dir,
    color_map,
    title="Per-Patient Top-k SOZ Metrics (from cv_fold_metrics.json)\n- Best Model Applied to Held-Out Patient (Nested CV) -",
)

    plot_soz_rank_boxplot_points_colored_by_seizure_from_df_all(
    root=root,
    df_best=df_best,
    out_dir=out_dir,
    color_map=color_map
)


    print("[INFO] Terminé.")

    nodes_csv = "results_grid_search_kwta_20s/results_20251121_044808/results/CHUM__Patient_01/series/seiz_11_seq_028_nodes.csv"
    seiz_csv = "results_grid_search_kwta_20s/results_20251121_044808/results/CHUM__Patient_01/series/seiz_11_seq_028.csv"
    npz_file  = "results_grid_search_kwta_20s/results_20251121_044808/results/CHUM__Patient_01/series/seiz_11_seq_028.npz"

    #replay_prob_figure(seiz_csv, nodes_csv, npz_file,
                   #pat_name = "CHUM__Patient_01",
                   #out_path="best_confidence/reconstructed_plot.png",
                   #show=False,
                   #patient_color_map=color_map,)
    
    nodes_csv = "results_grid_search_kwta_20s/results_20251121_003137/results/CHUM__Patient_22/series/seiz_1_seq_004_nodes.csv"
    seiz_csv = "results_grid_search_kwta_20s/results_20251121_003137/results/CHUM__Patient_22/series/seiz_1_seq_004.csv"
    npz_file  = "results_grid_search_kwta_20s/results_20251121_003137/results/CHUM__Patient_22/series/seiz_1_seq_004.npz"

    #replay_prob_figure(seiz_csv, nodes_csv, npz_file,
                   #out_path="best_confidence/reconstructed_plot_bad.png",
                   #pat_name = "CHUM__Patient_01",
                   #show=False,
                   #patient_color_map=color_map,)
    

    # --- BCE vs courbe step idéale à partir des .npz des meilleurs modèles ---
    df_bce_all = collect_bce_from_npz_for_best_models(root, df_best)
    print(df_bce_all)

    # ne garder que la séquence du milieu pour chaque (patient, seizure)
    df_bce = keep_middle_seq_per_seizure(df_bce_all)

    bce_csv = out_dir / "timewise_onset_bce_from_npz_middle_seq_only.csv"
    df_bce.to_csv(bce_csv, index=False)
    print(f"[OK] BCE (séquence du milieu seulement) sauvegardée -> {bce_csv}")

    plot_bce_boxplot_per_patient(df_bce, out_dir, color_map)


    # ------------------------------------------------------
    #  FEATURES: fréquence des features sélectionnées
    # ------------------------------------------------------
    df_feat_long = collect_selected_features_for_best_models(root, df_best)

    # sauvegarde “long format” (1 ligne = 1 feature sélectionnée pour 1 patient)
    feat_long_csv = out_dir / "selected_features_best_model_per_patient_long.csv"
    df_feat_long.to_csv(feat_long_csv, index=False)
    print(f"[OK] Selected features (long) saved -> {feat_long_csv}")

    # barplot fréquence globale (combien de patients sélectionnent chaque feature)
    plot_selected_feature_frequencies(
        df_feat_long,
        out_dir,
        title="Selected feature frequency across patients\n(best model per patient)",
        top_n=None,         # mets un int si tu veux limiter (ex: 40)
        rotate_xticks=90,
    )



    # ------------------------------------------------------
    #  STATS GLOBALES : moyennes sur les données des figures
    # ------------------------------------------------------

    # 1) Average BCE across all folds (pour les données du graphique = df_bce)
    avg_bce = df_bce["bce_onset"].mean()

        # 2) Average pos / neg scores et confidence gap (sur df_best = 1 ligne/patient)
    avg_pos = df_best["mean_score_pos"].mean()
    avg_neg = df_best["mean_score_neg"].mean()

    # gap patient-wise = mean_score_pos - mean_score_neg pour chaque patient,
    # puis moyenne de ces gaps
    df_best["per_patient_gap"] = df_best["mean_score_pos"] - df_best["mean_score_neg"]
    avg_gap = df_best["per_patient_gap"].mean()


    # 3) Average min rank du premier SOZ
    #    -> on réutilise collect_first_soz_rank_for_best_models qui donne
    #       une ligne par groupe (patient, seizure, seq) avec first_soz_rank = min(rank)
    df_first_all = collect_first_soz_rank_for_best_models(root, df_best)
    avg_min_rank = df_first_all["first_soz_rank"].mean()

    print("\n[STATS] Global metrics on plotted data")
    print(f"  - Average BCE (middle seq only)         : {avg_bce:.4f}")
    print(f"  - Average mean positive score (SOZ)     : {avg_pos:.4f}")
    print(f"  - Average mean negative score (non-SOZ) : {avg_neg:.4f}")
    print(f"  - Average confidence gap                : {avg_gap:.4f}")
    print(f"  - Average min first-SOZ rank            : {avg_min_rank:.4f}")

    # Sauvegarde dans un petit CSV récapitulatif
    summary_df = pd.DataFrame({
        "avg_bce_middle_seq": [avg_bce],
        "avg_mean_score_pos": [avg_pos],
        "avg_mean_score_neg": [avg_neg],
        "avg_confidence_gap": [avg_gap],
        "avg_first_soz_min_rank": [avg_min_rank],
    })
    summary_csv = out_dir / "global_summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Global summary metrics saved -> {summary_csv}")



if __name__ == "__main__":
    main()



