#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sélectionne, pour chaque patient, le meilleur couple (results_dir, exp_dir)
en MAXIMISANT un F1@topK (top 10% ou top 20%) CALCULÉ directement à partir de
cv_val_predictions_ranked.csv (dans chaque exp*).

Puis reproduit les mêmes figures :
  - boxplot mean_score_pos / mean_score_neg / confidence_gap
  - barplot du rang du premier SOZ (médiane ± IQR) + fond nb électrodes
  - barplot rang de toutes les électrodes SOZ (médiane ± IQR) + fond nb électrodes + nb SOZ
  - boxplot des metrics top-k (Precision/Recall/F1 @10% et @20%) + AUC_group_mean

Contrainte maintenue :
  - pour chaque patient, pour chaque seizure_id : au moins 2 électrodes SOZ distinctes
    (via node_index si présent, sinon lignes) dans cv_val_predictions_ranked.csv.

NOUVEAU :
  - Exporte un CSV de traçabilité (config sélectionnée + métriques utilisées pour les plots)
  - Exporte un grand CSV concaténant toutes les prédictions utilisées (uniquement best combo par patient)
  - (Optionnel) Exporte un grand CSV "collapsed by node_index" (celui réellement ranké dans compute_patient_metrics...)

Exemple d'appel :

uv run z_final_results_good_outcome.py \
  --root /home/julieletallec/test/results_grid_search_kwta_20_10_burst \
  --out final_results_good_outcome \
  --select_f1_pct 10
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


# Nom du gros dossier d'expérience (identique à ton code)
big_exp_name = "classifier_experiments_augmented_balanced_LASSO_global_0.03_postprocess_new0.1"


# ----------------------------------------------------------
# Utils : recherche des dossiers & couleurs
# ----------------------------------------------------------
def patient_labels_2lines(pat: str) -> str:
    pat = str(pat).strip()

    # enlève underscores/espaces en fin (ex: "CHUM_Patient_16_")
    pat = pat.rstrip("_").rstrip()

    # cas historiques
    if "__" in pat:
        a, b = pat.split("__", 1)
        return f"{a}\n{b}"
    if "::" in pat:
        a, b = pat.split("::", 1)
        return f"{a}\n{b}"

    # cas canonisés (ex: "CHUM_Patient_16")
    if pat.count("_") >= 1:
        a, b = pat.split("_", 1)
        return f"{a}\n{b}"

    return pat


def find_results_dirs(root: Path):
    """Retourne la liste des sous-dossiers results_*"""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("results_")
    )


def build_patient_color_map(patients):
    """Construit un dict patient -> couleur (palette stable)."""
    unique_patients = sorted(pd.unique(patients))
    palette = sns.color_palette("husl", len(unique_patients))
    return {pat: palette[i] for i, pat in enumerate(unique_patients)}


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def make_patientline_jitterplot_triplet_level(
    df_best: pd.DataFrame,
    root: Path,
    out_dir: Path,
    color_map: dict,
    title: str = "Triplet-level confidence per patient (one x per patient)",
    jitter: float = 0.18,
    alpha: float = 0.25,
    s: float = 8,
):
    """
    Un point par (patient, seizure, electrode) (i.e. occurrence), mais
    regroupés par patient sur l'axe X (20 patients).

    3 subplots (un par métrique):
      - y_score SOZ
      - y_score non-SOZ
      - confidence_gap point-wise vs moyennes du groupe (patient,seizure,seq)
    """
    rows = []

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat} -> {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            continue

        df_pat["is_SOZ"] = pd.to_numeric(df_pat.get("is_SOZ", 0), errors="coerce").fillna(0).astype(int)
        df_pat["y_score"] = pd.to_numeric(df_pat.get("y_score", np.nan), errors="coerce")
        df_pat = df_pat.dropna(subset=["y_score"])
        if df_pat.empty:
            continue

        # identifiant électrode
        if "node_index" not in df_pat.columns:
            df_pat = df_pat.reset_index(drop=False).rename(columns={"index": "node_index"})

        group_keys = ["patient", "seizure_id", "seq_idx"]
        if not all(k in df_pat.columns for k in group_keys):
            raise ValueError(f"Missing group keys in {pred_path}: {set(group_keys) - set(df_pat.columns)}")

        grp = df_pat.groupby(group_keys, dropna=False)

        mean_pos = grp.apply(
            lambda g: float(np.mean(g.loc[g["is_SOZ"] == 1, "y_score"])) if (g["is_SOZ"] == 1).any() else np.nan
        )
        mean_neg = grp.apply(
            lambda g: float(np.mean(g.loc[g["is_SOZ"] == 0, "y_score"])) if (g["is_SOZ"] == 0).any() else np.nan
        )
        mean_pos.name = "mean_pos_group"
        mean_neg.name = "mean_neg_group"

        df_pat = df_pat.join(mean_pos, on=group_keys)
        df_pat = df_pat.join(mean_neg, on=group_keys)

        is_soz = df_pat["is_SOZ"].to_numpy(dtype=int)
        ys = df_pat["y_score"].to_numpy(dtype=float)
        mp = df_pat["mean_pos_group"].to_numpy(dtype=float)
        mn = df_pat["mean_neg_group"].to_numpy(dtype=float)

        pos_val = np.where(is_soz == 1, ys, np.nan)
        neg_val = np.where(is_soz == 0, ys, np.nan)

        gap_val = np.full_like(ys, np.nan, dtype=float)
        mask_soz = (is_soz == 1) & np.isfinite(mn)
        gap_val[mask_soz] = ys[mask_soz] - mn[mask_soz]
        mask_neg = (is_soz == 0) & np.isfinite(mp)
        gap_val[mask_neg] = mp[mask_neg] - ys[mask_neg]

        rows.append(
            pd.DataFrame(
                {
                    "patient": pat,
                    "pos_score": pos_val,
                    "neg_score": neg_val,
                    "gap": gap_val,
                }
            )
        )

    if not rows:
        print("[WARN] No triplet-level points -> plot skipped.")
        return

    df_plot = pd.concat(rows, ignore_index=True)

    patients = sorted(df_plot["patient"].unique())
    x_map = {p: i for i, p in enumerate(patients)}

    metrics = [
        ("pos_score", "SOZ scores (points)"),
        ("neg_score", "non-SOZ scores (points)"),
        ("gap", "confidence gap per point"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 0.55 * len(patients)), 6), sharey=False)

    for ax, (col, ax_title) in zip(axes, metrics):
        for p in patients:
            vals = df_plot.loc[df_plot["patient"] == p, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            x0 = x_map[p]
            xs = x0 + np.random.uniform(-jitter, jitter, size=vals.size)

            ax.scatter(
                xs,
                vals,
                s=s,
                alpha=alpha,
                color=color_map.get(p, "grey"),
                linewidth=0,
            )

        ax.set_title(ax_title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.6, len(patients) - 0.4)
        ax.set_xticks(range(len(patients)))

        xtick_labels = []
        for p in patients:
            if "__" in p:
                part1, part2 = p.split("__", 1)
            else:
                part1, part2 = "", p
            xtick_labels.append(f"{part1}\n{part2}")

        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)

    axes[0].set_ylabel("Score / Gap")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out_path = out_dir / "jitterplot_triplet_level_per_patient_3panels.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure per-patient jitterplot -> {out_path}")


# ----------------------------------------------------------
# 0) Contrainte: min SOZ electrodes (distinct node_index) par seizure
# ----------------------------------------------------------
def patient_has_min_soz_electrodes_per_seizure_from_df(
    df_pat: pd.DataFrame,
    min_soz_electrodes: int = 2,
    count_unique_node_index: bool = True,
) -> bool:
    """
    Vérifie que pour ce patient, pour chaque seizure_id, il existe au moins
    `min_soz_electrodes` électrodes SOZ (is_SOZ==1).

    - Si count_unique_node_index=True : on compte des électrodes distinctes via node_index.
    - Sinon: compte brut de lignes SOZ.
    """
    if df_pat.empty:
        return False

    needed = {"seizure_id", "is_SOZ"}
    if not needed.issubset(df_pat.columns):
        return False

    df_pat = df_pat.copy()
    df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)

    df_soz = df_pat[df_pat["is_SOZ"] == 1]
    if df_soz.empty:
        return False

    if count_unique_node_index and "node_index" in df_soz.columns:
        counts = df_soz.groupby("seizure_id")["node_index"].nunique()
    else:
        counts = df_soz.groupby("seizure_id").size()

    all_seizures = df_pat["seizure_id"].unique()
    counts = counts.reindex(all_seizures, fill_value=0)

    return bool((counts >= int(min_soz_electrodes)).all())


# ----------------------------------------------------------
# 1) Metrics calculées depuis cv_val_predictions_ranked.csv
#     - collapse par node_index au sein d'un groupe (patient, seizure, seq)
# ----------------------------------------------------------
def _collapse_group_by_node(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Robustesse: si une électrode apparaît plusieurs fois dans un même (patient,seizure,seq),
    on la collapse en 1 ligne via node_index:
      - y_score = max
      - is_SOZ = max (OR)
      - rank   = min (si présent)
    Si node_index absent, on renvoie df_group tel quel (copie).
    """
    df_group = df_group.copy()

    if "node_index" not in df_group.columns:
        df_group["y_score"] = pd.to_numeric(df_group.get("y_score", np.nan), errors="coerce")
        df_group["is_SOZ"] = pd.to_numeric(df_group.get("is_SOZ", 0), errors="coerce").fillna(0).astype(int)
        if "rank" in df_group.columns:
            df_group["rank"] = pd.to_numeric(df_group["rank"], errors="coerce")
        return df_group

    if "y_score" not in df_group.columns:
        df_group["y_score"] = np.nan
    df_group["y_score"] = pd.to_numeric(df_group["y_score"], errors="coerce")

    if "is_SOZ" not in df_group.columns:
        df_group["is_SOZ"] = 0
    df_group["is_SOZ"] = pd.to_numeric(df_group["is_SOZ"], errors="coerce").fillna(0).astype(int)

    agg = {"y_score": "max", "is_SOZ": "max"}
    if "rank" in df_group.columns:
        df_group["rank"] = pd.to_numeric(df_group["rank"], errors="coerce")
        agg["rank"] = "min"

    df_c = (
        df_group
        .groupby("node_index", as_index=False)
        .agg(agg)
    )
    return df_c


def precision_recall_f1_topk(
    scores: np.ndarray,
    labels: np.ndarray,
    frac: float,
    fp_mode: str = "classic",  # "classic" ou "before_last_soz"
) -> Tuple[float, float, float]:
    """
    scores: (N,), labels: (N,) 0/1
    frac: 0.10 ou 0.20
    fp_mode:
      - "classic": FP = #nonSOZ dans top-k
      - "before_last_soz": FP = #nonSOZ STRICTEMENT avant le dernier SOZ dans top-k
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    k = max(1, int(math.ceil(frac * n)))

    order = np.argsort(scores)[::-1]
    top_idx = order[:k]

    top_labels = labels[top_idx].astype(int)

    tp = int(top_labels.sum())
    fn = int(labels.sum() - tp)

    if fp_mode == "classic":
        fp = int((top_labels == 0).sum())

    elif fp_mode == "before_last_soz":
        if tp == 0:
            fp = k
        else:
            last_soz_pos = int(np.where(top_labels == 1)[0].max())
            fp = int((top_labels[:last_soz_pos] == 0).sum())

    else:
        raise ValueError("fp_mode must be 'classic' or 'before_last_soz'")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_patient_metrics_from_predictions_df(df_pat: pd.DataFrame) -> Dict[str, float]:
    """
    Calcule des métriques patient-wise en moyenne sur les groupes (seizure_id, seq_idx).

    Retourne notamment:
      - mean_score_pos, mean_score_neg, confidence_gap (moyenne sur groupes)
      - Precision@top_10pct, Recall@top_10pct, F1@top_10pct
      - Precision@top_20pct, Recall@top_20pct, F1@top_20pct
      - AUC_group_mean (moyenne AUC sur groupes où les 2 classes existent)
      - num_groups_used (debug)
    """
    needed = {"patient", "seizure_id", "seq_idx", "is_SOZ", "y_score"}
    missing = needed - set(df_pat.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions for patient: {missing}")

    df_pat = df_pat.copy()
    df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
    df_pat["y_score"] = pd.to_numeric(df_pat["y_score"], errors="coerce")

    group_keys = ["patient", "seizure_id", "seq_idx"]
    grp = df_pat.groupby(group_keys)

    pos_means, neg_means, gaps = [], [], []
    p10s, r10s, f10s = [], [], []
    p20s, r20s, f20s = [], [], []
    aucs = []
    n_groups = 0
    n_auc_groups = 0

    for _, g in grp:
        g2 = _collapse_group_by_node(g)
        g2 = g2.dropna(subset=["y_score"])
        if g2.empty:
            continue

        scores = g2["y_score"].to_numpy(dtype=float)
        labels = g2["is_SOZ"].to_numpy(dtype=int)

        pos_mean = float(np.mean(scores[labels == 1])) if (labels == 1).any() else np.nan
        neg_mean = float(np.mean(scores[labels == 0])) if (labels == 0).any() else np.nan
        gap = (pos_mean - neg_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan

        pos_means.append(pos_mean)
        neg_means.append(neg_mean)
        gaps.append(gap)

        p10, r10, f10 = precision_recall_f1_topk(scores, labels, 0.10, fp_mode="before_last_soz")
        p20, r20, f20 = precision_recall_f1_topk(scores, labels, 0.20, fp_mode="before_last_soz")

        p10s.append(p10); r10s.append(r10); f10s.append(f10)
        p20s.append(p20); r20s.append(r20); f20s.append(f20)

        if len(np.unique(labels)) >= 2:
            try:
                auc = float(roc_auc_score(labels, scores))
                aucs.append(auc)
                n_auc_groups += 1
            except Exception:
                pass

        n_groups += 1

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    out = {
        "mean_score_pos": nanmean_safe(pos_means),
        "mean_score_neg": nanmean_safe(neg_means),
        "confidence_gap": nanmean_safe(gaps),

        "Precision@top_10pct": nanmean_safe(p10s),
        "Recall@top_10pct": nanmean_safe(r10s),
        "F1@top_10pct": nanmean_safe(f10s),

        "Precision@top_20pct": nanmean_safe(p20s),
        "Recall@top_20pct": nanmean_safe(r20s),
        "F1@top_20pct": nanmean_safe(f20s),

        "AUC_group_mean": nanmean_safe(aucs),

        "num_groups_used": float(n_groups),
        "num_auc_groups_used": float(n_auc_groups),
    }
    return out


# ----------------------------------------------------------
# 2) Sélection du meilleur modèle par patient (par F1 calculé depuis CSV)
# ----------------------------------------------------------
def collect_best_model_per_patient_by_f1(
    root: Path,
    select_f1_pct: int = 10,
    require_min_soz_per_seizure: Optional[int] = 2,
    count_unique_node_index: bool = True,
) -> pd.DataFrame:
    """
    Parcourt tous les results_*/<big_exp_name>/exp*/cv_val_predictions_ranked.csv

    Pour chaque patient trouvé dans chaque fichier, calcule:
      - mean_score_pos / mean_score_neg / confidence_gap
      - Precision/Recall/F1 @10% et @20%
      - AUC_group_mean

    Puis conserve, pour chaque patient, le couple (results_dir, experiment)
    qui maximise F1@top_{select_f1_pct}pct, sous contrainte min SOZ par seizure (si activée).

    Returns: DataFrame 1 ligne par patient (best combo)
    """
    assert select_f1_pct in (10, 20), "select_f1_pct doit être 10 ou 20"

    f1_key = f"F1@top_{select_f1_pct}pct"
    best_per_patient: Dict[str, Dict] = {}

    for results_dir in find_results_dirs(root):
        clf_root = results_dir / big_exp_name
        if not clf_root.is_dir():
            continue

        print(f"[SCAN] {results_dir.name}")

        for exp_dir in sorted(clf_root.iterdir()):
            if not exp_dir.is_dir():
                continue

            pred_path = exp_dir / "cv_val_predictions_ranked.csv"
            if not pred_path.is_file():
                continue

            try:
                df_pred = pd.read_csv(pred_path)
            except Exception as e:
                print(f"  [WARN] Impossible de lire {pred_path}: {e}")
                continue

            needed = {"patient", "seizure_id", "seq_idx", "is_SOZ", "y_score"}
            if not needed.issubset(df_pred.columns):
                print(f"  [WARN] Colonnes manquantes dans {pred_path}: {needed - set(df_pred.columns)}")
                continue

            df_pred["patient"] = df_pred["patient"].astype(str)

            for pat in sorted(df_pred["patient"].unique()):
                df_pat = df_pred[df_pred["patient"] == pat].copy()
                if df_pat.empty:
                    continue

                if require_min_soz_per_seizure is not None:
                    ok = patient_has_min_soz_electrodes_per_seizure_from_df(
                        df_pat,
                        min_soz_electrodes=require_min_soz_per_seizure,
                        count_unique_node_index=count_unique_node_index,
                    )
                    if not ok:
                        continue

                try:
                    metrics = compute_patient_metrics_from_predictions_df(df_pat)
                except Exception as e:
                    print(f"  [WARN] metrics failed for {pat} in {pred_path}: {e}")
                    continue

                score = metrics.get(f1_key, np.nan)
                if not np.isfinite(score):
                    continue

                current_best = best_per_patient.get(pat)
                better = (current_best is None) or (score > current_best["criterion_value"])

                if better:
                    best_per_patient[pat] = {
                        "patient": pat,
                        "results_dir": results_dir.name,
                        "experiment": exp_dir.name,
                        "criterion_name": f1_key,
                        "criterion_value": float(score),

                        "mean_score_pos": float(metrics.get("mean_score_pos", np.nan)),
                        "mean_score_neg": float(metrics.get("mean_score_neg", np.nan)),
                        "confidence_gap": float(metrics.get("confidence_gap", np.nan)),

                        "Precision@top_10pct": float(metrics.get("Precision@top_10pct", np.nan)),
                        "Recall@top_10pct": float(metrics.get("Recall@top_10pct", np.nan)),
                        "F1@top_10pct": float(metrics.get("F1@top_10pct", np.nan)),

                        "Precision@top_20pct": float(metrics.get("Precision@top_20pct", np.nan)),
                        "Recall@top_20pct": float(metrics.get("Recall@top_20pct", np.nan)),
                        "F1@top_20pct": float(metrics.get("F1@top_20pct", np.nan)),

                        "AUC_group_mean": float(metrics.get("AUC_group_mean", np.nan)),

                        "num_groups_used": float(metrics.get("num_groups_used", np.nan)),
                        "num_auc_groups_used": float(metrics.get("num_auc_groups_used", np.nan)),
                    }

    if not best_per_patient:
        raise RuntimeError(
            "Aucun patient sélectionné. Vérifie la contrainte min SOZ, les chemins, "
            "et la présence des colonnes requises dans cv_val_predictions_ranked.csv."
        )

    df_best = pd.DataFrame(list(best_per_patient.values()))
    df_best = df_best.sort_values("patient").reset_index(drop=True)
    return df_best


# ----------------------------------------------------------
# 2bis) NOUVEAU : Exports de traçabilité & predictions utilisées
# ----------------------------------------------------------
def export_audit_trace(df_best: pd.DataFrame, out_dir: Path, select_f1_pct: int) -> Path:
    """
    1 ligne par patient : config choisie + métriques utilisées pour les plots.
    (c'est df_best, mais avec ordre de colonnes stable)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_order = [
        "patient",
        "results_dir",
        "experiment",
        "criterion_name",
        "criterion_value",
        "mean_score_pos",
        "mean_score_neg",
        "confidence_gap",
        "Precision@top_10pct",
        "Recall@top_10pct",
        "F1@top_10pct",
        "Precision@top_20pct",
        "Recall@top_20pct",
        "F1@top_20pct",
        "AUC_group_mean",
        "num_groups_used",
        "num_auc_groups_used",
    ]
    cols_present = [c for c in cols_order if c in df_best.columns]
    audit = df_best[cols_present].copy()

    audit_path = out_dir / f"audit_best_config_and_metrics_top{select_f1_pct}.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[OK] Audit trace sauvegardée -> {audit_path}")
    return audit_path


def export_audit_with_top10_conf(
    df_best: pd.DataFrame,
    df_conf_top10: pd.DataFrame,
    out_dir: Path,
    select_f1_pct: int,
) -> Path:
    """
    Même audit, mais ajoute les valeurs "mean_score_pos/neg/gap" calculées uniquement
    sur les électrodes du TOP10% (à l'intérieur de chaque groupe).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_conf_top10 = df_conf_top10.copy()
    df_conf_top10 = df_conf_top10.rename(columns={
        "mean_score_pos": "mean_score_pos_TOP10",
        "mean_score_neg": "mean_score_neg_TOP10",
        "confidence_gap": "confidence_gap_TOP10",
    })

    audit = pd.merge(df_best, df_conf_top10, on="patient", how="left")
    audit_path = out_dir / f"audit_best_config_and_metrics_plus_TOP10_conf_top{select_f1_pct}.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[OK] Audit + TOP10 conf sauvegardée -> {audit_path}")
    return audit_path


def export_selected_predictions(
    root: Path,
    df_best: pd.DataFrame,
    out_dir: Path,
    big_exp_name: str,
    out_name: str = "predictions_selected_best_models_all_patients.csv",
) -> Path:
    """
    Concatène toutes les lignes de cv_val_predictions_ranked.csv réellement utilisées
    (uniquement best (results_dir, experiment) par patient).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for export: {pat} -> {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        df = df[df["patient"].astype(str) == pat].copy()
        if df.empty:
            continue

        df["selected_results_dir"] = res_dir
        df["selected_experiment"] = exp_dir
        df["selected_pred_path"] = str(pred_path)

        parts.append(df)

    if not parts:
        raise RuntimeError("Aucune prédiction sélectionnée à exporter (parts vide).")

    df_all = pd.concat(parts, ignore_index=True)

    # ordre colonnes lisible
    front = [
        "patient", "seizure_id", "seq_idx", "node_index", "is_SOZ", "y_score", "rank",
        "selected_results_dir", "selected_experiment", "selected_pred_path",
    ]
    cols = [c for c in front if c in df_all.columns] + [c for c in df_all.columns if c not in front]
    df_all = df_all[cols]

    out_path = out_dir / out_name
    df_all.to_csv(out_path, index=False)
    print(f"[OK] Export predictions best-models -> {out_path} (n={len(df_all)})")
    return out_path


def export_selected_predictions_collapsed(
    root: Path,
    df_best: pd.DataFrame,
    out_dir: Path,
    big_exp_name: str,
    out_name: str = "predictions_selected_best_models_collapsed_by_node.csv",
) -> Path:
    """
    Exporte la version réellement "rankable" par groupe, i.e. après collapse par node_index
    au sein de chaque (patient, seizure_id, seq_idx).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])
        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            continue

        df = pd.read_csv(pred_path)
        df = df[df["patient"].astype(str) == pat].copy()
        if df.empty:
            continue

        if "node_index" not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": "node_index"})

        df["is_SOZ"] = pd.to_numeric(df.get("is_SOZ", 0), errors="coerce").fillna(0).astype(int)
        df["y_score"] = pd.to_numeric(df.get("y_score", np.nan), errors="coerce")

        collapsed_groups = []
        for keys, g in df.groupby(group_keys, dropna=False):
            g2 = _collapse_group_by_node(g).dropna(subset=["y_score"]).copy()
            if g2.empty:
                continue
            for k, val in zip(group_keys, keys):
                g2[k] = val
            collapsed_groups.append(g2)

        if not collapsed_groups:
            continue

        dcc = pd.concat(collapsed_groups, ignore_index=True)
        dcc["selected_results_dir"] = res_dir
        dcc["selected_experiment"] = exp_dir
        dcc["selected_pred_path"] = str(pred_path)

        parts.append(dcc)

    if not parts:
        raise RuntimeError("Aucune prédiction collapsed à exporter (parts vide).")

    df_all = pd.concat(parts, ignore_index=True)

    out_path = out_dir / out_name
    df_all.to_csv(out_path, index=False)
    print(f"[OK] Export predictions collapsed -> {out_path} (n={len(df_all)})")
    return out_path


# ----------------------------------------------------------
# 3) Figures : boxplot confidence
# ----------------------------------------------------------
def _infer_electrode_label_col(df: pd.DataFrame) -> str:
    """
    Devine la colonne la plus probable pour afficher le nom d'électrode.
    Fallback sur node_index, puis sur index de ligne.
    """
    candidates = [
        "electrode_name", "electrode", "channel_name", "channel", "contact",
        "node_name", "node_label", "bipolar_name"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    if "node_index" in df.columns:
        return "node_index"
    return ""


def plot_scores_per_seizure_per_patient_chum(
    root: Path,
    df_best: pd.DataFrame,
    out_dir: Path,
    big_exp_name: str,
    chum_prefix: str = "CHUM",
    annotate: str = "soz",
    max_labels_per_seizure: int | None = 8,
    jitter_x: float = 0.18,
    alpha_non_soz: float = 0.45,
    alpha_soz: float = 0.95,
    point_size: float = 18,
    soz_point_size: float = 28,
    export_patient: str | None = None,
    export_seizure: str | int | None = None,
):
    """
    Pour chaque patient dont le nom commence par chum_prefix (ex: "CHUM"),
    produit un plot séparé:
      - x: index de seizure (1..N, trié)
      - y: y_score
      - 1 point par électrode (agrégé sur seq_idx si besoin)
      - SOZ en rouge
      - annotation du nom électrode à côté du point (SOZ ou ALL)

    Les données sont lues depuis:
      root / results_dir / big_exp_name / experiment / cv_val_predictions_ranked.csv
    en utilisant le meilleur modèle par patient (df_best).
    """
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfb = df_best.copy()
    dfb["patient"] = dfb["patient"].astype(str)
    dfb = dfb[dfb["patient"].str.startswith(chum_prefix)].reset_index(drop=True)

    if dfb.empty:
        print(f"[WARN] Aucun patient avec prefix '{chum_prefix}' dans df_best.")
        return

    for _, row in dfb.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat} -> {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            print(f"[WARN] No rows for patient {pat} in {pred_path}")
            continue

        if "seizure_id" not in df_pat.columns or "y_score" not in df_pat.columns:
            print(f"[WARN] Missing seizure_id/y_score for {pat} in {pred_path}")
            continue

        df_pat["is_SOZ"] = pd.to_numeric(df_pat.get("is_SOZ", 0), errors="coerce").fillna(0).astype(int)
        df_pat["y_score"] = pd.to_numeric(df_pat["y_score"], errors="coerce")
        df_pat = df_pat.dropna(subset=["y_score"])
        if df_pat.empty:
            print(f"[WARN] No finite y_score for {pat}")
            continue

        if "node_index" not in df_pat.columns:
            df_pat = df_pat.reset_index(drop=False).rename(columns={"index": "node_index"})

        label_col = _infer_electrode_label_col(df_pat)
        if label_col == "":
            df_pat["__electrode_label__"] = df_pat["node_index"].astype(str)
            label_col = "__electrode_label__"
        else:
            df_pat[label_col] = df_pat[label_col].astype(str)

        agg = (
            df_pat.groupby(["seizure_id", "node_index"], as_index=False)
            .agg(
                y_score=("y_score", "max"),
                is_SOZ=("is_SOZ", "max"),
                electrode_label=(label_col, "first"),
            )
        )

        # EXPORT CSV pour un patient + une seizure spécifique
        if (
            export_patient is not None
            and export_seizure is not None
            and str(pat) == str(export_patient)
        ):
            print(agg["seizure_id"].value_counts())
            sub = agg[agg["seizure_id"].astype(str) == str(export_seizure)].copy()

            if sub.empty:
                print(f"[WARN] Aucun résultat pour {pat}, seizure {export_seizure}")
            else:
                csv_out = out_dir / (
                    f"{pat.replace('::','__')}_seizure_{export_seizure}_electrode_scores.csv"
                )

                sub_out = (
                    sub[["electrode_label", "y_score", "is_SOZ"]]
                    .rename(columns={
                        "electrode_label": "electrode",
                        "y_score": "score",
                    })
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )

                sub_out.to_csv(csv_out, index=False)
                print(f"[OK] CSV exporté -> {csv_out}")

        seizure_ids = agg["seizure_id"].astype(str).unique().tolist()

        def _try_num(x):
            try:
                return float(x)
            except Exception:
                return None

        if all(_try_num(s) is not None for s in seizure_ids):
            seizure_ids_sorted = sorted(seizure_ids, key=lambda s: float(s))
        else:
            seizure_ids_sorted = sorted(seizure_ids)

        seizure_to_x = {sid: i + 1 for i, sid in enumerate(seizure_ids_sorted)}
        agg["x"] = agg["seizure_id"].astype(str).map(seizure_to_x).astype(float)

        fig_w = max(10, 0.55 * len(seizure_ids_sorted))
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        m0 = (agg["is_SOZ"].astype(int) == 0)
        x0 = agg.loc[m0, "x"].to_numpy(float) + np.random.uniform(-jitter_x, jitter_x, size=m0.sum())
        y0 = agg.loc[m0, "y_score"].to_numpy(float)
        ax.scatter(x0, y0, s=point_size, alpha=alpha_non_soz, linewidth=0)

        m1 = (agg["is_SOZ"].astype(int) == 1)
        x1 = agg.loc[m1, "x"].to_numpy(float) + np.random.uniform(-jitter_x, jitter_x, size=m1.sum())
        y1 = agg.loc[m1, "y_score"].to_numpy(float)
        ax.scatter(x1, y1, s=soz_point_size, alpha=alpha_soz, color="red", linewidth=0, zorder=3)

        annotate = (annotate or "soz").lower()
        if annotate not in ("soz", "all", "none"):
            annotate = "soz"

        if annotate != "none":
            for sid in seizure_ids_sorted:
                sub = agg[agg["seizure_id"].astype(str) == str(sid)].copy()
                if sub.empty:
                    continue

                if annotate == "soz":
                    sub = sub[sub["is_SOZ"].astype(int) == 1].copy()

                if sub.empty:
                    continue

                if max_labels_per_seizure is not None and len(sub) > int(max_labels_per_seizure):
                    sub = sub.sort_values("y_score", ascending=False).head(int(max_labels_per_seizure))

                for _, r in sub.iterrows():
                    x = float(seizure_to_x[str(sid)]) + np.random.uniform(-jitter_x, jitter_x)
                    y = float(r["y_score"])
                    txt = str(r["electrode_label"])
                    ax.text(
                        x + 0.03,
                        y,
                        txt,
                        fontsize=8,
                        color=("red" if int(r["is_SOZ"]) == 1 else "black"),
                        va="center",
                        ha="left",
                        zorder=4,
                    )

        ax.set_title(f"{pat} — Electrode scores per seizure (1 point/electrode)")
        ax.set_xlabel("Seizure (ordered index)")
        ax.set_ylabel("y_score")
        ax.set_xlim(0.5, len(seizure_ids_sorted) + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, len(seizure_ids_sorted) + 1))
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        out_path = out_dir / f"{pat.replace('::','__')}_scores_per_seizure_points_per_electrode.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Saved {pat} plot -> {out_path}")


def make_boxplot(
    df: pd.DataFrame,
    out_dir: Path,
    color_map: dict,
    title: str = "Confidence Metrics Distribution",
    out_name: str = "boxplot_confidence_gap.png",
):
    """
    Boxplot mean_score_pos / mean_score_neg / confidence_gap + points par patient.
    """
    dfp = df.copy()
    dfp["patient"] = dfp["patient"].astype(str)

    cols = ["mean_score_pos", "mean_score_neg", "confidence_gap"]
    xlabels = [
        "Average Positive Score (SOZ)\nMean over Seizures",
        "Average Negative Score (non-SOZ)\nMean over Seizures",
        "Confidence Gap (SOZ vs non-SOZ)\nMean over Seizures",
    ]

    data = []
    for c in cols:
        v = pd.to_numeric(dfp[c], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        data,
        tick_labels=xlabels,
        showfliers=False,
        widths=0.30,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=1.5),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    for j, c in enumerate(cols, start=1):
        vals = pd.to_numeric(dfp[c], errors="coerce").to_numpy(dtype=float)
        for pat, v in zip(dfp["patient"].tolist(), vals):
            if not np.isfinite(v):
                continue
            col = color_map.get(pat, "grey")
            x = j + np.random.uniform(-0.08, 0.08)
            ax.scatter(
                x,
                v,
                s=70,
                alpha=0.75,
                color=col,
                edgecolors="black",
                linewidths=0.3,
            )

    ax.set_ylim(-0.2, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / out_name

    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure sauvegardée -> {fig_path}")


# ----------------------------------------------------------
# 4) Figure top-k metrics
# ----------------------------------------------------------
def make_boxplot_topk_metrics(df: pd.DataFrame, out_dir: Path, color_map, title: str):
    metrics = [
        "Precision@top_10pct",
        "Recall@top_10pct",
        "F1@top_10pct",
        "Precision@top_20pct",
        "Recall@top_20pct",
        "F1@top_20pct",
        "AUC_group_mean",
    ]

    pretty_labels = [
        "Precision\n(top 10%)",
        "Recall\n(top 10%)",
        "F1\n(top 10%)",
        "Precision\n(top 20%)",
        "Recall\n(top 20%)",
        "F1\n(top 20%)",
        "AUC\n(group mean)",
    ]

    present = [m for m in metrics if m in df.columns]
    if not present:
        print("[WARN] Aucun metric top-k présent, figure ignorée.")
        return

    data = [df[m].astype(float).values for m in present]
    labels = [pretty_labels[metrics.index(m)] for m in present]
    patients = df["patient"].tolist()

    plt.figure(figsize=(10, 6))
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


# ----------------------------------------------------------
# 5) Rang du PREMIER SOZ + counts
# ----------------------------------------------------------
def plot_soz_rank_mix_first_and_all(
    df_first: pd.DataFrame,
    df_all: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str = "SOZ detection ranks per patient (First vs All) — median ± IQR",
    show_seizure_count: bool = True,
    point_size: float = 70,
    capsize: float = 4,
    x_offset: float = 0.14,
    bg_width: float = 0.82,
):
    """
    Mix des 2 plots:
      - fond: nb contacts total (clair) + nb contacts SOZ (plus foncé)
      - 2 points (First SOZ + All SOZ) par patient, avec IQR en barres d'erreur
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.merge(
        df_first,
        df_all,
        on="patient",
        how="inner",
        suffixes=("_first", "_all"),
    ).copy()

    if df.empty:
        print("[WARN] Merge df_first/df_all vide -> plot skipped.")
        return

    patients = df["patient"].astype(str).tolist()
    x = np.arange(len(patients))

    total_contacts = pd.to_numeric(df["max_electrodes_all"], errors="coerce").fillna(0).astype(int).values
    soz_contacts   = pd.to_numeric(df["max_soz_electrodes"], errors="coerce").fillna(0).astype(int).values

    first_med = pd.to_numeric(df["first_soz_median_rank"], errors="coerce").values
    first_q25 = pd.to_numeric(df["first_soz_q25_rank"], errors="coerce").values
    first_q75 = pd.to_numeric(df["first_soz_q75_rank"], errors="coerce").values

    all_med = pd.to_numeric(df["all_soz_median_rank"], errors="coerce").values
    all_q25 = pd.to_numeric(df["all_soz_q25_rank"], errors="coerce").values
    all_q75 = pd.to_numeric(df["all_soz_q75_rank"], errors="coerce").values

    if "num_seizures_all" in df.columns:
        num_seizures = pd.to_numeric(df["num_seizures_all"], errors="coerce").fillna(0).astype(int).values
    else:
        num_seizures = pd.to_numeric(df.get("num_seizures", 0), errors="coerce").fillna(0).astype(int).values

    fig = plt.figure(figsize=(max(12, 0.45 * len(patients)), 6))

    for i, pat in enumerate(patients):
        base = color_map.get(pat, (0.8, 0.8, 0.8))
        plt.bar(i, total_contacts[i], width=bg_width, color=(*base, 0.20), edgecolor="none", zorder=0)
        plt.bar(i, soz_contacts[i], width=bg_width, color=(*base, 0.55), edgecolor="none", zorder=1)

    def _plot_point_iqr(xpos, med, q25, q75, color, marker, label):
        med = np.asarray(med, float)
        q25 = np.asarray(q25, float)
        q75 = np.asarray(q75, float)

        valid = np.isfinite(med) & np.isfinite(q25) & np.isfinite(q75)
        if not np.any(valid):
            return

        lower = med - q25
        upper = q75 - med

        plt.errorbar(
            xpos[valid],
            med[valid],
            yerr=np.vstack([lower[valid], upper[valid]]),
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=capsize,
            capthick=1.0,
            zorder=5,
        )

        plt.scatter(
            xpos[valid],
            med[valid],
            s=point_size,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            marker=marker,
            zorder=6,
            label=label,
        )

    x_first = x - x_offset
    x_all   = x + x_offset

    _plot_point_iqr(x_first, first_med, first_q25, first_q75, color="black", marker="o", label="First SOZ (median ± IQR)")
    _plot_point_iqr(x_all,   all_med,   all_q25,   all_q75,   color="white", marker="o", label="All SOZ (median ± IQR)")

    if show_seizure_count:
        y_text = max(5, int(np.nanmax(total_contacts)) + 2)
        for i in range(len(patients)):
            plt.text(i, y_text, f"S:{int(num_seizures[i])}", ha="center", va="bottom", fontsize=10, zorder=10)

    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("SOZ detection rank (median ± IQR)\nwith total contacts (light) and SOZ contacts (dark)")
    plt.grid(axis="y", alpha=0.3)

    y_max_candidates = [
        np.nanmax(total_contacts) if len(total_contacts) else 0,
        np.nanmax([np.nanmax(first_q75), np.nanmax(all_q75)]) if (np.isfinite(first_q75).any() or np.isfinite(all_q75).any()) else 0,
    ]
    ymax = float(np.nanmax(y_max_candidates)) if len(y_max_candidates) else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    plt.ylim(0, ymax * 1.08)

    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 170]
    if len(ticks) > 0:
        plt.yticks(ticks)

    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

    fig_path = out_dir / "barplot_mix_first_vs_all_soz_rank_points_IQR_with_electrode_background.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Mix plot saved -> {fig_path}")


def compute_first_soz_rank_and_counts(df_best: pd.DataFrame, root: Path) -> pd.DataFrame:
    records = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir = row["results_dir"]
        exp_dir = row["experiment"]

        csv_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not csv_path.is_file():
            print(f"[WARN] {csv_path} introuvable, on saute {pat}")
            continue

        df = pd.read_csv(csv_path)
        df_pat = df[df["patient"].astype(str) == str(pat)].copy()
        if df_pat.empty:
            print(f"[WARN] aucune ligne pour {pat} dans {csv_path}")
            continue

        num_seizures = df_pat["seizure_id"].nunique()
        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"])
        max_electrodes = grp["node_index"].nunique().max() if "node_index" in df_pat.columns else grp.size().max()

        first_ranks = []
        for _, g in grp:
            g_soz = g[g["is_SOZ"] == 1]
            if g_soz.empty:
                continue
            if "rank" in g_soz.columns:
                first_ranks.append(int(pd.to_numeric(g_soz["rank"], errors="coerce").min()))
            else:
                g2 = g.dropna(subset=["y_score"]).copy()
                if g2.empty:
                    continue
                g2 = g2.sort_values("y_score", ascending=False)
                g2["rank_tmp"] = np.arange(1, len(g2) + 1)
                g2_soz = g2[g2["is_SOZ"] == 1]
                if g2_soz.empty:
                    continue
                first_ranks.append(int(g2_soz["rank_tmp"].min()))

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
                "max_electrodes": int(max_electrodes) if np.isfinite(max_electrodes) else 0,
            }
        )

    df_first = pd.DataFrame(records)
    return df_first.sort_values("patient").reset_index(drop=True)


def plot_first_soz_rank_barplot(
    df_first: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str = "First SOZ rank per patient (median + IQR)",
):
    patients = df_first["patient"].tolist()
    median_ranks = df_first["first_soz_median_rank"].values
    q25 = df_first["first_soz_q25_rank"].values
    q75 = df_first["first_soz_q75_rank"].values
    num_seizures = df_first["num_seizures"].values
    max_electrodes = df_first["max_electrodes"].values

    x = np.arange(len(patients))
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    for i, (pat, n_elec) in enumerate(zip(patients, max_electrodes)):
        bg_color = (*color_map.get(pat, (0.8, 0.8, 0.8)), 0.25)
        plt.bar(i, n_elec, color=bg_color, edgecolor="none")

    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, color=c, edgecolor="black", linewidth=1.0)
        offset = 160
        plt.text(i, offset, f"S:{int(num_seizures[i])}", ha="center", va="bottom", fontsize=10)

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

    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("First SOZ Channel Detection Rank\n(vs Total Number of Implanted Electrodes)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    ymax = max(np.nanmax(max_electrodes), np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks))
    plt.ylim(0, ymax * 1.05)

    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 170]
    plt.yticks(ticks)

    fig_path = out_dir / "barplot_first_soz_rank_median_IQR_background_electrodes.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure barplot avec background électrodes -> {fig_path}")


# ----------------------------------------------------------
# 6) Rang de TOUTES les électrodes SOZ
# ----------------------------------------------------------
def compute_all_soz_rank_and_counts(df_best: pd.DataFrame, root: Path) -> pd.DataFrame:
    records = []

    for _, row in df_best.iterrows():
        pat = row["patient"]
        res_dir = row["results_dir"]
        exp_dir = row["experiment"]

        csv_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not csv_path.is_file():
            print(f"[WARN] {csv_path} introuvable, on saute {pat}")
            continue

        df = pd.read_csv(csv_path)
        df_pat = df[df["patient"].astype(str) == str(pat)].copy()
        if df_pat.empty:
            print(f"[WARN] aucune ligne pour {pat} dans {csv_path}")
            continue

        num_seizures = df_pat["seizure_id"].nunique()
        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"])

        if "node_index" in df_pat.columns:
            max_electrodes = int(grp["node_index"].nunique().max())
        else:
            max_electrodes = int(grp.size().max())

        all_ranks = []
        max_soz_electrodes = 0
        num_groups_with_soz = 0

        for _, g in grp:
            g_soz = g[g["is_SOZ"] == 1]
            if g_soz.empty:
                continue

            num_groups_with_soz += 1

            if "rank" in g_soz.columns:
                rr = pd.to_numeric(g_soz["rank"], errors="coerce").dropna().astype(int).tolist()
            else:
                g2 = g.dropna(subset=["y_score"]).copy()
                if g2.empty:
                    continue
                g2 = g2.sort_values("y_score", ascending=False)
                g2["rank_tmp"] = np.arange(1, len(g2) + 1)
                rr = g2[g2["is_SOZ"] == 1]["rank_tmp"].astype(int).tolist()

            all_ranks.extend(rr)

            if "node_index" in g_soz.columns:
                n_soz = int(g_soz["node_index"].nunique())
            else:
                n_soz = int(len(g_soz))
            max_soz_electrodes = max(max_soz_electrodes, n_soz)

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
    return df_all.sort_values("patient").reset_index(drop=True)


def plot_all_soz_rank_barplot(
    df_all: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str = "All SOZ rank per patient (median + IQR)",
):
    patients = df_all["patient"].tolist()
    median_ranks = df_all["all_soz_median_rank"].values
    q25 = df_all["all_soz_q25_rank"].values
    q75 = df_all["all_soz_q75_rank"].values
    num_seizures = df_all["num_seizures"].values
    max_electrodes = df_all["max_electrodes"].values
    max_soz_electrodes = df_all["max_soz_electrodes"].values

    x = np.arange(len(patients))
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    for i, pat in enumerate(patients):
        base_color = color_map.get(pat, (0.8, 0.8, 0.8))
        plt.bar(i, max_electrodes[i], width=0.8, color=(*base_color, 0.20), edgecolor="none")
        if np.isfinite(max_soz_electrodes[i]):
            plt.bar(i, max_soz_electrodes[i], width=0.8, color=(*base_color, 0.65), edgecolor="none")

    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, width=0.25, color=c, edgecolor="black", linewidth=1.0)
        offset = 160
        plt.text(i, offset, f"S:{int(num_seizures[i])}", ha="center", va="bottom", fontsize=10)

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

    xtick_labels = []
    for p in patients:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.xticks(x, xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("All SOZ Channel Detection Rank\n(vs Total Number of Implanted Electrodes and SOZ Electrodes)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    ymax = max(np.nanmax(max_electrodes), np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks))
    plt.ylim(0, ymax * 1.05)

    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 170]
    plt.yticks(ticks)

    fig_path = out_dir / "barplot_all_soz_rank_median_IQR_background_electrodes_with_soz_count.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure barplot (ALL SOZ, avec nb d'électrodes SOZ) -> {fig_path}")


# ----------------------------------------------------------
# 7) Boxplot SOZ ranks + points colored by seizure + background electrodes
# ----------------------------------------------------------
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

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
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

        if all(k in df_pat.columns for k in group_keys) and "node_index" in df_pat.columns:
            grp = df_pat.groupby(group_keys)
            max_electrodes = int(grp["node_index"].nunique().max())
        else:
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


# ----------------------------------------------------------
# BCE onset (inchangé)
# ----------------------------------------------------------
def onset_curve_bce(p_smooth, t_true, eps=1e-7):
    p_smooth = np.asarray(p_smooth, dtype=float)
    T = len(p_smooth)
    if t_true is None or t_true < 0 or T == 0:
        return np.nan

    y_true = np.zeros(T, dtype=float)
    y_true[t_true:] = 1.0

    p = np.clip(p_smooth, eps, 1.0 - eps)
    bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(bce.mean())


def collect_bce_from_npz_for_best_models(root: Path, df_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir_name = str(row["results_dir"])

        patient_dir_name = pat.replace("::", "__")
        series_dir = root / res_dir_name / "results" / patient_dir_name / "series"

        if not series_dir.is_dir():
            print(f"[WARN] Pas de series_dir pour {pat} : {series_dir}")
            continue

        for npz_path in sorted(series_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Impossible de lire {npz_path}: {e}")
                continue

            if "p_graph_smooth" in data:
                p_smooth = np.asarray(data["p_graph_smooth"], dtype=float)
            elif "p_graph" in data:
                p_smooth = np.asarray(data["p_graph"], dtype=float)
            else:
                print(f"[WARN] p_graph_smooth/p_graph manquant dans {npz_path}, on saute.")
                continue

            t_true = int(data.get("t_true", -1))
            if t_true < 0:
                t_true = None

            bce_onset = onset_curve_bce(p_smooth, t_true)

            seizure_id = str(data.get("seizure_id", "?"))
            seq_index = int(data.get("seq_index", data.get("seq_idx", -1)))

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


def keep_middle_seq_per_seizure(df_bce: pd.DataFrame) -> pd.DataFrame:
    def _pick_middle(g):
        g = g.sort_values("seq_index")
        mid = len(g) // 2
        return g.iloc[[mid]]

    return (
        df_bce
        .groupby(["patient", "seizure_id"], group_keys=False)
        .apply(_pick_middle)
        .reset_index(drop=True)
    )


def plot_bce_boxplot_per_patient(df_bce: pd.DataFrame, out_dir: Path, color_map: dict):
    pats = sorted(df_bce["patient"].unique())
    data = [df_bce[df_bce["patient"] == p]["bce_onset"].dropna().values for p in pats]

    xtick_labels = []
    for p in pats:
        if "__" in p:
            part1, part2 = p.split("__", 1)
        else:
            part1, part2 = "", p
        xtick_labels.append(f"{part1}\n{part2}")

    plt.figure(figsize=(10, 4))

    bp = plt.boxplot(
        data,
        showmeans=False,
        patch_artist=True,
        showfliers=False
    )

    for i, p in enumerate(pats):
        c = color_map.get(p, "grey")
        plt.setp(bp["boxes"][i], facecolor=(0, 0, 0, 0), edgecolor=c, linewidth=1)
        plt.setp(bp["whiskers"][2*i:2*i+2], color=c, linewidth=1)
        plt.setp(bp["caps"][2*i:2*i+2], color=c, linewidth=1)
        plt.setp(bp["medians"][i], color="orange", linewidth=1.5)

    for i, p in enumerate(pats):
        values = df_bce[df_bce["patient"] == p]["bce_onset"].dropna().values
        c = color_map.get(p, "grey")
        for v in values:
            jitter = np.random.uniform(-0.12, 0.12)
            plt.scatter(i + 1 + jitter, v, color=c, linewidth=0.4, s=15, alpha=0.75)

    plt.xticks(range(1, len(pats) + 1), xtick_labels, rotation=90, fontsize=8)
    plt.ylabel("Time-wise BCE vs\nIdeal Onset Step")
    plt.title("Per-Patient Onset Curve Loss (Graph-Level Aggregated Ictal Activity Probability)\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome Patients")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)

    fig_path = out_dir / "boxplot_bce_onset_per_patient.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure BCE sauvegardée -> {fig_path}")


def compute_confidence_metrics_topk_for_best_models(
    df_best: pd.DataFrame,
    root: Path,
    frac: float = 0.10,
    fp_mode: str = "before_last_soz",
) -> pd.DataFrame:
    rows = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat} -> {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            continue

        needed = set(group_keys + ["is_SOZ", "y_score"])
        if not needed.issubset(df_pat.columns):
            print(f"[WARN] Missing columns for {pat} in {pred_path}: {needed - set(df_pat.columns)}")
            continue

        df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
        df_pat["y_score"] = pd.to_numeric(df_pat["y_score"], errors="coerce")
        df_pat = df_pat.dropna(subset=["y_score"])
        if df_pat.empty:
            continue

        pos_means, neg_means, gaps = [], [], []

        for _, g in df_pat.groupby(group_keys, dropna=False):
            g2 = _collapse_group_by_node(g).dropna(subset=["y_score"])
            if g2.empty:
                continue

            scores = g2["y_score"].to_numpy(dtype=float)
            labels = g2["is_SOZ"].to_numpy(dtype=int)

            n = len(scores)
            if n == 0:
                continue

            k = max(1, int(math.ceil(frac * n)))
            order = np.argsort(scores)[::-1]
            top_idx = order[:k]

            top_scores = scores[top_idx]
            top_labels = labels[top_idx]

            pos = float(np.mean(top_scores[top_labels == 1])) if (top_labels == 1).any() else np.nan
            neg = float(np.mean(top_scores[top_labels == 0])) if (top_labels == 0).any() else np.nan
            gap = (pos - neg) if np.isfinite(pos) and np.isfinite(neg) else np.nan

            pos_means.append(pos)
            neg_means.append(neg)
            gaps.append(gap)

        def nanmean_safe(x):
            x = np.asarray(x, dtype=float)
            return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

        rows.append({
            "patient": pat,
            "mean_score_pos": nanmean_safe(pos_means),
            "mean_score_neg": nanmean_safe(neg_means),
            "confidence_gap": nanmean_safe(gaps),
        })

    if not rows:
        print("[WARN] No top-k confidence metrics -> plot skipped.")
        return pd.DataFrame(columns=["patient", "mean_score_pos", "mean_score_neg", "confidence_gap"])

    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)


def plot_top10_scores_distribution_per_patient(
    df_best: pd.DataFrame,
    root: Path,
    out_dir: Path,
    color_map: dict,
    frac: float = 0.10,
    title: str = "Distribution of electrode scores in the top 10% (per group) - per patient",
    out_name: str = "boxplot_top10_scores_distribution_per_patient.png",
    jitter: float = 0.15,
    alpha: float = 0.6,
    point_size: float = 12,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for _, row in df_best.iterrows():
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        pred_path = root / res_dir / big_exp_name / exp_dir / "cv_val_predictions_ranked.csv"
        if not pred_path.is_file():
            print(f"[WARN] Missing predictions for {pat} -> {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        df_pat = df[df["patient"].astype(str) == pat].copy()
        if df_pat.empty:
            continue

        needed = set(group_keys + ["is_SOZ", "y_score"])
        if not needed.issubset(df_pat.columns):
            print(f"[WARN] Missing columns for {pat} in {pred_path}: {needed - set(df_pat.columns)}")
            continue

        df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
        df_pat["y_score"] = pd.to_numeric(df_pat["y_score"], errors="coerce")

        for _, g in df_pat.groupby(group_keys):
            g2 = _collapse_group_by_node(g).dropna(subset=["y_score"])
            if g2.empty:
                continue

            scores = g2["y_score"].to_numpy(dtype=float)
            labels = g2["is_SOZ"].to_numpy(dtype=int)

            n = len(scores)
            if n == 0:
                continue

            k = max(1, int(math.ceil(frac * n)))
            order = np.argsort(scores)[::-1]
            top_idx = order[:k]

            top_scores = scores[top_idx]
            top_labels = labels[top_idx]

            tmp = pd.DataFrame({
                "patient": pat,
                "y_score_top": top_scores,
                "is_SOZ": top_labels,
            })
            rows.append(tmp)

    if not rows:
        print("[WARN] No top-k scores collected -> plot skipped.")
        return

    df_plot = pd.concat(rows, ignore_index=True)
    patients = sorted(df_plot["patient"].unique())

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(patients)), 6))

    data = []
    for p in patients:
        vals = df_plot.loc[df_plot["patient"] == p, "y_score_top"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)

    ax.boxplot(
        data,
        tick_labels=[patient_labels_2lines(p) for p in patients],
        showfliers=False,
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=1.5),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    for i, p in enumerate(patients, start=1):
        sub = df_plot[df_plot["patient"] == p]
        vals = sub["y_score_top"].to_numpy(dtype=float)
        labs = sub["is_SOZ"].to_numpy(dtype=int)

        xs = i + np.random.uniform(-jitter, jitter, size=len(vals))
        base_col = color_map.get(p, "grey")

        m0 = (labs == 0) & np.isfinite(vals)
        ax.scatter(xs[m0], vals[m0], s=point_size, alpha=alpha, color=base_col, linewidth=0, zorder=2)

        m1 = (labs == 1) & np.isfinite(vals)
        ax.scatter(xs[m1], vals[m1], s=point_size * 1.6, alpha=0.9, color="red", marker="o", linewidth=0, zorder=3)

    ax.set_ylabel(f"y_score of electrodes in top {int(frac*100)}% (within each group)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    fig_path = out_dir / out_name
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Top-{int(frac*100)}% score distribution plot -> {fig_path}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyse du meilleur modèle par patient (sélection par F1@topK depuis predictions CSV).")
    ap.add_argument("--root", required=True, help="Dossier racine contenant les results_*")
    ap.add_argument("--out", default="best_f1", help="Dossier de sortie pour le CSV + figures")

    ap.add_argument("--select_f1_pct", type=int, choices=[10, 20], default=10,
                    help="Sélectionne le best model per patient en maximisant F1@top_10pct ou F1@top_20pct")
    ap.add_argument("--min_soz_per_seizure", type=int, default=2,
                    help="Contrainte: min # électrodes SOZ distinctes par seizure (default=2). Mets 0 pour désactiver.")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out + f"/best_F1_top{args.select_f1_pct}pct").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"Le dossier root n'existe pas: {root}")

    min_soz = None if args.min_soz_per_seizure <= 0 else int(args.min_soz_per_seizure)

    print(f"[INFO] Root = {root}")
    print(f"[INFO] Selection criterion = F1@top_{args.select_f1_pct}pct (from cv_val_predictions_ranked.csv)")
    print(f"[INFO] Constraint min SOZ electrodes per seizure = {min_soz}")

    df_best = collect_best_model_per_patient_by_f1(
        root=root,
        select_f1_pct=args.select_f1_pct,
        require_min_soz_per_seizure=min_soz,
        count_unique_node_index=True,
    )

    # Sauvegarde CSV "best"
    csv_path = out_dir / "best_model_per_patient_selected_by_F1.csv"
    df_best.to_csv(csv_path, index=False)
    print(f"[OK] Tableau sauvegardé -> {csv_path}")

    # NOUVEAU : Audit trace (config + métriques)
    export_audit_trace(df_best, out_dir, args.select_f1_pct)

    # NOUVEAU : Exports des prédictions réellement utilisées (brutes + collapsed)
    export_selected_predictions(root=root, df_best=df_best, out_dir=out_dir, big_exp_name=big_exp_name)
    export_selected_predictions_collapsed(root=root, df_best=df_best, out_dir=out_dir, big_exp_name=big_exp_name)

    # Palette
    color_map = build_patient_color_map(df_best["patient"])

    plot_scores_per_seizure_per_patient_chum(
        root=root,
        df_best=df_best,
        out_dir=out_dir,
        big_exp_name=big_exp_name,
        annotate="soz",
        export_patient="CHUM__Patient_02",
        export_seizure="?",
    )

    # Figure 1 : boxplot confiance
    make_boxplot(
        df_best,
        out_dir,
        color_map,
        title="Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\nand Their Separation (Confidence Gap)\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome"
    )

    df_conf_top10 = compute_confidence_metrics_topk_for_best_models(
        df_best=df_best,
        root=root,
        frac=0.10,
    )

    make_boxplot(
        df_conf_top10,
        out_dir,
        color_map,
        title="Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Groups)\n"
            "and Their Separation (Confidence Gap) - TOP 10% ELECTRODES ONLY\n"
            "- Best Model Applied to Held-Out Patient (Nested CV) -\n"
            "GOOD Surgery Outcome",
        out_name="boxplot_confidence_gap_TOP10_only.png",
    )

    # NOUVEAU : Audit enrichi avec les conf TOP10 (celles du plot TOP10-only)
    export_audit_with_top10_conf(df_best, df_conf_top10, out_dir, args.select_f1_pct)

    make_patientline_jitterplot_triplet_level(
        df_best=df_best,
        root=root,
        out_dir=out_dir,
        color_map=color_map,
        title="Triplet-level SOZ/non-SOZ confidence per patient\n(one point per (patient,seizure,electrode))\n- Best Model Applied to Held-Out Patient (Nested CV) -",
    )

    # Figure 2 : rang premier SOZ
    df_first = compute_first_soz_rank_and_counts(df_best, root)
    plot_first_soz_rank_barplot(
        df_first,
        color_map,
        out_dir,
        title="Per-Patient First SOZ Channel Detection Rank (Median ± IQR)\nwith Total Implanted Electrode Count and Number of Evaluated Seizures\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome",
    )

    # Figure 3 : rang toutes SOZ
    df_all_soz = compute_all_soz_rank_and_counts(df_best, root)
    plot_all_soz_rank_barplot(
        df_all_soz,
        color_map,
        out_dir,
        title="Per-Patient All SOZ Channel Detection Rank (Median ± IQR)\nwith Total Implanted Electrode, SOZ Electrode Counts and Number of Evaluated Seizures\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome",
    )

    plot_soz_rank_mix_first_and_all(
        df_first=df_first,
        df_all=df_all_soz,
        color_map=color_map,
        out_dir=out_dir,
        title="Per-Patient SOZ Detection Ranks (First vs All)\nMedian ± IQR with Total & SOZ Electrode Counts\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome",
    )

    make_boxplot_topk_metrics(
        df_best,
        out_dir,
        color_map,
        title="Per-Patient Top-k SOZ Metrics\n- Best Model Applied to Held-Out Patient (Nested CV) -\nGOOD Surgery Outcome",
    )

    plot_soz_rank_boxplot_points_colored_by_seizure_from_df_all(
        root=root,
        df_best=df_best,
        out_dir=out_dir,
        color_map=color_map
    )

    print("[INFO] Terminé (figures ranking).")

    # ------------------------------------------------------
    # BCE onset (depuis .npz) pour les meilleurs modèles
    # ------------------------------------------------------
    df_bce_all = collect_bce_from_npz_for_best_models(root, df_best)
    df_bce = keep_middle_seq_per_seizure(df_bce_all)

    bce_csv = out_dir / "timewise_onset_bce_from_npz_middle_seq_only.csv"
    df_bce.to_csv(bce_csv, index=False)
    print(f"[OK] BCE (middle seq only) sauvegardée -> {bce_csv}")

    plot_bce_boxplot_per_patient(df_bce, out_dir, color_map)

    plot_top10_scores_distribution_per_patient(
        df_best=df_best,
        root=root,
        out_dir=out_dir,
        color_map=color_map,
        frac=0.10,
        title="Per-Patient Distribution of Electrode Scores in the Top 10%\n- Best Model Applied to Held-Out Patient (Nested CV) -",
    )


if __name__ == "__main__":
    main()