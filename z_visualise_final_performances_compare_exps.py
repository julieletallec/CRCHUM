#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare plusieurs big_exp_name côte à côte, avec labels personnalisés.

Pour chaque big_exp_name :
  - parcourt results_*/<big_exp_name>/exp*/cv_fold_metrics.json
  - sélectionne le meilleur couple (results_dir, experiment) par patient selon --criterion et --mode
  - extrait les métriques:
      mean_score_pos_global, mean_score_neg_global, confidence_gap_global
      Precision@top_10pct, Recall@top_10pct, Precision@top_20pct, Recall@top_20pct, AUC_group_mean
  - lit selected_features_GLOBAL.txt dans le meilleur exp_dir et agrège les fréquences

Figures comparatives (côte à côte par exp) :
  - Boxplots top-k metrics (AUC/Precision/Recall…) groupés par métrique
  - Boxplots confidence (pos/neg/gap) groupés par métrique
  - Barplot features sélectionnées (x=feature, barres groupées par exp)

Exemple :
uv run z_visualise_final_performances_compare_exps.py \
  --root /home/julieletallec/test/results_grid_search_kwta_20_10_burst \
  --out figures_compare_calib_bestconfidencegap \
  --criterion  confidence_gap_global\
  --mode max \
  --exp classifier_experiments_augmented_balanced_nozeroval=Baseline \
  --exp classifier_experiments_augmented_balanced_LASSO_global_0.03=LASSO_0.03 \
  --exp classifier_experiments_augmented_balanced_LASSO_global_0.03_postprocess=LASSO_0.03_calib \


"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Utils
# -----------------------------

def find_results_dirs(root: Path) -> List[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("results_"))


def build_exp_color_map(exp_labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
    palette = sns.color_palette("tab10", max(3, len(exp_labels)))
    return {lab: palette[i % len(palette)] for i, lab in enumerate(exp_labels)}


def patient_has_min_soz_electrodes_per_seizure(
    root: Path,
    big_exp_name: str,
    results_dir_name: str,
    exp_dir_name: str,
    patient: str,
    min_soz_electrodes: int = 2,
    count_unique_node_index: bool = True,
) -> bool:
    pred_path = (
        root / results_dir_name / big_exp_name / exp_dir_name / "cv_val_predictions_ranked.csv"
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

    if count_unique_node_index and "node_index" in df_soz.columns:
        counts = df_soz.groupby("seizure_id")["node_index"].nunique()
    else:
        counts = df_soz.groupby("seizure_id").size()

    all_seizures = df_pat["seizure_id"].unique()
    counts = counts.reindex(all_seizures, fill_value=0)

    return bool((counts >= min_soz_electrodes).all())


def collect_best_model_per_patient(
    root: Path,
    big_exp_name: str,
    criterion: str = "confidence_gap_global",
    mode: str = "max",
    require_min_soz_per_seizure: int | None = 2,
    count_unique_node_index: bool = True,
) -> pd.DataFrame:
    assert mode in ("max", "min")

    best_per_patient = {}  # patient -> dict

    for results_dir in find_results_dirs(root):
        clf_root = results_dir / big_exp_name
        if not clf_root.is_dir():
            continue

        print(f"[SCAN] {results_dir.name} / {big_exp_name}")

        for exp_dir in sorted(clf_root.iterdir()):  # stable
            if not exp_dir.is_dir():
                continue

            metrics_path = exp_dir / "cv_fold_metrics.json"
            if not metrics_path.is_file():
                continue

            try:
                folds_metrics = json.loads(metrics_path.read_text())
            except Exception as e:
                print(f"  [WARN] cannot read {metrics_path}: {e}")
                continue

            for fold_entry in folds_metrics:
                score = fold_entry.get(criterion, None)
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    continue

                val_pats = str(fold_entry.get("val_patients", "")).split(",")
                mean_pos = fold_entry.get("mean_score_pos_global", None)
                mean_neg = fold_entry.get("mean_score_neg_global", None)
                conf_gap = fold_entry.get("confidence_gap_global", None)

                for pat in val_pats:
                    pat = pat.strip()
                    if not pat:
                        continue

                    # contrainte SOZ (identique à avant)
                    if require_min_soz_per_seizure is not None:
                        ok = patient_has_min_soz_electrodes_per_seizure(
                            root=root,
                            big_exp_name=big_exp_name,
                            results_dir_name=results_dir.name,
                            exp_dir_name=exp_dir.name,
                            patient=pat,
                            min_soz_electrodes=require_min_soz_per_seizure,
                            count_unique_node_index=count_unique_node_index,
                        )
                        if not ok:
                            continue

                    current_best = best_per_patient.get(pat)

                    better = False
                    if current_best is None:
                        better = True
                    elif mode == "max" and float(score) > current_best["criterion_value"]:
                        better = True
                    elif mode == "min" and float(score) < current_best["criterion_value"]:
                        better = True
                    # tie-break stable: si égalité parfaite, on garde le 1er (comportement historique)
                    # -> donc on NE CHANGE RIEN ici.

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
            f"Aucun patient trouvé pour criterion={criterion} big_exp_name={big_exp_name}. "
            "Vérifie cv_fold_metrics.json et la contrainte SOZ."
        )

    df = pd.DataFrame(list(best_per_patient.values())).sort_values("patient").reset_index(drop=True)
    return df


# -----------------------------
# 2) Features
# -----------------------------

def collect_selected_features_for_best_models(
    root: Path,
    df_best_all: pd.DataFrame,
    features_filename: str = "selected_features_GLOBAL.txt",
) -> pd.DataFrame:
    """
    DF long: exp_label, patient, feature
    """
    rows = []
    for _, row in df_best_all.iterrows():
        exp_label = str(row["exp_label"])
        big_exp_name = str(row["big_exp_name"])
        pat = str(row["patient"])
        res_dir = str(row["results_dir"])
        exp_dir = str(row["experiment"])

        feat_path = root / res_dir / big_exp_name / exp_dir / features_filename
        if not feat_path.is_file():
            print(f"[WARN] Missing {features_filename} for {pat} in {exp_label}: {feat_path}")
            continue

        try:
            txt = feat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Cannot read {feat_path}: {e}")
            continue

        feats = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        for f in feats:
            rows.append({"exp_label": exp_label, "patient": pat, "feature": f})

    if not rows:
        raise RuntimeError("No selected features collected.")
    return pd.DataFrame(rows)


# -----------------------------
# 3) Boxplots comparatifs sans points patients
# -----------------------------

from matplotlib.colors import to_rgba

from matplotlib.colors import to_rgba

def grouped_boxplot(
    df_long,
    metric_order,
    exp_order,
    exp_color_map,
    title,
    ylabel,
    out_path,
    ylim=None,
    box_alpha=0.55,
):
    dfp = df_long.dropna(subset=["value"]).copy()
    dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
    dfp = dfp.dropna(subset=["value"])
    if dfp.empty:
        print(f"[WARN] Skipped (no data): {out_path.name}")
        return

    palette = {lab: exp_color_map.get(lab, (0.3, 0.3, 0.3)) for lab in exp_order}

    plt.figure(figsize=(max(12, 1.35 * len(metric_order)), 6))
    ax = sns.boxplot(
        data=dfp,
        x="metric_name",
        y="value",
        hue="exp_label",
        order=metric_order,
        hue_order=exp_order,
        showfliers=False,
        palette=palette,     # seaborn gère le mapping hue->couleur correctement
        linewidth=1.8,
        saturation=1.0,
    )

    # --- Ajuste style SANS casser le mapping hue/couleur ---
    # seaborn peut mettre les boxes dans ax.patches (souvent) plutôt que ax.artists
    patches = list(ax.patches) if hasattr(ax, "patches") else []
    if not patches and hasattr(ax, "artists"):
        patches = list(ax.artists)

    for p in patches:
        fc = p.get_facecolor()  # RGBA déjà correcte (assignée par seaborn)
        # on garde la même couleur, on ajuste seulement alpha + contour
        p.set_facecolor((fc[0], fc[1], fc[2], box_alpha))
        p.set_edgecolor((fc[0], fc[1], fc[2], 1.0))
        p.set_linewidth(2.0)

    # Recolorer whiskers/caps/median = plus tricky.
    # Version robuste : on les laisse en noir (sinon risque de mismatch).
    # Si tu veux absolument les recolorer, je peux te donner une variante
    # qui associe chaque ligne au patch via sa position, mais c'est plus long.

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.grid(axis="y", alpha=0.3)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_path}")



def plot_topk_metrics_compare(df_best_all: pd.DataFrame, out_dir: Path, exp_color_map: Dict[str, Tuple[float, float, float]]):
    metrics = [
        "Precision@top_10pct",
        "Recall@top_10pct",
        "Precision@top_20pct",
        "Recall@top_20pct",
        "AUC_group_mean",
    ]
    rows = []
    for m in metrics:
        if m not in df_best_all.columns:
            continue
        sub = df_best_all[["exp_label", m]].copy()
        sub = sub.rename(columns={m: "value"})
        sub["metric_name"] = m
        rows.append(sub)

    if not rows:
        print("[WARN] No top-k metrics -> skipped.")
        return

    df_long = pd.concat(rows, ignore_index=True)
    exp_order = list(df_best_all["exp_label"].drop_duplicates())

    grouped_boxplot(
        df_long=df_long,
        metric_order=metrics,
        exp_order=exp_order,
        exp_color_map=exp_color_map,
        title="Top-k SOZ Metrics (best model per patient) — comparison across experiments",
        ylabel="Metric value",
        out_path=out_dir / "boxplot_topk_metrics_compare.png",
        ylim=(0, 1),
    )

    df_long.to_csv(out_dir / "topk_metrics_compare_long.csv", index=False)


def plot_confidence_metrics_compare(df_best_all: pd.DataFrame, out_dir: Path, exp_color_map: Dict[str, Tuple[float, float, float]]):
    metrics = ["mean_score_pos", "mean_score_neg", "confidence_gap"]

    rows = []
    for m in metrics:
        if m not in df_best_all.columns:
            continue
        sub = df_best_all[["exp_label", m]].copy()
        sub = sub.rename(columns={m: "value"})
        sub["metric_name"] = m
        rows.append(sub)

    if not rows:
        print("[WARN] No confidence metrics -> skipped.")
        return

    df_long = pd.concat(rows, ignore_index=True)
    exp_order = list(df_best_all["exp_label"].drop_duplicates())

    grouped_boxplot(
        df_long=df_long,
        metric_order=metrics,
        exp_order=exp_order,
        exp_color_map=exp_color_map,
        title="Confidence metrics (best model per patient) — comparison across experiments",
        ylabel="Score",
        out_path=out_dir / "boxplot_confidence_compare.png",
        ylim=(0, 1),
    )

    df_long.to_csv(out_dir / "confidence_metrics_compare_long.csv", index=False)


# -----------------------------
# 4) Features compare (barres groupées)
# -----------------------------

def plot_selected_features_compare(
    df_feat_long: pd.DataFrame,
    out_dir: Path,
    exp_order: List[str],
    exp_color_map: Dict[str, Tuple[float, float, float]],
    top_n: int | None = 40,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # dédoublonne patient-feature par exp
    df_u = df_feat_long.drop_duplicates(subset=["exp_label", "patient", "feature"]).copy()

    counts = (
        df_u.groupby(["exp_label", "feature"])["patient"]
            .nunique()
            .reset_index(name="count_patients")
    )

    # top global
    top = (
        counts.groupby("feature")["count_patients"]
              .sum()
              .sort_values(ascending=False)
    )
    if top_n is not None:
        top = top.head(int(top_n))
    features_order = list(top.index)

    counts = counts[counts["feature"].isin(features_order)].copy()

    palette = {k: exp_color_map.get(k, (0.3, 0.3, 0.3)) for k in exp_order}

    plt.figure(figsize=(max(12, 0.45 * len(features_order)), 6))
    sns.barplot(
        data=counts,
        x="feature",
        y="count_patients",
        hue="exp_label",
        order=features_order,
        hue_order=exp_order,
        palette=palette,
        dodge=True,
    )

    plt.title("Selected feature frequency (best model per patient) — comparison across experiments")
    plt.ylabel("Number of patients")
    plt.xlabel("Feature")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=90, ha="right")
    plt.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    out_path = out_dir / "barplot_selected_features_compare.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_path}")

    counts.to_csv(out_dir / "selected_features_compare_counts.csv", index=False)


# -----------------------------
# Parsing CLI: --exp big_exp=label
# -----------------------------

def parse_exp_arg(s: str) -> Tuple[str, str]:
    """
    Parse "big_exp_name=Label"
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            "Format attendu pour --exp: big_exp_name=Label (ex: classifier_xxx=Baseline)"
        )
    big, lab = s.split("=", 1)
    big = big.strip()
    lab = lab.strip()
    if not big or not lab:
        raise argparse.ArgumentTypeError("big_exp_name et Label doivent être non vides.")
    return big, lab


# -----------------------------
# MAIN
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare multiple experiments side-by-side (no patient points).")
    ap.add_argument("--root", required=True, help="Dossier racine contenant les results_*")
    ap.add_argument("--out", default="figures_compare", help="Dossier de sortie")
    ap.add_argument("--criterion", default="AUC_group_mean")
    ap.add_argument("--mode", choices=["max", "min"], default="max")
    ap.add_argument(
        "--exp",
        dest="exps",
        action="append",
        type=parse_exp_arg,
        required=True,
        help="Exp à comparer: big_exp_name=Label (répéter pour en passer plusieurs)",
    )
    ap.add_argument("--require-min-soz-per-seizure", type=int, default=2)
    ap.add_argument("--features-filename", default="selected_features_GLOBAL.txt")
    ap.add_argument("--top-features", type=int, default=40, help="Top features global (0 = toutes)")

    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"Le dossier root n'existe pas: {root}")

    # args.exps : List[Tuple[big_exp_name, label]]
    exp_pairs = args.exps
    if not exp_pairs:
        raise SystemExit("Aucun --exp fourni. Exemple: --exp big_exp_name=LABEL")

    exp_labels = [lab for (_, lab) in exp_pairs]

    print(f"[INFO] Root = {root}")
    print("[INFO] Experiments:")
    for big, lab in exp_pairs:
        print(f"  - {lab}  <=  {big}")

    # colors by exp label
    exp_color_map = build_exp_color_map(exp_labels)

    # collect df_best for each exp
    df_best_list = []
    for big_exp_name, exp_label in exp_pairs:
        df_best = collect_best_model_per_patient(
            root=root,
            big_exp_name=big_exp_name,
            criterion=args.criterion,
            mode=args.mode,
            require_min_soz_per_seizure=args.require_min_soz_per_seizure,
        )

        # IMPORTANT: garder ces 2 colonnes pour features + plots
        df_best["big_exp_name"] = big_exp_name
        df_best["exp_label"] = exp_label

        df_best_list.append(df_best)

        csv_path = out_dir / f"best_per_patient__{exp_label}__{args.criterion}.csv"
        df_best.to_csv(csv_path, index=False)
        print(f"[OK] Saved -> {csv_path}")

    df_best_all = pd.concat(df_best_list, ignore_index=True)
    df_best_all.to_csv(out_dir / "best_per_patient_ALL_EXPS.csv", index=False)

    exp_order = exp_labels  # keep CLI order

    # plots
    plot_topk_metrics_compare(df_best_all, out_dir, exp_color_map)
    plot_confidence_metrics_compare(df_best_all, out_dir, exp_color_map)

    # features
    df_feat_long = collect_selected_features_for_best_models(
        root=root,
        df_best_all=df_best_all,
        features_filename=args.features_filename,
    )
    df_feat_long.to_csv(out_dir / "selected_features_long_all_exps.csv", index=False)

    top_n = None if args.top_features == 0 else int(args.top_features)
    plot_selected_features_compare(df_feat_long, out_dir, exp_order, exp_color_map, top_n=top_n)

    print("[INFO] Done.")



if __name__ == "__main__":
    main()
