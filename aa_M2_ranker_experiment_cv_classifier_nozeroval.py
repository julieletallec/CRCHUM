#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for LGBMClassifier on electrode features, with
Leave-One-Patient-Out cross-validation (LOPO).

Pour chaque expérience dans le config:
  - Faire une CV LOPO (1 patient en validation à chaque fold)
  - Entraîner un LGBMClassifier à chaque fold
  - Agréger les prédictions sur tous les patients
  - Sauvegarder:
      * cv_val_predictions_ranked.csv
      * cv_global_metrics.json
      * cv_fold_metrics.csv / .json
      * cv_metrics_per_patient.csv / .json
      * config_used.yaml (config de base + expérience)

NB: y_score = proba prédite P(SOZ=1 | features),
     utilisée pour le ranking et interprétable comme "confiance positive".
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, log_evaluation, early_stopping


# ------------------------ helpers: io ------------------------ #

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def save_config(cfg: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# ------------------------ helpers: preprocessing ------------------------ #

def per_group_zscore(df: pd.DataFrame, group_keys: List[str], feature_cols: List[str]) -> pd.DataFrame:
    """Z-score features within each (patient, seizure, seq) group."""
    def _z(g):
        g = g.copy()
        vals = g[feature_cols].astype(float)
        mu = vals.mean(axis=0)
        sd = vals.std(axis=0).replace(0, 1.0)
        g[feature_cols] = (vals - mu) / sd
        return g

    return df.groupby(group_keys, group_keys=False).apply(_z).reset_index(drop=True)


def global_standardize(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[feature_cols].astype(float))
    Xva = scaler.transform(val_df[feature_cols].astype(float))
    df_tr = train_df.copy()
    df_va = val_df.copy()
    df_tr[feature_cols] = Xtr
    df_va[feature_cols] = Xva
    return df_tr, df_va, scaler


def filter_train_groups_with_positive(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """Keep only groups that contain at least one SOZ electrode."""
    g = df.groupby(group_keys)["is_SOZ"].sum().reset_index(name="npos")
    keep = g[g["npos"] > 0]
    return df.merge(keep[group_keys], on=group_keys, how="inner")


def build_matrices(
    df: pd.DataFrame,
    group_keys: List[str],
    label_col: str,
    drop_extras: List[str],
):
    """
    Returns:
      X: 2D numpy (n_samples x n_features)
      y: labels (0/1)
      feature_cols: list of feature names
      meta: dataframe with group+electrode info & label
    """
    non_feature = set(group_keys + [label_col, "electrode_name", "node_index"]) | set(drop_extras or [])
    feature_cols = [c for c in df.columns if c not in non_feature]

    df2 = df.copy().reset_index(drop=True)
    X = df2[feature_cols].astype(float).values
    y = df2[label_col].astype(int).values

    meta = df2[group_keys + ["node_index", "electrode_name", label_col]].copy()
    return X, y, feature_cols, meta


# ------------------------ helpers: metrics ------------------------ #
# (inchangé, tes trois versions groupwise_metrics__, _, et groupwise_metrics)
# Je garde seulement la dernière (groupwise_metrics) qui est utilisée dans le code.

from math import comb

def groupwise_metrics(
    df_pred: pd.DataFrame,
    k_list=None,               # laissé pour compatibilité, mais non utilisé ici
    p_list=(0.10, 0.20),       # proportions: 10%, 20%, etc.
) -> Dict[str, Any]:
    """
    df_pred doit contenir :
      - patient, seizure_id, seq_idx
      - y_true  (0/1, 1 = SOZ)
      - y_score (score du modèle, plus haut = plus probable SOZ)

    Retourne un dict avec :
      - Pour chaque p dans p_list :
          * Precision@top_XXpct
          * Recall@top_XXpct
          * SOZ_enrichment@top_XXpct
      - AUC_group_mean
      - num_groups_eval
      - Brier (global)
      - mean_score_pos_global / mean_score_neg_global / confidence_gap_global
      - confidence_gap_group_mean / median / min / max / num_groups_with_gap
        (gaps calculés INTRA-GROUPE puis moyennés)
    """
    gcols = ["patient", "seizure_id", "seq_idx"]

    # ------------------------------------------------------------
    # 1) Métriques top-α% par groupe (comme avant)
    # ------------------------------------------------------------
    metrics: Dict[str, List[float]] = {}

    p_list = tuple(p_list) if p_list is not None else ()
    for p in p_list:
        pct = int(round(p * 100))
        metrics[f"Precision@top_{pct}pct"] = []
        metrics[f"Recall@top_{pct}pct"] = []
        metrics[f"SOZ_enrichment@top_{pct}pct"] = []

    aucs = []
    group_gaps: List[float] = []  # nouveaux gaps (pos−neg) par groupe

    # Boucle sur les groupes (patient, seizure, seq)
    for _, g in df_pred.groupby(gcols):
        g = g.sort_values("y_score", ascending=False)
        rels = g["y_true"].astype(int).tolist()
        y = g["y_true"].values
        scores = g["y_score"].values

        N = len(rels)
        total_pos = sum(rels)
        total_neg = N - total_pos

        has_pos = total_pos > 0
        has_neg = total_neg > 0

        # AUC par groupe (uniquement s'il y a au moins un positif ET un négatif)
        if has_pos and has_neg:
            try:
                aucs.append(roc_auc_score(y, scores))
            except Exception:
                pass

        # ---- nouveau : confidence gap PAR GROUPE ----
        if has_pos and has_neg:
            pos_scores = scores[y == 1]
            neg_scores = scores[y == 0]
            gap = float(np.mean(pos_scores) - np.mean(neg_scores))
            group_gaps.append(gap)

        # --- métriques top-α% intra-groupe --- #
        if N > 0 and len(p_list) > 0:
            base_prev = total_pos / N if N > 0 else 0.0  # prévalence globale SOZ dans le groupe

            for p in p_list:
                pct = int(round(p * 100))
                # nombre d'électrodes dans le top-α%
                k_p = int(np.ceil(p * N))
                k_p = max(1, min(k_p, N))  # au moins 1, au plus N

                tp_p = sum(rels[:k_p])
                fp_p = k_p - tp_p

                # Precision : % de SOZ dans top-α%
                prec_p = tp_p / k_p if k_p > 0 else float("nan")
                metrics[f"Precision@top_{pct}pct"].append(prec_p)

                # Recall : % des SOZ capturés dans top-α%
                if total_pos > 0:
                    rec_p = tp_p / total_pos
                else:
                    # même convention qu'avant : groupe sans SOZ -> 1.0
                    rec_p = 1.0
                metrics[f"Recall@top_{pct}pct"].append(rec_p)

                # Enrichment : (prévalence SOZ dans top-α%) / (prévalence globale SOZ)
                if base_prev > 0 and not np.isnan(prec_p):
                    enrich_p = prec_p / base_prev
                else:
                    enrich_p = float("nan")
                metrics[f"SOZ_enrichment@top_{pct}pct"].append(enrich_p)

    # ------------------------------------------------------------
    # 2) Agrégation des métriques top-α%
    # ------------------------------------------------------------
    out: Dict[str, Any] = {}
    for m, vals in metrics.items():
        if len(vals):
            out[m] = float(np.nanmean(vals))
        else:
            out[m] = float("nan")

    out["AUC_group_mean"] = float(np.mean(aucs)) if len(aucs) else float("nan")
    out["num_groups_eval"] = int(df_pred.groupby(gcols).ngroups)

    # ------------------------------------------------------------
    # 3) Métriques globales de calibration / confiance
    #    (version globale comme avant, sur TOUTES les électrodes)
    # ------------------------------------------------------------
    y_all = df_pred["y_true"].astype(int).values
    s_all = df_pred["y_score"].astype(float).values

    if len(y_all) > 0:
        # Brier score global
        out["Brier"] = float(np.mean((y_all - s_all) ** 2))
    else:
        out["Brier"] = float("nan")

    pos_scores_all = s_all[y_all == 1]
    neg_scores_all = s_all[y_all == 0]

    if len(pos_scores_all) > 0:
        out["mean_score_pos_global"] = float(np.mean(pos_scores_all))
    else:
        out["mean_score_pos_global"] = float("nan")

    if len(neg_scores_all) > 0:
        out["mean_score_neg_global"] = float(np.mean(neg_scores_all))
    else:
        out["mean_score_neg_global"] = float("nan")

    if (
        not np.isnan(out["mean_score_pos_global"])
        and not np.isnan(out["mean_score_neg_global"])
    ):
        out["confidence_gap_global"] = float(
            out["mean_score_pos_global"] - out["mean_score_neg_global"]
        )
    else:
        out["confidence_gap_global"] = float("nan")

    # ------------------------------------------------------------
    # 4) Confidence gap MOYENNÉ PAR GROUPE  (ce qu'on veut vraiment)
    # ------------------------------------------------------------
    if len(group_gaps) > 0:
        gaps = np.array(group_gaps, dtype=float)
        out["confidence_gap_group_mean"] = float(np.nanmean(gaps))
        out["confidence_gap_group_median"] = float(np.nanmedian(gaps))
        out["confidence_gap_group_min"] = float(np.nanmin(gaps))
        out["confidence_gap_group_max"] = float(np.nanmax(gaps))
        out["confidence_gap_group_num_groups"] = int(len(gaps))
    else:
        out["confidence_gap_group_mean"] = float("nan")
        out["confidence_gap_group_median"] = float("nan")
        out["confidence_gap_group_min"] = None
        out["confidence_gap_group_max"] = None
        out["confidence_gap_group_num_groups"] = 0

    return out

def groupwise_metrics_(
    df_pred: pd.DataFrame,
    k_list=(1, 3, 5, 10, 20),
    p_list=(0.10, 0.20),  # proportions (10%, 20%)
):
    """
    df_pred doit contenir :
      - patient, seizure_id, seq_idx
      - y_true  (0/1, 1 = SOZ)
      - y_score (score du modèle, plus haut = plus probable SOZ)

    Retourne un dict avec :
      - NDCG@k, MAP@k, Recall@k
      - Hit@k, Precision@k, FPR@k, Specificity@k
      - Hit@k_corrected, Precision@k_corrected, FPR@k_corrected, Specificity@k_corrected
      - FirstSOZ_* (stats rang)
      - Hit/Precision/Recall/Enrichment pour top-α% (α dans p_list)
    """
    def dcg(rels):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))

    def ndcg_at_k(rels, k):
        k = min(k, len(rels))
        rels_k = rels[:k]
        ideal = sorted(rels, reverse=True)[:k]
        denom = dcg(ideal)
        return (dcg(rels_k) / denom) if denom > 0 else 0.0

    def ap_at_k(rels, k):
        k = min(k, len(rels))
        hits = 0
        s = 0.0
        for i in range(k):
            if rels[i] == 1:
                hits += 1
                s += hits / (i + 1)
        tot_pos = sum(rels[:k])
        return s / max(1, tot_pos)

    def recall_at_k(rels, k):
        tot = sum(rels)
        if tot == 0:
            # convention: groupe sans SOZ -> 1.0
            return 1.0
        k = min(k, len(rels))
        return sum(rels[:k]) / tot

    gcols = ["patient", "seizure_id", "seq_idx"]
    metrics: Dict[str, List[float]] = {}

    # top-k fixes
    for k in k_list:
        metrics[f"NDCG@{k}"] = []
        metrics[f"MAP@{k}"] = []
        metrics[f"Recall@{k}"] = []
        metrics[f"Hit@{k}"] = []
        metrics[f"Precision@{k}"] = []
        metrics[f"FPR@{k}"] = []
        metrics[f"Specificity@{k}"] = []

        metrics[f"Hit@{k}_corrected"] = []
        metrics[f"Precision@{k}_corrected"] = []
        metrics[f"FPR@{k}_corrected"] = []
        metrics[f"Specificity@{k}_corrected"] = []

    # top-% (proportions)
    p_list = tuple(p_list) if p_list is not None else ()
    for p in p_list:
        pct = int(round(p * 100))
        metrics[f"Hit@top_{pct}pct"] = []
        metrics[f"Precision@top_{pct}pct"] = []
        metrics[f"Recall@top_{pct}pct"] = []
        metrics[f"SOZ_enrichment@top_{pct}pct"] = []

    aucs = []
    first_soz_ranks = []

    for _, g in df_pred.groupby(gcols):
        g = g.sort_values("y_score", ascending=False)
        rels = g["y_true"].astype(int).tolist()
        y = g["y_true"].values
        scores = g["y_score"].values

        N = len(rels)
        total_pos = sum(rels)
        total_neg = N - total_pos

        has_pos = total_pos > 0
        has_neg = total_neg > 0

        if has_pos and has_neg:
            try:
                aucs.append(roc_auc_score(y, scores))
            except Exception:
                pass

        # top-k
        for k in k_list:
            k_eff = min(k, N)

            metrics[f"NDCG@{k}"].append(ndcg_at_k(rels, k))
            metrics[f"MAP@{k}"].append(ap_at_k(rels, k))
            metrics[f"Recall@{k}"].append(recall_at_k(rels, k))

            if k_eff == 0:
                continue

            tp_k = sum(rels[:k_eff])
            fp_k = k_eff - tp_k

            hit_k = 1.0 if tp_k > 0 else 0.0
            metrics[f"Hit@{k}"].append(hit_k)

            prec_k = tp_k / k_eff
            metrics[f"Precision@{k}"].append(prec_k)

            if total_neg > 0:
                fpr_k = fp_k / total_neg
                tn_k = total_neg - fp_k
                spec_k = tn_k / total_neg
            else:
                fpr_k = float("nan")
                spec_k = float("nan")

            metrics[f"FPR@{k}"].append(fpr_k)
            metrics[f"Specificity@{k}"].append(spec_k)

            # corrections vs hasard
            if total_pos > 0 and k_eff > 0:
                if k_eff > total_neg:
                    p_zero_pos = 0.0
                else:
                    try:
                        p_zero_pos = comb(total_neg, k_eff) / comb(N, k_eff)
                    except ValueError:
                        p_zero_pos = 0.0
                hit_exp = 1.0 - p_zero_pos
            else:
                hit_exp = 0.0

            if hit_exp < 1.0:
                hit_corr = (hit_k - hit_exp) / (1.0 - hit_exp) if (1.0 - hit_exp) > 0 else float("nan")
            else:
                hit_corr = float("nan")
            metrics[f"Hit@{k}_corrected"].append(hit_corr)

            if N > 0:
                prec_exp = total_pos / N
            else:
                prec_exp = 0.0

            if prec_exp < 1.0:
                prec_corr = (prec_k - prec_exp) / (1.0 - prec_exp) if (1.0 - prec_exp) > 0 else float("nan")
            else:
                prec_corr = float("nan")
            metrics[f"Precision@{k}_corrected"].append(prec_corr)

            if total_neg > 0 and N > 0:
                fpr_exp = k_eff / N
                if fpr_exp < 1.0:
                    fpr_corr = (fpr_k - fpr_exp) / (1.0 - fpr_exp) if (1.0 - fpr_exp) > 0 else float("nan")
                else:
                    fpr_corr = float("nan")

                if not np.isnan(fpr_corr):
                    spec_corr = 1.0 - fpr_corr
                else:
                    spec_corr = float("nan")
            else:
                fpr_corr = float("nan")
                spec_corr = float("nan")

            metrics[f"FPR@{k}_corrected"].append(fpr_corr)
            metrics[f"Specificity@{k}_corrected"].append(spec_corr)

        # top-% (proportionnel)
        if N > 0 and len(p_list) > 0:
            base_prev = total_pos / N if N > 0 else 0.0
            for p in p_list:
                pct = int(round(p * 100))
                k_p = int(np.ceil(p * N))
                k_p = max(1, min(k_p, N))

                tp_p = sum(rels[:k_p])
                fp_p = k_p - tp_p

                hit_p = 1.0 if tp_p > 0 else 0.0
                metrics[f"Hit@top_{pct}pct"].append(hit_p)

                prec_p = tp_p / k_p if k_p > 0 else float("nan")
                metrics[f"Precision@top_{pct}pct"].append(prec_p)

                if total_pos > 0:
                    rec_p = tp_p / total_pos
                else:
                    rec_p = 1.0
                metrics[f"Recall@top_{pct}pct"].append(rec_p)

                if base_prev > 0 and not np.isnan(prec_p):
                    enrich_p = prec_p / base_prev
                else:
                    enrich_p = float("nan")
                metrics[f"SOZ_enrichment@top_{pct}pct"].append(enrich_p)

        # rang du premier SOZ
        if has_pos:
            first_idx = np.argmax(rels)
            first_soz_ranks.append(first_idx + 1)

    out: Dict[str, Any] = {}
    for m, vals in metrics.items():
        if len(vals):
            out[m] = float(np.nanmean(vals))
        else:
            out[m] = float("nan")

    out["AUC_group_mean"] = float(np.mean(aucs)) if len(aucs) else float("nan")
    out["num_groups_eval"] = int(df_pred.groupby(gcols).ngroups)

    if first_soz_ranks:
        first_soz_ranks = np.array(first_soz_ranks, dtype=float)
        out["FirstSOZ_rank_mean"] = float(np.mean(first_soz_ranks))
        out["FirstSOZ_rank_median"] = float(np.median(first_soz_ranks))
        out["FirstSOZ_rank_min"] = int(np.min(first_soz_ranks))
        out["FirstSOZ_rank_max"] = int(np.max(first_soz_ranks))
        out["FirstSOZ_num_groups"] = int(len(first_soz_ranks))
    else:
        out["FirstSOZ_rank_mean"] = float("nan")
        out["FirstSOZ_rank_median"] = float("nan")
        out["FirstSOZ_rank_min"] = None
        out["FirstSOZ_rank_max"] = None
        out["FirstSOZ_num_groups"] = 0

    return out


# ------------------------ main experiment loop (LOPO) ------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Run multiple LGBMClassifier experiments with LOPO CV.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    features_path = Path(cfg["features_path"]).expanduser()

    # Par défaut: dossier "experiments_cv" à côté du fichier de features
    output_root = Path(cfg.get("output_root", features_path.parent / "experiments_cv")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    group_keys = cfg.get("group_keys", ["patient", "seizure_id", "seq_idx"])
    label_col = cfg.get("label_col", "is_SOZ")

    random_state = int(cfg.get("random_state", 42))
    eval_at = list(cfg.get("eval_at", [3, 5, 10, 20]))

    base_lgbm_params = cfg.get("base_lgbm_params", {})
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise SystemExit("No experiments defined under 'experiments' in config.")

    print(f"[INFO] Loading features from {features_path}")
    df = load_table(features_path)

    drop_extras = cfg.get("drop_extra_cols", [])

    # si is_flat_zero existe, on l'exclut des features (mais on peut l'utiliser pour filtrer)
    if "is_flat_zero" in df.columns and "is_flat_zero" not in drop_extras:
        drop_extras.append("is_flat_zero")

    # Basic checks
    needed = set(group_keys + [label_col, "node_index", "electrode_name"])
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in features table: {missing}")

    # Feature columns (commun à tous les expés)
    non_feature = set(group_keys + [label_col, "electrode_name", "node_index"]) | set(drop_extras or [])
    feature_cols = [c for c in df.columns if c not in non_feature]

    # LOPO CV sur les patients
    patients_all = df["patient"].astype(str).values
    unique_pats = np.unique(patients_all)
    logo = LeaveOneGroupOut()

    print(f"[INFO] LOPO CV over {len(unique_pats)} patients: {unique_pats.tolist()}")

    # Run each experiment with LOPO
    for exp in experiments:
        name = exp["name"]
        print("\n" + "=" * 30)
        print(f"[EXP] {name} (LOPO CV)")
        print("=" * 30)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = output_root / f"{name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # merge base config + experiment for trace
        merged_cfg = dict(cfg)
        merged_cfg["experiment"] = exp
        save_config(merged_cfg, exp_dir)

        # options d'expérience
        per_group_standardize = bool(exp.get("per_group_standardize", True))
        global_standardize_flag = bool(exp.get("global_standardize", False))
        train_only_pos = bool(exp.get("train_only_groups_with_positive", True))

        # hyperparams LightGBM
        params = dict(base_lgbm_params)
        params.update(exp.get("lgbm_params", {}))
        # enlever d'éventuels paramètres spécifiques au ranker
        params.pop("lambdarank_truncation_level", None)

        es_rounds = exp.get("early_stopping_rounds", cfg.get("early_stopping_rounds", 80))
        log_period = exp.get("log_period", cfg.get("log_period", 50))

        all_eval_rows = []
        fold_metrics = []

        for fold_idx, (tr_idx, va_idx) in enumerate(logo.split(df, groups=patients_all), start=1):
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[va_idx].copy()
            val_pats = sorted(df_va["patient"].astype(str).unique())
            print(f"\n[EXP:{name}] Fold {fold_idx} | val patients: {val_pats}")

            # filter groups without any SOZ in train
            if train_only_pos:
                before = len(df_tr)
                df_tr = filter_train_groups_with_positive(df_tr, group_keys)
                print(f"[EXP:{name}] train rows (>=1 SOZ): {before} -> {len(df_tr)}")

            # enlever du training les électrodes SOZ qui sont full zéro
            if "is_flat_zero" in df_tr.columns:
                before = len(df_tr)
                df_tr = df_tr[~((df_tr["is_SOZ"] == 1) & (df_tr["is_flat_zero"] == True))].copy()
                removed = before - len(df_tr)
                print(f"[EXP:{name}] Fold {fold_idx}: removed {removed} flat-zero SOZ rows from TRAIN")
            
            
            
            # enlever du TEST les électrodes SOZ qui sont flat-zero
            if "is_flat_zero" in df_va.columns:
                before = len(df_va)
                df_va = df_va[~((df_va["is_SOZ"] == 1) & (df_va["is_flat_zero"] == True))].copy()
                removed = before - len(df_va)
                if removed > 0:
                    print(
                        f"[EXP:{name}] Fold {fold_idx}: removed {removed} flat-zero SOZ rows from VAL"
                    )




            # per-group z-score
            if per_group_standardize:
                df_tr = per_group_zscore(df_tr, group_keys, feature_cols)
                df_va = per_group_zscore(df_va, group_keys, feature_cols)
                print(f"[EXP:{name}] Fold {fold_idx}: per-group z-score")

            # global standardization
            if global_standardize_flag:
                df_tr, df_va, _ = global_standardize(df_tr, df_va, feature_cols)
                print(f"[EXP:{name}] Fold {fold_idx}: global standardization")

            # build matrices
            Xtr, ytr, featcols, meta_tr = build_matrices(
                df_tr, group_keys, label_col, drop_extras
            )
            Xva, yva, _, meta_va = build_matrices(
                df_va, group_keys, label_col, drop_extras
            )

            clf = LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",   # 👈 AJOUT TRÈS IMPORTANT
                    random_state=random_state,
                    **params,
                )


            callbacks = []
            if es_rounds and es_rounds > 0:
                callbacks.append(early_stopping(stopping_rounds=int(es_rounds)))
            if log_period and log_period > 0:
                callbacks.append(log_evaluation(period=int(log_period)))

            print(f"[EXP:{name}] Fold {fold_idx}: training (classifier)...")
            clf.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                callbacks=callbacks,
            )

            # proba d'être SOZ (classe 1)
            y_score = clf.predict_proba(Xva)[:, 1]
            df_fold = meta_va.copy()
            df_fold["y_score"] = y_score
            df_fold["fold"] = fold_idx
            df_fold["val_patient"] = df_fold["patient"]
            all_eval_rows.append(df_fold)

            eval_df_fold = df_fold[group_keys + ["is_SOZ", "y_score"]].rename(
                columns={"is_SOZ": "y_true"}
            )
            m_fold = groupwise_metrics(eval_df_fold, k_list=tuple(eval_at))
            m_fold["fold"] = fold_idx
            m_fold["val_patients"] = ",".join(val_pats)
            fold_metrics.append(m_fold)

            print(f"[EXP:{name}] Fold {fold_idx} metrics:")
            for k, v in m_fold.items():
                if k in ("fold", "val_patients"):
                    continue
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # ---- aggregate across all folds / patients ---- #
        all_eval_df = pd.concat(all_eval_rows, ignore_index=True)
        all_eval_df = all_eval_df.sort_values(
            group_keys + ["y_score"],
            ascending=[True, True, True, False]
        )
        all_eval_df["rank"] = all_eval_df.groupby(group_keys)["y_score"].rank(
            method="first", ascending=False
        ).astype(int)

        global_eval_df = all_eval_df[group_keys + ["is_SOZ", "y_score"]].rename(
            columns={"is_SOZ": "y_true"}
        )
        global_metrics = groupwise_metrics(global_eval_df, k_list=tuple(eval_at))

        # per-patient metrics
        per_patient_rows = []
        per_patient_dict = {}
        for pat, df_pat in global_eval_df.groupby("patient"):
            m_pat = groupwise_metrics(df_pat, k_list=tuple(eval_at))
            m_pat["patient"] = pat
            per_patient_rows.append(m_pat)
            per_patient_dict[str(pat)] = m_pat

        # ---- save all outputs ---- #
        all_eval_df.to_csv(exp_dir / "cv_val_predictions_ranked.csv", index=False)

        pd.DataFrame(fold_metrics).to_csv(exp_dir / "cv_fold_metrics.csv", index=False)
        with open(exp_dir / "cv_fold_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        df_pat = pd.DataFrame(per_patient_rows)
        df_pat.to_csv(exp_dir / "cv_metrics_per_patient.csv", index=False)
        with open(exp_dir / "cv_metrics_per_patient.json", "w") as f:
            json.dump(per_patient_dict, f, indent=2)

        with open(exp_dir / "cv_global_metrics.json", "w") as f:
            json.dump(global_metrics, f, indent=2)

        print(f"\n[EXP:{name}] GLOBAL CV metrics:")
        for k, v in global_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"[EXP:{name}] CV outputs saved to {exp_dir}")

    print("\n[INFO] All experiments (LOPO CV) done.")


if __name__ == "__main__":
    main()
