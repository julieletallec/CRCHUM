#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for LGBMClassifier on electrode features, with
Leave-One-Patient-Out cross-validation (LOPO).

NOUVEAU (à la place de Platt calibration):
  - "Logit shift" post-processing: p = sigmoid(raw + delta)
  - delta est choisi par fold sur un split de calibration pris DANS le TRAIN seulement,
    pour booster les positifs tout en gardant les négatifs bas (contrainte sur mean(neg)).

Toujours présent:
  - Feature selection LASSO (LogisticRegression L1) UNE SEULE FOIS PAR EXPÉRIENCE
    (fit sur tout le dataset), puis appliquée à tous les folds LOPO.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

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


def _clean_feature_matrix_for_lasso(
    df_any: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Nettoyage minimal pour éviter que LASSO casse:
      - Remplacer inf/-inf par NaN
      - Imputer NaN avec la médiane (sur df_any)
      - Retirer les colonnes constantes (sur df_any)
    Retourne:
      df_clean, feature_cols_clean, medians (pour réutiliser ensuite sur folds)
    """
    df2 = df_any.copy()

    df2[feature_cols] = df2[feature_cols].replace([np.inf, -np.inf], np.nan)

    med = df2[feature_cols].median(axis=0, numeric_only=True)
    df2[feature_cols] = df2[feature_cols].fillna(med)

    nunique = df2[feature_cols].nunique(dropna=False)
    keep_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]
    if len(keep_cols) == 0:
        keep_cols = list(feature_cols)

    drop_cols = [c for c in feature_cols if c not in keep_cols]
    if drop_cols:
        df2 = df2.drop(columns=drop_cols)

    return df2, keep_cols, med


def lasso_feature_select_global(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    y_col: str = "is_SOZ",
    C: float = 0.05,
    max_features: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[List[str], pd.Series]:
    """
    Fit une LogisticRegression L1 sur TOUT le dataset pour choisir 1 set de features.
    Retourne:
      selected_cols, medians_used (pour imputer les folds pareil)
    """
    df_clean, feature_cols2, med = _clean_feature_matrix_for_lasso(df_all, feature_cols)

    X = df_clean[feature_cols2].astype(float).values
    y = df_clean[y_col].astype(int).values

    base = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=float(C),
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        random_state=random_state,
    )
    base.fit(X, y)

    selector = SelectFromModel(base, prefit=True, threshold=1e-12)
    mask = selector.get_support()
    selected = [c for c, keep in zip(feature_cols2, mask) if keep]

    if len(selected) == 0:
        selected = list(feature_cols2)

    if max_features is not None and len(selected) > int(max_features):
        coefs = np.abs(base.coef_.ravel())
        ranked = sorted(
            [(feature_cols2[i], coefs[i]) for i in range(len(feature_cols2)) if mask[i]],
            key=lambda x: x[1],
            reverse=True,
        )
        selected = [name for name, _ in ranked[: int(max_features)]]

    return selected, med


def apply_global_impute_and_select(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    feature_cols_all: List[str],
    selected_cols: List[str],
    medians: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Applique le même nettoyage que pour LASSO:
      - replace inf
      - impute NaN avec medians (appris globalement)
      - garde seulement selected_cols
    """
    df_tr2 = df_tr.copy()
    df_va2 = df_va.copy()

    df_tr2[feature_cols_all] = df_tr2[feature_cols_all].replace([np.inf, -np.inf], np.nan)
    df_va2[feature_cols_all] = df_va2[feature_cols_all].replace([np.inf, -np.inf], np.nan)

    df_tr2[feature_cols_all] = df_tr2[feature_cols_all].fillna(medians)
    df_va2[feature_cols_all] = df_va2[feature_cols_all].fillna(medians)

    keep = list(selected_cols)
    drop_cols = [c for c in feature_cols_all if c not in keep and c in df_tr2.columns]
    if drop_cols:
        df_tr2 = df_tr2.drop(columns=drop_cols, errors="ignore")
        df_va2 = df_va2.drop(columns=drop_cols, errors="ignore")

    return df_tr2, df_va2, keep


def build_matrices(
    df: pd.DataFrame,
    group_keys: List[str],
    label_col: str,
    drop_extras: List[str],
):
    non_feature = set(group_keys + [label_col, "electrode_name", "node_index"]) | set(drop_extras or [])
    feature_cols = [c for c in df.columns if c not in non_feature]

    df2 = df.copy().reset_index(drop=True)
    X = df2[feature_cols].astype(float).values
    y = df2[label_col].astype(int).values

    meta = df2[group_keys + ["node_index", "electrode_name", label_col]].copy()
    return X, y, feature_cols, meta


# ------------------------ helpers: postprocess (logit shift) ------------------------ #

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def choose_delta_logit_shift(
    raw_scores: np.ndarray,
    y: np.ndarray,
    delta_grid: np.ndarray,
    neg_mean_cap: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """
    Choisit delta pour p = sigmoid(raw + delta) afin de :
      - PRIORITÉ: maximiser mean(p|y=1)
      - sous contrainte mean(p|y=0) <= neg_mean_cap si possible.

    Si aucune valeur ne respecte la contrainte:
      - on minimise d'abord la violation (mean_neg - cap, tronquée à 0),
      - puis on maximise mean_pos (tie-break: minimise mean_neg).
    """
    y = y.astype(int)
    pos_mask = y == 1
    neg_mask = y == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0, {
            "ok": 0.0,
            "reason": 1.0,
            "best_mean_pos": float("nan"),
            "best_mean_neg": float("nan"),
            "best_violation": float("nan"),
        }

    candidates = []
    for d in delta_grid:
        p = sigmoid(raw_scores + float(d))
        mean_pos = float(np.mean(p[pos_mask]))
        mean_neg = float(np.mean(p[neg_mask]))
        violation = max(0.0, mean_neg - float(neg_mean_cap))
        candidates.append((float(d), mean_pos, mean_neg, violation))

    feasible = [c for c in candidates if c[2] <= neg_mean_cap]

    if feasible:
        # faisable: MAX mean_pos, tie-break MIN mean_neg
        d, mp, mn, vio = max(feasible, key=lambda t: (t[1], -t[2]))
        return d, {
            "ok": 1.0,
            "feasible": 1.0,
            "best_mean_pos": float(mp),
            "best_mean_neg": float(mn),
            "best_violation": float(vio),
        }

    # infaisable: MIN violation, puis MAX mean_pos, tie-break MIN mean_neg
    min_vio = min(candidates, key=lambda t: t[3])[3]
    near = [c for c in candidates if abs(c[3] - min_vio) <= 1e-12]
    d, mp, mn, vio = max(near, key=lambda t: (t[1], -t[2]))
    return d, {
        "ok": 1.0,
        "feasible": 0.0,
        "best_mean_pos": float(mp),
        "best_mean_neg": float(mn),
        "best_violation": float(vio),
    }



# ------------------------ helpers: metrics ------------------------ #

def groupwise_metrics(
    df_pred: pd.DataFrame,
    k_list=None,
    p_list=(0.10, 0.20),
) -> Dict[str, Any]:
    gcols = ["patient", "seizure_id", "seq_idx"]

    metrics: Dict[str, List[float]] = {}
    p_list = tuple(p_list) if p_list is not None else ()
    for p in p_list:
        pct = int(round(p * 100))
        metrics[f"Precision@top_{pct}pct"] = []
        metrics[f"Recall@top_{pct}pct"] = []
        metrics[f"SOZ_enrichment@top_{pct}pct"] = []

    aucs = []
    group_gaps: List[float] = []

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

            pos_scores = scores[y == 1]
            neg_scores = scores[y == 0]
            gap = float(np.mean(pos_scores) - np.mean(neg_scores))
            group_gaps.append(gap)

        if N > 0 and len(p_list) > 0:
            base_prev = total_pos / N if N > 0 else 0.0

            for p in p_list:
                pct = int(round(p * 100))
                k_p = int(np.ceil(p * N))
                k_p = max(1, min(k_p, N))

                tp_p = sum(rels[:k_p])
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

    out: Dict[str, Any] = {}
    for m, vals in metrics.items():
        out[m] = float(np.nanmean(vals)) if len(vals) else float("nan")

    out["AUC_group_mean"] = float(np.mean(aucs)) if len(aucs) else float("nan")
    out["num_groups_eval"] = int(df_pred.groupby(gcols).ngroups)

    y_all = df_pred["y_true"].astype(int).values
    s_all = df_pred["y_score"].astype(float).values
    out["Brier"] = float(np.mean((y_all - s_all) ** 2)) if len(y_all) else float("nan")

    pos_scores_all = s_all[y_all == 1]
    neg_scores_all = s_all[y_all == 0]

    out["mean_score_pos_global"] = float(np.mean(pos_scores_all)) if len(pos_scores_all) else float("nan")
    out["mean_score_neg_global"] = float(np.mean(neg_scores_all)) if len(neg_scores_all) else float("nan")

    if not np.isnan(out["mean_score_pos_global"]) and not np.isnan(out["mean_score_neg_global"]):
        out["confidence_gap_global"] = float(out["mean_score_pos_global"] - out["mean_score_neg_global"])
    else:
        out["confidence_gap_global"] = float("nan")

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


# ------------------------ main experiment loop (LOPO) ------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Run multiple LGBMClassifier experiments with LOPO CV.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    features_path = Path(cfg["features_path"]).expanduser()
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
    if "is_flat_zero" in df.columns and "is_flat_zero" not in drop_extras:
        drop_extras.append("is_flat_zero")

    needed = set(group_keys + [label_col, "node_index", "electrode_name"])
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in features table: {missing}")

    non_feature = set(group_keys + [label_col, "electrode_name", "node_index"]) | set(drop_extras or [])
    feature_cols_base = [c for c in df.columns if c not in non_feature]

    patients_all = df["patient"].astype(str).values
    unique_pats = np.unique(patients_all)
    logo = LeaveOneGroupOut()
    print(f"[INFO] LOPO CV over {len(unique_pats)} patients: {unique_pats.tolist()}")

    for exp in experiments:
        name = exp["name"]
        print("\n" + "=" * 30)
        print(f"[EXP] {name} (LOPO CV)")
        print("=" * 30)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = output_root / f"{name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        merged_cfg = dict(cfg)
        merged_cfg["experiment"] = exp
        save_config(merged_cfg, exp_dir)

        per_group_standardize = bool(exp.get("per_group_standardize", True))
        global_standardize_flag = bool(exp.get("global_standardize", False))
        train_only_pos = bool(exp.get("train_only_groups_with_positive", True))

        # -------- Feature selection config (global LASSO once per experiment) --------
        fs_cfg = exp.get("feature_selection", cfg.get("feature_selection", {})) or {}
        fs_enabled = bool(fs_cfg.get("enabled", False))
        fs_method = str(fs_cfg.get("method", "")).lower() if fs_enabled else ""

        # -------- Postprocess config (logit shift auto) --------
        pp_cfg = exp.get("postprocess", cfg.get("postprocess", {})) or {}
        pp_enabled = bool(pp_cfg.get("enabled", True))
        pp_method = str(pp_cfg.get("method", "logit_shift")).lower()

        pp_val_size = float(pp_cfg.get("val_size", cfg.get("val_size", 0.1)))
        pp_group_key = str(pp_cfg.get("group_key", "patient"))
        neg_mean_cap = float(pp_cfg.get("neg_mean_cap", 0.05))

        delta_min = float(pp_cfg.get("delta_min", -2.0))
        delta_max = float(pp_cfg.get("delta_max", 6.0))
        delta_steps = int(pp_cfg.get("delta_steps", 81))
        delta_grid = np.linspace(delta_min, delta_max, delta_steps)

        # ------------------- GLOBAL FEATURE SELECTION (once per experiment) -------------------
        selected_cols_global = None
        medians_global = None
        if fs_enabled:
            if fs_method != "lasso":
                raise ValueError(f"Unknown feature selection method: {fs_method}")

            C = float(fs_cfg.get("C", 0.05))
            max_features = fs_cfg.get("max_features", None)
            if max_features is not None:
                max_features = int(max_features)

            selected_cols_global, medians_global = lasso_feature_select_global(
                df_all=df,
                feature_cols=feature_cols_base,
                y_col=label_col,
                C=C,
                max_features=max_features,
                random_state=random_state,
            )

            sel_path = exp_dir / "selected_features_GLOBAL.txt"
            with open(sel_path, "w") as f:
                for c in selected_cols_global:
                    f.write(c + "\n")

            print(
                f"[EXP:{name}] GLOBAL LASSO selected {len(selected_cols_global)} features "
                f"(C={C}, max_features={max_features})"
            )
            print(f"[EXP:{name}] saved selected features -> {sel_path.name}")

        params = dict(base_lgbm_params)
        params.update(exp.get("lgbm_params", {}))
        params.pop("lambdarank_truncation_level", None)

        es_rounds = exp.get("early_stopping_rounds", cfg.get("early_stopping_rounds", 80))
        log_period = exp.get("log_period", cfg.get("log_period", 50))

        all_eval_rows = []
        fold_metrics = []
        fold_pp_rows = []

        for fold_idx, (tr_idx, va_idx) in enumerate(logo.split(df, groups=patients_all), start=1):
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[va_idx].copy()
            val_pats = sorted(df_va["patient"].astype(str).unique())
            print(f"\n[EXP:{name}] Fold {fold_idx} | val patients: {val_pats}")

            feature_cols = list(feature_cols_base)

            if train_only_pos:
                before = len(df_tr)
                df_tr = filter_train_groups_with_positive(df_tr, group_keys)
                print(f"[EXP:{name}] train rows (>=1 SOZ): {before} -> {len(df_tr)}")

            if "is_flat_zero" in df_tr.columns:
                before = len(df_tr)
                df_tr = df_tr[~((df_tr[label_col] == 1) & (df_tr["is_flat_zero"] == True))].copy()
                removed = before - len(df_tr)
                print(f"[EXP:{name}] Fold {fold_idx}: removed {removed} flat-zero SOZ rows from TRAIN")

            if "is_flat_zero" in df_va.columns:
                before = len(df_va)
                df_va = df_va[~((df_va[label_col] == 1) & (df_va["is_flat_zero"] == True))].copy()
                removed = before - len(df_va)
                if removed > 0:
                    print(f"[EXP:{name}] Fold {fold_idx}: removed {removed} flat-zero SOZ rows from VAL")

            # per-group z-score
            if per_group_standardize:
                df_tr = per_group_zscore(df_tr, group_keys, feature_cols)
                df_va = per_group_zscore(df_va, group_keys, feature_cols)
                print(f"[EXP:{name}] Fold {fold_idx}: per-group z-score")

            # global standardization
            if global_standardize_flag:
                df_tr, df_va, _ = global_standardize(df_tr, df_va, feature_cols)
                print(f"[EXP:{name}] Fold {fold_idx}: global standardization")

            # Apply GLOBAL LASSO feature set
            if fs_enabled and selected_cols_global is not None and medians_global is not None:
                df_tr, df_va, feature_cols = apply_global_impute_and_select(
                    df_tr=df_tr,
                    df_va=df_va,
                    feature_cols_all=feature_cols_base,
                    selected_cols=selected_cols_global,
                    medians=medians_global,
                )
                print(f"[EXP:{name}] Fold {fold_idx}: applied GLOBAL LASSO feature set ({len(feature_cols)} feats)")

            Xtr, ytr, featcols, meta_tr = build_matrices(df_tr, group_keys, label_col, drop_extras)
            Xva, yva, _, meta_va = build_matrices(df_va, group_keys, label_col, drop_extras)

            clf = LGBMClassifier(
                objective="binary",
                class_weight="balanced",
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

            # Base scores = raw logits
            va_raw = clf.predict(Xva, raw_score=True)

            # ------------------- Postprocess: logit shift chosen on train-calib split -------------------
            delta_used = 0.0
            if pp_enabled and pp_method == "logit_shift":
                if pp_group_key not in df_tr.columns:
                    raise ValueError(f"postprocess.group_key='{pp_group_key}' not in df_tr columns")

                groups_pp = df_tr[pp_group_key].astype(str).values
                gss = GroupShuffleSplit(n_splits=1, test_size=pp_val_size, random_state=random_state)

                fit_idx, cal_idx = next(gss.split(df_tr, groups=groups_pp))
                df_cal = df_tr.iloc[cal_idx].copy()

                Xcal, ycal, _, _ = build_matrices(df_cal, group_keys, label_col, drop_extras)
                cal_raw = clf.predict(Xcal, raw_score=True)

                npos = int((ycal == 1).sum())
                nneg = int((ycal == 0).sum())

                if npos == 0 or nneg == 0:
                    print(
                        f"[EXP:{name}] Fold {fold_idx}: postprocess skipped "
                        f"(calib has pos={npos}, neg={nneg}). Using sigmoid(raw)."
                    )
                    delta_used = 0.0
                else:
                    delta_used, diag = choose_delta_logit_shift(
                        raw_scores=cal_raw,
                        y=ycal,
                        delta_grid=delta_grid,
                        neg_mean_cap=neg_mean_cap,
                    )
                    print(
                        f"[EXP:{name}] Fold {fold_idx}: logit_shift chose delta={delta_used:.3f} "
                        f"(mean_pos={diag['best_mean_pos']:.3f}, mean_neg={diag['best_mean_neg']:.3f}, "
                        f"feasible={int(diag.get('feasible', 0.0))}, cap={neg_mean_cap}, "
                        f"violation={diag.get('best_violation', float('nan')):.3f})"
                    )

                fold_pp_rows.append(
                    {
                        "fold": fold_idx,
                        "val_patients": ",".join(val_pats),
                        "pp_method": pp_method,
                        "pp_group_key": pp_group_key,
                        "pp_val_size": pp_val_size,
                        "neg_mean_cap": neg_mean_cap,
                        "delta_used": float(delta_used),
                        "delta_min": float(delta_min),
                        "delta_max": float(delta_max),
                        "delta_steps": int(delta_steps),
                        "calib_pos": int(npos) if "npos" in locals() else None,
                        "calib_neg": int(nneg) if "nneg" in locals() else None,
                    }
                )

            # Final score used for evaluation/output
            y_score = sigmoid(va_raw + float(delta_used))

            df_fold = meta_va.copy()
            df_fold["y_score"] = y_score
            df_fold["fold"] = fold_idx
            df_fold["val_patient"] = df_fold["patient"]
            all_eval_rows.append(df_fold)

            eval_df_fold = df_fold[group_keys + ["is_SOZ", "y_score"]].rename(columns={"is_SOZ": "y_true"})
            m_fold = groupwise_metrics(eval_df_fold, k_list=tuple(eval_at))
            m_fold["fold"] = fold_idx
            m_fold["val_patients"] = ",".join(val_pats)
            fold_metrics.append(m_fold)

            print(f"[EXP:{name}] Fold {fold_idx} metrics:")
            for k, v in m_fold.items():
                if k in ("fold", "val_patients"):
                    continue
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        all_eval_df = pd.concat(all_eval_rows, ignore_index=True)
        all_eval_df = all_eval_df.sort_values(group_keys + ["y_score"], ascending=[True, True, True, False])
        all_eval_df["rank"] = all_eval_df.groupby(group_keys)["y_score"].rank(method="first", ascending=False).astype(int)

        global_eval_df = all_eval_df[group_keys + ["is_SOZ", "y_score"]].rename(columns={"is_SOZ": "y_true"})
        global_metrics = groupwise_metrics(global_eval_df, k_list=tuple(eval_at))

        per_patient_rows = []
        per_patient_dict = {}
        for pat, df_pat in global_eval_df.groupby("patient"):
            m_pat = groupwise_metrics(df_pat, k_list=tuple(eval_at))
            m_pat["patient"] = pat
            per_patient_rows.append(m_pat)
            per_patient_dict[str(pat)] = m_pat

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

        if len(fold_pp_rows) > 0:
            pd.DataFrame(fold_pp_rows).to_csv(exp_dir / "cv_fold_postprocess_params.csv", index=False)
            with open(exp_dir / "cv_fold_postprocess_params.json", "w") as f:
                json.dump(fold_pp_rows, f, indent=2)
            print(f"[EXP:{name}] saved postprocess params -> cv_fold_postprocess_params.*")

        print(f"\n[EXP:{name}] GLOBAL CV metrics:")
        for k, v in global_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"[EXP:{name}] CV outputs saved to {exp_dir}")

    print("\n[INFO] All experiments (LOPO CV) done.")


if __name__ == "__main__":
    main()
