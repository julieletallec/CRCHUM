#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline tout-en-un:

1) Lit le CSV best_model_per_patient_selected_by_F1.csv
2) Trouve le combo (results_dir, experiment) le plus fréquent (ou --pick)
3) Localise 2 configs:
   - M1 config: <root>/<results_dir>/config_used.yaml (ou recherche récursive)
   - M2 config: <root>/<results_dir>/<big_experiment>/<experiment>/config_used.yaml (ou recherche récursive)
4) Trouve les features (par défaut):
   - <root>/<results_dir>/features/features_augmented.parquet
   - sinon fallback: recherche récursive d'un parquet contenant "features" dans le nom
5) Entraîne M2 UNE FOIS sur TOUS les patients (pas de CV) en reprenant les options du config M2:
   - per_group_standardize, global_standardize, train_only_groups_with_positive
   - feature_selection (LASSO global)
   - postprocess (logit_shift sur split calib patient-grouped)
6) Sauvegarde dans --out:
   - config_used__M1.yaml
   - config_used__M2.yaml
   - model_M2.joblib
   - artifacts.json (delta, selected_features, medians, scaler_used, etc.)
   - selected_features.txt
   - train_predictions_ranked.csv (scores sur le dataset complet)
   - summary.txt

Usage:
uv run 11_train_best_M2_on_all_good_patients.py \
  --csv /home/julieletallec/test/figures_grid_search_kwta_20_10_burst_nozerovalF1_specialFP_new0.1/best_F1_top10pct/best_model_per_patient_selected_by_F1.csv \
  --root /home/julieletallec/test/results_grid_search_kwta_20_10_burst \
  --big_experiment classifier_experiments_augmented_balanced_LASSO_global_0.03_postprocess_new0.1 \
  --out /home/julieletallec/test/M2_singlefit_out

Optionnel:
  --pick results_XXX,expYYY
  --features /path/to/features_augmented.parquet   (force un chemin features)
  --no_train   (ne fait que trouver/copier les configs)
"""

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMClassifier, log_evaluation, early_stopping
import joblib


# ----------------------------
#  Part A: combo + configs
# ----------------------------

def most_frequent_combo(df: pd.DataFrame) -> tuple[str, str, int]:
    required = {"results_dir", "experiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    counts = (
        df.groupby(["results_dir", "experiment"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )
    if counts.empty:
        raise RuntimeError("Aucun combo (results_dir, experiment) trouvé dans le CSV.")

    top = counts.iloc[0]
    return str(top["results_dir"]), str(top["experiment"]), int(top["count"])


def find_config_exact_or_search(base: Path, filename: str = "config_used.yaml") -> Optional[Path]:
    direct = base / filename
    if direct.is_file():
        return direct

    if not base.is_dir():
        return None

    hits = [p for p in base.rglob(filename) if p.is_file()]
    if not hits:
        return None

    hits.sort(key=lambda p: len(p.parts))  # plus proche
    return hits[0]


# ----------------------------
#  Part B: find features
# ----------------------------

def find_features_parquet(fold_dir: Path) -> Optional[Path]:
    """
    Heuristique:
      1) fold_dir/features/features_augmented.parquet
      2) fold_dir/features/*.parquet
      3) recherche récursive d'un parquet contenant "features" dans le nom
    """
    p1 = fold_dir / "features" / "features_augmented.parquet"
    if p1.is_file():
        return p1

    p2_dir = fold_dir / "features"
    if p2_dir.is_dir():
        candidates = sorted([p for p in p2_dir.glob("*.parquet") if p.is_file()])
        if candidates:
            return candidates[0]

    candidates = sorted([p for p in fold_dir.rglob("*.parquet") if "feature" in p.name.lower()])
    if candidates:
        candidates.sort(key=lambda p: len(p.parts))
        return candidates[0]

    return None


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


# ----------------------------
#  Part C: preprocessing + training single-fit
# ----------------------------

def per_group_zscore(df: pd.DataFrame, group_keys: List[str], feature_cols: List[str]) -> pd.DataFrame:
    def _z(g):
        g = g.copy()
        vals = g[feature_cols].astype(float)
        mu = vals.mean(axis=0)
        sd = vals.std(axis=0).replace(0, 1.0)
        g[feature_cols] = (vals - mu) / sd
        return g
    return df.groupby(group_keys, group_keys=False).apply(_z).reset_index(drop=True)


def filter_groups_with_positive(df: pd.DataFrame, group_keys: List[str], label_col: str) -> pd.DataFrame:
    g = df.groupby(group_keys)[label_col].sum().reset_index(name="npos")
    keep = g[g["npos"] > 0]
    return df.merge(keep[group_keys], on=group_keys, how="inner")


def build_feature_cols(df: pd.DataFrame, group_keys: List[str], label_col: str, drop_extras: List[str]) -> List[str]:
    non_feature = set(group_keys + [label_col, "electrode_name", "node_index"]) | set(drop_extras or [])
    return [c for c in df.columns if c not in non_feature]


def clean_for_lasso(df_any: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    df2 = df_any.copy()
    df2[feature_cols] = df2[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = df2[feature_cols].median(axis=0, numeric_only=True)
    df2[feature_cols] = df2[feature_cols].fillna(med)

    nunique = df2[feature_cols].nunique(dropna=False)
    keep_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]
    if not keep_cols:
        keep_cols = list(feature_cols)

    drop_cols = [c for c in feature_cols if c not in keep_cols]
    if drop_cols:
        df2 = df2.drop(columns=drop_cols)

    return df2, keep_cols, med


def lasso_feature_select_global(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    C: float,
    max_features: Optional[int],
    random_state: int,
) -> Tuple[List[str], pd.Series]:
    df_clean, feature_cols2, med = clean_for_lasso(df_all, feature_cols)

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
    if not selected:
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


def apply_impute_and_select(df_any: pd.DataFrame, feature_cols_all: List[str], selected_cols: List[str], med: pd.Series) -> pd.DataFrame:
    df2 = df_any.copy()
    df2[feature_cols_all] = df2[feature_cols_all].replace([np.inf, -np.inf], np.nan)
    df2[feature_cols_all] = df2[feature_cols_all].fillna(med)

    drop_cols = [c for c in feature_cols_all if c not in selected_cols and c in df2.columns]
    if drop_cols:
        df2 = df2.drop(columns=drop_cols, errors="ignore")
    return df2


def global_standardize(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].astype(float))
    out = df.copy()
    out[feature_cols] = X
    return out, scaler


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def choose_delta_logit_shift(raw_scores: np.ndarray, y: np.ndarray, delta_grid: np.ndarray, neg_mean_cap: float) -> Tuple[float, Dict[str, float]]:
    y = y.astype(int)
    pos_mask = y == 1
    neg_mask = y == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0, {"ok": 0.0}

    candidates = []
    for d in delta_grid:
        p = sigmoid(raw_scores + float(d))
        mean_pos = float(np.mean(p[pos_mask]))
        mean_neg = float(np.mean(p[neg_mask]))
        violation = max(0.0, mean_neg - float(neg_mean_cap))
        candidates.append((float(d), mean_pos, mean_neg, violation))

    feasible = [c for c in candidates if c[2] <= neg_mean_cap]
    if feasible:
        d, mp, mn, vio = max(feasible, key=lambda t: (t[1], -t[2]))
        return d, {"ok": 1.0, "feasible": 1.0, "best_mean_pos": mp, "best_mean_neg": mn, "best_violation": vio}

    min_vio = min(candidates, key=lambda t: t[3])[3]
    near = [c for c in candidates if abs(c[3] - min_vio) <= 1e-12]
    d, mp, mn, vio = max(near, key=lambda t: (t[1], -t[2]))
    return d, {"ok": 1.0, "feasible": 0.0, "best_mean_pos": mp, "best_mean_neg": mn, "best_violation": vio}


def train_single_fit_from_m2_config(
    cfg_m2: Dict[str, Any],
    features_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    group_keys = cfg_m2.get("group_keys", ["patient", "seizure_id", "seq_idx"])
    label_col = cfg_m2.get("label_col", "is_SOZ")
    drop_extras = cfg_m2.get("drop_extra_cols", []) or []
    random_state = int(cfg_m2.get("random_state", 42))

    df = load_table(features_path)

    needed = set(group_keys + [label_col, "node_index", "electrode_name"])
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in features table: {missing}")

    # Base feature cols (avant FS)
    feature_cols_base = build_feature_cols(df, group_keys, label_col, drop_extras)

    # --- récupérer l'experiment dans le config_used.yaml M2 ---
    # Si cfg_m2 a déjà un champ "experiment", on l'utilise.
    # Sinon, il peut avoir une liste "experiments" -> on prend le premier (ou tu peux adapter).
    exp = cfg_m2.get("experiment", None)
    if exp is None:
        exps = cfg_m2.get("experiments", [])
        if not exps:
            raise RuntimeError("Le config M2 ne contient ni 'experiment' ni 'experiments'.")
        exp = exps[0]

    # flags preprocessing
    per_group_standardize = bool(exp.get("per_group_standardize", True))
    global_standardize_flag = bool(exp.get("global_standardize", False))
    train_only_pos = bool(exp.get("train_only_groups_with_positive", True))

    # Feature selection (LASSO global)
    fs_cfg = exp.get("feature_selection", cfg_m2.get("feature_selection", {})) or {}
    fs_enabled = bool(fs_cfg.get("enabled", False))
    fs_method = str(fs_cfg.get("method", "")).lower() if fs_enabled else ""

    selected_cols = None
    medians = None

    # optionnel: remove flat-zero SOZ (même logique)
    if "is_flat_zero" in df.columns:
        before = len(df)
        df = df[~((df[label_col] == 1) & (df["is_flat_zero"] == True))].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"[TRAIN] removed {removed} flat-zero SOZ rows")

    if train_only_pos:
        before = len(df)
        df = filter_groups_with_positive(df, group_keys, label_col)
        print(f"[TRAIN] keep only groups with >=1 SOZ: {before} -> {len(df)}")

    if per_group_standardize:
        df = per_group_zscore(df, group_keys, feature_cols_base)
        print("[TRAIN] per-group z-score")

    scaler = None
    if global_standardize_flag:
        df, scaler = global_standardize(df, feature_cols_base)
        print("[TRAIN] global standardization")

    if fs_enabled:
        if fs_method != "lasso":
            raise ValueError(f"Unknown feature selection method: {fs_method}")

        C = float(fs_cfg.get("C", 0.05))
        max_features = fs_cfg.get("max_features", None)
        if max_features is not None:
            max_features = int(max_features)

        selected_cols, medians = lasso_feature_select_global(
            df_all=df,
            feature_cols=feature_cols_base,
            y_col=label_col,
            C=C,
            max_features=max_features,
            random_state=random_state,
        )
        print(f"[TRAIN] GLOBAL LASSO selected {len(selected_cols)} features (C={C}, max_features={max_features})")

        (out_dir / "selected_features.txt").write_text("\n".join(selected_cols) + "\n", encoding="utf-8")

        df = apply_impute_and_select(df, feature_cols_base, selected_cols, medians)

    # Build X/y
    feature_cols_final = build_feature_cols(df, group_keys, label_col, drop_extras)
    X = df[feature_cols_final].astype(float).values
    y = df[label_col].astype(int).values

    # LightGBM params
    base_lgbm_params = cfg_m2.get("base_lgbm_params", {}) or {}
    params = dict(base_lgbm_params)
    params.update(exp.get("lgbm_params", {}) or {})
    params.pop("lambdarank_truncation_level", None)  # sécurité

    es_rounds = exp.get("early_stopping_rounds", cfg_m2.get("early_stopping_rounds", 0))
    log_period = exp.get("log_period", cfg_m2.get("log_period", 0))

    clf = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=random_state,
        **params,
    )

    # Postprocess config
    pp_cfg = exp.get("postprocess", cfg_m2.get("postprocess", {})) or {}
    pp_enabled = bool(pp_cfg.get("enabled", True))
    pp_method = str(pp_cfg.get("method", "logit_shift")).lower()

    pp_val_size = float(pp_cfg.get("val_size", cfg_m2.get("val_size", 0.1)))
    pp_group_key = str(pp_cfg.get("group_key", "patient"))
    neg_mean_cap = float(pp_cfg.get("neg_mean_cap", 0.05))
    delta_min = float(pp_cfg.get("delta_min", -2.0))
    delta_max = float(pp_cfg.get("delta_max", 6.0))
    delta_steps = int(pp_cfg.get("delta_steps", 81))
    delta_grid = np.linspace(delta_min, delta_max, delta_steps)

    # Calib split (patient-grouped) uniquement pour choisir delta (après training)
    delta_used = 0.0
    delta_diag: Dict[str, Any] = {"enabled": pp_enabled, "method": pp_method}

    # Train (option: early stopping nécessite un eval_set)
    callbacks = []
    if es_rounds and es_rounds > 0:
        callbacks.append(early_stopping(stopping_rounds=int(es_rounds)))
    if log_period and log_period > 0:
        callbacks.append(log_evaluation(period=int(log_period)))

    # Si early stopping demandé, on doit créer un split interne train/val
    if callbacks:
        groups = df[group_keys[0]].astype(str).values if group_keys else df["patient"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=pp_val_size, random_state=random_state)
        tr_idx, va_idx = next(gss.split(df, groups=groups))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=callbacks)
    else:
        clf.fit(X, y)

    # Choix du delta sur split calib (sur dataset complet, patient-grouped)
    if pp_enabled and pp_method == "logit_shift":
        if pp_group_key not in df.columns:
            raise ValueError(f"postprocess.group_key='{pp_group_key}' not in df columns")

        groups_pp = df[pp_group_key].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=pp_val_size, random_state=random_state)
        fit_idx, cal_idx = next(gss.split(df, groups=groups_pp))

        Xcal = X[cal_idx]
        ycal = y[cal_idx]
        cal_raw = clf.predict(Xcal, raw_score=True)

        npos = int((ycal == 1).sum())
        nneg = int((ycal == 0).sum())

        if npos == 0 or nneg == 0:
            delta_used = 0.0
            delta_diag.update({"skipped": True, "calib_pos": npos, "calib_neg": nneg})
        else:
            delta_used, diag = choose_delta_logit_shift(cal_raw, ycal, delta_grid, neg_mean_cap)
            delta_diag.update(diag)
            delta_diag.update({"calib_pos": npos, "calib_neg": nneg})
        delta_diag.update({
            "delta_used": float(delta_used),
            "neg_mean_cap": float(neg_mean_cap),
            "delta_min": float(delta_min),
            "delta_max": float(delta_max),
            "delta_steps": int(delta_steps),
            "pp_val_size": float(pp_val_size),
            "pp_group_key": pp_group_key,
        })

    # Produire des scores sur tout le dataset pour sanity check
    raw_all = clf.predict(X, raw_score=True)
    score_all = sigmoid(raw_all + float(delta_used))

    pred_df = df[group_keys + ["node_index", "electrode_name", label_col]].copy()
    pred_df["y_score"] = score_all
    pred_df = pred_df.sort_values(group_keys + ["y_score"], ascending=[True, True, True, False])
    pred_df["rank"] = pred_df.groupby(group_keys)["y_score"].rank(method="first", ascending=False).astype(int)

    pred_df.to_csv(out_dir / "train_predictions_ranked.csv", index=False)

    # Sauvegardes
    joblib.dump(clf, out_dir / "model_M2.joblib")

    artifacts = {
        "features_path": str(features_path),
        "feature_cols_base_n": int(len(feature_cols_base)),
        "feature_cols_final_n": int(len(feature_cols_final)),
        "feature_cols_final": feature_cols_final,
        "lasso_selected_cols": selected_cols,
        "medians_available": bool(medians is not None),
        "scaler_used": bool(scaler is not None),
        "postprocess": delta_diag,
    }
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    # IMPORTANT: sauver medians & scaler si utilisés
    if medians is not None:
        medians.to_csv(out_dir / "medians_global.csv", header=True)
    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")

    return artifacts


# ----------------------------
#  Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV best_model_per_patient_selected_by_F1.csv")
    ap.add_argument("--root", required=True, help="Dossier racine contenant results_*")
    ap.add_argument("--big_experiment", required=True, help="Nom du dossier big experiment")
    ap.add_argument("--out", required=True, help="Dossier de sortie (DOIT être un dossier)")
    ap.add_argument("--pick", default="", help="Forcer un combo: 'results_dir,experiment'")
    ap.add_argument("--features", default="", help="Chemin features (optionnel) pour forcer")
    ap.add_argument("--no_train", action="store_true", help="Ne pas entraîner, seulement trouver/copier configs")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser()
    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()

    if not csv_path.is_file():
        raise SystemExit(f"CSV introuvable: {csv_path}")
    if not root.is_dir():
        raise SystemExit(f"Root introuvable: {root}")

    if out_dir.exists() and out_dir.is_file():
        raise SystemExit(f"--out doit être un dossier, pas un fichier: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.pick.strip():
        try:
            results_dir, experiment = [x.strip() for x in args.pick.split(",", 1)]
        except Exception:
            raise SystemExit("--pick doit être au format: results_dir,experiment")
        count = int((df["results_dir"].astype(str).eq(results_dir) & df["experiment"].astype(str).eq(experiment)).sum())
    else:
        results_dir, experiment, count = most_frequent_combo(df)

    fold_dir = root / results_dir
    big_exp_dir = fold_dir / args.big_experiment
    exp_dir = big_exp_dir / experiment

    fold_cfg = find_config_exact_or_search(fold_dir, "config_used.yaml")
    exp_cfg = find_config_exact_or_search(exp_dir, "config_used.yaml")

    print("\n[RESULT] Combo sélectionné")
    print(f"  results_dir : {results_dir}")
    print(f"  experiment  : {experiment}")
    print(f"  occurrences : {count}")

    print("\n[CONFIG] M1 (fold-level)")
    print(f"  {fold_cfg if fold_cfg else 'NOT FOUND'}")
    print("\n[CONFIG] M2 (experiment-level)")
    print(f"  {exp_cfg if exp_cfg else 'NOT FOUND'}")

    # copie configs
    if fold_cfg:
        shutil.copy2(fold_cfg, out_dir / "config_used__M1.yaml")
    if exp_cfg:
        shutil.copy2(exp_cfg, out_dir / "config_used__M2.yaml")

    # features path
    if args.features.strip():
        features_path = Path(args.features).expanduser()
    else:
        features_path = find_features_parquet(fold_dir)

    if features_path is None or not Path(features_path).is_file():
        raise SystemExit(
            "Features introuvables. Donne --features /path/to/features_augmented.parquet "
            "ou vérifie fold_dir/features/."
        )

    print(f"\n[FEATURES] Using: {features_path}")

    # si pas d'entrainement demandé
    if args.no_train:
        (out_dir / "summary.txt").write_text(
            "\n".join([
                f"results_dir={results_dir}",
                f"experiment={experiment}",
                f"occurrences={count}",
                f"fold_dir={fold_dir}",
                f"big_exp_dir={big_exp_dir}",
                f"exp_dir={exp_dir}",
                f"M1_config={fold_cfg if fold_cfg else 'NOT_FOUND'}",
                f"M2_config={exp_cfg if exp_cfg else 'NOT_FOUND'}",
                f"features_path={features_path}",
                ""
            ]),
            encoding="utf-8"
        )
        print(f"[OK] Configs + summary -> {out_dir}")
        return

    # charger cfg M2
    if not exp_cfg:
        raise SystemExit("Impossible d'entraîner M2: config experiment-level introuvable.")
    cfg_m2 = yaml.safe_load(Path(exp_cfg).read_text(encoding="utf-8"))

    print("\n[TRAIN] Start single-fit training for M2...")
    artifacts = train_single_fit_from_m2_config(cfg_m2, Path(features_path), out_dir)

    # summary
    (out_dir / "summary.txt").write_text(
        "\n".join([
            f"results_dir={results_dir}",
            f"experiment={experiment}",
            f"occurrences={count}",
            f"fold_dir={fold_dir}",
            f"big_exp_dir={big_exp_dir}",
            f"exp_dir={exp_dir}",
            f"M1_config={fold_cfg if fold_cfg else 'NOT_FOUND'}",
            f"M2_config={exp_cfg if exp_cfg else 'NOT_FOUND'}",
            f"features_path={features_path}",
            f"model={out_dir / 'model_M2.joblib'}",
            ""
        ]),
        encoding="utf-8"
    )

    print(f"\n[OK] Training complete. Outputs -> {out_dir}")
    print("  model_M2.joblib")
    print("  artifacts.json")
    print("  train_predictions_ranked.csv")
    print("  config_used__M1.yaml / config_used__M2.yaml")


if __name__ == "__main__":
    main()
