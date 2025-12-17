#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrateur pour lancer automatiquement :
  1) le script de features
  2) le script d'expériences LGBMRanker (LOPO CV)
pour tous les sous-dossiers "results_YYYYMMDD_HHMMSS" d'un grand dossier.

Usage (exemple) :
uv run run_full_M2_experiment_cv.py \
      --features-script /home/julieletallec/test/aa_features_script.py \
      --ranker-script /home/julieletallec/test/aa_ranker_experiment_cv_classifier_nozeroval.py \
      --root /home/julieletallec/test/results_grid_search_kwta_20_10_burst
"""

import argparse
import re
import subprocess
from pathlib import Path
import copy
import yaml


# =========================
#  Configs de base
# =========================

# --- Config de base pour le script de features --- #
BASE_FEATURES_CONFIG = {
    # results_root et output_dir seront remplis dynamiquement
    "results_root": "",
    "patients": [
        "CHUM__Patient_01", "CHUM__Patient_02", "CHUM__Patient_07",
        "CHUM__Patient_11", "CHUM__Patient_14", "CHUM__Patient_22",
        "ds004100__sub-HUP074", "ds004100__sub-HUP082", "ds004100__sub-HUP089",
        "ds004100__sub-HUP097", "ds004100__sub-HUP107", "ds004100__sub-HUP111",
        "ds004100__sub-HUP123", "ds004100__sub-HUP126", "ds004100__sub-HUP130",
        "ds004100__sub-HUP141", "ds004100__sub-HUP144", "ds004100__sub-HUP148",
        "ds004100__sub-HUP150", "ds004100__sub-HUP157", "ds004100__sub-HUP173",
        "ds004100__sub-HUP180", "ds004100__sub-HUP185",
    ],
    "file_extensions": ["csv"],
    "output_dir": "",
    "save_long_as": "dataset_long.parquet",
    "save_features_as": "features_augmented.parquet",
    "manifest_name": "selection_manifest.csv",
}

# --- Config de base pour le script d'expériences LOPO --- #
BASE_EXPERIMENTS_CONFIG = {
    # features_path et output_root seront remplis dynamiquement
    "features_path": "",
    "output_root": "",

    "group_keys": ["patient", "seizure_id", "seq_idx"],
    "label_col": "is_SOZ",
    "drop_extra_cols": [],

    "val_size": 0.2,
    "random_state": 42,

    "eval_at": [3, 5, 10, 20],
    "early_stopping_rounds": 80,
    "log_period": 50,

    # Params LightGBM par défaut pour le classifier
    "base_lgbm_params": {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_child_samples": 20,
        "n_jobs": 4,
        # le script forcera objective="binary" dans LGBMClassifier
    },

    # --- Liste réduite à 5 expériences pour le classifier --- #
    "experiments": [
        # 1) Baseline : group-normalization, modèle "standard"
        {
            "name": "exp1_binary_groupnorm_baseline",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "lgbm_params": {
                # rien de spécial, utilise base_lgbm_params
            },
        },

        # 2) Modèle plus profond, learning rate plus faible
        {
            "name": "exp2_binary_deeper_lowlr",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "lgbm_params": {
                "num_leaves": 127,
                "learning_rate": 0.03,
                "n_estimators": 1200,
                # reste comme base_lgbm_params
            },
        },

        # 3) Même taille que baseline mais plus régularisé
        {
            "name": "exp3_binary_more_reg",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "reg_lambda": 0.2,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # 4) Pas de normalisation par groupe, on garde tous les groupes (y compris sans SOZ)
        {
            "name": "exp4_binary_no_norm_keep_all_groups",
            "per_group_standardize": False,
            "global_standardize": False,
            "train_only_groups_with_positive": False,
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },

        # 5) Petit modèle avec fort dropout de features (colsample) + sample
        {
            "name": "exp5_binary_small_model_heavy_dropout",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "lgbm_params": {
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "colsample_bytree": 0.4,
                "subsample": 0.7,
                "min_child_samples": 60,
            },
        },
    ],
}




BASE_EXPERIMENTS_CONFIG_ = {
    # features_path et output_root seront remplis dynamiquement
    "features_path": "",
    "output_root": "",

    "group_keys": ["patient", "seizure_id", "seq_idx"],
    "label_col": "is_SOZ",
    "drop_extra_cols": [],

    "val_size": 0.2,
    "random_state": 42,

    "eval_at": [3, 5, 10, 20],
    "early_stopping_rounds": 80,
    "log_period": 50,

    "base_lgbm_params": {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_child_samples": 20,
        "n_jobs": 4,
    },

    # --- Liste des expériences, recopiée de ton YAML --- #
    "experiments": [
        {
            "name": "exp1_lambdarank_groupnorm",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "lambdarank",
            "metric": "ndcg",
            "lgbm_params": {},
        },
        {
            "name": "exp2_xendcg_groupnorm_trunc20",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 20},
        },
        {
            "name": "exp3_xendcg_deeper_lowlr",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 127,
                "learning_rate": 0.03,
                "n_estimators": 1200,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp4_xendcg_more_reg",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "reg_lambda": 0.2,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp5_xendcg_no_norm",
            "per_group_standardize": False,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 20},
        },
        {
            "name": "exp6_xendcg_global_only",
            "per_group_standardize": False,
            "global_standardize": True,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 20},
        },
        {
            "name": "exp7_xendcg_group_plus_global",
            "per_group_standardize": True,
            "global_standardize": True,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 20},
        },
        {
            "name": "exp8_xendcg_small_leaves",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "min_child_samples": 50,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp9_xendcg_trunc10",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 10},
        },
        {
            "name": "exp10_xendcg_many_trees_strong_reg",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.02,
                "n_estimators": 2000,
                "reg_lambda": 0.5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp11_shallow_trees_strong_bagging",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 15,
                "max_depth": 4,
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "bagging_freq": 1,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp12_very_deep_few_trees",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 255,
                "max_depth": -1,
                "n_estimators": 400,
                "learning_rate": 0.05,
                "min_child_samples": 30,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp13_xendcg_strong_L1_L2",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.03,
                "n_estimators": 1500,
                "reg_alpha": 0.5,
                "reg_lambda": 0.8,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp14_xendcg_underfit_control",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 31,
                "learning_rate": 0.1,
                "n_estimators": 300,
                "reg_lambda": 1.0,
                "min_child_samples": 80,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp15_lambdarank_trunc10",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "lambdarank",
            "metric": "ndcg",
            "lgbm_params": {"lambdarank_truncation_level": 10},
        },
        {
            "name": "exp16_xendcg_no_truncation",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {},
        },
        {
            "name": "exp17_xendcg_keep_all_groups",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": False,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {"lambdarank_truncation_level": 20},
        },
        {
            "name": "exp18_lambdarank_keep_all_groups",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": False,
            "objective": "lambdarank",
            "metric": "ndcg",
            "lgbm_params": {},
        },
        {
            "name": "exp19_xendcg_heavy_feature_dropout",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 1200,
                "colsample_bytree": 0.4,
                "subsample": 0.8,
                "bagging_freq": 1,
                "lambdarank_truncation_level": 20,
            },
        },
        {
            "name": "exp20_xendcg_small_model_heavy_dropout",
            "per_group_standardize": True,
            "global_standardize": False,
            "train_only_groups_with_positive": True,
            "objective": "rank_xendcg",
            "metric": "xendcg",
            "lgbm_params": {
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "colsample_bytree": 0.4,
                "subsample": 0.7,
                "min_child_samples": 60,
                "lambdarank_truncation_level": 20,
            },
        },
    ],
}


# =========================
#  Fonctions utilitaires
# =========================

def is_results_dir(path: Path) -> bool:
    """
    Retourne True si le dossier ressemble à "results_YYYYMMDD_HHMMSS".
    Tu peux simplifier si tu veux juste startswith("results_").
    """
    if not path.is_dir():
        return False
    return re.match(r"^results_\d{8}_\d{6}$", path.name) is not None


def run_pipeline_on_one_dir(
    run_dir: Path,
    features_script: Path,
    ranker_script: Path,
    force: bool = False,
):
    print(f"\n=== Traitement du dossier : {run_dir} ===")

    # --- Chemins dynamiques pour ce run --- #
    results_root = run_dir / "results"
    features_dir = run_dir / "features"
    features_parquet = features_dir / "features_augmented.parquet"
    ranker_out_root = run_dir / "classifier_experiments_augmented_balanced_nozeroval"
    """
    # Si on ne force pas et que les features existent déjà, on peut éventuellement skipper
    if features_parquet.exists() and not force:
        print(f"[INFO] Features déjà présents : {features_parquet} (utiliser --force pour recalculer)")
    else:
        # 1) Construire config de features
        feat_cfg = copy.deepcopy(BASE_FEATURES_CONFIG)
        feat_cfg["results_root"] = str(results_root)
        feat_cfg["output_dir"] = str(features_dir)

        features_config_path = run_dir / "features_config_auto.yaml"
        with open(features_config_path, "w") as f:
            yaml.safe_dump(feat_cfg, f, sort_keys=False, allow_unicode=True)

        print(f"[INFO] Écriture du config features : {features_config_path}")
        print(f"[INFO] Lancement du script features...")
        subprocess.run(
            ["python", str(features_script), "--config", str(features_config_path)],
            check=True,
        )
    """
    # 1) Construire config de features
    feat_cfg = copy.deepcopy(BASE_FEATURES_CONFIG)
    feat_cfg["results_root"] = str(results_root)
    feat_cfg["output_dir"] = str(features_dir)

    features_config_path = run_dir / "features_config_auto.yaml"
    with open(features_config_path, "w") as f:
        yaml.safe_dump(feat_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Écriture du config features : {features_config_path}")
    print(f"[INFO] Lancement du script features...")
    subprocess.run(
        ["python", str(features_script), "--config", str(features_config_path)],
        check=True,
    )



    # Vérifier que les features existent
    if not features_parquet.exists():
        print(f"[ERREUR] {features_parquet} introuvable après le script de features. On passe ce dossier.")
        return

    # 2) Construire config d'expériences
    exp_cfg = copy.deepcopy(BASE_EXPERIMENTS_CONFIG)
    exp_cfg["features_path"] = str(features_parquet)
    exp_cfg["output_root"] = str(ranker_out_root)

    experiments_config_path = run_dir / "experiments_config_auto.yaml"
    with open(experiments_config_path, "w") as f:
        yaml.safe_dump(exp_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Écriture du config expériences : {experiments_config_path}")
    print(f"[INFO] Lancement du script LOPO CV...")
    subprocess.run(
        ["python", str(ranker_script), "--config", str(experiments_config_path)],
        check=True,
    )

    print(f"[OK] Pipeline terminée pour {run_dir}")


# =========================
#  Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Lancer la pipeline (features + LOPO CV) sur tous les dossiers results_*")
    parser.add_argument("--root", required=True, help="Grand dossier, ex: results_grid_search_topkmean_fast")
    parser.add_argument("--features-script", required=True, help="Chemin vers le script de features (celui avec main() et --config)")
    parser.add_argument("--ranker-script", required=True, help="Chemin vers le script d'expériences LOPO (celui avec main() et --config)")
    parser.add_argument("--force", action="store_true", help="Recalculer les features même si features.parquet existe déjà")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    features_script = Path(args.features_script).expanduser()
    ranker_script = Path(args.ranker_script).expanduser()

    if not root.is_dir():
        raise SystemExit(f"Le dossier root n'existe pas : {root}")
    if not features_script.is_file():
        raise SystemExit(f"Script features introuvable : {features_script}")
    if not ranker_script.is_file():
        raise SystemExit(f"Script ranker introuvable : {ranker_script}")

    # Lister tous les sous-dossiers results_*
    subdirs = sorted([p for p in root.iterdir() if is_results_dir(p)])
    if not subdirs:
        print(f"[WARN] Aucun sous-dossier 'results_YYYYMMDD_HHMMSS' trouvé dans {root}")
        return

    print(f"[INFO] Dossiers trouvés :")
    for d in subdirs:
        print(f"  - {d.name}")

    # Lancer la pipeline pour chacun
    for d in subdirs:
        run_pipeline_on_one_dir(d, features_script, ranker_script, force=args.force)

    print("\n[INFO] Tous les dossiers ont été traités.")


if __name__ == "__main__":
    main()
