import os
import copy
import itertools
import subprocess
import yaml  # pip install pyyaml

BASE_CONFIG = "config_grid_search_topkmean.yaml"      # ta config de base
SCRIPT = "aa_M1_scriptt.py"            # ton script principal
OUT_CONFIG_DIR = "grid_configs"  # où on écrit les configs générées

os.makedirs(OUT_CONFIG_DIR, exist_ok=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def set_by_path(cfg, path, value):
    """
    path de type 'train.lr' ou 'model.k_ratio'
    """
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def main():
    base_cfg = load_yaml(BASE_CONFIG)
    base_name = base_cfg["experiment"].get("name", "experiment")

    # ----------------------------
    # GRILLE SANS k-WTA :
    # - pooling_mode = "topk_mean"
    # - on varie k_ratio (top k% des électrodes)
    # - + lr, epochs si tu veux
    # ----------------------------
    grid = {
        # pooling topk_mean : différentes fractions de noeuds utilisés
        "model.k_ratio": [0.05, 0.10, 0.20, 0.30],

        # hyperparams d'entraînement
        "train.lr":      [1e-3, 3e-4],
        "train.epochs":  [20, 30, 50],
    }

    # on force le pooling_mode dans la config copiée (au cas où le YAML de base ne l’ait pas déjà)
    force_pooling_mode = "topk_mean"

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]

    for combo in itertools.product(*values_list):
        # construire un tag lisible
        tag_parts = []
        for k, v in zip(keys, combo):
            short_name = k.split(".")[-1]
            if isinstance(v, float):
                v_str = f"{v}".replace(".", "p")
            else:
                v_str = str(v)
            tag_parts.append(f"{short_name}={v_str}")
        tag = "__".join(tag_parts)

        print("=" * 80)
        print(f"[GRID] Nouvelle config : {tag}")
        print("=" * 80)

        cfg = copy.deepcopy(base_cfg)

        # on s’assure que le k-WTA est neutralisé pour tous les runs
        cfg["model"]["k_max_ratio"] = 1.0
        cfg["model"]["k_new_ratio"] = 1.0
        cfg["model"]["alpha_stay"]  = 0.0
        cfg["model"]["abs_min"]     = 0.0

        # et que le pooling est bien topk_mean
        cfg["model"]["pooling_mode"] = force_pooling_mode

        # appliquer les valeurs de la grille (k_ratio, lr, epochs)
        for k, v in zip(keys, combo):
            set_by_path(cfg, k, v)

        # nom d'expérience lisible
        cfg["experiment"]["name"] = f"{base_name}__{tag}"

        # sauver la config spécifique
        cfg_path = os.path.join(OUT_CONFIG_DIR, f"config__{tag}.yaml")
        save_yaml(cfg, cfg_path)
        print(f"[GRID] Config écrite -> {cfg_path}")

        # lancer le script principal
        cmd = ["python", SCRIPT, "--config", cfg_path]
        print(f"[GRID] Lancement : {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[GRID] ⚠️ Échec pour {tag} (returncode={result.returncode})")

if __name__ == "__main__":
    main()
