import os
import copy
import itertools
import subprocess
import yaml  # pip install pyyaml

BASE_CONFIG = "config_grid_search_kwta.yaml"      # ta config de base (celle que tu as collée)
SCRIPT = "aa_M1_scriptt.py"            # ton script principal
OUT_CONFIG_DIR = "grid_configs"  # dossier pour sauvegarder les configs générées

os.makedirs(OUT_CONFIG_DIR, exist_ok=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def set_by_path(cfg, path, value):
    """
    path de type 'train.lr' ou 'model.k_max_ratio'
    """
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def main():
    # 1) Config de base
    base_cfg = load_yaml(BASE_CONFIG)
    base_name = base_cfg["experiment"].get("name", "experiment")

    # 2) Grille d'hyperparamètres à explorer
    #    >>> ADAPTE LES LISTES ICI SELON CE QUE TU VEUX TESTER <<<
    grid = {
        # modèle
        "model.k_max_ratio":  [0.05, 0.10, 0.20],
        "model.k_new_ratio":  [0.05, 0.10],
        "model.alpha_stay":   [0.5, 0.7, 0.9],
        # entraînement
        "train.epochs":       [20, 30],          # tu peux mettre [50] si tu ne veux pas varier
        "train.lr":           [1e-3, 3e-4, 1e-4]
    }

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]

    for combo in itertools.product(*values_list):
        # 3) Construire un tag lisible pour cette combinaison
        tag_parts = []
        for k, v in zip(keys, combo):
            short_name = k.split(".")[-1]
            if isinstance(v, float):
                # éviter les '.' dans le nom de fichier
                v_str = f"{v}".replace(".", "p")
            else:
                v_str = str(v)
            tag_parts.append(f"{short_name}={v_str}")
        tag = "__".join(tag_parts)

        print("=" * 80)
        print(f"[GRID] Nouvelle config : {tag}")
        print("=" * 80)

        # 4) Copier la config de base et appliquer les modifications
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            set_by_path(cfg, k, v)

        # Optionnel mais pratique : nom d'expérience unique
        cfg["experiment"]["name"] = f"{base_name}__{tag}"

        # 5) Sauvegarder la nouvelle config
        cfg_path = os.path.join(OUT_CONFIG_DIR, f"config__{tag}.yaml")
        save_yaml(cfg, cfg_path)
        print(f"[GRID] Config écrite -> {cfg_path}")

        # 6) Lancer ton script principal avec cette config
        cmd = ["python", SCRIPT, "--config", cfg_path]
        print(f"[GRID] Lancement : {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[GRID] ⚠️ Échec pour {tag} (returncode={result.returncode})")

if __name__ == "__main__":
    main()
