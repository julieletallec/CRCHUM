import os, yaml, torch, random, numpy as np

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_device(cfg) -> str:
    want = cfg.get("experiment", {}).get("device", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
