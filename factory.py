from torch import optim
from models.model_backbone import GraphBackboneNodeGRU

def build_model(cfg, inferred_in_dim: int | None):
    return GraphBackboneNodeGRU.from_config(cfg, inferred_in_dim)

def build_optimizer(cfg, model):
    t = cfg.get("train", {})
    lr = float(t.get("lr", 1e-3))
    wd = float(t.get("weight_decay", 0.0))
    opt_name = str(t.get("optimizer", "adamw")).lower()
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
