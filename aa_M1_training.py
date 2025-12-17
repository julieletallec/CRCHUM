import os, warnings, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import time
from utils.config import ensure_dir
from utils.factory import build_model, build_optimizer
from aa_M1_sequence_dataset_copy import SequenceDataset, collate_sequences
from aa_M1_pipeline_m1_extract_nodeprobs import _infer_in_dim_from_dataset, _normalize_collate_output

def _binarize(y_list, device):
    with torch.no_grad():
        return [y.view(-1).to(device=device, dtype=torch.float32) for y in y_list]


from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler




def _make_train_loader(cfg, patient_key: str):
    data = cfg["data"]
    splits_all_train = data["splits_all_train"]
    fold_idx = int(data.get("fold_idx", 0))
    train_splits = data.get("train_splits", ["train"]) or ["train"]
    if isinstance(train_splits, str):
        train_splits = [train_splits]

    ds_list = []
    for sk in train_splits:
        try:
            ds_k = SequenceDataset(splits_all_train, patient_key, sk, fold_idx=fold_idx)
            if len(ds_k) == 0:
                warnings.warn(f"[{patient_key}] Split '{sk}' empty.")
            else:
                ds_list.append(ds_k)
        except KeyError as e:
            warnings.warn(f"[{patient_key}] Split '{sk}' not found: {e}")

    if not ds_list:
        raise ValueError(f"[{patient_key}] No valid split in {train_splits}.")

    ds = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)

    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", False))

    dl = DataLoader(
        ds,
        batch_size=int(data.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=num_workers,          # = 0
        pin_memory=pin_memory,            # = False
        persistent_workers=False,         # 🔴 très important quand num_workers=0
    )
    return ds, dl


def _make_train_loader_(cfg, patient_key: str):
    data = cfg["data"]
    splits_all_train = data["splits_all_train"]
    fold_idx = int(data.get("fold_idx", 0))
    train_splits = data.get("train_splits", ["train"]) or ["train"]
    if isinstance(train_splits, str):
        train_splits = [train_splits]

    ds_list = []
    for sk in train_splits:
        try:
            ds_k = SequenceDataset(splits_all_train, patient_key, sk, fold_idx=fold_idx)
            if len(ds_k) == 0:
                warnings.warn(f"[{patient_key}] Split '{sk}' empty.")
            else:
                ds_list.append(ds_k)
        except KeyError as e:
            warnings.warn(f"[{patient_key}] Split '{sk}' not found: {e}")

    if not ds_list:
        raise ValueError(f"[{patient_key}] No valid split in {train_splits}.")

    ds = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)

    dl = DataLoader(
        ds,
        batch_size=int(data.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", True)),
    )
    return ds, dl

def _ckpt_path_for(cfg, patient_key: str):
    exp = cfg["experiment"]
    ckpt_dir = os.path.join(exp["output_dir"], cfg["train"].get("ckpt_subdir", "checkpoints"))
    
    ensure_dir(ckpt_dir)
    name = f"{patient_key.replace('::','__')}_{exp.get('specificity','exp')}.pt"
    return os.path.join(ckpt_dir, name)

def train_one_patient(cfg, patient_key, device, ckpt_path: str | None = None):
    if ckpt_path is None:
        exp_dir = cfg["experiment"]["output_dir"]
        ckpt_dir = cfg["train"].get("ckpt_dir", os.path.join(exp_dir, "checkpoints"))
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = f"{patient_key.replace('::','__')}_{cfg['experiment'].get('specificity','exp')}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    else:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    ds, dl = _make_train_loader(cfg, patient_key)

    # infer in_dim if requested
    m = cfg["model"]
    inferred_in_dim = None
    if str(m.get("in_dim", "auto")) == "auto":
        inferred_in_dim = _infer_in_dim_from_dataset(ds)

    model = build_model(cfg, inferred_in_dim).to(device)
    #model = torch.compile(model, mode="reduce-overhead")

    print(f"[DEBUG] device = {device}, first param device = {next(model.parameters()).device}")
    optimizer = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg["train"].get("amp", True) and device.type == "cuda"))

    epochs = int(cfg["train"].get("epochs", 10))
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    max_batches = int(cfg["train"].get("max_batches_per_epoch", 0))
    max_T = cfg["train"].get("max_timesteps", None)

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, steps = 0.0, 0

        # stats de temps
        t_epoch_start = time.time()
        t_loader_total = 0.0
        t_forward_total = 0.0
        t_backward_total = 0.0

        for batch_idx, packet in enumerate(dl):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            t0 = time.time()
            # ----------------- DATA / COLLATE -----------------
            batches_time, seq_ids_time, y_graph_time, B = _normalize_collate_output(packet)
            if B <= 0:
                continue

            if max_T is not None:
                batches_time = batches_time[:max_T]
                seq_ids_time = seq_ids_time[:max_T]
                y_graph_time = y_graph_time[:max_T]
                if len(batches_time) == 0:
                    continue

            # petit debug sur le 1er batch pour voir la taille réelle
            if steps == 0:
                T = len(batches_time)
                num_graphs_t0 = batches_time[0].num_graphs
                num_nodes_t0 = batches_time[0].num_nodes
                print(f"[DEBUG] batch: B={B}, T={T}, graphs_t0={num_graphs_t0}, nodes_t0={num_nodes_t0}")

            batches_time = [bt.to(device) for bt in batches_time]
            y_graph_time = _binarize(y_graph_time, device)
            t1 = time.time()
            t_loader_total += (t1 - t0)

            # ----------------- FORWARD -----------------
            t_fwd0 = time.time()
            
            with torch.amp.autocast(
                device.type,
                enabled=(cfg["train"].get("amp", True) and device.type == "cuda")
            ):
                p_graph_time = model.forward_sequence(
                    batches_time, seq_ids_time, return_node_probs=False
                )

            if len(p_graph_time) == 0:
                continue



            # concatène tous les graphes de tous les temps en un seul vecteur
            PgTB = torch.cat([p.view(-1) for p in p_graph_time], dim=0)           # shape [N_tot]
            YgTB = torch.cat(y_graph_time, dim=0).to(device=PgTB.device).float()  # shape [N_tot]

            with torch.amp.autocast(device.type, enabled=False):
                loss = F.binary_cross_entropy(PgTB.float(), YgTB)
            t_fwd1 = time.time()
            t_forward_total += (t_fwd1 - t_fwd0)
            """
            # ---------- FAKE FORWARD : pas de modèle, juste pour mesurer ----------
            YgTB = torch.cat(y_graph_time, dim=0).to(device=device).float()
            PgTB = torch.rand_like(YgTB)

            with torch.amp.autocast(device.type, enabled=False):
                loss = F.binary_cross_entropy(PgTB.float(), YgTB)


            t_fwd1 = time.time()
            t_forward_total += (t_fwd1 - t_fwd0)
            # ⚠️ PAS DE BACKWARD NI OPTIMIZER ICI
            running_loss += float(loss.item())
            steps += 1
            continue  # on saute le bloc backward
            """
            # ----------------- BACKWARD -----------------
            t_bwd0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            t_bwd1 = time.time()
            t_backward_total += (t_bwd1 - t_bwd0)

            running_loss += float(loss.item())
            steps += 1

        t_epoch = time.time() - t_epoch_start
        print(
            f"[{patient_key}] epoch {ep:03d} | loss {running_loss / max(1, steps):.4f} | "
            f"epoch {t_epoch:.1f}s (loader {t_loader_total:.1f}s, fwd {t_forward_total:.1f}s, bwd {t_backward_total:.1f}s)"
        )

    ckpt_path = _ckpt_path_for(cfg, patient_key)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": inferred_in_dim if inferred_in_dim is not None else m.get("in_dim"),
        "hid": int(m.get("hid", 128)),
        "model": "nodegru_simple"
    }, ckpt_path)

    return ckpt_path

