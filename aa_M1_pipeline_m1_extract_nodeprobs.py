#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
UTILISATION:
uv run pipeline_m1_extract_nodeprobs.py   --splits "/home/julieletallec/smb_share/TeamMembers/LeTallec_Julie/processed_data/gnn/pyg/splits_patient_specific_cv.json"   --hid 128 --epochs 10 --batch-size 4 --lr 1e-3   --seconds-pre 20 --seconds-post 20 --epoch-sec 1.0   --smooth-W 5 --thr 0.5 --min-stable 3   --ckpt-dir checkpoints_m1   --out-dir nodeprobs_centered20s10EPOCHS



uv run pipeline_m1_extract_nodeprobs.py   --splits "splits_patient_specific_cv.json"   --hid 128 --epochs 10 --batch-size 4 --lr 1e-3   --seconds-pre 20 --seconds-post 20 --epoch-sec 1.0   --smooth-W 5 --thr 0.5 --min-stable 3   --ckpt-dir checkpoints_m1   --out-dir nodeprobs_centered20s10EPOCHS

"""



"""
pipeline_m1_extract_nodeprobs.py

Objectifs:
- Entraîner un modèle M1 (NodeGRU) par patient sur TOUTES les séquences listées dans le JSON (train+val+test).
- Afficher un rapport des séquences utilisées (seizure, seq_index, onset_idx, onset_% basés sur phase / first_ictal_epoch).
- Appliquer M1 en inférence uniquement sur UNE fenêtre centrée par crise (onset au milieu), de seconds_pre à seconds_post.
- Exporter les probabilités nodales par électrode/temps (CSV par crise + Parquet agrégé par patient).
- Ajouter true_ictal (vérité terrain) et pred_ictal (logique modèle: p_graph_time -> lissage + min_stable).
- ➕ (ajout) Colonnes d'export: `is_SOZ` (0/1) et `success` (0/1, d'après g.y; vide si inconnu).

Entrées:
- JSON de splits (CV ou simple) structurant les séquences par patient.
- Modules existants:
    - sequence_dataset.py (SequenceDataset, collate_sequences)
    - model_backbone.py  (GraphBackboneNodeGRU)

Sorties:
- checkpoints/<patient>_nodegru_hid{hid}.pt
- <out_dir>/<patient>/seizure_XXXX.csv
- <out_dir>/<patient>/all_nodeprobs.parquet
"""

import os
import json
import argparse
from typing import Any, List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- Vos modules existants ---
from aa_M1_sequence_dataset_copy import SequenceDataset, collate_sequences
from aa_M1_model_backbone_embedding import GraphBackboneNodeGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================= Helpers collate/signatures =======================

def _normalize_collate_output(packet: Any):
    """Compat signature pour collate_sequences.
    Retourne: batches_time, seq_ids_time (ou None), y_graph_time, B
    """
    if not isinstance(packet, (tuple, list)):
        raise ValueError(f"collate_sequences must return tuple/list, got: {type(packet)}")
    if len(packet) == 6:
        batches_time, seq_ids_time, y_graph_time, _, _, B = packet; return batches_time, seq_ids_time, y_graph_time, int(B)
    if len(packet) == 5:
        batches_time, seq_ids_time, y_graph_time, _, B = packet; return batches_time, seq_ids_time, y_graph_time, int(B)
    if len(packet) == 4:
        batches_time, seq_ids_time, y_graph_time, B = packet; return batches_time, seq_ids_time, y_graph_time, int(B)
    if len(packet) == 3:
        batches_time, y_graph_time, B = packet; return batches_time, None, y_graph_time, int(B)
    if len(packet) == 2:
        batches_time, y_graph_time = packet; return batches_time, None, y_graph_time, 1
    raise ValueError(f"Unknown collate signature (len={len(packet)})")


def _infer_in_dim_from_dataset(ds: SequenceDataset) -> int:
    """Récupère in_dim à partir du premier graphe de la première séquence."""
    seq0 = ds[0]
    if not seq0:
        raise ValueError("Sequence 0 is empty.")
    g0 = seq0[0]
    x = getattr(g0, "x", None)
    if x is None:
        raise ValueError("Graph has no node features 'x'. Cannot infer in_dim.")
    return int(x.shape[1])


# ======================= Lecture/repérage onset (phase / first_ictal_epoch) =======================

def _get_phase_from_graph(g) -> Optional[str]:
    """Renvoie 'ictal' / 'preictal' si détectable, sinon None."""
    meta = getattr(g, "meta", {}) or {}
    ph = meta.get("phase", None)
    if ph is not None:
        return str(ph).strip().lower()

    # fallback: essayer via un chemin stocké
    pth = meta.get("path", None) or getattr(g, "path", None)
    if pth is not None:
        p = str(pth).lower()
        if "ictal" in p:
            return "ictal"
        if "preictal" in p:
            return "preictal"
    return None


def _find_onset_index(seq: List[Any]) -> Optional[int]:
    """Trouve l'index du premier epoch ictal.
    Priorité 1: meta['phase'] == 'ictal'
    Priorité 2: flag 'first_ictal_epoch' == True
    Sinon: None.
    """
    # phase
    for t, g in enumerate(seq):
        if _get_phase_from_graph(g) == "ictal":
            return t
    # flag explicite
    for t, g in enumerate(seq):
        if getattr(g, "first_ictal_epoch", False):
            return t
    return None


# ======================= Lissage & détection onset (logique modèle) =======================

def moving_average_1d(arr: np.ndarray, W: int) -> np.ndarray:
    if W <= 1 or arr.size == 0:
        return arr
    kernel = np.ones(W, dtype=float) / float(W)
    return np.convolve(arr, kernel, mode="same")


def detect_onset_simple(p: np.ndarray, thr: float = 0.5, min_stable: int = 3) -> int:
    """Retourne l'index du premier run de longueur >= min_stable où p >= thr (lissé en amont)."""
    if p.size == 0:
        return -1
    T = p.size
    above = (p >= thr).astype(np.int32)
    run = 0
    for t in range(T):
        if above[t]:
            run += 1
            if run >= min_stable:
                return t - min_stable + 1
        else:
            run = 0
    # fallback: point de proba max
    return int(np.argmax(p))


# ======================= Split: fusion train+val+test =======================

def build_all_in_train_split(base_splits_path: str, patient_key: str) -> str:
    """
    Construit un JSON temporaire où TOUT (train+val+test) est en 'train' pour ce patient,
    en conservant EXACTEMENT la structure des items (list/dict/scalaires).
    Déduplication robuste via json.dumps(sort_keys=True).
    """
    if not os.path.exists(base_splits_path):
        raise FileNotFoundError(f"[splits] Fichier introuvable: {base_splits_path}")

    with open(base_splits_path, "r") as f:
        node = json.load(f)

    if patient_key not in node:
        raise ValueError(f"Patient '{patient_key}' introuvable dans {base_splits_path}")

    folds = node[patient_key]

    def _extend_unique(dest_list: List[Any], seen_keys: set, items: List[Any]):
        for it in items:
            try:
                key = json.dumps(it, sort_keys=True)
            except TypeError:
                key = str(it)
            if key not in seen_keys:
                seen_keys.add(key)
                dest_list.append(it)

    all_entries: List[Any] = []
    seen = set()

    if isinstance(folds, list):
        for fold in folds:  # CV
            for k in ("train", "val", "test"):
                if k in fold and isinstance(fold[k], list):
                    _extend_unique(all_entries, seen, fold[k])
    elif isinstance(folds, dict):
        for k in ("train", "val", "test"):
            if k in folds and isinstance(folds[k], list):
                _extend_unique(all_entries, seen, folds[k])
    else:
        raise ValueError("Format de splits inconnu pour ce patient.")

    flat = {
        patient_key: {
            "train": all_entries,
            "val": [],
            "test": [],
            "__all_in_train": True,
        }
    }

    tmpdir = os.path.join(os.path.dirname(base_splits_path), "tmp_all_train")
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path = os.path.join(tmpdir, f"split_{patient_key.replace('::','__')}_ALLTRAIN.json")
    with open(tmp_path, "w") as f:
        json.dump(flat, f)
    return tmp_path


# ======================= Entraînement M1 =======================

def _print_training_sequences_report(ds: SequenceDataset, patient_key: str):
    """
    Imprime le rapport des séquences utilisées pour entraîner M1:
    seq_index | seizure | T | onset_idx | onset_%
    (onset basé sur phase/first_ictal_epoch, pas sur y outcome)
    """
    print(f"\n[TRAIN SEQUENCES] Patient={patient_key}")
    print("  seq_index | seizure |  T | onset_idx | onset_%")
    for seq_idx in range(len(ds)):
        seq = ds[seq_idx]
        if not seq:
            print(f"  {seq_idx:9d} |   NA    |  0 |     NA    |   NA")
            continue
        meta0 = getattr(seq[0], "meta", {}) or {}
        seizure_clin = meta0.get("seizure", None)
        T = len(seq)
        t_true = _find_onset_index(seq)
        if t_true is None:
            print(f"  {seq_idx:9d} | {str(seizure_clin):7s} | {T:3d} |    none   |   none")
        else:
            onset_pct = (t_true / max(1, T - 1)) * 100.0
            print(f"  {seq_idx:9d} | {str(seizure_clin):7s} | {T:3d} | {t_true:8d} | {onset_pct:6.2f}%")


def train_m1_for_patient(
    splits_all_train: str,
    patient_key: str,
    hid: int = 128,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    ckpt_dir: str = "checkpoints",
) -> str:
    """
    Entraîne un NodeGRU (M1) sur TOUTES les séquences du patient (split unique 'train').
    Affiche le rapport des séquences utilisées. Sauvegarde le checkpoint et retourne son chemin.
    """
    ds = SequenceDataset(splits_all_train, patient_key, "train")
    if len(ds) == 0:
        raise ValueError(f"[{patient_key}] Dataset vide (train). Vérifie les splits.")
    _print_training_sequences_report(ds, patient_key)

    in_dim = _infer_in_dim_from_dataset(ds)
    net = GraphBackboneNodeGRU(in_dim=in_dim, hid=hid).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = torch.nn.BCELoss()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_sequences, num_workers=0)

    for ep in range(1, epochs + 1):
        net.train()
        ep_loss, steps = 0.0, 0
        for packet in dl:
            batches_time, seq_ids_time, y_graph_time, B = _normalize_collate_output(packet)
            if B <= 0:
                continue
            batches_time = [bt.to(DEVICE) for bt in batches_time]
            y_graph_time = [y.to(DEVICE) for y in y_graph_time]

            # Forward: proba globale p_g(t) pour BCE temporelle
            p_graph_time = net.forward_sequence(batches_time, seq_ids_time)  # List[Tensor]

            losses = []
            for p_t, y_t in zip(p_graph_time, y_graph_time):
                if p_t.numel() == 0:
                    continue
                y_t = y_t.to(p_t.device).view(-1)
                m = min(p_t.numel(), y_t.numel())
                if m == 0:
                    continue
                losses.append(torch.nn.functional.binary_cross_entropy(p_t[:m], y_t[:m]))
            if not losses:
                continue

            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())
            steps += 1

        print(f"[{patient_key}] epoch {ep:03d} | train_loss {ep_loss / max(1, steps):.4f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{patient_key.replace('::','__')}_nodegru_hid{hid}.pt")
    torch.save(
        {"state_dict": net.state_dict(), "in_dim": in_dim, "hid": hid, "model": "nodegru"},
        ckpt_path,
    )
    return ckpt_path


def _load_m1_from_ckpt(ckpt_path: str) -> GraphBackboneNodeGRU:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    hid = int(ckpt.get("hid", 128))
    net = GraphBackboneNodeGRU(in_dim=in_dim, hid=hid).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()
    return net


# ======================= Inférence: 1 fenêtre centrée par SEIZURE =======================

def infer_one_centered_window_per_seizure(
    splits_all_train: str,
    patient_key: str,
    ckpt_path: str,
    seconds_pre: int = 20,
    seconds_post: int = 20,
    epoch_sec: float = 1.0,        # durée (s) d'un pas/epoch/graphe
    smooth_W: int = 5,
    thr: float = 0.5,
    min_stable: int = 3,
    out_dir: str = "m1_nodeprobs_centered",
):
    """
    Pour chaque CRISE clinique (meta['seizure']), exporte UNE SEULE fenêtre centrée
    sur l'onset réel (premier epoch ictal), avec seconds_pre avant et seconds_post après.
    - Choisit la séquence (parmi celles du JSON) qui couvre le mieux cette fenêtre.
    - Sauvegarde 1 CSV par crise (seizure_XXXX.csv) + Parquet agrégé.
    - Colonnes: t_rel (pas), t_rel_sec (s), electrode_name, is_SOZ (0/1), success (0/1),
      p_node, true_ictal, pred_ictal.
    - pred_ictal suit EXACTEMENT la logique modèle: lissage p_graph + détection onset (thr, min_stable),
      puis pred_ictal[t] = 1 pour t >= t_pred, 0 sinon.
    """
    from torch_geometric.data import Batch

    os.makedirs(out_dir, exist_ok=True)
    patient_dir = os.path.join(out_dir, patient_key.replace('::','__'))
    os.makedirs(patient_dir, exist_ok=True)

    ds = SequenceDataset(splits_all_train, patient_key, "train")
    if len(ds) == 0:
        print(f"[WARN] {patient_key}: aucune séquence trouvée.")
        return

    net = _load_m1_from_ckpt(ckpt_path)
    net.eval()

    # Regrouper les seq_index par ID de seizure clinique
    by_seizure: Dict[int, List[int]] = {}
    for seq_idx in range(len(ds)):
        seq = ds[seq_idx]
        if not seq:
            continue
        meta0 = getattr(seq[0], "meta", {}) or {}
        seiz_id = meta0.get("seizure", None)
        if seiz_id is None:
            continue
        by_seizure.setdefault(int(seiz_id), []).append(seq_idx)

    if not by_seizure:
        print(f"[WARN] {patient_key}: aucune crise clinique trouvée (meta['seizure'] manquant ?).")
        return

    steps_pre  = int(round(seconds_pre  / epoch_sec))
    steps_post = int(round(seconds_post / epoch_sec))

    all_rows = []

    with torch.no_grad():
        for seiz_id, seq_indices in sorted(by_seizure.items()):
            # Choisir la meilleure séquence pour couvrir [t_true-steps_pre, t_true+steps_post]
            best = None  # (coverage, seq_idx, t0, t1, t_true, node_names, soz_bool)
            for seq_idx in seq_indices:
                seq = ds[seq_idx]
                if not seq:
                    continue

                # onset réel (premier ictal via phase/flag)
                t_true = _find_onset_index(seq)
                if t_true is None:
                    continue

                T = len(seq)
                t0_want = t_true - steps_pre
                t1_want = t_true + steps_post
                t0 = max(0, t0_want)
                t1 = min(T - 1, t1_want)
                if t1 <= t0:
                    continue
                coverage = (t1 - t0 + 1)

                g0 = seq[0]
                node_names = getattr(g0, "node_names", None) or [f"ch{e}" for e in range(g0.x.size(0))]
                node_is_soz = getattr(g0, "node_is_soz", None)
                if node_is_soz is not None:
                    vec = node_is_soz.view(-1).cpu().numpy().astype(int)
                    def _m(v): return True if v==1 else (False if v==0 else None)
                    soz_bool = [_m(int(v)) for v in vec]
                else:
                    soz_bool = [None] * len(node_names)

                cand = (coverage, seq_idx, t0, t1, t_true, node_names, soz_bool)
                if (best is None) or (coverage > best[0]):
                    best = cand

                full_len = (steps_pre + steps_post + 1)
                if coverage >= full_len:
                    break  # full coverage atteint

            if best is None:
                print(f"[WARN] {patient_key} | seizure {seiz_id}: aucune séquence couvrant l'onset.")
                continue

            coverage, seq_idx, t0, t1, t_true, node_names, soz_bool = best
            seq = ds[seq_idx]

            # Forward sur la fenêtre choisie
            sub_seq = seq[t0:t1+1]
            batches_time = [Batch.from_data_list([g]).to(DEVICE) for g in sub_seq]
            # On demande p_graph_time ET p_node_time
            p_graph_time, p_node_time = net.forward_sequence(
                batches_time, seq_ids_time=None, return_node_probs=True
            )

            # p_node_time -> matrice (T_win, N)
            P_list = []
            for t in range(len(p_node_time)):
                nodes_list = p_node_time[t]
                if not nodes_list:
                    continue
                p_nodes = nodes_list[0].detach().cpu().view(-1).numpy()
                P_list.append(p_nodes)
            if not P_list:
                print(f"[WARN] {patient_key} | seizure {seiz_id}: p_node_time vide.")
                continue
            P = np.stack(P_list, axis=0)  # (T_win, N)

            # p_graph_time -> vecteur (T_win,) puis lissage + onset modèle
            pg_raw = []
            for t in range(len(p_graph_time)):
                pt = p_graph_time[t]
                if isinstance(pt, torch.Tensor):
                    pg_raw.append(float(pt.view(-1).mean().detach().cpu().item()))
                else:
                    pg_raw.append(float(pt))
            pg_raw = np.asarray(pg_raw, dtype=np.float32)
            pg_smooth = moving_average_1d(pg_raw, smooth_W)
            t_pred_model = detect_onset_simple(pg_smooth, thr=thr, min_stable=min_stable)

            # true_ictal(t) via phase des graphs de sub_seq
            true_ictal = []
            for g in sub_seq:
                true_ictal.append(1 if _get_phase_from_graph(g) == "ictal" else 0)
            true_ictal = np.asarray(true_ictal, dtype=int)

            # pred_ictal(t): 1 pour t >= t_pred_model, sinon 0
            if t_pred_model < 0:
                pred_ictal = np.zeros_like(true_ictal, dtype=int)
            else:
                pred_ictal = np.zeros_like(true_ictal, dtype=int)
                pred_ictal[t_pred_model:] = 1

            # ------ New: succès patient (graph-level), dérivé de g.y ------
            # Convention build_pyg_graphs: y = 1 (good outcome), 0 (bad outcome), -1 (inconnu)
            g0_for_y = seq[0]
            y_tensor = getattr(g0_for_y, "y", None)
            success_val = None
            if isinstance(y_tensor, torch.Tensor) and y_tensor.numel() > 0:
                y_item = int(y_tensor.view(-1)[0].item())
                if y_item == 1:
                    success_val = 1
                elif y_item == 0:
                    success_val = 0
                else:
                    success_val = None  # inconnu -> vide dans le CSV

            # Sauvegarde CSV par crise
            rows = []
            for i in range(P.shape[0]):
                t_abs = t0 + i           # index absolu dans la séquence choisie
                t_rel = t_abs - t_true   # onset au milieu => 0
                t_rel_sec = t_rel * epoch_sec
                for e, name in enumerate(node_names):
                    # is_SOZ 0/1 (vide si inconnu)
                    sb = soz_bool[e]
                    is_soz_int = (1 if sb is True else (0 if sb is False else None))
                    rows.append({
                        "patient": patient_key,
                        "seizure": int(seiz_id),
                        "seq_index": int(seq_idx),          # séquence sélectionnée
                        "t_rel": int(t_rel),                # en pas
                        "t_rel_sec": float(t_rel_sec),      # en secondes
                        "t_abs": int(t_abs),
                        "t0_abs": int(t0),
                        "t1_abs": int(t1),
                        "electrode_id": int(e),
                        "electrode_name": name,
                        "is_SOZ": is_soz_int,               # 0/1
                        "success": success_val,             # 0/1 selon y (good/bad), vide si inconnu
                        "p_node": float(P[i, e]),
                        "true_ictal": int(true_ictal[i]),
                        "pred_ictal": int(pred_ictal[i]),
                    })
            df = pd.DataFrame(rows)
            out_csv = os.path.join(patient_dir, f"seizure_{int(seiz_id):04d}.csv")
            df.to_csv(out_csv, index=False)
            print(f"[OK] {patient_key} | seizure {seiz_id}: window [{t0}..{t1}] (onset_true={t_true}, onset_pred={t_pred_model}) -> {out_csv}")
            all_rows.append(df)

    # Parquet agrégé par patient
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_parquet(os.path.join(patient_dir, "all_nodeprobs.parquet"), index=False)
        print(f"[OK] Export parquet -> {patient_dir}")


# ======================= CLI =======================

def main():
    ap = argparse.ArgumentParser(description="M1 per-patient training on ALL sequences + centered-window inference per seizure (with true_ictal & pred_ictal).")
    # Chemin du JSON de splits (CV ou simple)
    ap.add_argument("--splits", type=str, required=True,
                    help="Chemin vers le fichier JSON des splits (ex: .../gnn/pyg/splits_patient_specific_cv.json)")
    # Filtre optionnel sur les patients (contains)
    ap.add_argument("--patients-like", type=str, default=None,
                    help="Filtre substring sur le nom du patient. Si None: tous les patients du JSON.")
    # Hyperparamètres M1 (fixes pour tous les patients)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Fenêtre centrée (en secondes) pour l'inférence
    ap.add_argument("--seconds-pre", type=int, default=20)
    ap.add_argument("--seconds-post", type=int, default=20)
    ap.add_argument("--epoch-sec", type=float, default=1.0, help="Durée (s) d'un pas/epoch/graphe.")
    # Logique modèle (lissage & détection onset)
    ap.add_argument("--smooth-W", type=int, default=5, help="Taille de fenêtre du moving average sur p_graph.")
    ap.add_argument("--thr", type=float, default=0.5, help="Seuil de décision sur p_graph lissé.")
    ap.add_argument("--min-stable", type=int, default=3, help="Run minimal au-dessus du seuil pour valider l'onset.")
    # Dossiers de sortie
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--out-dir", type=str, default="m1_nodeprobs_centered20s")
    args = ap.parse_args()

    if not os.path.exists(args.splits):
        raise FileNotFoundError(f"[splits] Introuvable: {args.splits}")

    # Charger les patients
    with open(args.splits, "r") as f:
        obj = json.load(f)
    patients = list(obj.keys())
    if args.patients_like:
        patients = [p for p in patients if args.patients_like in p]
    if not patients:
        print("[WARN] Aucun patient sélectionné (filtre ?).")
        return

    # Snapshot HP
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "m1_hyperparams.json"), "w") as f:
        json.dump({
            "model": "nodegru",
            "hid": args.hid,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seconds_pre": args.seconds_pre,
            "seconds_post": args.seconds_post,
            "epoch_sec": args.epoch_sec,
            "smooth_W": args.smooth_W,
            "thr": args.thr,
            "min_stable": args.min_stable,
        }, f, indent=2)

    # Boucle patients
    for patient_key in patients:
        print(f"\n===== PATIENT {patient_key} =====")
        # 1) Split temporaire "tout en train" (train+val+test)
        split_all_train = build_all_in_train_split(args.splits, patient_key)

        # 2) Entraîner M1 sur toutes les séquences listées
        ckpt_path = train_m1_for_patient(
            splits_all_train=split_all_train,
            patient_key=patient_key,
            hid=args.hid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            ckpt_dir=args.ckpt_dir,
        )
        print(f"[SAVE] M1 checkpoint: {ckpt_path}")

        # 3) Inférence: 1 fenêtre centrée (seconds_pre/seconds_post) par SEIZURE
        infer_one_centered_window_per_seizure(
            splits_all_train=split_all_train,
            patient_key=patient_key,
            ckpt_path=ckpt_path,
            seconds_pre=args.seconds_pre,
            seconds_post=args.seconds_post,
            epoch_sec=args.epoch_sec,
            smooth_W=args.smooth_W,
            thr=args.thr,
            min_stable=args.min_stable,
            out_dir=args.out_dir,
        )

    print("\n[DONE] Tous les patients traités.")


if __name__ == "__main__":
    main()
