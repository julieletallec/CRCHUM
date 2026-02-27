#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrateur M1 (pas de grid-search):
- charge UNE config M1 (config_used.yaml)
- entraîne un modèle M1 par patient (train_one_patient)
- évalue et génère les outputs comme avant (evaluate_and_plot_on_test + plots)
- la liste de patients est fournie au runtime (ou via fichier txt)

Usage:
uv run 12_trainandrun_best_M1_on_all_bad_patients.py \
  --config_m1 /home/julieletallec/test/M2_singlefit_out/config_used__M1.yaml \
  --out_root /home/julieletallec/test/M1_singleconfig_runs \
  --patients "CHUM::Patient_09,CHUM::Patient_16,CHUM::Patient_17,CHUM::Patient_21,ds004100::sub-HUP080,ds004100::sub-HUP112,ds004100::sub-HUP114,ds004100::sub-HUP133,ds004100::sub-HUP138,ds004100::sub-HUP151,ds004100::sub-HUP162,ds004100::sub-HUP171,ds004100::sub-HUP172,ds004100::sub-HUP181,ds004100::sub-HUP187,ds004100::sub-HUP188"


Optionnel:
  --splits_json /path/to/splits_patient_specific_cv.json   (override cfg["data"]["splits_all_train"])
  --if_exists skip|overwrite
"""

import os
import json
import csv
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import yaml

from utils.config import resolve_device, set_seed, ensure_dir

# Tes modules existants
from aa_M1_training import train_one_patient
from aa_M1_evaluation import (
    evaluate_and_plot_on_test,
    plot_global_summary,
    plot_per_sequence_variability,
)


# -----------------------
# Helpers
# -----------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def parse_patients_list(patients_arg: str) -> List[str]:
    # accepte "a,b,c" ou "a b c"
    if not patients_arg:
        return []
    s = patients_arg.replace("\n", " ").replace(",", " ").strip()
    return [x for x in (p.strip() for p in s.split(" ")) if x]


def read_patients_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


def ensure_out_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = out_root / f"results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_patient_keys_from_splits_json(splits_json: Path) -> List[str]:
    with open(splits_json, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    return []


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_m1", required=True, help="Chemin vers config_used.yaml (M1 unique)")
    ap.add_argument("--out_root", required=True, help="Dossier racine où créer results_YYYYMMDD_HHMMSS/")

    # Patients: soit --patients, soit --patients_file, soit 'all from splits'
    ap.add_argument("--patients", default="", help="Liste patients: 'a,b,c' ou 'a b c'")
    ap.add_argument("--patients_file", default="", help="Fichier txt: 1 patient par ligne")
    ap.add_argument("--patients_like", default="", help="Filtre substring si tu veux subset")

    # Override splits JSON
    ap.add_argument("--splits_json", default="", help="Override cfg['data']['splits_all_train']")

    # comportement checkpoints existants
    ap.add_argument("--if_exists", default="", choices=["", "skip", "overwrite"],
                    help="Override cfg['train']['if_exists'] (skip/overwrite). Si vide: garde config.")

    args = ap.parse_args()

    cfg_path = Path(args.config_m1).expanduser()
    out_root = Path(args.out_root).expanduser()

    if not cfg_path.is_file():
        raise SystemExit(f"config_m1 introuvable: {cfg_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(cfg_path)

    # Seed + device
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    device_str = resolve_device(cfg)
    device = torch.device(device_str)

    # Override splits
    if args.splits_json.strip():
        splits_json = Path(args.splits_json).expanduser()
        if not splits_json.is_file():
            raise SystemExit(f"splits_json introuvable: {splits_json}")
        cfg.setdefault("data", {})
        cfg["data"]["splits_all_train"] = str(splits_json)
    else:
        splits_json = Path(cfg["data"]["splits_all_train"]).expanduser()
        if not splits_json.is_file():
            raise SystemExit(f"splits_all_train introuvable (cfg): {splits_json}")

    # Override if_exists
    if args.if_exists:
        cfg.setdefault("train", {})
        cfg["train"]["if_exists"] = args.if_exists

    # Patients list
    patients: List[str] = []
    if args.patients_file.strip():
        pf = Path(args.patients_file).expanduser()
        if not pf.is_file():
            raise SystemExit(f"patients_file introuvable: {pf}")
        patients.extend(read_patients_file(pf))

    patients.extend(parse_patients_list(args.patients))

    # Si rien fourni: tous ceux du JSON
    if not patients:
        patients = get_patient_keys_from_splits_json(splits_json)

    # filtre substring optionnel
    if args.patients_like.strip():
        patients = [p for p in patients if args.patients_like in p]

    # dédoublonnage en gardant l'ordre
    seen = set()
    patients = [p for p in patients if not (p in seen or seen.add(p))]

    if not patients:
        raise SystemExit("Aucun patient sélectionné (patients/patients_file/patients_like).")

    # Créer dossier results timestampé
    exp_dir = ensure_out_dir(out_root)
    print(f"[INFO] exp_dir = {exp_dir}")

    # Sauver config utilisée *avec overrides* (reproductible)
    save_yaml(cfg, exp_dir / "config_used.yaml")

    # Injecter output_dir + ckpt_dir comme avant
    cfg.setdefault("experiment", {})
    cfg["experiment"]["output_dir"] = str(exp_dir)

    ts = exp_dir.name.replace("results_", "")
    ckpt_dir = exp_dir / f"checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("train", {})
    cfg["train"]["ckpt_dir"] = str(ckpt_dir)

    print(f"[INFO] checkpoints -> {ckpt_dir}")
    print(f"[INFO] splits_json -> {splits_json}")
    print(f"[INFO] patients ({len(patients)}): {patients}")

    # Pre-read JSON keys to avoid spam errors
    with open(splits_json, "r") as f:
        all_splits_data = json.load(f)
    available = set(all_splits_data.keys())

    all_results: List[Dict[str, Any]] = []
    all_per_seq_rows: List[Dict[str, Any]] = []

    for patient_key in patients:
        if patient_key not in available:
            print(f"[{patient_key}] ⚠️ absent de {splits_json}; skip.")
            continue

        # ckpt name identique à ton runner
        ckpt_name = f"{patient_key.replace('::','__')}_{cfg['experiment'].get('specificity','exp')}.pt"
        ckpt_path = ckpt_dir / ckpt_name
        print(ckpt_path)

        # training per patient
        if ckpt_path.exists() and str(cfg["train"].get("if_exists", "skip")).lower() == "skip":
            print(f"[{patient_key}] checkpoint exists -> skip training ({ckpt_path.name})")
        else:
            print(f"[{patient_key}] training...")
            train_one_patient(cfg, patient_key, device, ckpt_path=str(ckpt_path))

        # evaluation / inference (comme avant)
        print(f"[{patient_key}] evaluation...")
        res, per_seq = evaluate_and_plot_on_test(cfg, patient_key, str(ckpt_path), device)
        if res:
            all_results.append(res)
        if per_seq:
            all_per_seq_rows.extend(per_seq)

    # Outputs comme ton script
    results_dir = exp_dir / cfg.get("eval", {}).get("results_subdir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    per_patient_csv = results_dir / "metrics_per_patient.csv"
    per_patient_json = results_dir / "metrics_per_patient.json"

    per_patient_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    with open(per_patient_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["patient","fold_idx","split_mode","n_sequences","loss_bce_graph","roc_auc_graph","results_dir"])
        for r in all_results:
            loss = r.get("loss_bce_graph")
            auc = r.get("roc_auc_graph")
            writer.writerow([
                r.get("patient"),
                r.get("fold_idx"),
                r.get("split_mode"),
                r.get("n_sequences"),
                "" if loss is None else f"{loss:.6f}",
                "" if auc is None else f"{auc:.6f}",
                r.get("results_dir","")
            ])

    print(f"[GLOBAL] CSV -> {per_patient_csv}")
    print(f"[GLOBAL] JSON -> {per_patient_json}")

    # Per-sequence csv (si présent)
    per_sequence_csv = results_dir / "metrics_per_sequence.csv"
    with open(per_sequence_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["patient","seizure_id","seq_index","len_seq","auc_seq","bce_seq","t_true","t_pred","delay"])
        for r in all_per_seq_rows:
            if r is None:
                continue
            auc_seq = r.get("auc_seq")
            bce_seq = r.get("bce_seq")
            w.writerow([
                r.get("patient"),
                r.get("seizure_id"),
                r.get("seq_index"),
                r.get("len_seq"),
                "" if (auc_seq is None or (isinstance(auc_seq,float) and math.isnan(auc_seq))) else f"{auc_seq:.6f}",
                "" if (bce_seq is None or (isinstance(bce_seq,float) and math.isnan(bce_seq))) else f"{bce_seq:.6f}",
                "" if r.get("t_true") is None else r["t_true"],
                "" if r.get("t_pred") is None else r["t_pred"],
                "" if r.get("delay") is None else r["delay"],
            ])
    print(f"[GLOBAL] per-sequence CSV -> {per_sequence_csv}")

    # figures globales
    try:
        plot_global_summary([r for r in all_results if r], out_png=str(results_dir / "summary_metrics.png"))
    except Exception as e:
        print(f"[GLOBAL] ⚠️ summary figure failed: {e}")

    try:
        plot_per_sequence_variability(all_per_seq_rows, out_png=str(results_dir / "summary_per_sequence.png"))
    except Exception as e:
        print(f"[GLOBAL] ⚠️ per-sequence figure failed: {e}")

    print(f"\n[DONE] M1 run terminé -> {exp_dir}")


if __name__ == "__main__":
    main()
