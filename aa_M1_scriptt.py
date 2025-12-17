import os, json, argparse, torch, csv, math
from datetime import datetime

from utils.config import load_config, resolve_device, set_seed, ensure_dir
from aa_M1_training import train_one_patient
from aa_M1_evaluation import (
    evaluate_and_plot_on_test,
    load_patient_splits_from_json,
    normalize_requested_splits,
    plot_global_summary,
    plot_per_sequence_variability,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device_str = resolve_device(cfg)
    device = torch.device(device_str)


    # Dossier principal défini dans le YAML
    base_results_dir = cfg["experiment"]["output_dir"]

    # Ajouter un timestamp au dossier de résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_results_dir, f"results_{timestamp}")
    ensure_dir(exp_dir)

    # Sauvegarder la config utilisée dans ce dossier
    config_copy_path = os.path.join(exp_dir, "config_used.yaml")
    try:
        import shutil
        shutil.copyfile(args.config, config_copy_path)
        print(f"[INFO] Copie de la config sauvegardée -> {config_copy_path}")
    except Exception as e:
        print(f"[WARN] Impossible de copier le fichier YAML: {e}")

    # 3) IMPORTANT: propager le chemin horodaté dans la config
    cfg["experiment"]["output_dir"] = exp_dir
    # --- Créer un sous-dossier checkpoints horodaté ---
    ckpt_dir = os.path.join(exp_dir, f"checkpoints_{timestamp}")
    ensure_dir(ckpt_dir)
    cfg["train"]["ckpt_dir"] = ckpt_dir  # injecté dans la config pour le train()

    patients = list(cfg["run"].get("patients", []))
    splits_json = cfg["data"]["splits_all_train"]

    # aggregate holders
    all_results, all_per_seq_rows = [], []

    # pre-read patient keys present in JSON
    try:
        with open(splits_json, "r") as f:
            all_splits_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot read splits JSON: {splits_json} ({e})")

    for patient_key in patients:
        available = set(all_splits_data.keys())
        if patient_key not in available:
            print(f"[{patient_key}] ⚠️ not in {splits_json}; skipping.")
            continue

        # training (per patient)
        #ckpt_dir = os.path.join(exp_dir, cfg["train"].get("ckpt_subdir", "checkpoints"))
        ckpt_dir = cfg["train"].get("ckpt_dir", os.path.join(exp_dir, "checkpoints"))

        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = f"{patient_key.replace('::','__')}_{cfg['experiment'].get('specificity','exp')}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        if os.path.exists(ckpt_path) and str(cfg["train"].get("if_exists", "skip")).lower() == "skip":
            print(f"[{patient_key}] checkpoint exists, skip training.")
        else:
            ckpt_path = train_one_patient(cfg, patient_key, device, ckpt_path=ckpt_path)

        # evaluation
        res, per_seq = evaluate_and_plot_on_test(cfg, patient_key, ckpt_path, device)
        if res:
            all_results.append(res)
        all_per_seq_rows.extend(per_seq or [])

    # write per-patient metrics
    results_dir = os.path.join(exp_dir, cfg["eval"].get("results_subdir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    per_patient_csv = os.path.join(results_dir, "metrics_per_patient.csv")
    per_patient_json = os.path.join(results_dir, "metrics_per_patient.json")
    with open(per_patient_json, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(per_patient_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["patient","fold_idx","split_mode","n_sequences","loss_bce_graph","roc_auc_graph","results_dir"])
        for r in all_results:
            loss_str = "" if r.get("loss_bce_graph") is None else f"{r['loss_bce_graph']:.6f}"
            auc_str = "" if r.get("roc_auc_graph") is None else f"{r['roc_auc_graph']:.6f}"
            writer.writerow([
                r.get("patient"), r.get("fold_idx"), r.get("split_mode"), r.get("n_sequences"), loss_str, auc_str, r.get("results_dir","")
            ])
    print(f"[GLOBAL] CSV -> {os.path.abspath(per_patient_csv)}")
    print(f"[GLOBAL] JSON -> {os.path.abspath(per_patient_json)}")

    # write per-sequence metrics
    per_sequence_csv = os.path.join(results_dir, "metrics_per_sequence.csv")
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
    print(f"[GLOBAL] per-sequence CSV -> {os.path.abspath(per_sequence_csv)}")

    # figures
    try:
        plot_global_summary([r for r in all_results if r], out_png=os.path.join(results_dir, "summary_metrics.png"))
    except Exception as e:
        print(f"[GLOBAL] ⚠️ summary figure failed: {e}")
    try:
        plot_per_sequence_variability(all_per_seq_rows, out_png=os.path.join(results_dir, "summary_per_sequence.png"))
    except Exception as e:
        print(f"[GLOBAL] ⚠️ per-sequence figure failed: {e}")

if __name__ == "__main__":
    main()
