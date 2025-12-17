import os, math, json, numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from models.model_backbone import GraphBackboneNodeGRU
from aa_M1_sequence_dataset_copy import SequenceDataset, collate_sequences

# -------------------------- helpers --------------------------

def step_vector_from_onset(T: int, t_pred: int | None) -> np.ndarray:
    """Construit un vecteur binaire (T,) = 0 avant t_pred, 1 à partir de t_pred."""
    y = np.zeros((T,), dtype=np.float32)
    if t_pred is not None and 0 <= t_pred < T:
        y[t_pred:] = 1.0
    return y



def get_electrode_names_from_graph(g) -> list[str] | None:
    """
    Essaie plusieurs clés/attributs pour obtenir les noms d'électrodes par nœud.
    Retourne une liste[str] de longueur N si dispo, sinon None.
    """
    # attributs directs possibles
    for attr in ("electrode_name", "electrode_names", "node_names", "names"):
        if hasattr(g, attr):
            v = getattr(g, attr)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            if torch.is_tensor(v):
                return [str(x) for x in v.tolist()]

    # meta dict
    meta = getattr(g, "meta", {}) or {}
    for k in ("electrode_name", "electrode_names", "node_names", "names"):
        if k in meta:
            v = meta[k]
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            if torch.is_tensor(v):
                return [str(x) for x in v.tolist()]

    return None


def get_outcome_from_graph(g) -> str | None:
    """
    Récupère l'outcome patient s'il est présent (dans g.outcome ou g.meta['outcome']).
    """
    if hasattr(g, "outcome"):
        try:
            return str(getattr(g, "outcome"))
        except Exception:
            pass
    meta = getattr(g, "meta", {}) or {}
    if "outcome" in meta and meta["outcome"] is not None:
        return str(meta["outcome"])
    return None


def load_patient_splits_from_json(splits_json_path: str, patient_key: str):
    with open(splits_json_path, "r") as f:
        data = json.load(f)
    if patient_key not in data:
        return set()
    splits_dict = data[patient_key]
    if isinstance(splits_dict, dict):
        return set(splits_dict.keys())
    return set()

def normalize_requested_splits(requested, available):
    if requested is None:
        return tuple(sorted(available))
    if isinstance(requested, (list, tuple)):
        keep = [s for s in requested if s in available]
        return tuple(keep)
    return (requested,) if requested in available else tuple()

class SimpleSeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]

def select_one_sequence_per_seizure(splits_json, patient_key, eval_splits=("test",), fold_idx=0, seed=0):
    rng = np.random.default_rng(seed)
    chosen_sequences, log_rows = [], []
    by_seiz = {}

    for sk in eval_splits:
        try:
            ds_k = SequenceDataset(splits_json, patient_key, sk, fold_idx=fold_idx)
        except Exception:
            continue
        for i in range(len(ds_k)):
            seq = ds_k[i]
            if not seq:
                continue
            meta0 = getattr(seq[0], "meta", {}) or {}
            seiz_id = meta0.get("seizure", None) or f"no_seiz_{sk}"
            by_seiz.setdefault(seiz_id, []).append((sk, i, seq))

    for seiz_id, bucket in by_seiz.items():
        sk, i, seq = bucket[rng.integers(len(bucket))]
        chosen_sequences.append(seq)
        log_rows.append((seiz_id, sk, i, len(seq)))

    log_rows.sort(key=lambda x: str(x[0]))
    return chosen_sequences, log_rows

def get_soz_mask_from_graph(g) -> torch.Tensor | None:
    for attr in ("node_is_soz", "soz_mask"):
        if hasattr(g, attr):
            t = getattr(g, attr)
            if torch.is_tensor(t):
                return t.bool().view(-1)
    meta = getattr(g, "meta", None)
    if isinstance(meta, dict):
        for k in ("node_is_soz", "soz_mask"):
            if k in meta:
                val = meta[k]
                if torch.is_tensor(val):
                    return val.bool().view(-1)
                elif isinstance(val, (list, tuple)):
                    return torch.tensor(val, dtype=torch.bool).view(-1)
    return None

def get_phase_from_graph(g) -> str:
    meta = getattr(g, "meta", {}) or {}
    return str(meta.get("phase", "preictal")).lower()

def moving_average_1d(x: np.ndarray, W: int) -> np.ndarray:
    if W <= 1:
        return x
    return np.convolve(x, np.ones(W) / W, mode="same")

def detect_onset_simple(x: np.ndarray, thr: float = 0.5, min_stable: int = 3) -> int:
    T = len(x)
    for t in range(T - min_stable + 1):
        if np.all(x[t:t+min_stable] >= thr):
            return t
    return -1

def detect_onset_adaptive(x: np.ndarray, pre_window=20, k_sigma=2.0, thr_abs_min=0.55, thr_abs_max=0.85, frac_of_range=0.60, min_stable=3, dmin=0.02):
    T = len(x)
    if T == 0:
        return -1
    w = max(5, min(pre_window, T))
    base = x[:w]
    mu, sigma = float(np.mean(base)), float(np.std(base))
    p95 = float(np.percentile(base, 95))
    H0 = max(mu + k_sigma * sigma, p95, thr_abs_min)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    H_cap_range = x_min + frac_of_range * max(1e-6, (x_max - x_min))
    H = min(H0, thr_abs_max, H_cap_range)
    L = 0.8 * H
    count = 0
    for t in range(T):
        if x[t] >= H:
            if t >= 2 and ((x[t] - x[t-2]) / 2.0) < dmin:
                count = 0
                continue
            count += 1
            if count >= min_stable:
                t0 = t - min_stable + 1
                tail = x[t0 : min(t0 + 3, T)]
                if np.all(tail >= L):
                    return t0
        else:
            count = 0
    if T >= 2:
        dp = np.diff(x)
        t_peak = int(np.argmax(dp)) + 1
        return t_peak
    return -1

def _stack_probs(seq_list):
    if len(seq_list) == 0:
        return torch.empty(0, 0)
    return torch.stack([p.view(-1) for p in seq_list], dim=1)

# ------------------------------ plotting ------------------------------

def plot_probs(
    per_node_probs: torch.Tensor,
    save_path: Path,
    soz_mask: torch.Tensor | None = None,
    title: str = "",
    t_true: int | None = None,
    t_pred: int | None = None,
    p_graph=None,
    p_graph_smooth=None,
):
    """
    - Électrodes SOZ en rouge, non-SOZ en noir.
    - p_graph (noir) et p_graph_smooth (magenta pointillé).
    - Annotations texte 'onset réel' et 'onset prédit' au niveau des barres.
    """
    per_node_probs = per_node_probs.detach().cpu()
    N, T = (per_node_probs.shape if per_node_probs.numel() > 0 else (0, 0))
    x = range(T)

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    if T == 0 or N == 0:
        ax.set_title(title or "séquence vide")
        ax.set_xlabel("temps (pas)")
        ax.set_ylabel("p_v(t)")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(); return

    # --- normaliser le masque SOZ
    if soz_mask is not None and torch.is_tensor(soz_mask):
        soz_mask = soz_mask.view(-1).to(dtype=torch.bool)
        if soz_mask.numel() != N:
            soz_mask = None
    else:
        soz_mask = None

    # --- tracer les courbes par nœud (SOZ rouge / non-SOZ noir)
    legend_handles = []
    if soz_mask is None:
        for n in range(N):
            ax.plot(x, per_node_probs[n].numpy(), color="black", alpha=0.25, linewidth=1.0, zorder=1)
    else:
        sm = soz_mask.cpu().numpy().astype(bool)
        for n in range(N):
            if sm[n]:
                ax.plot(x, per_node_probs[n].numpy(), color="red", alpha=1.0, linewidth=1.6, zorder=1)
            else:
                ax.plot(x, per_node_probs[n].numpy(), color="black", alpha=0.25, linewidth=1.0, zorder=1)
        from matplotlib.lines import Line2D
        legend_handles += [
            Line2D([0],[0], color="red",   lw=2, label="SOZ"),
            Line2D([0],[0], color="black", lw=2, label="non-SOZ"),
        ]

    # --- proba graphe (brute & lissée)
    if p_graph is not None and len(p_graph) > 0:
        ln1, = ax.plot(range(len(p_graph)), p_graph, linewidth=2.2, color="black", label="p_graph", zorder=3)
        legend_handles.append(ln1)
    if p_graph_smooth is not None and len(p_graph_smooth) > 0:
        ln2, = ax.plot(range(len(p_graph_smooth)), p_graph_smooth, linewidth=2.0,
                       linestyle="--", color="magenta", label="p_graph (lissé)", zorder=3)
        legend_handles.append(ln2)

    # --- onsets + annotations
    if t_true is not None and 0 <= t_true < T:
        ax.axvline(t_true, linestyle="--", color="green", linewidth=1.8, zorder=2)
        ax.annotate(
            "onset réel",
            xy=(t_true, 1.0), xycoords=("data", "axes fraction"),
            xytext=(5, 5), textcoords="offset points",
            color="green", fontsize=9, ha="left", va="bottom"
        )
    if t_pred is not None:
        xmax = max(T, (len(p_graph_smooth) if p_graph_smooth is not None else T))
        if 0 <= t_pred < xmax:
            ax.axvline(t_pred, linestyle=":", color="orange", linewidth=2.0, zorder=2)
            ax.annotate(
                "onset prédit",
                xy=(t_pred, 0.95), xycoords=("data", "axes fraction"),
                xytext=(5, -10), textcoords="offset points",
                color="orange", fontsize=9, ha="left", va="top"
            )

    # --- mise en forme
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("temps (pas)")
    ax.set_ylabel("p_v(t)")
    ax.set_title(title or "Courbes de probabilité par électrode")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_global_summary(per_patient_rows, out_png="summary_metrics.png"):
    rows = [r for r in per_patient_rows if r is not None]
    if not rows:
        print("[GLOBAL] no results to plot."); return
    import matplotlib.pyplot as plt
    patients = [r["patient"].replace("::", "·") for r in rows]
    aucs = [r.get("roc_auc_graph") for r in rows]
    losses = [r.get("loss_bce_graph") for r in rows]
    ns = [max(1, int(r.get("n_sequences", 1))) for r in rows]
    import numpy as np
    auc_vals = np.array([x for x in aucs if x is not None], dtype=float)
    loss_vals = np.array([x for x in losses if x is not None], dtype=float)
    macro_auc = float(np.mean(auc_vals)) if auc_vals.size else float("nan")
    macro_loss = float(np.mean(loss_vals)) if loss_vals.size else float("nan")
    def _weighted_mean(vals, weights):
        vv, ww = [], []
        for v, w in zip(vals, weights):
            if v is not None:
                vv.append(v); ww.append(w)
        if not vv: return float("nan")
        vv, ww = np.array(vv, float), np.array(ww, float)
        return float(np.sum(vv * ww) / np.sum(ww))
    micro_auc = _weighted_mean(aucs, ns)
    micro_loss = _weighted_mean(losses, ns)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    idx = np.arange(len(patients))
    auc_plot_vals = [a if a is not None else np.nan for a in aucs]
    ax1.bar(idx, auc_plot_vals); ax1.axhline(0.5, linestyle="--")
    ax1.set_ylabel("ROC-AUC (graph)"); ax1.set_xticks(idx); ax1.set_xticklabels(patients, rotation=80, ha="right")
    ax1.set_ylim(0.0, 1.0); ax1.set_title("AUC per patient")

    ax2 = fig.add_subplot(gs[1, 0])
    loss_plot_vals = [l if l is not None else np.nan for l in losses]
    ax2.bar(idx, loss_plot_vals); ax2.set_ylabel("BCE (graph)")
    ax2.set_xticks(idx); ax2.set_xticklabels(patients, rotation=80, ha="right")
    ax2.set_title("Loss per patient")

    ax3 = fig.add_subplot(gs[2, 0]); ax3.axis("off")
    txt = [
        "Global summary",
        f"• Macro AUC: {macro_auc:.3f}" if not math.isnan(macro_auc) else "• Macro AUC: n/a",
        f"• Micro AUC: {micro_auc:.3f}" if not math.isnan(micro_auc) else "• Micro AUC: n/a",
        f"• Macro Loss: {macro_loss:.4f}" if not math.isnan(macro_loss) else "• Macro Loss: n/a",
        f"• Micro Loss: {micro_loss:.4f}" if not math.isnan(micro_loss) else "• Micro Loss: n/a",
        f"• Patients: {len(rows)}",
        f"• Total sequences: {int(np.sum(ns))}",
    ]
    ax3.text(0.01, 0.95, "\n".join(txt), va="top")

    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(); print(f"[GLOBAL] -> {os.path.abspath(out_png)}")

def plot_per_sequence_variability(per_seq_rows, out_png="summary_per_sequence.png"):
    from collections import defaultdict
    if not per_seq_rows:
        print("[GLOBAL] no per-sequence results."); return
    import matplotlib.pyplot as plt, numpy as np
    auc_by_patient = defaultdict(list)
    bce_by_patient = defaultdict(list)
    delays = []
    for r in per_seq_rows:
        p = r.get("patient","?")
        a = r.get("auc_seq", None)
        l = r.get("bce_seq", None)
        d = r.get("delay", None)
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            auc_by_patient[p].append(a)
        if l is not None and not (isinstance(l, float) and np.isnan(l)):
            bce_by_patient[p].append(l)
        if d is not None:
            delays.append(d)

    patients_auc = sorted(auc_by_patient.keys())
    patients_bce = sorted(bce_by_patient.keys())

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 1.6])

    ax1 = fig.add_subplot(gs[0,0])
    auc_data = [auc_by_patient[p] for p in patients_auc]
    if auc_data:
        ax1.boxplot(auc_data, showfliers=False)
        ax1.set_xticklabels([p.replace("::","·") for p in patients_auc], rotation=70, ha="right")
        ax1.set_ylim(0.0, 1.0); ax1.axhline(0.5, linestyle="--")
        ax1.set_title("Per-sequence ROC-AUC by patient"); ax1.set_ylabel("AUC")
    else:
        ax1.text(0.5,0.5,"No AUC data", ha="center")

    ax2 = fig.add_subplot(gs[1,0])
    bce_data = [bce_by_patient[p] for p in patients_bce]
    if bce_data:
        ax2.boxplot(bce_data, showfliers=False)
        ax2.set_xticklabels([p.replace("::","·") for p in patients_bce], rotation=70, ha="right")
        ax2.set_title("Per-sequence BCE by patient"); ax2.set_ylabel("BCE")
    else:
        ax2.text(0.5,0.5,"No BCE data", ha="center")

    ax3 = fig.add_subplot(gs[2,0])
    if len(delays) > 0:
        ax3.hist(delays, bins=31)
        ax3.set_title("Onset delay distribution (t_pred - t_true)")
        ax3.set_xlabel("Delay (steps; positive = late)"); ax3.set_ylabel("#sequences"); ax3.axvline(0, linestyle="--")
    else:
        ax3.text(0.5,0.5,"No onset delay data", ha="center")

    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(); print(f"[GLOBAL] -> {os.path.abspath(out_png)}")

# ------------------------------ evaluation ------------------------------

def load_model_from_ckpt_(ckpt_path: str, device: torch.device) -> GraphBackboneNodeGRU:
    ckpt = torch.load(ckpt_path, map_location=device)
    net = GraphBackboneNodeGRU(in_dim=ckpt["in_dim"], hid=ckpt["hid"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
def load_model_from_ckpt(ckpt_path: str, device: torch.device, cfg) -> GraphBackboneNodeGRU:
    ckpt = torch.load(ckpt_path, map_location=device)
    # on reconstruit le modèle avec in_dim + hid du checkpoint
    net = GraphBackboneNodeGRU(in_dim=ckpt["in_dim"], hid=ckpt["hid"]).to(device)
    net.load_state_dict(ckpt["state_dict"])

    # >>>> IMPORTANT : remettre les hyperparams du YAML <<<<
    m = cfg.get("model", {})

    net.k_max_ratio = float(m.get("k_max_ratio", net.k_max_ratio))
    net.k_max_fixed = None if m.get("k_max_fixed") in (None, "null") else int(m["k_max_fixed"])

    net.k_new_ratio = float(m.get("k_new_ratio", net.k_new_ratio))
    net.k_new_fixed = None if m.get("k_new_fixed") in (None, "null") else int(m["k_new_fixed"])

    net.alpha_stay  = float(m.get("alpha_stay", net.alpha_stay))
    net.abs_min     = float(m.get("abs_min", net.abs_min))

    net.pooling_mode = str(m.get("pooling_mode", net.pooling_mode))
    net.k_ratio      = float(m.get("k_ratio", net.k_ratio))

    net.eval()
    return net



def evaluate_and_plot_on_test(cfg, patient_key: str, ckpt_path: str, device: torch.device):
    exp = cfg["experiment"]; ev = cfg["eval"]; data = cfg["data"]
    results_dir = os.path.join(exp["output_dir"], ev.get("results_subdir", "results"))
    out_base = Path(results_dir) / patient_key.replace("::","__")
    out_base.mkdir(parents=True, exist_ok=True)


    # dossier pour les séries sauvegardées
    series_dir = out_base / ev.get("series_subdir", "series")
    series_dir.mkdir(parents=True, exist_ok=True)
    save_series = bool(ev.get("save_series", False))
    series_formats = [s.lower() for s in ev.get("series_format", ["npz"])]
    save_nodes = bool(ev.get("save_nodes", True))
    save_nodes_wide_csv = bool(ev.get("save_nodes_wide_csv", False))




    fold_idx = int(data.get("fold_idx", 0))
    splits_json = data["splits_all_train"]

    print(f"\n===== [TEST] {patient_key} =====")
    DEVICE = device
    #net = load_model_from_ckpt(ckpt_path, DEVICE)
    net = load_model_from_ckpt(ckpt_path, DEVICE, cfg)

    available = load_patient_splits_from_json(splits_json, patient_key)
    if not available:
        print(f"[{patient_key}] no split available in {splits_json}")
        return None, []

    eval_mode = ev.get("mode", "test_only")
    if eval_mode == "one_per_seizure":
        usable = normalize_requested_splits(["test","train","val"], available)
        seqs, _ = select_one_sequence_per_seizure(
            splits_json, patient_key, eval_splits=usable, fold_idx=fold_idx, seed=cfg["experiment"].get("seed", 0)
        )
        if not seqs:
            print(f"[{patient_key}] no sequences for one_per_seizure"); return None, []
        ds_eval = SimpleSeqDataset(seqs)
    else:
        sk = "test" if "test" in available else next(iter(sorted(available)))
        try:
            ds_eval = SequenceDataset(splits_json, patient_key, sk, fold_idx=fold_idx)
        except Exception as e:
            print(f"[{patient_key}] cannot load split '{sk}': {e}"); return None, []

    dl = DataLoader(
        ds_eval,
        batch_size=int(ev.get("batch_size", 1)),
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    all_losses, all_scores, all_labels = [], [], []
    per_sequence_metrics, seq_counter = [], 0

    with torch.no_grad():
        for packet in dl:
            batches_time, seq_ids_time, y_graph_time, B = packet
            if B <= 0:
                continue
            batches_time = [bt.to(DEVICE) for bt in batches_time]
            y_graph_time = [y.to(DEVICE) for y in y_graph_time]

            p_graph_time, _, p_node_seq = net.forward_sequence(
                batches_time, seq_ids_time, return_node_probs=True
            )

            losses = []
            for p_t, y_t in zip(p_graph_time, y_graph_time):
                if p_t.numel() == 0:
                    continue
                m = min(p_t.numel(), y_t.numel())
                losses.append(F.binary_cross_entropy(p_t[:m], y_t[:m]))
                all_scores += p_t[:m].cpu().tolist()
                all_labels += y_t[:m].cpu().tolist()
            if losses:
                all_losses.append(torch.stack(losses).mean().item())

            for b, seq_list in p_node_seq.items():
                probs_b = _stack_probs(seq_list)  # (N_b, T_b)
                seq_graphs = ds_eval[seq_counter]
                g0 = seq_graphs[0]
                soz_mask = get_soz_mask_from_graph(g0)
                phases = [get_phase_from_graph(g) for g in seq_graphs]
                t_true = next((i for i, ph in enumerate(phases) if ph == "ictal"), None)
                
                electrode_names = get_electrode_names_from_graph(g0)  # List[str] or None
                outcome = get_outcome_from_graph(g0)                  # str or None
                N_b, T_b = probs_b.size(0), probs_b.size(1)

                # normalisations
                if electrode_names is not None and len(electrode_names) != N_b:
                    electrode_names = None
                if soz_mask is None:
                    soz_mask_np = np.zeros((N_b,), dtype=bool)
                else:
                    soz_mask_np = soz_mask.detach().cpu().view(-1).numpy().astype(bool)

                pg = []
                for pt in p_graph_time:
                    if pt.numel() > 0:
                        pg.append(float(pt.view(-1)[0].item()))
                pg = np.asarray(pg, dtype=np.float32)
                pg_smooth = moving_average_1d(pg, W=int(ev.get("smooth_W", 5)))

                onset_cfg = ev.get("onset", {})
                method = onset_cfg.get("method", "adaptive")
                if method == "simple":
                    thr = float(onset_cfg.get("simple", {}).get("thr", 0.5))
                    min_stable = int(onset_cfg.get("simple", {}).get("min_stable", 3))
                    t_pred = detect_onset_simple(pg_smooth, thr=thr, min_stable=min_stable)
                else:
                    a = onset_cfg.get("adaptive", {})
                    t_pred = detect_onset_adaptive(
                        pg_smooth,
                        pre_window=int(a.get("pre_window", 20)),
                        k_sigma=float(a.get("k_sigma", 2.0)),
                        thr_abs_min=float(a.get("thr_abs_min", 0.55)),
                        thr_abs_max=float(a.get("thr_abs_max", 0.85)),
                        frac_of_range=float(a.get("frac_of_range", 0.60)),
                        min_stable=int(a.get("min_stable", 3)),
                        dmin=float(a.get("dmin", 0.02)),
                    )
                if t_pred < 0 and pg_smooth.size > 0:
                    t_pred = int(np.argmax(pg_smooth))
                if t_pred is not None and t_pred < 0:
                    t_pred = None

                seiz_id = getattr(g0, "meta", {}).get("seizure", None) or getattr(g0, "seizure", "?")
                title = f"{patient_key} | seiz#{seiz_id} | seq#{seq_counter:03d} (N={probs_b.size(0)}, T={probs_b.size(1)})"
                save_path = out_base / f"seiz_{seiz_id}_seq_{seq_counter:03d}.png"

                # per-sequence metrics
                try:
                    from sklearn.metrics import roc_auc_score as _roc_auc_score
                    y_true_seq, y_pred_seq = [], []
                    for p_t, y_t in zip(p_graph_time, y_graph_time):
                        if p_t.numel() == 0:
                            continue
                        m = min(p_t.numel(), y_t.numel())
                        y_true_seq += y_t[:m].detach().cpu().tolist()
                        y_pred_seq += p_t[:m].detach().cpu().tolist()
                    if len(set(y_true_seq)) > 1:
                        auc_seq = float(_roc_auc_score(y_true_seq, y_pred_seq))
                    else:
                        auc_seq = float("nan")
                    eps = 1e-7
                    yp = torch.tensor(y_pred_seq, dtype=torch.float32).clamp(min=eps, max=1.0-eps)
                    yt = torch.tensor(y_true_seq, dtype=torch.float32)
                    bce_seq = float(F.binary_cross_entropy(yp, yt).item())
                except Exception:
                    auc_seq = float("nan"); bce_seq = float("nan")

                delay = None
                if t_true is not None and t_pred is not None:
                    delay = int(t_pred - t_true)

                per_sequence_metrics.append({
                    "patient": patient_key,
                    "seq_index": int(seq_counter),
                    "seizure_id": str(seiz_id),
                    "len_seq": int(len(seq_graphs)),
                    "auc_seq": auc_seq,
                    "bce_seq": bce_seq,
                    "t_true": int(t_true) if t_true is not None else None,
                    "t_pred": int(t_pred) if t_pred is not None else None,
                    "delay": delay,
                })
                
                # ---------- SAVE SERIES (per sequence) ----------
                if save_series:
                    # -- 1) Construire y_graph_true (T,) aligné aux pas de p_graph_time
                    y_graph_true = []
                    for p_t, y_t in zip(p_graph_time, y_graph_time):
                        if p_t.numel() == 0:
                            continue
                        m = min(p_t.numel(), y_t.numel())
                        y_graph_true.append(float(y_t[:m].view(-1)[0].item()))
                    y_graph_true = np.asarray(y_graph_true, dtype=np.float32)

                    # -- 2) Longueur commune pour sauvegarde “time-series”
                    Tsave = int(max(len(pg), len(pg_smooth), len(y_graph_true)))
                    def _pad_to(x, T):
                        x = np.asarray(x, dtype=np.float32)
                        if x.shape[0] >= T: return x[:T]
                        if x.ndim == 1:
                            z = np.zeros((T,), dtype=np.float32); z[:x.shape[0]] = x; return z
                        return x

                    pg_pad     = _pad_to(pg, Tsave)
                    pgs_pad    = _pad_to(pg_smooth, Tsave)
                    y_true_pad = _pad_to(y_graph_true, Tsave)
                    y_pred_onset = step_vector_from_onset(Tsave, t_pred)  # 0 avant t_pred, 1 après

                    # -- 3) chemins
                    base_stem = f"seiz_{seiz_id}_seq_{seq_counter:03d}"
                    npz_path       = series_dir / f"{base_stem}.npz"
                    csv_wide_path  = series_dir / f"{base_stem}.csv"          # time-series global (wide)
                    nodes_csv_path = series_dir / f"{base_stem}_nodes.csv"    # mapping des nœuds

                    # -- 4) NPZ (compact, rechargement rapide)
                    if "npz" in series_formats:
                        payload = {
                            # méta séquence
                            "patient": patient_key,
                            "seizure_id": str(seiz_id),
                            "seq_index": int(seq_counter),
                            "outcome": "" if outcome is None else str(outcome),
                            "t_true": -1 if t_true is None else int(t_true),
                            "t_pred": -1 if t_pred is None else int(t_pred),

                            # séries supervisées & prédites
                            "y_graph_true": y_true_pad.astype(np.float32),   # label vrai
                            "y_pred_onset": y_pred_onset.astype(np.float32), # label prédit (step à partir de t_pred)
                            "p_graph": pg_pad.astype(np.float32),
                            "p_graph_smooth": pgs_pad.astype(np.float32),

                            # infos nœuds
                            "is_SOZ": soz_mask_np.astype(np.bool_),          # (N,)
                        }
                        if save_nodes:
                            payload["per_node_probs"] = probs_b.detach().cpu().numpy().astype(np.float32)  # (N,T_b)
                            if electrode_names is not None:
                                maxlen = max(1, max(len(s) for s in electrode_names))
                                payload["electrode_names"] = np.array(electrode_names, dtype=f"<U{maxlen}")
                            else:
                                payload["electrode_names"] = np.array([], dtype="<U1")

                        np.savez_compressed(npz_path, **payload)

                    # -- 5) CSV “wide” (une ligne par t, toutes les électrodes en colonnes)
                    if "csv" in series_formats and save_nodes and save_nodes_wide_csv:
                        import csv as _csv
                        # colonnes électrodes: node_000[:name], node_001[:name], ...
                        if electrode_names is not None:
                            node_headers = [f"node_{i:03d}:{electrode_names[i]}" for i in range(N_b)]
                        else:
                            node_headers = [f"node_{i:03d}" for i in range(N_b)]

                        with open(csv_wide_path, "w", newline="") as f:
                            w = _csv.writer(f)
                            # méta en tête (commentées)
                            w.writerow(["#patient", patient_key])
                            w.writerow(["#seizure_id", seiz_id])
                            w.writerow(["#seq_index", seq_counter])
                            w.writerow(["#outcome", "" if outcome is None else outcome])
                            w.writerow(["#t_true", "" if t_true is None else t_true])
                            w.writerow(["#t_pred", "" if t_pred is None else t_pred])
                            # header
                            w.writerow(["t", "y_graph_true", "y_pred_onset", "p_graph", "p_graph_smooth"] + node_headers)

                            # data
                            probs_np = probs_b.detach().cpu().numpy()  # (N,T_b)
                            for t_i in range(Tsave):
                                yg  = y_true_pad[t_i] if t_i < y_true_pad.shape[0] else ""
                                yp  = y_pred_onset[t_i]
                                pg_i  = pg_pad[t_i]  if t_i < pg_pad.shape[0]  else ""
                                pgs_i = pgs_pad[t_i] if t_i < pgs_pad.shape[0] else ""
                                if t_i < probs_np.shape[1]:
                                    row_nodes = probs_np[:, t_i].tolist()
                                else:
                                    row_nodes = [""] * probs_np.shape[0]
                                w.writerow([t_i, yg, yp, pg_i, pgs_i] + row_nodes)

                    # -- 6) CSV mapping des nœuds (index -> nom, SOZ)
                    if "csv" in series_formats and save_nodes:
                        import csv as _csv
                        with open(nodes_csv_path, "w", newline="") as f:
                            w = _csv.writer(f)
                            w.writerow(["node_index", "electrode_name", "is_SOZ"])
                            for i in range(N_b):
                                name_i = electrode_names[i] if (electrode_names is not None) else ""
                                w.writerow([i, name_i, int(soz_mask_np[i])])

                    # -- 7) garder les chemins dans les métriques
                    if per_sequence_metrics:
                        per_sequence_metrics[-1]["series_npz"] = str(npz_path) if "npz" in series_formats else None
                        per_sequence_metrics[-1]["series_csv"] = str(csv_wide_path) if ("csv" in series_formats and save_nodes_wide_csv) else None
                        per_sequence_metrics[-1]["nodes_csv"]  = str(nodes_csv_path) if ("csv" in series_formats and save_nodes) else None
                # ---------- END SAVE SERIES ----------




                plot_probs(
                    probs_b,
                    save_path,
                    soz_mask=soz_mask,
                    title=title,
                    t_true=t_true,
                    t_pred=t_pred,
                    p_graph=pg,
                    p_graph_smooth=pg_smooth,
                )
                print(f"  figure: {save_path}")
                seq_counter += 1

    test_loss = sum(all_losses)/len(all_losses) if all_losses else float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    print("\n===== Summary =====")
    print(f"[{patient_key}] sequences: {seq_counter}")
    print(f"[{patient_key}] BCE(graph): {test_loss:.4f}")
    print(f"[{patient_key}] ROC-AUC(graph): {auc:.3f}")
    print(f"Figures -> {out_base.resolve()}")

    return {
        "patient": patient_key,
        "split_mode": cfg["eval"].get("mode", "test_only"),
        "fold_idx": int(cfg["data"].get("fold_idx", 0)),
        "n_sequences": int(seq_counter),
        "loss_bce_graph": None if math.isnan(test_loss) else float(test_loss),
        "roc_auc_graph": None if math.isnan(auc) else float(auc),
        "results_dir": str(out_base.resolve()),
        "per_sequence_metrics": per_sequence_metrics,
    }, per_sequence_metrics
