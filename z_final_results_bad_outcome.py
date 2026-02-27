#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualisation à partir d'un seul CSV (preds_ranked.csv) généré par M2-on-M1.

Le CSV doit contenir au minimum:
  patient, seizure_id, seq_idx, is_SOZ, m2_score
Optionnel mais recommandé:
  node_index, rank (sinon on recalcule)

Les calculs reproduisent ta logique:
  - collapse par node_index au sein de (patient,seizure,seq) via max(score) et OR(is_SOZ)
  - top-k metrics avec fp_mode="before_last_soz"
  - rang du premier SOZ (median ± IQR) + background nb électrodes
  - rang de toutes les SOZ (median ± IQR) + background nb électrodes + nb SOZ
  - confidence: mean_pos, mean_neg, confidence_gap + min/max (boxplot 3 panels)

NOUVEAU (TRACE / AUDIT):
  - audit_patient_metrics.csv : 1 ligne par patient (topk metrics, confidence, min/max, etc.)
  - predictions_trace_collapsed_ranked.csv : toutes les prédictions utilisées pour ranking/figures
      (collapse node_index dans chaque (patient,seizure,seq), rank recalculé, flags top10/top20)
  - per_patient_predictions/<PAT>_predictions_trace.csv : (optionnel) 1 fichier par patient

Usage:
uv run z_visualise_bad_patients.py \
  --csv /home/.../preds_ranked.csv \
  --out /home/.../figures_m2_on_bad_pat \
  --series_dir /home/.../results/results \
  --bce_middle_seq_only \
  --bce_ylim01 \
  --min_soz_per_seizure 1


uv run z_final_results_bad_outcome.py \
  --csv /home/julieletallec/test/M2_on_M1_outputs/preds_ranked.csv \
  --out /home/julieletallec/test/final_results_bad_outcome \
  --series_dir /home/julieletallec/test/M1_singleconfig_runs/results/results \
  --bce_middle_seq_only \
  --bce_ylim01 \
  --min_soz_per_seizure 1

"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from matplotlib.lines import Line2D


# -------------------------------
# Helpers (format/palette)
# -------------------------------

def canon_patient(p: str) -> str:
    return str(p).replace("::", "_").replace("__", "_").strip()


def build_patient_color_map(patients: List[str]):
    unique = sorted(pd.unique(patients))
    palette = sns.color_palette("husl", len(unique))
    return {pat: palette[i] for i, pat in enumerate(unique)}


def patient_labels_2lines(pat: str) -> str:
    pat = str(pat).strip()
    pat = pat.rstrip("_").rstrip()

    if "__" in pat:
        a, b = pat.split("__", 1)
        return f"{a}\n{b}"
    if "::" in pat:
        a, b = pat.split("::", 1)
        return f"{a}\n{b}"

    if pat.count("_") >= 1:
        a, b = pat.split("_", 1)
        return f"{a}\n{b}"

    return pat


# -------------------------------
# Scatter per patient/seizure (optionnel)
# -------------------------------

def plot_per_patient_seizure_scores_with_labels_and_optional_csv(
    df: pd.DataFrame,
    out_dir: Path,
    score_col: str = "m2_score",
    electrode_col: str = "electrode_name",
    patient_filter_prefix: str = "CHUM",
    patient: str | None = None,
    seizure_id: str | int | None = None,
    export_csv: bool = False,
    csv_suffix: str = "electrode_scores",
    annotate: bool = True,
    max_labels: int = 200,
    figsize=(12, 6),
    dpi=200,
    drop_unknown_seizure: bool = False,
    unknown_seizure_token: str = "?",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()

    if not {"patient", "seizure_id", "is_SOZ", score_col}.issubset(d.columns):
        raise ValueError(f"CSV doit contenir au minimum: patient, seizure_id, is_SOZ, {score_col}")

    d["patient"] = d["patient"].astype(str).str.strip()
    d["seizure_id"] = d["seizure_id"].astype(str).str.strip()
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    def _is_chum(p: str) -> bool:
        p = str(p)
        return (
            p.startswith(patient_filter_prefix)
            or p.startswith(patient_filter_prefix + "_")
            or p.startswith(patient_filter_prefix + "__")
        )

    d = d[d["patient"].apply(_is_chum)].copy()

    if drop_unknown_seizure:
        d = d[d["seizure_id"] != str(unknown_seizure_token)].copy()

    if patient is not None:
        patient = str(patient).strip()
        d = d[d["patient"] == patient].copy()

    if seizure_id is not None:
        seizure_id = str(seizure_id).strip()
        d = d[d["seizure_id"] == seizure_id].copy()

    if d.empty:
        print("[WARN] Aucun point après filtrage (CHUM/patient/seizure).")
        return

    if electrode_col not in d.columns:
        if "node_index" in d.columns:
            electrode_col_eff = "node_index"
        else:
            d = d.reset_index(drop=False).rename(columns={"index": "row_index"})
            electrode_col_eff = "row_index"
    else:
        electrode_col_eff = electrode_col

    if "node_index" in d.columns:
        d["node_index"] = pd.to_numeric(d["node_index"], errors="coerce")
        agg = {score_col: "max", "is_SOZ": "max", electrode_col_eff: "first"}
        d = (
            d.groupby(["patient", "seizure_id", "node_index"], as_index=False, dropna=False)
             .agg(agg)
        )

    if export_csv and patient is not None and seizure_id is not None:
        sub = d[(d["patient"] == str(patient)) & (d["seizure_id"] == str(seizure_id))].copy()
        if sub.empty:
            print(f"[WARN] Aucun résultat pour {patient}, seizure {seizure_id} (export CSV annulé).")
        else:
            cols = ["patient", "seizure_id", electrode_col_eff, score_col, "is_SOZ"]
            if "node_index" in sub.columns and "node_index" not in cols:
                cols.insert(2, "node_index")
            sub_out = sub[cols].rename(columns={electrode_col_eff: "electrode"})
            sub_out = sub_out.sort_values(by=score_col, ascending=False)

            csv_path = out_dir / f"{patient}_seizure_{seizure_id}_{csv_suffix}.csv"
            sub_out.to_csv(csv_path, index=False)
            print(f"[OK] CSV export -> {csv_path}")

    def _sort_key(s: str):
        s = str(s).strip()
        try:
            return (0, int(float(s)))
        except Exception:
            return (1, s)

    for pat, dp in d.groupby("patient", dropna=False):
        dp = dp.copy()
        dp["seizure_id"] = dp["seizure_id"].astype(str).str.strip()

        seizs = sorted(dp["seizure_id"].unique().tolist(), key=_sort_key)
        x_map = {str(s).strip(): i for i, s in enumerate(seizs)}

        fig, ax = plt.subplots(figsize=figsize)

        for seiz, g in dp.groupby("seizure_id", dropna=False):
            seiz = str(seiz).strip()
            if seiz not in x_map:
                continue

            x0 = x_map[seiz]
            ys = g[score_col].to_numpy(dtype=float)
            soz = g["is_SOZ"].to_numpy(dtype=int)

            xs = x0 + np.random.uniform(-0.18, 0.18, size=len(g))

            m0 = (soz == 0) & np.isfinite(ys)
            ax.scatter(xs[m0], ys[m0], s=18, alpha=0.55, color="grey", linewidth=0)

            m1 = (soz == 1) & np.isfinite(ys)
            ax.scatter(xs[m1], ys[m1], s=26, alpha=0.90, color="red", linewidth=0)

            if annotate and len(g) <= max_labels:
                names = g[electrode_col_eff].astype(str).tolist()
                for xi, yi, nm, is_soz in zip(xs, ys, names, soz):
                    if not np.isfinite(yi):
                        continue
                    ax.text(
                        xi, yi,
                        nm,
                        fontsize=7,
                        color=("red" if int(is_soz) == 1 else "black"),
                        alpha=0.9,
                    )

        ax.set_title(f"{pat} — electrode scores per seizure")
        ax.set_xlabel("Seizure #")
        ax.set_ylabel(score_col)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(range(len(seizs)))
        ax.set_xticklabels(seizs, rotation=0)

        fig.tight_layout()
        fig_path = out_dir / f"{pat}_seizure_scores_{score_col}.png"
        plt.savefig(fig_path, dpi=dpi)
        plt.close(fig)
        print(f"[OK] Plot patient -> {fig_path}")


# -------------------------------
# Core collapse / constraints
# -------------------------------

def patient_has_min_soz_electrodes_per_seizure_from_df(
    df_pat: pd.DataFrame,
    min_soz_electrodes: int = 2,
    count_unique_node_index: bool = True,
) -> bool:
    if df_pat.empty:
        return False

    if "seizure_id" not in df_pat.columns or "is_SOZ" not in df_pat.columns:
        return False

    df_pat = df_pat.copy()
    df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)

    df_soz = df_pat[df_pat["is_SOZ"] == 1]
    if df_soz.empty:
        return False

    if count_unique_node_index and "node_index" in df_soz.columns:
        counts = df_soz.groupby("seizure_id")["node_index"].nunique()
    else:
        counts = df_soz.groupby("seizure_id").size()

    all_seizures = df_pat["seizure_id"].unique()
    counts = counts.reindex(all_seizures, fill_value=0)

    return bool((counts >= int(min_soz_electrodes)).all())


def _collapse_group_by_node(df_group: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Collapse par node_index (si dispo) au sein d'un groupe:
      - score = max
      - is_SOZ = max
      - rank = min si présent
    """
    df_group = df_group.copy()

    if score_col not in df_group.columns:
        df_group[score_col] = np.nan
    df_group[score_col] = pd.to_numeric(df_group[score_col], errors="coerce")

    if "is_SOZ" not in df_group.columns:
        df_group["is_SOZ"] = 0
    df_group["is_SOZ"] = pd.to_numeric(df_group["is_SOZ"], errors="coerce").fillna(0).astype(int)

    if "node_index" not in df_group.columns:
        if "rank" in df_group.columns:
            df_group["rank"] = pd.to_numeric(df_group["rank"], errors="coerce")
        return df_group

    agg = {score_col: "max", "is_SOZ": "max"}
    if "rank" in df_group.columns:
        df_group["rank"] = pd.to_numeric(df_group["rank"], errors="coerce")
        agg["rank"] = "min"

    df_c = df_group.groupby("node_index", as_index=False).agg(agg)
    return df_c


# -------------------------------
# Top-k metrics
# -------------------------------

def precision_recall_f1_topk(
    scores: np.ndarray,
    labels: np.ndarray,
    frac: float,
    fp_mode: str = "before_last_soz",
) -> Tuple[float, float, float]:
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    k = max(1, int(math.ceil(frac * n)))
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    top_labels = labels[top_idx].astype(int)

    tp = int(top_labels.sum())
    fn = int(labels.sum() - tp)

    if fp_mode == "classic":
        fp = int((top_labels == 0).sum())
    elif fp_mode == "before_last_soz":
        if tp == 0:
            fp = k
        else:
            last_soz_pos = int(np.where(top_labels == 1)[0].max())
            fp = int((top_labels[:last_soz_pos] == 0).sum())
    else:
        raise ValueError("fp_mode must be 'classic' or 'before_last_soz'")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_patient_metrics_from_predictions_df(df_pat: pd.DataFrame, score_col: str, fp_mode: str) -> Dict[str, float]:
    needed = {"patient", "seizure_id", "seq_idx", "is_SOZ", score_col}
    missing = needed - set(df_pat.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_pat = df_pat.copy()
    df_pat["is_SOZ"] = pd.to_numeric(df_pat["is_SOZ"], errors="coerce").fillna(0).astype(int)
    df_pat[score_col] = pd.to_numeric(df_pat[score_col], errors="coerce")

    group_keys = ["patient", "seizure_id", "seq_idx"]
    grp = df_pat.groupby(group_keys, dropna=False)

    pos_means, neg_means, gaps = [], [], []
    p10s, r10s, f10s = [], [], []
    p20s, r20s, f20s = [], [], []
    aucs = []
    n_groups = 0
    n_auc_groups = 0

    for _, g in grp:
        g2 = _collapse_group_by_node(g, score_col=score_col).dropna(subset=[score_col])
        if g2.empty:
            continue

        scores = g2[score_col].to_numpy(dtype=float)
        labels = g2["is_SOZ"].to_numpy(dtype=int)

        pos_mean = float(np.mean(scores[labels == 1])) if (labels == 1).any() else np.nan
        neg_mean = float(np.mean(scores[labels == 0])) if (labels == 0).any() else np.nan
        gap = (pos_mean - neg_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan

        pos_means.append(pos_mean)
        neg_means.append(neg_mean)
        gaps.append(gap)

        p10, r10, f10 = precision_recall_f1_topk(scores, labels, 0.10, fp_mode=fp_mode)
        p20, r20, f20 = precision_recall_f1_topk(scores, labels, 0.20, fp_mode=fp_mode)
        p10s.append(p10); r10s.append(r10); f10s.append(f10)
        p20s.append(p20); r20s.append(r20); f20s.append(f20)

        if len(np.unique(labels)) >= 2:
            try:
                auc = float(roc_auc_score(labels, scores))
                aucs.append(auc)
                n_auc_groups += 1
            except Exception:
                pass

        n_groups += 1

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    return {
        "mean_score_pos": nanmean_safe(pos_means),
        "mean_score_neg": nanmean_safe(neg_means),
        "confidence_gap": nanmean_safe(gaps),

        "Precision@top_10pct": nanmean_safe(p10s),
        "Recall@top_10pct": nanmean_safe(r10s),
        "F1@top_10pct": nanmean_safe(f10s),

        "Precision@top_20pct": nanmean_safe(p20s),
        "Recall@top_20pct": nanmean_safe(r20s),
        "F1@top_20pct": nanmean_safe(f20s),

        "AUC_group_mean": nanmean_safe(aucs),
        "num_groups_used": float(n_groups),
        "num_auc_groups_used": float(n_auc_groups),
    }


# -------------------------------
# Rank computations (first/all SOZ)
# -------------------------------

def ensure_rank_per_group(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Si 'rank' absent, calcule rank dans chaque groupe (patient,seizure,seq)
    après tri score desc.
    """
    df = df.copy()
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        return df

    keys = ["patient", "seizure_id", "seq_idx"]
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    def _assign_rank(g):
        g = g.sort_values(score_col, ascending=False).copy()
        g["rank"] = np.arange(1, len(g) + 1)
        return g

    return df.groupby(keys, group_keys=False, dropna=False).apply(_assign_rank)


def compute_first_soz_rank_and_counts(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    records = []
    df = ensure_rank_per_group(df, score_col=score_col)

    for pat, df_pat in df.groupby("patient"):
        df_pat = df_pat.copy()
        num_seizures = df_pat["seizure_id"].nunique()

        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"], dropna=False)
        if "node_index" in df_pat.columns:
            max_electrodes = int(grp["node_index"].nunique().max())
        else:
            max_electrodes = int(grp.size().max())

        first_ranks = []
        for _, g in grp:
            g_soz = g[pd.to_numeric(g["is_SOZ"], errors="coerce").fillna(0).astype(int) == 1]
            if g_soz.empty:
                continue
            first_ranks.append(int(pd.to_numeric(g_soz["rank"], errors="coerce").min()))

        if first_ranks:
            arr = np.array(first_ranks, dtype=float)
            mean_rank = float(np.mean(arr))
            median_rank = float(np.median(arr))
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = float(q75 - q25)
        else:
            mean_rank = median_rank = q25 = q75 = iqr = np.nan

        records.append({
            "patient": pat,
            "first_soz_mean_rank": mean_rank,
            "first_soz_median_rank": median_rank,
            "first_soz_q25_rank": q25,
            "first_soz_q75_rank": q75,
            "first_soz_iqr_rank": iqr,
            "num_groups_with_soz": int(len(first_ranks)),
            "num_seizures": int(num_seizures),
            "max_electrodes": int(max_electrodes),
        })

    return pd.DataFrame(records).sort_values("patient").reset_index(drop=True)


def compute_all_soz_rank_and_counts(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    records = []
    df = ensure_rank_per_group(df, score_col=score_col)

    for pat, df_pat in df.groupby("patient"):
        df_pat = df_pat.copy()
        num_seizures = df_pat["seizure_id"].nunique()

        grp = df_pat.groupby(["patient", "seizure_id", "seq_idx"], dropna=False)
        if "node_index" in df_pat.columns:
            max_electrodes = int(grp["node_index"].nunique().max())
        else:
            max_electrodes = int(grp.size().max())

        all_ranks = []
        max_soz_electrodes = 0
        num_groups_with_soz = 0

        for _, g in grp:
            g_soz = g[pd.to_numeric(g["is_SOZ"], errors="coerce").fillna(0).astype(int) == 1]
            if g_soz.empty:
                continue
            num_groups_with_soz += 1

            rr = pd.to_numeric(g_soz["rank"], errors="coerce").dropna().astype(int).tolist()
            all_ranks.extend(rr)

            if "node_index" in g_soz.columns:
                n_soz = int(g_soz["node_index"].nunique())
            else:
                n_soz = int(len(g_soz))
            max_soz_electrodes = max(max_soz_electrodes, n_soz)

        if all_ranks:
            arr = np.array(all_ranks, dtype=float)
            mean_rank = float(np.mean(arr))
            median_rank = float(np.median(arr))
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = float(q75 - q25)
        else:
            mean_rank = median_rank = q25 = q75 = iqr = np.nan

        records.append({
            "patient": pat,
            "all_soz_mean_rank": mean_rank,
            "all_soz_median_rank": median_rank,
            "all_soz_q25_rank": q25,
            "all_soz_q75_rank": q75,
            "all_soz_iqr_rank": iqr,
            "num_groups_with_soz": int(num_groups_with_soz),
            "num_seizures": int(num_seizures),
            "max_electrodes": int(max_electrodes),
            "max_soz_electrodes": int(max_soz_electrodes),
        })

    return pd.DataFrame(records).sort_values("patient").reset_index(drop=True)


# -------------------------------
# Confidence gap (seizure-level) + TOP10% only (nouveau demandé)
# -------------------------------

def compute_patient_confidence_gap_seizure_level(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Pour chaque (patient,seizure): pos_mean/neg_mean/gap sur TOUTES électrodes (collapse node_index si dispo),
    puis patient-level = nanmean sur seizures.
    """
    needed = {"patient", "seizure_id", "is_SOZ", score_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["patient"] = d["patient"].astype(str)
    d["seizure_id"] = d["seizure_id"].astype(str)
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    if "node_index" in d.columns:
        d["node_index"] = pd.to_numeric(d["node_index"], errors="coerce")
        d = (
            d.groupby(["patient", "seizure_id", "node_index"], as_index=False)
             .agg({score_col: "max", "is_SOZ": "max"})
        )

    seiz_rows = []
    for (pat, seiz), g in d.groupby(["patient", "seizure_id"], dropna=False):
        scores = g[score_col].to_numpy(dtype=float)
        labels = g["is_SOZ"].to_numpy(dtype=int)

        pos = float(np.mean(scores[labels == 1])) if (labels == 1).any() else np.nan
        neg = float(np.mean(scores[labels == 0])) if (labels == 0).any() else np.nan
        gap = (pos - neg) if np.isfinite(pos) and np.isfinite(neg) else np.nan

        seiz_rows.append({"patient": pat, "seizure_id": seiz, "pos_mean_seiz": pos, "neg_mean_seiz": neg, "gap_seiz": gap})

    df_seiz = pd.DataFrame(seiz_rows)

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    return (
        df_seiz.groupby("patient", as_index=False)
              .agg({"pos_mean_seiz": nanmean_safe, "neg_mean_seiz": nanmean_safe, "gap_seiz": nanmean_safe})
              .rename(columns={"pos_mean_seiz": "mean_score_pos", "neg_mean_seiz": "mean_score_neg", "gap_seiz": "confidence_gap"})
    )


def compute_patient_confidence_gap_seizure_level_topk(
    df: pd.DataFrame,
    score_col: str,
    frac: float = 0.10,
    suffix: str = "_top10",
) -> pd.DataFrame:
    """
    Même idée que compute_patient_confidence_gap_seizure_level, mais en ne gardant
    que les électrodes dans le top-k (frac) au sein de chaque seizure (après collapse node_index).
    Puis patient-level = nanmean sur seizures.

    Retourne:
      patient,
      mean_score_pos{suffix},
      mean_score_neg{suffix},
      confidence_gap{suffix}
    """
    needed = {"patient", "seizure_id", "is_SOZ", score_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["patient"] = d["patient"].astype(str)
    d["seizure_id"] = d["seizure_id"].astype(str)
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    if "node_index" in d.columns:
        d["node_index"] = pd.to_numeric(d["node_index"], errors="coerce")
        d = (
            d.groupby(["patient", "seizure_id", "node_index"], as_index=False)
             .agg({score_col: "max", "is_SOZ": "max"})
        )

    seiz_rows = []
    for (pat, seiz), g in d.groupby(["patient", "seizure_id"], dropna=False):
        scores = g[score_col].to_numpy(dtype=float)
        labels = g["is_SOZ"].to_numpy(dtype=int)
        n = len(scores)
        if n == 0:
            continue

        k = max(1, int(math.ceil(frac * n)))
        order = np.argsort(scores)[::-1]
        top_idx = order[:k]

        top_scores = scores[top_idx]
        top_labels = labels[top_idx]

        pos = float(np.mean(top_scores[top_labels == 1])) if (top_labels == 1).any() else np.nan
        neg = float(np.mean(top_scores[top_labels == 0])) if (top_labels == 0).any() else np.nan
        gap = (pos - neg) if np.isfinite(pos) and np.isfinite(neg) else np.nan

        seiz_rows.append({"patient": pat, "seizure_id": seiz, "pos_mean_seiz": pos, "neg_mean_seiz": neg, "gap_seiz": gap})

    df_seiz = pd.DataFrame(seiz_rows)
    if df_seiz.empty:
        return pd.DataFrame(columns=["patient", f"mean_score_pos{suffix}", f"mean_score_neg{suffix}", f"confidence_gap{suffix}"])

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    df_pat = (
        df_seiz.groupby("patient", as_index=False)
              .agg({"pos_mean_seiz": nanmean_safe, "neg_mean_seiz": nanmean_safe, "gap_seiz": nanmean_safe})
              .rename(columns={
                  "pos_mean_seiz": f"mean_score_pos{suffix}",
                  "neg_mean_seiz": f"mean_score_neg{suffix}",
                  "gap_seiz": f"confidence_gap{suffix}",
              })
    )
    return df_pat


def plot_confidence_gap_boxplot(
    df_pat: pd.DataFrame,
    out_path,
    color_map: dict,
    title: str,
    col_pos: str = "mean_score_pos",
    col_neg: str = "mean_score_neg",
    col_gap: str = "confidence_gap",
):
    dfp = df_pat.copy()
    dfp["patient"] = dfp["patient"].astype(str)
    dfp["patient_canon"] = dfp["patient"].map(canon_patient)

    cols = [col_pos, col_neg, col_gap]
    xlabels = [
        "Average Positive Score (SOZ)\nMean over Seizures",
        "Average Negative Score (non-SOZ)\nMean over Seizures",
        "Confidence Gap (SOZ vs non-SOZ)\nMean over Seizures",
    ]

    data = []
    for c in cols:
        v = pd.to_numeric(dfp[c], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.boxplot(
        data,
        tick_labels=xlabels,
        showfliers=False,
        widths=0.30,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=1.5),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    for j, c in enumerate(cols, start=1):
        vals = pd.to_numeric(dfp[c], errors="coerce").to_numpy(dtype=float)
        for pat_canon, v in zip(dfp["patient_canon"].tolist(), vals):
            if not np.isfinite(v):
                continue
            col = color_map.get(pat_canon, "grey")
            x = j + np.random.uniform(-0.08, 0.08)
            ax.scatter(x, v, s=70, alpha=0.75, color=col, edgecolors="black", linewidths=0.3)

    ax.set_ylim(-0.2, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    pats = sorted(dfp[["patient", "patient_canon"]].drop_duplicates()["patient_canon"].tolist())
    handles = []
    for pcanon in pats:
        orig = dfp.loc[dfp["patient_canon"] == pcanon, "patient"].iloc[0]
        label = patient_labels_2lines(orig)
        handles.append(
            Line2D([0], [0],
                   marker="o",
                   color="none",
                   markerfacecolor=color_map.get(pcanon, "grey"),
                   markeredgecolor="black",
                   markeredgewidth=0.3,
                   markersize=8,
                   label=label)
        )

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] confidence gap boxplot -> {out_path}")


# -------------------------------
# Plots: topk metrics & ranks
# -------------------------------

def make_boxplot_topk_metrics(df_best: pd.DataFrame, out_dir: Path, color_map: dict, title: str):
    metrics = [
        "Precision@top_10pct", "Recall@top_10pct", "F1@top_10pct",
        "Precision@top_20pct", "Recall@top_20pct", "F1@top_20pct",
        "AUC_group_mean",
    ]
    labels = [
        "Precision\n(top 10%)", "Recall\n(top 10%)", "F1\n(top 10%)",
        "Precision\n(top 20%)", "Recall\n(top 20%)", "F1\n(top 20%)",
        "AUC\n(group mean)",
    ]

    data = [df_best[m].astype(float).values for m in metrics]
    patients = df_best["patient"].tolist()

    plt.figure(figsize=(max(10, len(metrics) * 1.25), 6))
    plt.boxplot(data, labels=labels, showmeans=False, showfliers=False)

    for i, m in enumerate(metrics):
        x_center = i + 1
        vals = df_best[m].astype(float).values
        for val, pat in zip(vals, patients):
            if np.isnan(val):
                continue
            jitter = np.random.uniform(-0.12, 0.12)
            plt.scatter(x_center + jitter, val, color=color_map.get(pat, "grey"), s=55, alpha=0.75, linewidth=0.4)

    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    fig_path = Path(out_dir) / "boxplot_topk_metrics.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] {fig_path}")


def plot_first_soz_rank_barplot(df_first: pd.DataFrame, color_map: dict, out_dir: Path, title: str):
    patients = df_first["patient"].tolist()
    median_ranks = df_first["first_soz_median_rank"].values
    q25 = df_first["first_soz_q25_rank"].values
    q75 = df_first["first_soz_q75_rank"].values
    num_seizures = df_first["num_seizures"].values
    max_electrodes = df_first["max_electrodes"].values

    x = np.arange(len(patients))
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    for i, (pat, n_elec) in enumerate(zip(patients, max_electrodes)):
        c = color_map.get(pat, (0.8, 0.8, 0.8, 1.0))
        plt.bar(i, n_elec, color=(c[0], c[1], c[2], 0.25), edgecolor="none")

    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, color=c, edgecolor="black", linewidth=1.0)
        plt.text(i, 177, f"S:{int(num_seizures[i])}", ha="center", va="top", fontsize=9)

    lower_err = median_ranks - q25
    upper_err = q75 - median_ranks
    valid = ~np.isnan(median_ranks)

    plt.errorbar(
        x[valid],
        median_ranks[valid],
        yerr=np.vstack([lower_err[valid], upper_err[valid]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        capthick=1.0,
    )

    plt.xticks(x, [patient_labels_2lines(p) for p in patients], rotation=90, fontsize=8)
    plt.ylabel("First SOZ Channel Detection Rank\n(vs Total Number of Implanted Electrodes)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    ymax = max(np.nanmax(max_electrodes), np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks))
    plt.ylim(0, ymax * 1.05)

    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 190]
    plt.yticks(ticks)

    fig_path = Path(out_dir) / "barplot_first_soz_rank_median_IQR_background_electrodes.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] {fig_path}")


def plot_all_soz_rank_barplot(df_all: pd.DataFrame, color_map: dict, out_dir: Path, title: str):
    patients = df_all["patient"].tolist()
    median_ranks = df_all["all_soz_median_rank"].values
    q25 = df_all["all_soz_q25_rank"].values
    q75 = df_all["all_soz_q75_rank"].values
    num_seizures = df_all["num_seizures"].values
    max_electrodes = df_all["max_electrodes"].values
    max_soz_electrodes = df_all["max_soz_electrodes"].values

    x = np.arange(len(patients))
    plt.figure(figsize=(max(10, len(patients) * 0.4), 6))

    for i, pat in enumerate(patients):
        c = color_map.get(pat, (0.8, 0.8, 0.8, 1.0))
        plt.bar(i, max_electrodes[i], width=0.8, color=(c[0], c[1], c[2], 0.20), edgecolor="none")
        plt.bar(i, max_soz_electrodes[i], width=0.8, color=(c[0], c[1], c[2], 0.65), edgecolor="none")

    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, width=0.25, color=c, edgecolor="black", linewidth=1.0)
        plt.text(i, 177, f"S:{int(num_seizures[i])}", ha="center", va="top", fontsize=9)

    lower_err = median_ranks - q25
    upper_err = q75 - median_ranks
    valid = ~np.isnan(median_ranks)

    plt.errorbar(
        x[valid],
        median_ranks[valid],
        yerr=np.vstack([lower_err[valid], upper_err[valid]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        capthick=1.0,
    )

    plt.xticks(x, [patient_labels_2lines(p) for p in patients], rotation=90, fontsize=8)
    plt.ylabel("All SOZ Channel Detection Rank\n(vs Total Number of Implanted Electrodes and SOZ Electrodes)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    ymax = max(np.nanmax(max_electrodes), np.nanmax(q75) if np.isfinite(q75).any() else np.nanmax(median_ranks))
    plt.ylim(0, ymax * 1.05)

    ticks = np.arange(0, int(ymax) + 1, 10)
    ticks = ticks[ticks != 190]
    plt.yticks(ticks)

    fig_path = Path(out_dir) / "barplot_all_soz_rank_median_IQR_background_electrodes_with_soz_count.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] {fig_path}")

def plot_first_and_all_soz_rank_overlay(
    df_first: pd.DataFrame,
    df_all: pd.DataFrame,
    color_map: dict,
    out_dir: Path,
    title: str,
):
    """
    UN SEUL GRAPH:
      - background bar = max_electrodes (total)
      - ● noir plein  = First SOZ median ± IQR
      - ○ blanc bord noir = All SOZ median ± IQR
      - légende à l'extérieur (droite)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge propre
    a = df_first[[
        "patient", "max_electrodes", "num_seizures",
        "first_soz_median_rank", "first_soz_q25_rank", "first_soz_q75_rank"
    ]].copy()

    b = df_all[[
        "patient",
        "all_soz_median_rank", "all_soz_q25_rank", "all_soz_q75_rank"
    ]].copy()

    d = a.merge(b, on="patient", how="inner").sort_values("patient").reset_index(drop=True)
    if d.empty:
        print("[WARN] overlay plot skipped (no merged patients).")
        return

    patients = d["patient"].tolist()
    x = np.arange(len(patients))

    max_electrodes = d["max_electrodes"].to_numpy(dtype=float)
    num_seizures = d["num_seizures"].to_numpy(dtype=int)

    f_med = d["first_soz_median_rank"].to_numpy(dtype=float)
    f_q25 = d["first_soz_q25_rank"].to_numpy(dtype=float)
    f_q75 = d["first_soz_q75_rank"].to_numpy(dtype=float)

    a_med = d["all_soz_median_rank"].to_numpy(dtype=float)
    a_q25 = d["all_soz_q25_rank"].to_numpy(dtype=float)
    a_q75 = d["all_soz_q75_rank"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, len(patients) * 0.45), 6))

    # -------------------------------------------------
    # Background bars (total electrodes)
    # -------------------------------------------------
    for i, pat in enumerate(patients):
        c = color_map.get(pat, (0.8, 0.8, 0.8, 1.0))
        ax.bar(
            i,
            max_electrodes[i],
            color=(c[0], c[1], c[2], 0.20),
            edgecolor="none",
            width=0.8,
            zorder=1,
        )

        # label seizures en haut
        ax.text(
            i,
            177,
            f"S:{num_seizures[i]}",
            ha="center",
            va="top",
            fontsize=10,
        )

    # -------------------------------------------------
    # Error bars (IQR)
    # -------------------------------------------------
    dx = -0.15  # pas de décalage horizontal, comme ton exemple
    dy = 0.15

    # FIRST SOZ ●
    valid_f = np.isfinite(f_med)
    ax.errorbar(
        x[valid_f] + dx,
        f_med[valid_f],
        yerr=np.vstack([
            (f_med - f_q25)[valid_f],
            (f_q75 - f_med)[valid_f]
        ]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )

    # ALL SOZ ○
    valid_a = np.isfinite(a_med)
    ax.errorbar(
        x[valid_a] + dy,
        a_med[valid_a],
        yerr=np.vstack([
            (a_med - a_q25)[valid_a],
            (a_q75 - a_med)[valid_a]
        ]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )

    # -------------------------------------------------
    # Points
    # -------------------------------------------------
    ax.scatter(
        x[valid_f] + dx,
        f_med[valid_f],
        s=80,
        color="black",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="First SOZ (median ± IQR)",
    )

    ax.scatter(
        x[valid_a] + dy,
        a_med[valid_a],
        s=80,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="All SOZ (median ± IQR)",
    )

    # -------------------------------------------------
    # Axis / style
    # -------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        [patient_labels_2lines(p) for p in patients],
        rotation=90,
        fontsize=9,
    )

    ax.set_ylabel(
        "SOZ detection rank (median ± IQR)\n"
        "with total contacts (light) and SOZ contacts (dark)",
        fontsize=12,
    )

    ax.set_title(title, fontsize=14, pad=10)

    ax.grid(axis="y", alpha=0.3)

    ymax = max(np.nanmax(max_electrodes), np.nanmax(f_q75), np.nanmax(a_q75))
    ax.set_ylim(0, ymax * 1.08)

    # -------------------------------------------------
    # Legend outside
    # -------------------------------------------------
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "overlay_first_and_all_soz_rank_median_IQR_on_total_electrodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[OK] Overlay plot (styled like reference) generated.")
# -------------------------------
# Trace/Audit exports
# -------------------------------

def add_rank_and_topk_flags_collapsed(
    df: pd.DataFrame,
    score_col: str,
    fracs=(0.10, 0.20),
) -> pd.DataFrame:
    """
    DF 'trace' au niveau (patient,seizure_id,seq_idx,node_index collapsed):
      - collapse par node_index dans chaque groupe (patient,seizure_id,seq_idx)
      - calcule rank (1 = meilleur score) dans chaque groupe
      - ajoute k_top10/k_top20 et flags is_in_top10/is_in_top20
    """
    needed = {"patient", "seizure_id", "seq_idx", "is_SOZ", score_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"add_rank_and_topk_flags_collapsed: missing columns {missing}")

    d = df.copy()
    d["patient"] = d["patient"].astype(str)
    d["seizure_id"] = d["seizure_id"].astype(str)
    d["seq_idx"] = pd.to_numeric(d["seq_idx"], errors="coerce").fillna(0).astype(int)
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    group_keys = ["patient", "seizure_id", "seq_idx"]
    out_parts = []

    for keys, g in d.groupby(group_keys, dropna=False):
        g2 = _collapse_group_by_node(g, score_col=score_col).dropna(subset=[score_col]).copy()
        if g2.empty:
            continue

        pat, seiz, seq = keys
        g2["patient"] = pat
        g2["seizure_id"] = seiz
        g2["seq_idx"] = seq

        g2 = g2.sort_values(score_col, ascending=False).reset_index(drop=True)
        g2["rank"] = np.arange(1, len(g2) + 1)

        n = len(g2)
        for frac in fracs:
            k = max(1, int(math.ceil(frac * n)))
            pct = int(round(frac * 100))
            g2[f"k_top{pct}"] = k
            g2[f"is_in_top{pct}"] = (g2["rank"] <= k).astype(int)

        out_parts.append(g2)

    if not out_parts:
        return pd.DataFrame(columns=["patient","seizure_id","seq_idx","node_index","is_SOZ",score_col,"rank"])

    df_trace = pd.concat(out_parts, ignore_index=True)

    front = ["patient", "seizure_id", "seq_idx", "node_index", "is_SOZ", score_col, "rank"]
    for frac in fracs:
        pct = int(round(frac * 100))
        front += [f"k_top{pct}", f"is_in_top{pct}"]
    cols = [c for c in front if c in df_trace.columns] + [c for c in df_trace.columns if c not in front]
    return df_trace[cols]


def export_patient_metrics_and_predictions_trace(
    df_raw: pd.DataFrame,
    df_best: pd.DataFrame,
    out_dir: Path,
    score_col: str,
    export_per_patient: bool = True,
) -> dict:
    """
    Exporte:
      1) audit_patient_metrics.csv  (1 ligne/patient)
      2) predictions_trace_collapsed_ranked.csv (tous patients concaténés)
      3) (optionnel) per_patient_predictions/xxx.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_cols_first = [
        "patient",
        # all-electrodes confidence
        "mean_score_pos", "mean_score_neg", "confidence_gap",
        # TOP10% confidence (nouveau)
        "mean_score_pos_top10", "mean_score_neg_top10", "confidence_gap_top10",
        # min/max
        "min_score_pos", "min_score_neg", "min_gap",
        "max_score_pos", "max_score_neg", "max_gap",
        # topk metrics
        "Precision@top_10pct", "Recall@top_10pct", "F1@top_10pct",
        "Precision@top_20pct", "Recall@top_20pct", "F1@top_20pct",
        "AUC_group_mean",
        "num_groups_used", "num_auc_groups_used",
    ]
    audit_cols = [c for c in audit_cols_first if c in df_best.columns] + [c for c in df_best.columns if c not in audit_cols_first]
    audit_path = out_dir / "audit_patient_metrics.csv"
    df_best[audit_cols].to_csv(audit_path, index=False)

    allowed_patients = set(df_best["patient"].astype(str).tolist())
    df_used = df_raw[df_raw["patient"].astype(str).isin(allowed_patients)].copy()

    df_trace = add_rank_and_topk_flags_collapsed(df_used, score_col=score_col, fracs=(0.10, 0.20))
    trace_path = out_dir / "predictions_trace_collapsed_ranked.csv"
    df_trace.to_csv(trace_path, index=False)

    per_patient_dir = out_dir / "per_patient_predictions"
    if export_per_patient:
        per_patient_dir.mkdir(parents=True, exist_ok=True)
        for pat, dp in df_trace.groupby("patient", dropna=False):
            pat_name = str(pat).replace("/", "_")
            dp.to_csv(per_patient_dir / f"{pat_name}_predictions_trace.csv", index=False)

    return {
        "audit_patient_metrics": str(audit_path),
        "predictions_trace_collapsed_ranked": str(trace_path),
        "per_patient_dir": str(per_patient_dir) if export_per_patient else "",
    }


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="preds_ranked.csv (sortie M2-on-M1)")
    ap.add_argument("--out", required=True, help="Dossier de sortie figures + csv metrics")
    ap.add_argument("--score_col", default="m2_score", help="Colonne du score (default=m2_score)")
    ap.add_argument("--fp_mode", default="before_last_soz", choices=["classic", "before_last_soz"])
    ap.add_argument(
        "--min_soz_per_seizure",
        type=int,
        default=2,
        help="Contrainte: min # électrodes SOZ distinctes par seizure (default=2). Mets 0 pour désactiver.",
    )

    # NOTE: je laisse les args BCE/series_dir si tu les utilises ailleurs,
    # mais le script ci-dessous ne force pas leur usage.
    ap.add_argument("--series_dir", default=None, help="(optionnel) dossier series/*.npz (si tu veux BCE ensuite)")
    ap.add_argument("--bce_middle_seq_only", action="store_true")
    ap.add_argument("--bce_ylim01", action="store_true")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"CSV introuvable: {csv_path}")

    # 1) read + canonize
    df = pd.read_csv(csv_path)
    if "patient" in df.columns:
        df["patient_raw"] = df["patient"].astype(str)

    df["patient"] = df["patient"].map(canon_patient)

    if "seizure_id" in df.columns:
        df["seizure_id"] = df["seizure_id"].astype(str)
    if "seq_idx" in df.columns:
        df["seq_idx"] = pd.to_numeric(df["seq_idx"], errors="coerce").fillna(0).astype(int)

    if "is_SOZ" not in df.columns:
        raise SystemExit("La colonne is_SOZ est manquante dans le CSV.")
    df["is_SOZ"] = pd.to_numeric(df["is_SOZ"], errors="coerce").fillna(0).astype(int)

    if args.score_col not in df.columns:
        raise SystemExit(f"La colonne score_col='{args.score_col}' est absente du CSV.")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")

    # constraint
    min_soz = None if args.min_soz_per_seizure <= 0 else int(args.min_soz_per_seizure)

    # 2) patient metrics (topk + confidence all electrodes)
    rows = []
    for pat, df_pat in df.groupby("patient"):
        if min_soz is not None:
            ok = patient_has_min_soz_electrodes_per_seizure_from_df(
                df_pat,
                min_soz_electrodes=min_soz,
                count_unique_node_index=True,
            )
            if not ok:
                continue

        metrics = compute_patient_metrics_from_predictions_df(df_pat, score_col=args.score_col, fp_mode=args.fp_mode)
        rows.append({"patient": pat, **metrics})

    if not rows:
        raise SystemExit("Aucun patient retenu après contrainte min_soz_per_seizure.")

    df_best = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
    allowed_patients = set(df_best["patient"].astype(str).tolist())

    # 3) add min/max pos/neg/gap across groups (as before)
    extra = []
    group_keys = ["patient", "seizure_id", "seq_idx"]
    for pat, df_pat in df.groupby("patient"):
        if pat not in allowed_patients:
            continue

        grp = df_pat.groupby(group_keys, dropna=False)
        pos_vals, neg_vals, gaps = [], [], []

        for _, g in grp:
            g2 = _collapse_group_by_node(g, score_col=args.score_col).dropna(subset=[args.score_col])
            if g2.empty:
                continue

            scores = g2[args.score_col].to_numpy(dtype=float)
            labels = g2["is_SOZ"].to_numpy(dtype=int)

            pos = float(np.mean(scores[labels == 1])) if (labels == 1).any() else np.nan
            neg = float(np.mean(scores[labels == 0])) if (labels == 0).any() else np.nan
            gap = (pos - neg) if np.isfinite(pos) and np.isfinite(neg) else np.nan

            pos_vals.append(pos); neg_vals.append(neg); gaps.append(gap)

        def _nanmin(x):
            x = np.asarray(x, float)
            return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan

        def _nanmax(x):
            x = np.asarray(x, float)
            return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

        extra.append({
            "patient": pat,
            "min_score_pos": _nanmin(pos_vals),
            "min_score_neg": _nanmin(neg_vals),
            "min_gap": _nanmin(gaps),
            "max_score_pos": _nanmax(pos_vals),
            "max_score_neg": _nanmax(neg_vals),
            "max_gap": _nanmax(gaps),
        })

    df_best = df_best.merge(pd.DataFrame(extra), on="patient", how="left")

    # 4) NOUVEAU demandé: pos/neg/gap sur TOP10% uniquement
    df_top10_conf = compute_patient_confidence_gap_seizure_level_topk(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
        frac=0.10,
        suffix="_top10",
    )
    df_best = df_best.merge(df_top10_conf, on="patient", how="left")

    # 5) save patient metrics main table
    metrics_csv = out_dir / "patient_metrics_from_preds_ranked.csv"
    df_best.to_csv(metrics_csv, index=False)
    print(f"[OK] metrics table -> {metrics_csv}")

    # palette
    color_map = build_patient_color_map(df_best["patient"].astype(str).tolist())

    # 6) trace/audit exports
    trace_paths = export_patient_metrics_and_predictions_trace(
        df_raw=df,
        df_best=df_best,
        out_dir=out_dir,
        score_col=args.score_col,
        export_per_patient=True,
    )
    print("[OK] Trace exports:", trace_paths)

    # 7) figures
    make_boxplot_topk_metrics(
        df_best,
        out_dir,
        color_map,
        title=(
            "Per-Patient Top-k SOZ Metrics\n"
            "- Model Trained on All Good Surgery Outcome Patients and Applied on All Patients -\n"
            "BAD Surgery Outcome"
        ),
    )

    # ranks
    df_first = compute_first_soz_rank_and_counts(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
    )
    df_all = compute_all_soz_rank_and_counts(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
    )

    plot_first_soz_rank_barplot(
        df_first,
        color_map,
        out_dir,
        title=(
            "Per-Patient First SOZ Channel Detection Rank (Median ± IQR)\n"
            "with Total Implanted Electrode Count and Number of Evaluated Seizures\n"
            "- Model Trained on All Good Surgery Outcome Patients and Applied on All Patients -\n"
            "BAD Surgery Outcome"
        ),
    )

    plot_all_soz_rank_barplot(
        df_all,
        color_map,
        out_dir,
        title=(
            "Per-Patient All SOZ Channel Detection Rank (Median ± IQR)\n"
            "with Total Implanted Electrode, SOZ Electrode Counts and Number of Evaluated Seizures\n"
            "- Model Trained on All Good Surgery Outcome Patients and Applied on All Patients -\n"
            "BAD Surgery Outcome"
        ),
    )

    # NOUVEAU overlay plot (le graph que tu décris)
    plot_first_and_all_soz_rank_overlay(
        df_first=df_first,
        df_all=df_all,
        color_map=color_map,
        out_dir=out_dir,
        title=(
            "Per-Patient SOZ Detection Ranks (Median ± IQR) on Total Implanted Electrodes\n"
            "First SOZ vs All SOZ (2 points per patient)\n"
            "- BAD Surgery Outcome -"
        ),
    )

    # confidence plots (all electrodes)
    df_conf_all = compute_patient_confidence_gap_seizure_level(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
    )
    plot_confidence_gap_boxplot(
        df_conf_all,
        out_path=out_dir / "boxplot_confidence_gap_seizure_level.png",
        color_map={canon_patient(k): v for k, v in color_map.items()},
        title=(
            "Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\n"
            "and Their Separation (Confidence Gap)\n"
            "- BAD Surgery Outcome -"
        ),
        col_pos="mean_score_pos",
        col_neg="mean_score_neg",
        col_gap="confidence_gap",
    )

    # confidence plots (TOP10 only)
    if not df_top10_conf.empty:
        plot_confidence_gap_boxplot(
            df_top10_conf,
            out_path=out_dir / "boxplot_confidence_gap_seizure_level_TOP10.png",
            color_map={canon_patient(k): v for k, v in color_map.items()},
            title=(
                "Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\n"
                "and Their Separation (Confidence Gap) - TOP 10% ELECTRODES ONLY\n"
                "- BAD Surgery Outcome -"
            ),
            col_pos="mean_score_pos_top10",
            col_neg="mean_score_neg_top10",
            col_gap="confidence_gap_top10",
        )

    print("[DONE] Figures + exports générés.")


if __name__ == "__main__":
    main()