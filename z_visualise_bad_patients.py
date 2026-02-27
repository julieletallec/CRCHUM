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

Usage:
uv run z_visualise_bad_patients.py \
  --csv /home/julieletallec/test/M2_on_M1_outputs/preds_ranked.csv \
  --out /home/julieletallec/test/M2_on_M1_outputs/figures_m2_on_bad_pat_bce_top10 \
  --series_dir /home/julieletallec/test/M1_singleconfig_runs/results/results \
  --bce_middle_seq_only \
  --bce_ylim01 \
  --min_soz_per_seizure 1

Optionnel:
  --score_col m2_score     (ou y_score si tu renommes)
  --fp_mode classic|before_last_soz
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

import seaborn as sns
from matplotlib.lines import Line2D


# -------------------------------
# Helpers
# -------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_per_patient_seizure_scores_with_labels_and_optional_csv(
    df: pd.DataFrame,
    out_dir: Path,
    score_col: str = "m2_score",
    electrode_col: str = "electrode_name",     # <-- adapte à ton CSV (ou laisse, ça fallback)
    patient_filter_prefix: str = "CHUM",       # uniquement CHUM
    patient: str | None = None,                # si None -> tous les CHUM
    seizure_id: str | int | None = None,       # si donné -> seulement cette crise
    export_csv: bool = False,                  # export CSV pour (patient, seizure)
    csv_suffix: str = "electrode_scores",
    annotate: bool = True,                     # écrire les noms à côté des points
    max_labels: int = 200,                      # évite figures illisibles
    figsize=(12, 6),
    dpi=200,
    drop_unknown_seizure: bool = False,         # si True: enlève seizure_id non numériques (ex: "?")
    unknown_seizure_token: str = "?",           # token à considérer comme "unknown"
):
    """
    Pour chaque patient CHUM:
      - génère un plot séparé:
            x = seizure_id (trié numériquement si possible)
            y = score (score_col)
            1 point par électrode (après collapse node_index si présent)
            SOZ en rouge, non-SOZ en gris
            option: annoter avec le nom de l'électrode

    En plus, si patient + seizure_id + export_csv=True:
      - export un CSV avec (electrode, score, is_SOZ, node_index) pour cette crise
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("COLUMNS:", df.columns)
    # ---- copie + sanity checks
    d = df.copy()

    if not {"patient", "seizure_id", "is_SOZ", score_col}.issubset(d.columns):
        raise ValueError(f"CSV doit contenir au minimum: patient, seizure_id, is_SOZ, {score_col}")

    # ---- types / nettoyage
    d["patient"] = d["patient"].astype(str).str.strip()
    d["seizure_id"] = d["seizure_id"].astype(str).str.strip()
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    # Debug utile si tu veux voir les valeurs brutes
    # print(d["seizure_id"].value_counts(dropna=False))

    # ---- filtre CHUM uniquement (robuste à CHUM__Patient_XX vs CHUM_Patient_XX)
    def _is_chum(p: str) -> bool:
        p = str(p)
        return (
            p.startswith(patient_filter_prefix)
            or p.startswith(patient_filter_prefix + "_")
            or p.startswith(patient_filter_prefix + "__")
        )

    d = d[d["patient"].apply(_is_chum)].copy()

    # ---- option: drop unknown seizure ids ("?")
    if drop_unknown_seizure:
        d = d[d["seizure_id"] != str(unknown_seizure_token)].copy()

    # ---- filtre patient / seizure si demandé
    if patient is not None:
        patient = str(patient).strip()
        d = d[d["patient"] == patient].copy()

    if seizure_id is not None:
        seizure_id = str(seizure_id).strip()
        d = d[d["seizure_id"] == seizure_id].copy()

    if d.empty:
        print("[WARN] Aucun point après filtrage (CHUM/patient/seizure).")
        return

    # ---- colonne électrode (fallback)
    if electrode_col not in d.columns:
        if "node_index" in d.columns:
            electrode_col_eff = "node_index"
        else:
            d = d.reset_index(drop=False).rename(columns={"index": "row_index"})
            electrode_col_eff = "row_index"
    else:
        electrode_col_eff = electrode_col

    # ---- collapse par node_index au sein (patient, seizure_id) pour éviter doublons
    if "node_index" in d.columns:
        d["node_index"] = pd.to_numeric(d["node_index"], errors="coerce")
        agg = {score_col: "max", "is_SOZ": "max", electrode_col_eff: "first"}
        d = (
            d.groupby(["patient", "seizure_id", "node_index"], as_index=False, dropna=False)
             .agg(agg)
        )

    # ---- export CSV si demandé (uniquement quand patient + seizure_id donnés)
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

    # ---- helper pour trier seizure_id de façon numérique si possible (sinon à la fin)
    def _sort_key(s: str):
        s = str(s).strip()
        try:
            return (0, int(float(s)))  # "5", "5.0" -> ok
        except Exception:
            return (1, s)              # "?" etc -> après les numériques

    # ---- plots: 1 figure par patient
    for pat, dp in d.groupby("patient", dropna=False):
        dp = dp.copy()
        dp["seizure_id"] = dp["seizure_id"].astype(str).str.strip()  # sécurité anti-mix

        seizs = sorted(dp["seizure_id"].unique().tolist(), key=_sort_key)
        x_map = {str(s).strip(): i for i, s in enumerate(seizs)}

        fig, ax = plt.subplots(figsize=figsize)

        for seiz, g in dp.groupby("seizure_id", dropna=False):
            seiz = str(seiz).strip()
            if seiz not in x_map:
                print(f"[WARN] seizure_id '{seiz}' absent de x_map pour {pat} -> skip")
                continue

            x0 = x_map[seiz]
            ys = g[score_col].to_numpy(dtype=float)
            soz = g["is_SOZ"].to_numpy(dtype=int)

            # jitter léger pour séparer les électrodes
            xs = x0 + np.random.uniform(-0.18, 0.18, size=len(g))

            # non-SOZ
            m0 = (soz == 0) & np.isfinite(ys)
            ax.scatter(xs[m0], ys[m0], s=18, alpha=0.55, color="grey", linewidth=0)

            # SOZ
            m1 = (soz == 1) & np.isfinite(ys)
            ax.scatter(xs[m1], ys[m1], s=26, alpha=0.90, color="red", linewidth=0)

            # annotation (électrodes) : uniquement si pas trop de points
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




def canon_patient(p: str) -> str:
    return str(p).replace("::", "_").replace("__", "_").strip()



def build_patient_color_map(patients: List[str]):
    """
    Palette HUSL stable (même patient = même couleur).
    """
    unique = sorted(pd.unique(patients))
    palette = sns.color_palette("husl", len(unique))
    return {pat: palette[i] for i, pat in enumerate(unique)}



def patient_labels_2lines(pat: str) -> str:
    pat = str(pat).strip()

    # enlève underscores/espaces en fin (ex: "CHUM_Patient_16_")
    pat = pat.rstrip("_").rstrip()

    # cas historiques
    if "__" in pat:
        a, b = pat.split("__", 1)
        return f"{a}\n{b}"
    if "::" in pat:
        a, b = pat.split("::", 1)
        return f"{a}\n{b}"

    # cas canonisés (ex: "CHUM_Patient_16")
    if pat.count("_") >= 1:
        a, b = pat.split("_", 1)
        return f"{a}\n{b}"

    return pat

                    



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
    Robustesse: si une électrode apparaît plusieurs fois dans un même (patient,seizure,seq),
    on la collapse en 1 ligne via node_index:
      - score = max
      - is_SOZ = max (OR)
      - rank = min si présent
    Si node_index absent: renvoie df_group tel quel.
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

def onset_curve_bce(p_smooth, t_true, eps=1e-7):
    p_smooth = np.asarray(p_smooth, dtype=float)
    T = len(p_smooth)
    if t_true is None or t_true < 0 or T == 0:
        return np.nan
    y_true = np.zeros(T, dtype=float)
    y_true[t_true:] = 1.0
    p = np.clip(p_smooth, eps, 1.0 - eps)
    bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(bce.mean())


def canon_patient(p: str) -> str:
    return str(p).replace("::", "_").replace("__", "_").strip()

def onset_curve_bce(p_smooth, t_true, eps=1e-7):
    p_smooth = np.asarray(p_smooth, dtype=float)
    T = len(p_smooth)
    if t_true is None or t_true < 0 or T == 0:
        return np.nan
    y_true = np.zeros(T, dtype=float)
    y_true[t_true:] = 1.0
    p = np.clip(p_smooth, eps, 1.0 - eps)
    bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(bce.mean())


def collect_bce_from_npz_series_dir_(series_dir: Path, allowed_patients: set[str] | None = None) -> pd.DataFrame:
    rows = []
    series_dir = Path(series_dir)

    for npz_path in sorted(series_dir.rglob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        pat = str(data.get("patient", ""))
        if allowed_patients is not None and pat not in allowed_patients:
            continue

        if "p_graph_smooth" in data:
            p_smooth = np.asarray(data["p_graph_smooth"], dtype=float)
        elif "p_graph" in data:
            p_smooth = np.asarray(data["p_graph"], dtype=float)
        else:
            continue

        t_true = int(data.get("t_true", -1))
        if t_true < 0:
            t_true = None

        bce_onset = onset_curve_bce(p_smooth, t_true)
        seizure_id = str(data.get("seizure_id", "?"))
        seq_index = int(data.get("seq_index", data.get("seq_idx", -1)))

        rows.append({
            "patient": pat,
            "seizure_id": seizure_id,
            "seq_index": seq_index,
            "bce_onset": bce_onset,
            "t_true": t_true,
            "npz_path": str(npz_path),
        })

    if not rows:
        raise RuntimeError("Aucune BCE calculée: aucun .npz exploitable trouvé (p_graph_smooth/p_graph + t_true).")

    return pd.DataFrame(rows)


def keep_middle_seq_per_seizure(df_bce: pd.DataFrame) -> pd.DataFrame:
    def _pick_middle(g):
        g = g.sort_values("seq_index")
        mid = len(g) // 2
        return g.iloc[[mid]]
    return (
        df_bce.groupby(["patient", "seizure_id"], group_keys=False)
              .apply(_pick_middle)
              .reset_index(drop=True)
    )

def plot_bce_boxplot_per_patient(
    df_bce: pd.DataFrame,
    out_dir: Path,
    color_map: dict,
    ylim01: bool = True,
):
    """
    Boxplot des BCE temporelles par patient (une valeur par séquence).
    Style "comme la figure de référence":
      - boîtes blanches (facecolor) avec contours colorés
      - whiskers/caps plus épais
      - médiane orange plus épaisse
      - points plus gros
      - labels patients propres sur 2 lignes (CHUM\nPatient_16, ds004100\nsub-HUPxxx, etc.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ordre des patients (stable)
    pats = sorted(df_bce["patient"].astype(str).unique())

    # Données par patient
    data = [
        pd.to_numeric(df_bce.loc[df_bce["patient"].astype(str) == p, "bce_onset"], errors="coerce")
          .dropna()
          .to_numpy(dtype=float)
        for p in pats
    ]

    # Labels X: utilise ton helper existant (gère "__" et "::")
    xtick_labels = [patient_labels_2lines(p) for p in pats]

    # Figure plus "respirée" comme l'exemple
    #plt.figure(figsize=(min(14, len(pats) * 0.58), 4))
    plt.figure(figsize=(10, 4))

    # Boxplot (patch_artist=True pour pouvoir mettre facecolor)
    bp = plt.boxplot(
        data,
        showmeans=False,
        patch_artist=True,
        showfliers=False,
        widths=0.5,  # boîtes un peu plus fines comme la ref
    )

    # Style: épaisseurs
    box_lw = 1
    whisk_lw = 1
    cap_lw = 1
    median_lw = 1.5

    # Appliquer la couleur par patient sur chaque "groupe" de box
    for i, p in enumerate(pats):
        c = color_map.get(p, "grey")

        # Boîte: fond blanc + contour coloré
        plt.setp(bp["boxes"][i], facecolor="white", edgecolor=c, linewidth=box_lw)

        # Whiskers et caps: colorés et plus épais
        plt.setp(bp["whiskers"][2 * i : 2 * i + 2], color=c, linewidth=whisk_lw)
        plt.setp(bp["caps"][2 * i : 2 * i + 2], color=c, linewidth=cap_lw)

        # Médiane: orange plus épais
        plt.setp(bp["medians"][i], color="orange", linewidth=median_lw)

    # Points: plus gros, mêmes couleurs, jitter léger
    for i, p in enumerate(pats):
        vals = pd.to_numeric(
            df_bce.loc[df_bce["patient"].astype(str) == p, "bce_onset"],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        c = color_map.get(p, "grey")
        for v in vals:
            jitter = np.random.uniform(-0.10, 0.10)
            plt.scatter(
                i + 1 + jitter,
                v,
                color=c,
                s=35,           # plus gros
                alpha=0.85,
                linewidth=0.0,  # points pleins comme la ref
            )

    plt.xticks(range(1, len(pats) + 1), xtick_labels, rotation=90, fontsize=9)
    plt.ylabel("Time-wise BCE vs\nIdeal Onset Step", fontsize=11)
    plt.title(
        "Per-Patient Onset Curve Loss (Graph-Level Aggregated Ictal Activity Probability)\n"
        "- Same Model (Fixed Hyperparameters) Trained Separately and Applied on Each Patient -\n"
        "BAD Surgery Outcome Patients",
        fontsize=14,
        pad=10,
    )
    plt.grid(axis="y", alpha=0.3)

    if ylim01:
        plt.ylim(0, 1)

    fig_path = out_dir / "boxplot_bce_onset_per_patient.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Figure BCE sauvegardée -> {fig_path}")



def collect_bce_from_npz_series_dir(series_dir: Path, allowed_patients: set[str] | None = None) -> pd.DataFrame:
    rows = []
    series_dir = Path(series_dir)

    allowed_canon = {canon_patient(p) for p in allowed_patients} if allowed_patients is not None else None

    for npz_path in sorted(series_dir.rglob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        # patient dans le npz (ou fallback sur le nom du dossier)
        pat = str(data.get("patient", "")).strip()
        if not pat:
            # ex: .../CHUM_Patient_09/series/seiz_1_seq_001.npz
            pat = npz_path.parent.parent.name

        if allowed_patients is not None and canon_patient(pat) not in allowed_patients:
            continue


        # score temporel
        if "p_graph_smooth" in data:
            p_smooth = np.asarray(data["p_graph_smooth"], dtype=float)
        elif "p_graph" in data:
            p_smooth = np.asarray(data["p_graph"], dtype=float)
        else:
            continue

        t_true = int(data.get("t_true", -1))
        if t_true < 0:
            t_true = None

        bce_onset = onset_curve_bce(p_smooth, t_true)

        seizure_id = str(data.get("seizure_id", "?"))
        seq_index = int(data.get("seq_index", data.get("seq_idx", -1)))
        pat = canon_patient(pat)

        rows.append({
            "patient": pat,
            "seizure_id": seizure_id,
            "seq_index": seq_index,
            "bce_onset": bce_onset,
            "t_true": t_true,
            "npz_path": str(npz_path),
        })

    if not rows:
        raise RuntimeError("Aucune BCE calculée: aucun .npz exploitable trouvé (p_graph_smooth/p_graph + t_true).")

    return pd.DataFrame(rows)


def plot_topk_scores_distribution_per_patient(
    df: pd.DataFrame,
    out_dir: Path,
    color_map: dict,
    score_col: str = "m2_score",
    frac: float = 0.10,
    title: str = "Per-Patient Distribution of Electrode Scores in Top-k",
    out_name: str = "boxplot_top10_scores_distribution_per_patient.png",
    jitter: float = 0.12,
    alpha_nonsoz: float = 0.55,
    alpha_soz: float = 0.90,
    s_nonsoz: float = 14,
    s_soz: float = 22,
):
    """
    Collecte, pour chaque patient, les scores des électrodes appartenant au top-k (frac)
    DANS CHAQUE GROUPE (patient,seizure_id,seq_idx) après collapse par node_index.
    Puis plot distribution par patient (boxplot + points jitter).
    SOZ: rouge + un poil plus gros.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = {"patient", "seizure_id", "seq_idx", "is_SOZ", score_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"plot_topk_scores_distribution_per_patient: missing columns: {missing}")

    d = df.copy()
    d["patient"] = d["patient"].astype(str)
    d["seizure_id"] = d["seizure_id"].astype(str)
    d["seq_idx"] = pd.to_numeric(d["seq_idx"], errors="coerce").fillna(0).astype(int)
    d["is_SOZ"] = pd.to_numeric(d["is_SOZ"], errors="coerce").fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[score_col])

    rows = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    for (pat, seiz, seq), g in d.groupby(group_keys, dropna=False):
        g2 = _collapse_group_by_node(g, score_col=score_col)
        g2 = g2.dropna(subset=[score_col])
        if g2.empty:
            continue

        scores = g2[score_col].to_numpy(dtype=float)
        labels = g2["is_SOZ"].to_numpy(dtype=int)

        n = len(scores)
        if n == 0:
            continue

        k = max(1, int(math.ceil(frac * n)))
        order = np.argsort(scores)[::-1]
        top_idx = order[:k]

        top_scores = scores[top_idx]
        top_labels = labels[top_idx]

        rows.append(pd.DataFrame({
            "patient": pat,
            "score": top_scores,
            "is_SOZ": top_labels,
        }))

    if not rows:
        print("[WARN] No top-k scores collected -> plot skipped.")
        return

    df_plot = pd.concat(rows, ignore_index=True)

    # ordre stable des patients (comme tes autres plots)
    patients = sorted(df_plot["patient"].unique().tolist())

    # data par patient pour matplotlib boxplot
    data = []
    for p in patients:
        vals = df_plot.loc[df_plot["patient"] == p, "score"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(patients)), 6))

    ax.boxplot(
        data,
        tick_labels=[patient_labels_2lines(p) for p in patients],
        showfliers=False,
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=1.5),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    # points jitter
    for i, p in enumerate(patients, start=1):
        sub = df_plot[df_plot["patient"] == p]
        vals = sub["score"].to_numpy(dtype=float)
        labs = sub["is_SOZ"].to_numpy(dtype=int)

        xs = i + np.random.uniform(-jitter, jitter, size=len(vals))
        base_col = color_map.get(p, "grey")

        # non-SOZ
        m0 = (labs == 0) & np.isfinite(vals)
        ax.scatter(
            xs[m0], vals[m0],
            s=s_nonsoz, alpha=alpha_nonsoz,
            color=base_col, linewidth=0,
            zorder=2
        )

        # SOZ -> rouge + plus gros
        m1 = (labs == 1) & np.isfinite(vals)
        ax.scatter(
            xs[m1], vals[m1],
            s=s_soz, alpha=alpha_soz,
            color="red", linewidth=0,
            zorder=3
        )

    ax.set_title(title)
    ax.set_ylabel(f"y_score of electrodes in top {int(frac*100)}% (within each group)")
    ax.grid(axis="y", alpha=0.3)

    # si ton score est dans [0,1] garde ça, sinon commente
    ax.set_ylim(0, 1)

    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    fig_path = out_dir / out_name
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Top-{int(frac*100)}% score distribution -> {fig_path}")


def precision_recall_f1_topk(
    scores: np.ndarray,
    labels: np.ndarray,
    frac: float,
    fp_mode: str = "before_last_soz",
) -> Tuple[float, float, float]:
    """
    EXACTEMENT comme ton code:
      - k = ceil(frac * N)
      - tri desc par score
      - TP = #SOZ dans top-k
      - FN = #SOZ hors top-k
      - FP:
          classic: #nonSOZ dans top-k
          before_last_soz: #nonSOZ STRICTEMENT avant le dernier SOZ dans top-k
    """
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
        g2 = _collapse_group_by_node(g, score_col=score_col)
        g2 = g2.dropna(subset=[score_col])
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


def ensure_rank_per_group(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Si 'rank' absent, on le calcule dans chaque groupe (patient,seizure,seq) en triant par score desc,
    puis rank=1..N.
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
# Plots
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

    print("PATIENTS:", patients)

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
    fig_path = out_dir / "boxplot_topk_metrics.png"
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

    # background: max_electrodes
    for i, (pat, n_elec) in enumerate(zip(patients, max_electrodes)):
        c = color_map.get(pat, (0.8, 0.8, 0.8, 1.0))
        plt.bar(i, n_elec, color=(c[0], c[1], c[2], 0.25), edgecolor="none")

    # median bars
    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, color=c, edgecolor="black", linewidth=1.0)
        #plt.text(i, np.nanmax(max_electrodes) * 0.98, f"S:{int(num_seizures[i])}", ha="center", va="top", fontsize=9)
        plt.text(i, 177, f"S:{int(num_seizures[i])}", ha="center", va="top", fontsize=9)

    # IQR errorbars
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

    fig_path = out_dir / "barplot_first_soz_rank_median_IQR_background_electrodes.png"
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

    # backgrounds
    for i, pat in enumerate(patients):
        c = color_map.get(pat, (0.8, 0.8, 0.8, 1.0))
        plt.bar(i, max_electrodes[i], width=0.8, color=(c[0], c[1], c[2], 0.20), edgecolor="none")
        plt.bar(i, max_soz_electrodes[i], width=0.8, color=(c[0], c[1], c[2], 0.65), edgecolor="none")

    # median bars
    for i, (pat, med) in enumerate(zip(patients, median_ranks)):
        if np.isnan(med):
            continue
        c = color_map.get(pat, "grey")
        plt.bar(i, med, width=0.25, color=c, edgecolor="black", linewidth=1.0)
        plt.text(i, 177, f"S:{int(num_seizures[i])}", ha="center", va="top", fontsize=9)

    # IQR
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

    fig_path = out_dir / "barplot_all_soz_rank_median_IQR_background_electrodes_with_soz_count.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] {fig_path}")




import numpy as np
import pandas as pd

def compute_patient_confidence_gap_seizure_level(df: pd.DataFrame, score_col: str = "y_score") -> pd.DataFrame:
    """
    Pour chaque patient:
      - pour chaque seizure: pos_mean = mean(scores SOZ), neg_mean = mean(scores nonSOZ), gap = pos_mean - neg_mean
      - puis patient-level: moyenne des pos_mean sur les seizures, idem neg_mean, idem gap

    Retourne un DF avec:
      patient, mean_score_pos, mean_score_neg, confidence_gap
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

    # Optionnel mais recommandé: si node_index présent, on évite les doublons d'électrodes
    # au sein d'une seizure (si plusieurs lignes par électrode, on garde le max score)
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

        seiz_rows.append({
            "patient": pat,
            "seizure_id": seiz,
            "pos_mean_seiz": pos,
            "neg_mean_seiz": neg,
            "gap_seiz": gap,
        })

    df_seiz = pd.DataFrame(seiz_rows)

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    df_pat = (
        df_seiz.groupby("patient", as_index=False)
               .agg({
                   "pos_mean_seiz": nanmean_safe,
                   "neg_mean_seiz": nanmean_safe,
                   "gap_seiz": nanmean_safe,
               })
               .rename(columns={
                   "pos_mean_seiz": "mean_score_pos",
                   "neg_mean_seiz": "mean_score_neg",
                   "gap_seiz": "confidence_gap",
               })
    )
    return df_pat


import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_gap_boxplot(
    df_pat: pd.DataFrame,
    out_path,
    color_map: dict,
    title: str = "Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\nand Their Separation (Confidence Gap)\n- Same Model (Fixed Hyperparameters) Trained on Good Surgery Outcome Patients and Applied on All Patients\nBAD Surgery Outcome",
):
    """
    df_pat doit contenir:
      patient, mean_score_pos, mean_score_neg, confidence_gap

    - Couleurs cohérentes par patient (via clé canon_patient)
    - Légende à droite
    - Axe Y forcé [-0.2, 1]
    """
    dfp = df_pat.copy()
    dfp["patient"] = dfp["patient"].astype(str)
    dfp["patient_canon"] = dfp["patient"].map(canon_patient)

    cols = ["mean_score_pos", "mean_score_neg", "confidence_gap"]
    xlabels = [
        "Average Positive Score (SOZ)\nMean over Seizures",
        "Average Negative Score (non-SOZ)\nMean over Seizures",
        "Confidence Gap (SOZ vs non-SOZ)\nMean over Seizures",
    ]

    # Données pour boxplot (par colonne)
    data = []
    for c in cols:
        v = pd.to_numeric(dfp[c], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)

    fig, ax = plt.subplots(figsize=(11, 8))

    bp = ax.boxplot(
        data,
        tick_labels=xlabels,          # ✅ plus de warning Matplotlib
        showfliers=False,
        widths=0.30,
        patch_artist=True,
        medianprops=dict(color="orange", linewidth=1.5),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    # points patient par patient (même couleur sur les 3 colonnes)
    # on fait un jitter horizontal + transparence
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

    # ✅ légende patient à droite (1 entrée par patient)
    # On prend uniquement les patients présents, triés
    pats = sorted(dfp[["patient", "patient_canon"]].drop_duplicates()["patient_canon"].tolist())
    handles = []
    for pcanon in pats:
        # trouver un label "original" représentatif (première occurrence)
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

    ax.legend(
        handles=handles,
        #title="Patient",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] confidence gap boxplot -> {out_path}")




def make_boxplot_minmax_gap(df_best: pd.DataFrame, out_dir: Path, color_map: dict, title: str):
    """
    3 panels:
      - min_score_pos / min_score_neg / min_gap
      - max_score_pos / max_score_neg / max_gap
      - mean_score_pos / mean_score_neg / confidence_gap
    """
    cols = [
        ("min_score_pos", "min_score_neg", "min_gap", "Min scores & gap"),
        ("max_score_pos", "max_score_neg", "max_gap", "Max scores & gap"),
        ("mean_score_pos", "mean_score_neg", "confidence_gap", "Mean scores & gap"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    patients = df_best["patient"].tolist()

    for ax, (c1, c2, c3, t) in zip(axes, cols):
        data = [df_best[c1].astype(float).values,
                df_best[c2].astype(float).values,
                df_best[c3].astype(float).values]
        labels = [c1, c2, c3]
        ax.boxplot(data, labels=labels, showmeans=False, showfliers=False)

        for i, col in enumerate([c1, c2, c3]):
            x_center = i + 1
            vals = df_best[col].astype(float).values
            for val, pat in zip(vals, patients):
                if np.isnan(val):
                    continue
                jitter = np.random.uniform(-0.12, 0.12)
                ax.scatter(x_center + jitter, val, color=color_map.get(pat, "grey"), s=45, alpha=0.75, linewidth=0.3)

        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    fig_path = out_dir / "boxplot_confidence_gap.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {fig_path}")


def compute_patient_confidence_gap_seizure_level_topk(
    df: pd.DataFrame,
    score_col: str = "m2_score",
    frac: float = 0.10,
) -> pd.DataFrame:
    """
    Même idée que compute_patient_confidence_gap_seizure_level, mais en ne gardant
    que les électrodes dans le top-k (frac) AU SEIN de chaque seizure, après collapse node_index.

    Pour chaque (patient, seizure):
      - collapse par node_index (max score, OR is_SOZ)
      - top-k selon score_col
      - pos_mean = mean(scores SOZ dans top-k)
      - neg_mean = mean(scores nonSOZ dans top-k)
      - gap = pos_mean - neg_mean
    Puis patient-level = moyenne sur seizures (nanmean).
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

    # collapse par (patient, seizure, node_index) si possible
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

        seiz_rows.append({
            "patient": pat,
            "seizure_id": seiz,
            "pos_mean_seiz": pos,
            "neg_mean_seiz": neg,
            "gap_seiz": gap,
        })

    df_seiz = pd.DataFrame(seiz_rows)
    if df_seiz.empty:
        return pd.DataFrame(columns=["patient", "mean_score_pos", "mean_score_neg", "confidence_gap"])

    def nanmean_safe(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    df_pat = (
        df_seiz.groupby("patient", as_index=False)
               .agg({
                   "pos_mean_seiz": nanmean_safe,
                   "neg_mean_seiz": nanmean_safe,
                   "gap_seiz": nanmean_safe,
               })
               .rename(columns={
                   "pos_mean_seiz": "mean_score_pos",
                   "neg_mean_seiz": "mean_score_neg",
                   "gap_seiz": "confidence_gap",
               })
    )
    return df_pat



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
    ap.add_argument(
        "--series_dir",
        default=None,
        help="Dossier contenant les sous-dossiers patients avec /series/*.npz (pour BCE onset).",
    )
    ap.add_argument(
        "--bce_middle_seq_only",
        action="store_true",
        help="Garde uniquement la séquence du milieu par (patient,seizure) avant plot BCE.",
    )
    ap.add_argument(
        "--bce_ylim01",
        action="store_true",
        help="Force l'axe Y du plot BCE entre 0 et 1.",
    )

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"CSV introuvable: {csv_path}")

    # ------------------------------------------------------------
    # 1) Lire UNE seule fois le CSV et tout canoniser / typer
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # garde une copie "raw" si besoin (debug)
    if "patient" in df.columns:
        df["patient_raw"] = df["patient"].astype(str)

    # canonisation patient (IMPORTANT pour matcher df_best)
    df["patient"] = df["patient"].map(canon_patient)

    # types
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

    # contrainte min SOZ per seizure
    min_soz = None if args.min_soz_per_seizure <= 0 else int(args.min_soz_per_seizure)
    """

    plot_per_patient_seizure_scores_with_labels_and_optional_csv(
        df=df,
        out_dir=out_dir / "per_patient_seizure_scatter",
        score_col=args.score_col,
        electrode_col="electrode_name",  # <-- change si besoin
        annotate=True,
    )

    """
    # 2) Un patient + une crise, et export CSV
    plot_per_patient_seizure_scores_with_labels_and_optional_csv(
        df=df,
        out_dir=out_dir / "per_patient_seizure_scatter",
        score_col=args.score_col,
        electrode_col="electrode_name",  # <-- change si besoin
        patient="CHUM_Patient_21",        # (ou ton canon_patient)
        seizure_id="2",
        export_csv=True,
        annotate=True,
    )

    # ------------------------------------------------------------
    # 2) Compute patient metrics (mean + topk + auc)
    # ------------------------------------------------------------
    
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

        metrics = compute_patient_metrics_from_predictions_df(
            df_pat, score_col=args.score_col, fp_mode=args.fp_mode
        )
        rows.append({"patient": pat, **metrics})

    if not rows:
        raise SystemExit("Aucun patient retenu après contrainte min_soz_per_seizure.")

    df_best = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)

    # ------------------------------------------------------------
    # 3) Ajout MIN/MAX + gaps (sur groupes seizure/seq)
    # ------------------------------------------------------------
    extra = []
    group_keys = ["patient", "seizure_id", "seq_idx"]

    allowed_patients = set(df_best["patient"].astype(str).tolist())

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

            pos_vals.append(pos)
            neg_vals.append(neg)
            gaps.append(gap)

        def _nanmin(x):
            x = np.asarray(x, float)
            return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan

        def _nanmax(x):
            x = np.asarray(x, float)
            return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

        extra.append(
            {
                "patient": pat,
                "min_score_pos": _nanmin(pos_vals),
                "min_score_neg": _nanmin(neg_vals),
                "min_gap": _nanmin(gaps),
                "max_score_pos": _nanmax(pos_vals),
                "max_score_neg": _nanmax(neg_vals),
                "max_gap": _nanmax(gaps),
            }
        )

    df_extra = pd.DataFrame(extra)
    df_best = df_best.merge(df_extra, on="patient", how="left")

    # ------------------------------------------------------------
    # 4) Sauvegarde metrics table + palette couleurs
    # ------------------------------------------------------------
    metrics_csv = out_dir / "patient_metrics_from_preds_ranked.csv"
    df_best.to_csv(metrics_csv, index=False)
    print(f"[OK] metrics table -> {metrics_csv}")

    color_map = build_patient_color_map(df_best["patient"].astype(str).tolist())

    # ------------------------------------------------------------
    # 5) Figures
    # ------------------------------------------------------------
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

    df_first = compute_first_soz_rank_and_counts(
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

    df_all = compute_all_soz_rank_and_counts(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
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

    # Confidence gap seizure-level (utilise le même df canonisé, PAS de re-read)
    df_pat = compute_patient_confidence_gap_seizure_level(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
    )

    plot_confidence_gap_boxplot(
        df_pat,
        out_path=out_dir / "boxplot_confidence_gap_seizure_level.png",
        color_map=color_map,
        title=(
            "Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\n"
            "and Their Separation (Confidence Gap)\n"
            "- Model Trained on All Good Surgery Outcome Patients and Applied on All Patients -\n"
            "BAD Surgery Outcome"
        ),
    )
    df_pat_top10 = compute_patient_confidence_gap_seizure_level_topk(
        df[df["patient"].isin(allowed_patients)],
        score_col=args.score_col,
        frac=0.10,
    )

    plot_confidence_gap_boxplot(
        df_pat_top10,
        out_path=out_dir / "boxplot_confidence_gap_seizure_level_TOP10.png",
        color_map=color_map,
        title="Per-Patient SOZ and non-SOZ Confidence Scores (Mean over Seizures)\n"
            "and Their Separation (Confidence Gap) - TOP 10% ELECTRODES ONLY\n"
            "- Model Trained on All Good Surgery Outcome Patients and Applied on All Patients -\n"
            "BAD Surgery Outcome",
    )


    # NOUVEAU: distribution des scores dans le top10% (par groupe seizure/seq)
    plot_topk_scores_distribution_per_patient(
        df=df[df["patient"].isin(allowed_patients)],
        out_dir=out_dir,
        color_map=color_map,
        score_col=args.score_col,
        frac=0.10,
        title=(
            "Per-Patient Distribution of Electrode Scores in the Top 10%\n"
            "- Model Trained on Good Surgery Outcome Patients and Applied on Bad Surgery Outcome Patients -"        ),
        out_name="boxplot_top10_scores_distribution_per_patient.png",
    )

    # ------------------------------------------------------------
    # 6) BCE onset (npz)
    # ------------------------------------------------------------
    if args.series_dir is not None:
        allowed = set(df_best["patient"].astype(str).tolist())  # déjà canonisé
        df_bce_all = collect_bce_from_npz_series_dir(Path(args.series_dir), allowed_patients=allowed)

        if args.bce_middle_seq_only:
            df_bce = keep_middle_seq_per_seizure(df_bce_all)
        else:
            df_bce = df_bce_all

        bce_csv = out_dir / "timewise_onset_bce_from_npz.csv"
        df_bce.to_csv(bce_csv, index=False)
        print(f"[OK] BCE table -> {bce_csv}")

        plot_bce_boxplot_per_patient(df_bce, out_dir, color_map, ylim01=args.bce_ylim01)
    else:
        print("[INFO] series_dir non fourni -> BCE onset non calculée.")

    print("[DONE] Figures générées.")


    


if __name__ == "__main__":
    main()
