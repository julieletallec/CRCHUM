#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot p_v(t) par node, coloré SOZ vs non-SOZ, + moyenne par groupe.
- Entrée: un CSV "seq" (valeurs temporelles) + un CSV "nodes" (mapping node -> SOZ)
- Sortie: un PNG

Changements vs plot existant:
- pas de p_graph ni p_graph lissé
- garde onset true uniquement (ligne verticale)
- pas de onset prédit

Usage:
uv run z_visualise_activation_patterns_m1_output.py \
  --seq_csv results_grid_search_kwta_20_10_burst/results_20251212_193341/results/CHUM__Patient_01/series/seiz_7_seq_006.csv \
  --nodes_csv results_grid_search_kwta_20_10_burst/results_20251212_193341/results/CHUM__Patient_01/series/seiz_7_seq_006_nodes.csv \
  --out_png output_M1_good_outcome.png \
  --only_active_soz \
  --outcome GOOD


uv run z_visualise_activation_patterns_m1_output.py \
  --seq_csv M1_singleconfig_runs/results/results/ds004100__sub-HUP080/series/seiz_3_seq_002.csv \
  --nodes_csv M1_singleconfig_runs/results/results/ds004100__sub-HUP080/series/seiz_3_seq_002_nodes.csv \
  --out_png output_M1_bad_outcome.png \
  --outcome BAD


Optionnel:
  --t_true 5   (si l'onset true n'est pas dans le CSV)
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_meta_from_seq(seq_csv: Path) -> dict:
    meta = {}
    with open(seq_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            line = line[1:]
            parts = line.split(",", 1)
            if len(parts) == 2:
                k, v = parts[0].strip(), parts[1].strip()
                meta[k] = v
    return meta


def load_nodes(nodes_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv)
    required = {"node_index", "electrode_name", "is_SOZ"}
    if not required.issubset(df.columns):
        raise ValueError(f"[nodes] Colonnes inattendues: {list(df.columns)}")

    out = df[["node_index", "electrode_name", "is_SOZ"]].copy()
    out["node_index"] = pd.to_numeric(out["node_index"], errors="raise").astype(int)
    out["is_SOZ"] = pd.to_numeric(out["is_SOZ"], errors="coerce").fillna(0).astype(int).astype(bool)
    return out


def load_seq(seq_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(seq_csv, comment="#")
    if "t" not in df.columns:
        raise ValueError(f"[seq] Colonne 't' manquante. Colonnes: {list(df.columns)[:10]} ...")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    return df


def col_to_index(colname: str) -> int | None:
    try:
        left = colname.split(":", 1)[0]      # node_061
        return int(left.replace("node_", ""))
    except Exception:
        return None


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def plot_activation(
    seq_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    out_png: Path,
    meta: dict,
    title: str | None = None,
    t_true: float | None = None,
    t_pred: float | None = None,
    only_active_soz: bool = False,
):

    node_cols = [c for c in seq_df.columns if c.startswith("node_")]
    if not node_cols:
        raise ValueError("[plot] Aucune colonne node_* trouvée dans seq.csv")

    if "p_graph" not in seq_df.columns:
        raise ValueError("[plot] Colonne 'p_graph' absente du seq.csv (tu as dit que tu la voulais).")

    soz_map = dict(zip(nodes_df["node_index"].tolist(), nodes_df["is_SOZ"].tolist()))
    t = seq_df["t"].to_numpy()

    plt.figure(figsize=(12, 6), dpi=150)
    plt.grid(True, alpha=0.3)

    # channel-wise
    eps = 1e-6  # seuil pour "toujours à 0"

    for col in node_cols:
        idx = col_to_index(col)
        if idx is None:
            continue

        is_soz = bool(soz_map.get(idx, False))
        y = pd.to_numeric(seq_df[col], errors="coerce").to_numpy()

        # activité non nulle ?
        is_active = np.nanmax(np.abs(y)) > eps

        # règle finale
        plot_as_soz = is_soz and (is_active or not only_active_soz)

        if plot_as_soz:
            plt.plot(
                t, y,
                color="red",
                alpha=0.55,
                linewidth=1.8,
                zorder=2
            )
        else:
            plt.plot(
                t, y,
                color="gray",
                alpha=0.65,
                linewidth=1.0,
                zorder=1
            )


    # aggregated profile
    p_graph = pd.to_numeric(seq_df["p_graph"], errors="coerce").to_numpy()
    plt.plot(t, p_graph, color="black", linewidth=3.5, zorder=3)

    # y-range before annotations positions
    plt.ylim(0.0, 1.05)
    ymin, ymax = plt.gca().get_ylim()
    y_text = ymin + 0.6 * (ymax - ymin)

    # onset true (clinically annotated)
    if t_true is not None and np.isfinite(t_true):
        plt.axvline(t_true, color="red", linestyle="--", linewidth=2.2, zorder=4)
        plt.text(
            t_true + 0.05,
            y_text,
            "Clinically annotated seizure onset",
            color="red",
            rotation=90,
            va="center",
            ha="left",
            zorder=5
        )

    # onset predicted
    if t_pred is not None and np.isfinite(t_pred):
        plt.axvline(t_pred, color="black", linestyle="--", linewidth=2.2, zorder=4)
        plt.text(
            t_pred + 0.05,
            y_text,
            "Predicted seizure onset",
            color="black",
            rotation=90,
            va="center",
            ha="left",
            zorder=5
        )

    # title
    if title is None:
        patient = meta.get("patient", "")
        seizure_id = meta.get("seizure_id", "")
        title = (
            "Aggregated and Channel-Wise Ictal Activity Probabilities for Detecting Seizure Onset Time\n"
            f"(Output of Model 1 For {patient} (CHUM), Seizure {seizure_id})"
        )
    plt.title(title)

    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Ictal Activity (Activation Profile)")

    # legend: ONLY 3 items (no onset lines)
    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="SOZ electrodes"),
        Line2D([0], [0], color="gray", lw=2, label="nonSOZ electrodes"),
        Line2D([0], [0], color="black", lw=3.5, label="graph aggregated activation profile"),
    ]
    plt.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_csv", required=True)
    ap.add_argument("--nodes_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="", help="Override titre (sinon auto depuis meta)")
    ap.add_argument("--t_true", type=float, default=None, help="Override t_true (sinon meta #t_true)")
    ap.add_argument("--t_pred", type=float, default=None, help="Override t_pred (sinon meta #t_pred ou colonne t_pred)")
    ap.add_argument(
    "--only_active_soz",
    action="store_true",
    help="Afficher en rouge uniquement les électrodes SOZ dont l'activité n'est pas toujours nulle"
    )
    ap.add_argument("--outcome", required=True)

    args = ap.parse_args()

    seq_csv = Path(args.seq_csv).expanduser()
    nodes_csv = Path(args.nodes_csv).expanduser()
    out_png = Path(args.out_png).expanduser()
    outcome = args.outcome
    print(outcome)

    if not seq_csv.is_file():
        raise SystemExit(f"seq_csv introuvable: {seq_csv}")
    if not nodes_csv.is_file():
        raise SystemExit(f"nodes_csv introuvable: {nodes_csv}")

    meta = read_meta_from_seq(seq_csv)
    nodes_df = load_nodes(nodes_csv)
    seq_df = load_seq(seq_csv)

    # t_true
    t_true = args.t_true
    if t_true is None:
        t_true = _safe_float(meta.get("t_true"))

    # t_pred (priorité: override CLI > meta > colonne)
    t_pred = args.t_pred
    if t_pred is None:
        t_pred = _safe_float(meta.get("t_pred"))
    if t_pred is None and "t_pred" in seq_df.columns:
        # parfois constant sur toutes les lignes
        vals = pd.to_numeric(seq_df["t_pred"], errors="coerce").dropna().unique()
        if len(vals) > 0:
            t_pred = _safe_float(vals[0])

    title = args.title.strip() or ("Channel-Wise and Aggregated Ictal Activity Probabilities for Detecting Seizure Onset Time\n"
            f"(Output of Model 1 For One Seizure of One {outcome} Surgery Outcome Patient)")

    plot_activation(
        seq_df=seq_df,
        nodes_df=nodes_df,
        out_png=out_png,
        meta=meta,
        title=title,
        t_true=t_true,
        t_pred=t_pred,
        only_active_soz=args.only_active_soz,
    )


    print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    main()
