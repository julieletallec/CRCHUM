#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Appliquer M2 (pré-entraîné) sur les outputs M1 au format:
  <m1_root>/results/<PATIENT>/series/seiz_<id>_seq_<idx>.csv (+ _nodes.csv)

M2 files attendus:
  <m2_dir>/model_M2.joblib
  <m2_dir>/selected_features.txt
  <m2_dir>/artifacts.json (delta_used)

Usage:
uv run 13_run_best_M2_on_all_bad_patients.py \
  --m1_root /home/julieletallec/test/M1_singleconfig_runs/results/results \
  --m2_dir /home/julieletallec/test/M2_singlefit_out \
  --out /home/julieletallec/test/M2_on_M1_outputs/preds_ranked.csv

Optionnel:
  --patients "CHUM__Patient_09,ds004100__sub-HUP074"
  --patients_file /path/to/patients.txt
  --select_seq middle|first|all   (default: middle)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy


SEIZ_RE = re.compile(r"^seiz_(?P<seiz_id>[0-9]+|\?)_seq_(?P<seq_idx>[0-9]+)\.(?P<ext>csv|npz)$", re.IGNORECASE)


# -----------------------
# Utils
# -----------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def parse_patients_list(patients_arg: str) -> List[str]:
    if not patients_arg:
        return []
    s = patients_arg.replace("\n", " ").replace(",", " ").strip()
    return [x for x in (p.strip() for p in s.split(" ")) if x]


def read_patients_file(path: Path) -> List[str]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_selected_features(m2_dir: Path) -> List[str]:
    sel = m2_dir / "selected_features.txt"
    if not sel.is_file():
        raise SystemExit(f"selected_features.txt introuvable dans {m2_dir}")
    feats = [l.strip() for l in sel.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not feats:
        raise SystemExit(f"selected_features.txt est vide: {sel}")
    return feats


def read_delta_used(m2_dir: Path) -> float:
    art = m2_dir / "artifacts.json"
    if not art.is_file():
        return 0.0
    obj = load_json(art)

    for k in ["delta_used", "delta", "best_delta", "logit_shift_delta", "postprocess_delta", "delta_logit_shift"]:
        if k in obj and obj[k] is not None:
            try:
                return float(obj[k])
            except Exception:
                pass

    for k in ["postprocess", "logit_shift", "calibration", "inference"]:
        if k in obj and isinstance(obj[k], dict):
            for kk in ["delta_used", "delta", "best_delta", "logit_shift_delta"]:
                if kk in obj[k] and obj[k][kk] is not None:
                    try:
                        return float(obj[k][kk])
                    except Exception:
                        pass

    return 0.0


def load_m2_joblib(m2_dir: Path):
    model_path = m2_dir / "model_M2.joblib"
    if not model_path.is_file():
        raise SystemExit(f"Modèle introuvable: {model_path}")
    import joblib
    return joblib.load(model_path)


def predict_raw_sklearn(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s, dtype=float).reshape(-1)

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    p = model.predict(X)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# -----------------------
# Scan + read "series" files
# -----------------------

class Entry:
    def __init__(self, patient: str, seizure_id: str, seq_idx: int, data_path: Path, nodes_path: Optional[Path]):
        self.patient = patient
        self.seizure_id = seizure_id
        self.seq_idx = seq_idx
        self.data_path = data_path
        self.nodes_path = nodes_path


def parse_series_filename(p: Path) -> Optional[Tuple[str, int]]:
    m = SEIZ_RE.match(p.name)
    if not m:
        return None
    return m.group("seiz_id"), int(m.group("seq_idx"))


def scan_series(m1_root: Path, patients: List[str]) -> List[Entry]:
    """
    m1_root attendu:
      <m1_root>/<patient>/series/*.csv

    Si patients vide: on prend tous les sous-dossiers.
    """
    if not patients:
        patients = sorted([p.name for p in m1_root.iterdir() if p.is_dir()])

    out: List[Entry] = []
    for pat in patients:
        pdir = m1_root / pat
        if not pdir.is_dir():
            # fallback si on a donné CHUM::Patient_09 mais dossier CHUM__Patient_09
            pdir2 = m1_root / pat.replace("::", "__")
            if pdir2.is_dir():
                pdir = pdir2
            else:
                print(f"[WARN] patient dir introuvable: {pat}")
                continue

        sdir = pdir / "series"
        if not sdir.is_dir():
            print(f"[WARN] series/ introuvable pour {pat}: {sdir}")
            continue

        for csv_path in sorted(sdir.glob("seiz_*_seq_*.csv")):
            if csv_path.name.endswith("_nodes.csv"):
                continue
            parsed = parse_series_filename(csv_path)
            if not parsed:
                continue
            seiz_id, seq_idx = parsed
            nodes = csv_path.with_name(csv_path.stem + "_nodes.csv")
            out.append(Entry(pat, seiz_id, seq_idx, csv_path, nodes if nodes.is_file() else None))

    return out


def choose_sequences(entries: List[Entry], mode: str) -> List[Entry]:
    """
    Grouper par (patient, seizure_id).
    - middle: si 3 -> seq au milieu; sinon prend le milieu global
    - first: prend la plus petite seq
    - all: garde toutes
    """
    if mode == "all":
        return entries

    by: Dict[Tuple[str, str], List[Entry]] = {}
    for e in entries:
        by.setdefault((e.patient, e.seizure_id), []).append(e)

    chosen: List[Entry] = []
    for (pat, seiz), lst in by.items():
        lst.sort(key=lambda x: x.seq_idx)
        if mode == "first":
            chosen.append(lst[0])
        else:  # middle
            n = len(lst)
            if n == 3:
                chosen.append(lst[1])
            else:
                chosen.append(lst[n // 2])
    return chosen


def read_nodes_csv(nodes_path: Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_path)
    # ton fichier nodes est souvent: node_index, electrode_name, is_SOZ
    # mais on tolère quelques variantes
    if "node_index" not in df.columns:
        if "electrode_id" in df.columns:
            df = df.rename(columns={"electrode_id": "node_index"})
    if "electrode_name" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "electrode_name"})
    if "is_SOZ" not in df.columns:
        if "soz" in df.columns:
            df = df.rename(columns={"soz": "is_SOZ"})
    if "node_index" not in df.columns or "electrode_name" not in df.columns:
        raise ValueError(f"{nodes_path} ne contient pas node_index/electrode_name")
    if "is_SOZ" in df.columns:
        df["is_SOZ"] = pd.to_numeric(df["is_SOZ"], errors="coerce")
    df["node_index"] = pd.to_numeric(df["node_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["node_index"])
    df["node_index"] = df["node_index"].astype(int)
    return df


def _detect_header_and_delimiter(path: Path, max_probe_lines: int = 200):
    delimiters = [",", ";", "\t"]
    best = {"score": -1, "line_idx": None, "delim": None, "tokens": None}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [next(f, "") for _ in range(max_probe_lines)]

    for i, raw in enumerate(lines):
        if not raw:
            continue
        for delim in delimiters:
            toks = [t.strip().strip('"').strip("'") for t in raw.strip().split(delim)]
            score = sum(1 for t in toks if t.lower().startswith("node_"))
            if score > best["score"]:
                best = {"score": score, "line_idx": i, "delim": delim, "tokens": toks}

    if best["score"] <= 0 or best["line_idx"] is None:
        raise ValueError(f"Impossible de détecter des colonnes node_* dans {path.name}")
    return best["line_idx"], best["delim"], best["tokens"]


def read_signal_csv(data_path: Path):
    """
    Lit un CSV series du type:
      node_000:SEEG X1, node_001:SEEG X2, ...
    Retourne (vals_df, node_indices, node_names)
    """
    header_idx, sep, header_tokens = _detect_header_and_delimiter(data_path)

    first_node_col = None
    node_specs = []
    for j, tok in enumerate(header_tokens):
        tok_clean = tok.strip().strip('"').strip("'")
        if tok_clean.lower().startswith("node_"):
            if first_node_col is None:
                first_node_col = j
            if ":" in tok_clean:
                left, name = tok_clean.split(":", 1)
            else:
                left, name = tok_clean, tok_clean
            m = re.search(r"\d+", left)
            if not m:
                continue
            idx = int(m.group())
            node_specs.append((idx, name.strip()))

    if first_node_col is None or not node_specs:
        raise ValueError(f"Aucune colonne node_* extraite dans {data_path.name}")

    df = pd.read_csv(
        data_path,
        header=None,
        skiprows=header_idx + 1,
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    df_vals = df.iloc[:, first_node_col:].copy()

    if df_vals.shape[1] != len(node_specs):
        m = min(df_vals.shape[1], len(node_specs))
        df_vals = df_vals.iloc[:, :m]
        node_specs = node_specs[:m]

    node_indices = [idx for idx, _ in node_specs]
    node_names = [name for _, name in node_specs]
    df_vals = df_vals.apply(pd.to_numeric, errors="coerce")

    return df_vals, node_indices, node_names


def build_long_dataset(selected_entries: List[Entry]) -> pd.DataFrame:
    """
    Construit:
      patient, seizure_id, seq_idx, node_index, electrode_name, is_SOZ, t, value
    value = p_node(t)
    """
    rows = []

    for fe in selected_entries:
        nodes_df = None
        if fe.nodes_path is not None and fe.nodes_path.exists():
            nodes_df = read_nodes_csv(fe.nodes_path)

        vals_df, node_indices, node_names = read_signal_csv(fe.data_path)

        # mapping meta
        meta_by_node: Dict[int, Tuple[str, Optional[int]]] = {idx: (nm, None) for idx, nm in zip(node_indices, node_names)}
        if nodes_df is not None:
            for _, r in nodes_df.iterrows():
                idx = int(r["node_index"])
                nm = str(r["electrode_name"])
                soz = None
                if "is_SOZ" in r and not pd.isna(r["is_SOZ"]):
                    soz = int(r["is_SOZ"])
                meta_by_node[idx] = (nm, soz)

        T, _ = vals_df.shape
        for col_i, node_idx in enumerate(node_indices):
            col_vals = vals_df.iloc[:, col_i].values
            el_name, el_is_soz = meta_by_node.get(node_idx, (f"node_{node_idx}", None))
            rows.append(pd.DataFrame({
                "patient": fe.patient,
                "seizure_id": fe.seizure_id,
                "seq_idx": fe.seq_idx,
                "node_index": node_idx,
                "electrode_name": el_name,
                "is_SOZ": el_is_soz,
                "t": np.arange(T, dtype=int),
                "value": col_vals
            }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -----------------------
# Feature extraction
# -----------------------

def extract_features(df_long: pd.DataFrame) -> pd.DataFrame:
    feats = []
    group_cols = ["patient", "seizure_id", "seq_idx", "node_index", "electrode_name", "is_SOZ"]

    for key, sub in df_long.groupby(group_cols):
        patient, seiz_id, seq_idx, node_idx, el_name, is_soz = key
        values = sub["value"].astype(float).values
        t = sub["t"].astype(float).values
        n = len(values)
        if n < 3:
            continue

        vmean = float(np.nanmean(values))
        vstd  = float(np.nanstd(values))
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        q25, q75 = np.nanpercentile(values, [25, 75])
        iqr = float(q75 - q25)
        vmedian = float(np.nanmedian(values))
        vrange  = float(vmax - vmin)

        slopes = np.diff(values)
        slope_mean = float(np.nanmean(slopes)) if slopes.size else np.nan
        slope_std  = float(np.nanstd(slopes)) if slopes.size else np.nan

        try:
            area = float(np.trapezoid(values, t))
        except Exception:
            area = float(np.trapz(values))

        argmax = int(np.nanargmax(values))
        time_to_peak = float(t[argmax])

        try:
            t10 = t[np.where(values >= 0.1 * vmax)[0][0]]
            t90 = t[np.where(values >= 0.9 * vmax)[0][0]]
            rise_time = float(t90 - t10)
        except Exception:
            rise_time = np.nan

        try:
            after_peak = values[argmax:]
            t_after = t[argmax:]
            t90post = t_after[np.where(after_peak <= 0.9 * vmax)[0][0]]
            t50post = t_after[np.where(after_peak <= 0.5 * vmax)[0][0]]
            decay_time = float(t50post - t90post)
        except Exception:
            decay_time = np.nan

        fft_vals = np.abs(np.fft.rfft(values - np.nanmean(values)))
        fft_freqs = np.fft.rfftfreq(n, d=1)
        fft_peak_freq = float(fft_freqs[np.argmax(fft_vals)]) if fft_vals.size else np.nan
        fft_energy = float(np.nansum(fft_vals**2)) if fft_vals.size else np.nan
        p_spec = fft_vals / (np.nansum(fft_vals) + 1e-12) if fft_vals.size else np.array([1.0])
        spectral_entropy = float(entropy(p_spec + 1e-12))

        feats.append({
            "patient": patient,
            "seizure_id": seiz_id,
            "seq_idx": seq_idx,
            "node_index": int(node_idx),
            "electrode_name": el_name,
            "is_SOZ": is_soz,

            "mean": vmean, "std": vstd, "median": vmedian,
            "min": vmin, "max": vmax, "range": vrange, "iqr": iqr,
            "slope_mean": slope_mean, "slope_std": slope_std,
            "area": area, "time_to_peak": time_to_peak,
            "rise_time": rise_time, "decay_time": decay_time,
            "fft_peak_freq": fft_peak_freq, "fft_energy": fft_energy,
            "spectral_entropy": spectral_entropy,
        })

    return pd.DataFrame(feats)


def add_relative_features(df_feats: pd.DataFrame,
                          group_keys=("patient", "seizure_id", "seq_idx")) -> pd.DataFrame:
    df = df_feats.copy()
    group_keys = list(group_keys)
    meta_cols = set(group_keys + ["node_index", "electrode_name", "is_SOZ"])

    base_feature_cols = [
        c for c in df.columns
        if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    g = df.groupby(group_keys)

    for col in base_feature_cols:
        mu = g[col].transform("mean")
        sd = g[col].transform("std").replace(0, np.nan)
        rank = g[col].rank(method="average", pct=True)

        df[f"{col}_rel"] = df[col] - mu
        df[f"{col}_z"] = (df[col] - mu) / (sd + 1e-6)
        df[f"{col}_rank"] = rank

    return df


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_root", required=True, help="Dossier qui contient les dossiers patients (CHUM__Patient_XX) avec series/")
    ap.add_argument("--m2_dir", required=True, help="Dossier M2_singlefit_out")
    ap.add_argument("--patients", default="", help="Liste patients: 'a,b,c' ou 'a b c'")
    ap.add_argument("--patients_file", default="", help="Fichier txt: 1 patient par ligne")
    ap.add_argument("--select_seq", default="middle", choices=["middle", "first", "all"],
                    help="Quelle(s) séquence(s) utiliser par (patient,seizure). default=middle")
    ap.add_argument("--out", required=True, help="CSV de sortie (ranked)")
    args = ap.parse_args()

    m1_root = Path(args.m1_root).expanduser()
    m2_dir = Path(args.m2_dir).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not m1_root.is_dir():
        raise SystemExit(f"m1_root introuvable: {m1_root}")
    if not m2_dir.is_dir():
        raise SystemExit(f"m2_dir introuvable: {m2_dir}")

    # patients
    patients: List[str] = []
    if args.patients_file.strip():
        pf = Path(args.patients_file).expanduser()
        patients.extend(read_patients_file(pf))
    patients.extend(parse_patients_list(args.patients))

    # scan
    entries = scan_series(m1_root, patients)
    if not entries:
        raise SystemExit("Aucun fichier seiz_*_seq_*.csv trouvé. Vérifie m1_root.")

    selected_entries = choose_sequences(entries, args.select_seq)
    print(f"[INFO] entries total: {len(entries)} | selected: {len(selected_entries)} (mode={args.select_seq})")

    # build long + features
    df_long = build_long_dataset(selected_entries)
    if df_long.empty:
        raise SystemExit("df_long vide après lecture des series.")

    df_feats = extract_features(df_long)
    if df_feats.empty:
        raise SystemExit("df_feats vide (features).")

    df_feats = add_relative_features(df_feats)

    # M2
    selected_feats = load_selected_features(m2_dir)
    delta = read_delta_used(m2_dir)
    model = load_m2_joblib(m2_dir)

    # X
    for c in selected_feats:
        if c not in df_feats.columns:
            df_feats[c] = np.nan

    X_df = df_feats[selected_feats].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    med = X_df.median(axis=0, numeric_only=True)
    X_df = X_df.fillna(med)
    X = X_df.astype(float).values

    raw = predict_raw_sklearn(model, X)
    p = sigmoid(raw + float(delta))

    out_df = df_feats[["patient", "seizure_id", "seq_idx", "node_index", "electrode_name", "is_SOZ"]].copy()
    out_df["m2_raw"] = raw
    out_df["m2_score"] = p

    # ranks
    gcols = ["patient", "seizure_id", "seq_idx"]
    out_df = out_df.sort_values(gcols + ["m2_score"], ascending=[True, True, True, False])
    out_df["rank"] = out_df.groupby(gcols)["m2_score"].rank(method="first", ascending=False).astype(int)

    out_df.to_csv(out_path, index=False)
    print(f"[DONE] preds ranked -> {out_path}")
    print(f"[INFO] delta_used={delta:.6f} | features={len(selected_feats)} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
