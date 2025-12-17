#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy

# -------------------------------
# Parsing & scanning
# -------------------------------

SEIZURE_RE = re.compile(
    r"^seiz_(?P<seiz_id>[0-9]+|\?)+_seq_(?P<seq_idx>[0-9]+)\.(?P<ext>csv|npz)$",
    re.IGNORECASE,
)

def parse_filename(p: Path) -> Optional[Tuple[str, int]]:
    m = SEIZURE_RE.match(p.name)
    if not m:
        return None
    seiz_id = m.group("seiz_id")
    seq_idx = int(m.group("seq_idx"))
    return seiz_id, seq_idx

def find_series_root(patient_dir: Path) -> Path:
    series = patient_dir / "series"
    if not series.is_dir():
        raise FileNotFoundError(f"Répertoire 'series' introuvable sous {patient_dir}")
    return series

class FileEntry:
    def __init__(self, patient: str, seizure_id: str, seq_idx: int, data_path: Path, nodes_path: Optional[Path]):
        self.patient = patient
        self.seizure_id = str(seizure_id)
        self.seq_idx = int(seq_idx)
        self.data_path = data_path
        self.nodes_path = nodes_path

def scan_files(results_root: Path, patients: List[str], exts: List[str]) -> List[FileEntry]:
    entries: List[FileEntry] = []
    if not patients:
        patients = sorted([p.name for p in results_root.iterdir() if p.is_dir()])

    for patient in patients:
        patient_dir = results_root / patient
        if not patient_dir.is_dir():
            print(f"[WARN] patient '{patient}' introuvable sous {results_root}", file=sys.stderr)
            continue

        series_root = find_series_root(patient_dir)
        for ext in exts:
            for data_path in series_root.glob(f"seiz_*_seq_*.{ext}"):
                if data_path.name.endswith("_nodes.csv"):
                    continue
                parsed = parse_filename(data_path)
                if not parsed:
                    continue
                seiz_id, seq_idx = parsed
                nodes_candidate = data_path.with_name(data_path.stem + "_nodes.csv")
                nodes_path = nodes_candidate if nodes_candidate.exists() else None
                entries.append(FileEntry(patient, seiz_id, seq_idx, data_path, nodes_path))
    return entries

# -------------------------------
# Sélection "séquence du milieu"
# -------------------------------

def choose_middle_sequences(entries: List[FileEntry]) -> List[FileEntry]:
    selected: List[FileEntry] = []
    by_key: Dict[Tuple[str, str], List[FileEntry]] = {}
    for e in entries:
        by_key.setdefault((e.patient, e.seizure_id), []).append(e)

    for key, files in by_key.items():
        files.sort(key=lambda x: x.seq_idx)
        n = len(files)
        if n == 3:
            selected.append(files[1])
        elif n % 3 == 0:
            for i in range(0, n, 3):
                chunk = files[i:i+3]
                if len(chunk) == 3:
                    selected.append(chunk[1])
        else:
            mid = n // 2
            print(f"[WARN] {key} a {n} séquences; sélection milieu global: {files[mid].data_path.name}", file=sys.stderr)
            selected.append(files[mid])
    return selected




from typing import List, Dict, Tuple

def choose_first_sequences(entries: List[FileEntry]) -> List[FileEntry]:
    """
    Sélectionne la *première* séquence (plus petit seq_idx) pour chaque (patient, seizure_id).

    - Si un groupe contient 1 seul fichier → on le prend.
    - Si un groupe contient n fichiers → on prend simplement celui avec seq_idx minimal.
    """
    selected: List[FileEntry] = []
    by_key: Dict[Tuple[str, str], List[FileEntry]] = {}

    # Regroupement par (patient, seizure)
    for e in entries:
        by_key.setdefault((e.patient, e.seizure_id), []).append(e)

    # Pour chaque groupe → prendre la première séquence
    for key, files in by_key.items():
        # tri par seq_idx
        files.sort(key=lambda x: x.seq_idx)

        # première séquence
        first_file = files[0]
        selected.append(first_file)

    return selected


# -------------------------------
# I/O des CSV
# -------------------------------

def read_nodes_csv(nodes_path: Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_path)
    expected = {"node_index", "electrode_name", "is_SOZ"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{nodes_path} ne contient pas les colonnes attendues: {missing}")
    df["node_index"] = df["node_index"].astype(int)
    df["is_SOZ"] = df["is_SOZ"].astype(int)
    return df

from pathlib import Path
import pandas as pd
import numpy as np
import re

def _detect_header_and_delimiter(path: Path, max_probe_lines: int = 100):
    """
    Trouve:
      - l'index de la ligne d'entête (celle avec le plus de tokens 'node_*')
      - le séparateur (',', ';' ou '\\t')
      - les tokens nettoyés de cette ligne
    Retourne (header_line_idx, delimiter, header_tokens_cleaned)
    """
    delimiters = [",", ";", "\t"]
    best = {"score": -1, "line_idx": None, "delim": None, "tokens": None}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [next(f, "") for _ in range(max_probe_lines)]

    for i, raw in enumerate(lines):
        if not raw:
            continue
        for delim in delimiters:
            toks = [t.strip().strip('"').strip("'") for t in raw.strip().split(delim)]
            # score = combien de tokens commencent par node_
            score = sum(1 for t in toks if t.lower().startswith("node_"))
            if score > best["score"]:
                best = {"score": score, "line_idx": i, "delim": delim, "tokens": toks}

    if best["score"] <= 0 or best["line_idx"] is None:
        raise ValueError(
            f"Impossible de détecter une ligne d'entête avec des colonnes 'node_*' dans {path.name} "
            f"(scanné {max_probe_lines} lignes)."
        )
    return best["line_idx"], best["delim"], best["tokens"]


def read_signal_csv(data_path: Path):
    """
    Version robuste:
    - détecte la ligne d'entête et le séparateur
    - accepte des colonnes du type 'node_000:SEEG F23'
    Retourne: (df_vals, node_indices, node_names)
    """
    header_idx, sep, header_tokens = _detect_header_and_delimiter(data_path)

    # repère la 1re colonne 'node_*' et construit node_indices/names
    first_node_col = None
    node_specs = []  # [(index, name), ...]
    for j, tok in enumerate(header_tokens):
        tok_clean = tok.strip().strip('"').strip("'")
        if tok_clean.lower().startswith("node_"):
            if first_node_col is None:
                first_node_col = j
            # "node_044:SEEG G423" -> index=44, name="SEEG G423"
            if ":" in tok_clean:
                left, name = tok_clean.split(":", 1)
            else:
                left, name = tok_clean, tok_clean  # fallback
            m = re.search(r"\d+", left)
            if not m:
                continue
            idx = int(m.group())
            node_specs.append((idx, name.strip()))
    if first_node_col is None or not node_specs:
        raise ValueError(
            f"Entête détectée mais aucune colonne 'node_*' n'a été extraite dans {data_path.name}.\n"
            f"Tokens (début): {header_tokens[:10]}"
        )

    # lit le reste du fichier à partir de la ligne suivant l'entête
    df = pd.read_csv(
        data_path,
        header=None,
        skiprows=header_idx + 1,
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )

    # garde uniquement les colonnes node_* (à partir de first_node_col)
    df_vals = df.iloc[:, first_node_col:].copy()

    # sécurité si des colonnes manquent ou en trop
    if df_vals.shape[1] != len(node_specs):
        m = min(df_vals.shape[1], len(node_specs))
        df_vals = df_vals.iloc[:, :m]
        node_specs = node_specs[:m]

    node_indices = [idx for idx, _ in node_specs]
    node_names = [name for _, name in node_specs]
    df_vals = df_vals.apply(pd.to_numeric, errors="coerce")
    return df_vals, node_indices, node_names

# -------------------------------
# Construction du dataset long
# -------------------------------
def add_relative_features(
    df_feats: pd.DataFrame,
    group_keys = ("patient", "seizure_id", "seq_idx"),
) -> pd.DataFrame:
    """
    Ajoute des features relatives intra-groupe (patient, seizure, seq).

    Pour chaque feature f (ex: mean, std, time_to_peak, ...),
    on ajoute :
      - f_rel  = f - mean_groupe(f)
      - f_z    = (f - mean_groupe(f)) / std_groupe(f)
      - f_rank = rang de f dans le groupe (entre 0 et 1)

    On ne touche pas aux colonnes meta (patient, node_index, is_SOZ, etc.).
    """
    df = df_feats.copy()

    group_keys = list(group_keys)
    meta_cols = set(group_keys + ["node_index", "electrode_name", "is_SOZ", "is_flat_zero"])
    # on garde uniquement les features numériques "pures"
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


def build_long_dataset(selected_files: List[FileEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    manifest_rows = []

    for fe in selected_files:
        nodes_df = None
        if fe.nodes_path is not None and fe.nodes_path.exists():
            nodes_df = read_nodes_csv(fe.nodes_path)

        if fe.data_path.suffix.lower() == ".csv":
            vals_df, node_indices, node_names = read_signal_csv(fe.data_path)
        else:
            raise NotImplementedError(f"Format {fe.data_path.suffix} non implémenté")

        # mapping meta
        meta_by_node: Dict[int, Tuple[str, Optional[int]]] = {
            idx: (nm, None) for idx, nm in zip(node_indices, node_names)
        }
        if nodes_df is not None:
            for _, r in nodes_df.iterrows():
                idx = int(r["node_index"])
                nm = str(r["electrode_name"])
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
                "t": np.arange(T),
                "value": col_vals
            }))

        manifest_rows.append({
            "patient": fe.patient,
            "seizure_id": fe.seizure_id,
            "selected_seq_idx": fe.seq_idx,
            "data_file": str(fe.data_path),
            "nodes_file": str(fe.nodes_path) if fe.nodes_path else None,
        })

    df_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    return df_long, manifest

# -------------------------------
# Feature engineering
# -------------------------------


def extract_features(df_long: pd.DataFrame) -> pd.DataFrame:
    feats = []
    group_cols = ["patient", "seizure_id", "seq_idx", "node_index", "electrode_name", "is_SOZ"]

    for key, sub in df_long.groupby(group_cols):
        patient, seiz_id, seq_idx, node_idx, el_name, is_soz = key

        values = sub["value"].astype(float).values
        t = sub["t"].values
        n = len(values)
        if n < 3:
            continue

        # <- nouveau: détecter électrodes SOZ full zéro
        is_flat_zero = False
        if is_soz == 1 and np.nanmax(np.abs(values)) == 0:
            is_flat_zero = True
            print(patient, seiz_id, el_name)

        # stats
        vmean = float(np.nanmean(values))
        vstd  = float(np.nanstd(values))
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        q25, q75 = np.nanpercentile(values, [25, 75])
        iqr = float(q75 - q25)
        vmedian = float(np.nanmedian(values))
        vrange  = float(vmax - vmin)

        slopes = np.diff(values)
        slope_mean = float(np.nanmean(slopes))
        slope_std  = float(np.nanstd(slopes))
        area = float(np.trapezoid(values, t))
        argmax = int(np.argmax(values))
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
        fft_peak_freq = float(fft_freqs[np.argmax(fft_vals)])
        fft_energy = float(np.nansum(fft_vals**2))
        p_spec = fft_vals / (np.nansum(fft_vals) + 1e-12)
        spectral_entropy = float(entropy(p_spec + 1e-12))

        feats.append({
            "patient": patient,
            "seizure_id": seiz_id,
            "seq_idx": seq_idx,
            "node_index": node_idx,
            "electrode_name": el_name,
            "is_SOZ": is_soz,
            "is_flat_zero": is_flat_zero,  # <- NEW

            "mean": vmean, "std": vstd, "median": vmedian,
            "min": vmin, "max": vmax, "range": vrange, "iqr": iqr,
            "slope_mean": slope_mean, "slope_std": slope_std,
            "area": area, "time_to_peak": time_to_peak,
            "rise_time": rise_time, "decay_time": decay_time,
            "fft_peak_freq": fft_peak_freq, "fft_energy": fft_energy,
            "spectral_entropy": spectral_entropy,
        })

    return pd.DataFrame(feats)



def extract_features_(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    1 ligne = Patient × Seizure × Sequence × Electrode
    Features calculées sur p_v(t).
    """
    feats = []
    group_cols = ["patient", "seizure_id", "seq_idx", "node_index", "electrode_name", "is_SOZ"]

    for key, sub in df_long.groupby(group_cols):
        values = sub["value"].astype(float).values
        t = sub["t"].values
        n = len(values)
        if n < 3:
            continue














        # stats
        vmean = float(np.nanmean(values))
        vstd  = float(np.nanstd(values))
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        q25, q75 = np.nanpercentile(values, [25, 75])
        iqr = float(q75 - q25)
        vmedian = float(np.nanmedian(values))
        vrange  = float(vmax - vmin)

        # dynamique
        slopes = np.diff(values)
        slope_mean = float(np.nanmean(slopes))
        slope_std  = float(np.nanstd(slopes))
        area = float(np.trapezoid(values, t))
        argmax = int(np.argmax(values))
        time_to_peak = float(t[argmax])

        # montée 10->90%
        try:
            t10 = t[np.where(values >= 0.1 * vmax)[0][0]]
            t90 = t[np.where(values >= 0.9 * vmax)[0][0]]
            rise_time = float(t90 - t10)
        except Exception:
            rise_time = np.nan

        # descente 90->50% après le pic
        try:
            after_peak = values[argmax:]
            t_after = t[argmax:]
            t90post = t_after[np.where(after_peak <= 0.9 * vmax)[0][0]]
            t50post = t_after[np.where(after_peak <= 0.5 * vmax)[0][0]]
            decay_time = float(t50post - t90post)
        except Exception:
            decay_time = np.nan

        # fréquence / complexité
        fft_vals = np.abs(np.fft.rfft(values - np.nanmean(values)))
        fft_freqs = np.fft.rfftfreq(n, d=1)
        fft_peak_freq = float(fft_freqs[np.argmax(fft_vals)])
        fft_energy = float(np.nansum(fft_vals**2))
        p_spec = fft_vals / (np.nansum(fft_vals) + 1e-12)
        spectral_entropy = float(entropy(p_spec + 1e-12))

        feats.append({
            **dict(zip(group_cols, key)),
            "mean": vmean, "std": vstd, "median": vmedian,
            "min": vmin, "max": vmax, "range": vrange, "iqr": iqr,
            "slope_mean": slope_mean, "slope_std": slope_std,
            "area": area, "time_to_peak": time_to_peak,
            "rise_time": rise_time, "decay_time": decay_time,
            "fft_peak_freq": fft_peak_freq, "fft_energy": fft_energy,
            "spectral_entropy": spectral_entropy,
        })

    return pd.DataFrame(feats)

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline: scan + séquence du milieu + dataset long + features")
    ap.add_argument("--config", required=True, help="Chemin du config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_root = Path(cfg["results_root"]).expanduser()
    patients = list(cfg.get("patients", []))
    exts = list(cfg.get("file_extensions", ["csv"]))
    out_dir = Path(cfg.get("output_dir", results_root)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_long = cfg.get("save_long_as", "dataset_long.parquet")
    save_feats = cfg.get("save_features_as", "features.parquet")
    manifest_name = cfg.get("manifest_name", "selection_manifest.csv")

    # 1) scan
    entries = scan_files(results_root, patients, exts)
    if not entries:
        print("Aucun fichier trouvé. Vérifie 'results_root' et 'patients' dans config.yaml.", file=sys.stderr)
        sys.exit(2)

    # 2) séquence du milieu
    selected = choose_middle_sequences(entries)
    #selected = choose_first_sequences(entries)

    # 3) dataset long
    df_long, manifest = build_long_dataset(selected)

    # 4) sauvegarder dataset long
    long_path = out_dir / save_long
    try:
        if long_path.suffix.lower() == ".parquet":
            df_long.to_parquet(long_path, index=False)
        else:
            df_long.to_csv(long_path, index=False)
        print(f"[OK] dataset long -> {long_path}")
    except Exception as e:
        print(f"[WARN] écriture dataset long: {e}", file=sys.stderr)

    # 5) manifest
    manifest_path = out_dir / manifest_name
    manifest.to_csv(manifest_path, index=False)
    print(f"[OK] manifest -> {manifest_path}")

    # 6) features
    df_feats = extract_features(df_long)
    # 7) ajout des features relatives intra-groupe
    df_feats = add_relative_features(df_feats)

    feats_path = out_dir / save_feats
    try:
        if feats_path.suffix.lower() == ".parquet":
            df_feats.to_parquet(feats_path, index=False)
        else:
            df_feats.to_csv(feats_path, index=False)
        print(f"[OK] features -> {feats_path}")
    except Exception as e:
        print(f"[WARN] écriture features: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
